"""Executable buyer-agent evaluation loop for H-MOD seller-drift scenarios."""

import json
import os
import time
from typing import Any, Dict, List, Optional

from loguru import logger

from hmod.judge import judge_deal
from hmod.llm_reflection import LLMWeightReflector
from hmod.metrics import aggregate_dialogue_metrics, compute_cvr, compute_gsr, compute_t2da, subgroup_metrics
from hmod.objectives import BuyerObjectiveLibrary
from hmod.policy import LLMReflectionMetaController, RuleBuyerPolicy, RuleMetaController
from hmod.scenario import HMODScenario, load_scenarios
from hmod.simulator import DynamicSellerNegotiationSimulator, first_price


def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _initial_state(scenario: HMODScenario) -> Dict[str, Any]:
    case = scenario.case
    seller_initial = round(
        float(case["buyer_price"])
        + scenario.seller_persona.initial_ask_ratio
        * (float(case["seller_price"]) - float(case["buyer_price"])),
        2,
    )
    return {
        "task_background": {
            "item_name": case["item_name"],
            "buyer_price": float(case["buyer_price"]),
            "buyer_item_description": case["buyer_item_description"],
            "seller_price": float(case["seller_price"]),
            "seller_item_description": case["seller_item_description"],
        },
        "dialogue_context": [
            {
                "role": "assistant",
                "content": f"Hi, how much is the {case['item_name']}?",
            },
            {
                "role": "user",
                "content": (
                    f"Hi, this is a good {case['item_name']} and I am asking "
                    f"about ${seller_initial:.0f}."
                ),
            },
        ],
        "pre_goals": [],
        "pre_topics": [],
        "turn_id": 0,
        "last_seller_price": seller_initial,
    }


def _is_terminal(dialogue: List[Dict[str, str]]) -> bool:
    if not dialogue:
        return False
    last_turn = dialogue[-1]
    last = (last_turn.get("content", "") or "").lower()
    if (
        last_turn.get("role") == "user"
        and "if you cannot" in last
        and ("will pass" in last or "sell elsewhere" in last)
    ):
        return False
    terminal_markers = [
        "deal",
        "i can buy",
        "i'll buy",
        "i will buy",
        "i can sell",
        "sell it for",
        "sell to someone else",
        "will pass",
        "i will pass",
        "not going anywhere",
    ]
    return any(marker in last for marker in terminal_markers)


class HMODEvaluator:
    def __init__(
        self,
        mode: str,
        judge_model: str = "rule",
        use_llm_simulator: bool = False,
        audit_sample_size: int = 50,
        objective_file: Optional[str] = None,
        objective_id: Optional[str] = None,
        reflection_horizon: int = 3,
        controller_mode: str = "rule_scaffold",
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_api_key_env: str = "DEEPINFRA_API_KEY",
        llm_base_url: Optional[str] = None,
        llm_temperature: float = 0.0,
        llm_max_tokens: int = 500,
        llm_fallback_to_rule: bool = False,
        buyer_policy: Optional[Any] = None,
        experience_buffer: Optional[Any] = None,
        verbose: bool = False,
        turn_limit_mult: float = 1.0,
        hint_provider: Optional[Any] = None,
        meta_controller: Optional[Any] = None,
    ):
        if mode not in {"padpp_static", "hmod_dynamic", "hmod_no_mask"}:
            raise ValueError("mode must be one of padpp_static, hmod_dynamic, hmod_no_mask")
        if controller_mode not in {"rule_scaffold", "llm_reflection"}:
            raise ValueError("controller_mode must be one of rule_scaffold, llm_reflection")
        if objective_id and not objective_file:
            raise ValueError("--objective_id requires --objective_file")
        self.mode = mode
        self.controller_mode = controller_mode
        self.judge_model = judge_model
        self.verbose = verbose
        self.turn_limit_mult = max(0.1, float(turn_limit_mult))
        self.use_llm_simulator = use_llm_simulator
        self.audit_sample_size = audit_sample_size
        self.objective_file = objective_file
        self.objective_id = objective_id
        self.reflection_horizon = reflection_horizon
        objective_library = (
            BuyerObjectiveLibrary.from_file(objective_file) if objective_file else None
        )
        rule_controller = RuleMetaController(
            objective_library=objective_library,
            objective_id=objective_id,
            reflection_horizon=reflection_horizon,
        )
        if meta_controller is not None:
            # Pre-built controller injected by a caller (e.g. the two-agent
            # trainer/eval). It manages its own LLM agents and hints.
            self.meta_controller = meta_controller
        elif controller_mode == "llm_reflection" and mode != "padpp_static":
            reflector = LLMWeightReflector(
                model=llm_model,
                api_key=llm_api_key,
                api_key_env=llm_api_key_env,
                base_url=llm_base_url,
                temperature=llm_temperature,
                max_tokens=llm_max_tokens,
            )
            self.meta_controller = LLMReflectionMetaController(
                reflector=reflector,
                reflection_horizon=reflection_horizon,
                fallback_controller=rule_controller,
                fallback_to_rule=llm_fallback_to_rule,
            )
        else:
            self.meta_controller = rule_controller
        # buyer_policy can be the rule scaffold (default) or a NeuralBuyerPolicy
        # backed by the trained R-PADPP low policy (merged-pipeline path).
        self.buyer_policy = buyer_policy if buyer_policy is not None else RuleBuyerPolicy()

        # Grounding for the LLM reflection: general hints (from H-MOD hint
        # training) + per-(goal, drift) experience summary. Both feed the same
        # `experience_provider` slot, composed into one block.
        self.experience_buffer = experience_buffer
        self.hint_provider = hint_provider
        if isinstance(self.meta_controller, LLMReflectionMetaController):
            providers = []
            if hint_provider is not None:
                providers.append(hint_provider)
            if experience_buffer is not None:
                providers.append(
                    lambda macro_goal, drift_mode: (
                        (experience_buffer.summarize(macro_goal, drift_mode) or {}).get("text")
                    )
                )
            if providers:
                def _combined(macro_goal, drift_mode, _providers=providers):
                    parts = []
                    for prov in _providers:
                        try:
                            text = prov(macro_goal, drift_mode)
                        except Exception:
                            text = None
                        if text:
                            parts.append(text)
                    return "\n\n".join(parts) if parts else None

                self.meta_controller.experience_provider = _combined

    def run_dialogue(self, scenario: HMODScenario) -> Dict[str, Any]:
        simulator = DynamicSellerNegotiationSimulator(
            scenario,
            use_llm_response=self.use_llm_simulator,
            model_type=self.judge_model if self.judge_model != "rule" else "llama3",
        )
        state = _initial_state(scenario)
        weight_trace: List[Dict[str, Any]] = []
        violation_trace: List[Dict[str, Any]] = []
        current_weight: Optional[List[float]] = None

        # Effective turn limit (e.g. --turn_limit_mult 3 triples the dialogue
        # length). Used for the loop AND the metrics so they stay consistent.
        effective_turn_limit = max(1, int(round(scenario.turn_limit * self.turn_limit_mult)))

        if self.verbose:
            logger.info(
                f"===== Dialogue {scenario.id} | drift={scenario.drift_mode} | "
                f"macro_goal='{scenario.macro_goal}' | turn_limit={effective_turn_limit} ====="
            )
            logger.info(
                f"  [item] {scenario.case.get('item_name')} | buyer_target≈${scenario.target_price():.0f} "
                f"| buyer_ceiling≈${scenario.max_acceptable_price():.0f} | seller_list=${scenario.case.get('seller_price')}"
            )

        for turn in range(effective_turn_limit):
            state["turn_id"] = turn
            selected = self.meta_controller.select_local_weight(
                scenario=scenario,
                simulator_trace=simulator.get_trace(),
                previous_weight=current_weight,
                mode=self.mode,
                turn=turn,
                dialogue_history=state["dialogue_context"],
            )
            current_weight = selected["weight_vector"]
            weight_trace.append(
                {
                    "scenario_id": scenario.id,
                    "turn": turn,
                    "weight_vector": list(current_weight),
                    "w_t": list(current_weight),
                    "intent_state": selected["intent_state"],
                    "decision_summary": selected["decision_summary"],
                    "selected_objective_id": selected.get("selected_objective_id"),
                    "objective_cluster": selected.get("objective_cluster"),
                    "objective_source": selected.get("objective_source"),
                    "objective_match_reason": selected.get("objective_match_reason"),
                    "base_weight": selected.get("base_weight"),
                    "reflection_step": selected.get("reflection_step"),
                    "reflection_horizon": selected.get("reflection_horizon"),
                    "controller_mode": selected.get("controller_mode", self.controller_mode),
                    "llm_reflection": selected.get("llm_reflection"),
                    "llm_error": selected.get("llm_error"),
                    "mode": self.mode,
                }
            )

            decision = self.buyer_policy.select_action(
                scenario=scenario,
                state=state,
                weight=current_weight,
                mode=self.mode,
            )
            action = decision["action"]
            state["pred_goal"] = action
            state["last_buyer_price"] = action.get("price")
            state["dialogue_context"].append(
                {"role": "assistant", "content": decision["buyer_response"]}
            )

            violation_trace.append(
                {
                    "scenario_id": scenario.id,
                    "turn": turn,
                    "mode": self.mode,
                    "raw_action": decision["raw_action"],
                    "masked_action": decision["action"],
                    "blocked_violation": decision["blocked_violation"],
                    "actual_violation": decision["actual_violation"],
                    "violation_reason": decision["violation_reason"],
                    "max_acceptable_price": decision["max_acceptable_price"],
                }
            )

            seller_response = simulator.respond(state)
            state["dialogue_context"].append({"role": "user", "content": seller_response})
            state["last_seller_price"] = first_price(seller_response)

            if self.verbose:
                wt = [round(float(x), 2) for x in current_weight]
                reflected = "REFLECT" if selected.get("reflection_step") else "carry"
                sim_intent = simulator.get_trace().get("intent_state_by_turn", [])
                cur_intent = (sim_intent[-1].get("intent_state")
                              if sim_intent else selected.get("intent_state"))
                logger.info(
                    f"[turn {turn}] seller_intent={cur_intent} | w_t={wt} ({reflected}) | "
                    f"act={decision['action'].get('strategy')}"
                    + ("  [MASKED: over-ceiling]" if decision.get("blocked_violation") else "")
                )
                logger.info(f"   [Buyer]:  {decision['buyer_response']}")
                logger.info(f"   [Seller]: {seller_response}")

            if _is_terminal(state["dialogue_context"]):
                break

        judge_result = judge_deal(state["dialogue_context"], model_type=self.judge_model)
        turn_count = len([t for t in state["dialogue_context"] if t.get("role") == "assistant"]) - 1
        max_price = scenario.max_acceptable_price()
        gsr, gsr_components = compute_gsr(
            judge_result=judge_result,
            min_acceptable_price=max_price,
            turn_count=turn_count,
            turn_limit=effective_turn_limit,
            price_direction="at_most",
        )
        if judge_result.get("deal_price") is not None and float(judge_result["deal_price"]) > max_price:
            violation_trace.append(
                {
                    "scenario_id": scenario.id,
                    "turn": turn_count,
                    "mode": self.mode,
                    "raw_action": None,
                    "masked_action": None,
                    "blocked_violation": False,
                    "actual_violation": True,
                    "violation_reason": "final_deal_above_max_acceptable_ceiling",
                    "max_acceptable_price": max_price,
                }
            )

        sim_trace = simulator.get_trace()
        t2da = compute_t2da(
            weight_trace=weight_trace,
            t_drift=sim_trace.get("t_drift"),
            turn_limit=effective_turn_limit,
            expected_weight_shift=scenario.expected_weight_shift,
        )
        cvr = compute_cvr(violation_trace)

        if self.verbose:
            logger.info(
                f"===== Result {scenario.id}: deal={judge_result.get('deal')} "
                f"deal_price={judge_result.get('deal_price')} | GSR={gsr} "
                f"T2DA={t2da} CVR={cvr} | turns={turn_count} =====\n"
            )

        # Record this episode for cross-episode experience accumulation.
        if self.experience_buffer is not None:
            final_w = weight_trace[-1]["weight_vector"] if weight_trace else None
            intent_states = [w.get("intent_state") for w in weight_trace]
            try:
                self.experience_buffer.record_episode(
                    macro_goal=scenario.macro_goal,
                    drift_mode=scenario.drift_mode,
                    final_weight=final_w or [],
                    gsr=float(gsr),
                    deal=bool(judge_result.get("deal")),
                    deal_price=judge_result.get("deal_price"),
                    max_acceptable_price=max_price,
                    intent_states=intent_states,
                )
            except Exception:
                pass

        return {
            "scenario_id": scenario.id,
            "mode": self.mode,
            "drift_mode": scenario.drift_mode,
            "persona_type": scenario.seller_persona.type,
            "macro_goal": scenario.macro_goal,
            "buyer_intent_id": scenario.buyer_intent_id,
            "selected_objective_id": weight_trace[0].get("selected_objective_id")
            if weight_trace
            else None,
            "objective_cluster": weight_trace[0].get("objective_cluster")
            if weight_trace
            else None,
            "dialogue": state["dialogue_context"],
            "judge_result": judge_result,
            "gsr": gsr,
            "gsr_components": gsr_components,
            "t2da": t2da,
            "cvr": cvr,
            "simulator_trace": sim_trace,
            "weight_trace": weight_trace,
            "violation_trace": violation_trace,
        }

    def run(self, scenarios: List[HMODScenario],
            progress_prefix: Optional[str] = None) -> Dict[str, Any]:
        dialogues = []
        n = len(scenarios)
        for i, scenario in enumerate(scenarios):
            rec = self.run_dialogue(scenario)
            dialogues.append(rec)
            if progress_prefix is not None:
                logger.info(
                    f"[{progress_prefix}] {i + 1}/{n} {rec['scenario_id']}: "
                    f"deal={rec['judge_result'].get('deal')} GSR={rec['gsr']} "
                    f"t2da={(rec.get('t2da') or {}).get('t2da')} "
                    f"turns={rec['gsr_components']['turn_count']}"
                )
        return {
            "mode": self.mode,
            "controller_mode": self.controller_mode,
            "judge_model": self.judge_model,
            "objective_file": self.objective_file,
            "objective_id": self.objective_id,
            "reflection_horizon": self.reflection_horizon,
            "metrics": aggregate_dialogue_metrics(dialogues),
            "metrics_by_drift_mode": subgroup_metrics(dialogues, "drift_mode"),
            "metrics_by_persona": subgroup_metrics(dialogues, "persona_type"),
            "metrics_by_objective": subgroup_metrics(dialogues, "selected_objective_id"),
            "dialogues": dialogues,
        }


def run_and_write(
    scenario_file: str,
    mode: str,
    num_cases: Optional[int],
    output_dir: str,
    audit_sample_size: int = 50,
    judge_model: str = "rule",
    use_llm_simulator: bool = False,
    objective_file: Optional[str] = None,
    objective_id: Optional[str] = None,
    reflection_horizon: int = 3,
    controller_mode: str = "rule_scaffold",
    llm_model: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_api_key_env: str = "DEEPINFRA_API_KEY",
    llm_base_url: Optional[str] = None,
    llm_temperature: float = 0.0,
    llm_max_tokens: int = 500,
    llm_fallback_to_rule: bool = False,
    buyer_policy: Optional[Any] = None,
    experience_buffer: Optional[Any] = None,
    verbose: bool = False,
    turn_limit_mult: float = 1.0,
    hint_provider: Optional[Any] = None,
    meta_controller: Optional[Any] = None,
) -> Dict[str, Any]:
    scenarios = load_scenarios(scenario_file, limit=num_cases)
    run_id = f"{mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    evaluator = HMODEvaluator(
        mode=mode,
        judge_model=judge_model,
        use_llm_simulator=use_llm_simulator,
        audit_sample_size=audit_sample_size,
        objective_file=objective_file,
        objective_id=objective_id,
        reflection_horizon=reflection_horizon,
        controller_mode=controller_mode,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_api_key_env=llm_api_key_env,
        llm_base_url=llm_base_url,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
        llm_fallback_to_rule=llm_fallback_to_rule,
        buyer_policy=buyer_policy,
        experience_buffer=experience_buffer,
        verbose=verbose,
        turn_limit_mult=turn_limit_mult,
        hint_provider=hint_provider,
        meta_controller=meta_controller,
    )
    result = evaluator.run(scenarios)

    dialogue_rows = result["dialogues"]
    weight_rows = [row for rec in dialogue_rows for row in rec["weight_trace"]]
    violation_rows = [row for rec in dialogue_rows for row in rec["violation_trace"]]
    audit_rows = [
        {
            "scenario_id": rec["scenario_id"],
            "mode": rec["mode"],
            "controller_mode": result["controller_mode"],
            "drift_mode": rec["drift_mode"],
            "persona_type": rec["persona_type"],
            "buyer_intent_id": rec.get("buyer_intent_id"),
            "selected_objective_id": rec.get("selected_objective_id"),
            "objective_cluster": rec.get("objective_cluster"),
            "dialogue": rec["dialogue"],
            "llm_judge": rec["judge_result"],
            "gsr": rec["gsr"],
            "gsr_components": rec["gsr_components"],
            "human_deal": None,
            "human_deal_price": None,
            "human_success": None,
            "human_notes": "",
        }
        for rec in dialogue_rows[:audit_sample_size]
    ]

    metrics_payload = {
        "mode": result["mode"],
        "controller_mode": result["controller_mode"],
        "judge_model": result["judge_model"],
        "objective_file": result["objective_file"],
        "objective_id": result["objective_id"],
        "reflection_horizon": result["reflection_horizon"],
        "metrics": result["metrics"],
        "metrics_by_drift_mode": result["metrics_by_drift_mode"],
        "metrics_by_persona": result["metrics_by_persona"],
        "metrics_by_objective": result["metrics_by_objective"],
        "scenario_file": scenario_file,
        "num_dialogues": len(dialogue_rows),
        "human_audit_samples": len(audit_rows),
    }
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=2, ensure_ascii=False, default=_json_default)

    compact_dialogues = []
    for rec in dialogue_rows:
        compact = dict(rec)
        compact.pop("weight_trace", None)
        compact.pop("violation_trace", None)
        compact_dialogues.append(compact)

    write_jsonl(os.path.join(run_dir, "dialogues.jsonl"), compact_dialogues)
    write_jsonl(os.path.join(run_dir, "weight_trace.jsonl"), weight_rows)
    write_jsonl(os.path.join(run_dir, "violation_trace.jsonl"), violation_rows)
    write_jsonl(os.path.join(run_dir, "human_audit.jsonl"), audit_rows)
    metrics_payload["run_dir"] = run_dir
    return metrics_payload
