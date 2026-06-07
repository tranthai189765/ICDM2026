"""Simple buyer-mode policies and safety masking for H-MOD evaluation."""

import math
from typing import Any, Dict, List, Optional

from hmod.llm_reflection import LLMWeightReflector
from hmod.objectives import BuyerObjectiveLibrary, BuyerStrategyObjective, ObjectiveMapping
from hmod.scenario import HMODScenario, coerce_objective_weight


PRICE_ACTIONS = {"propose", "counter", "agree", "final_offer"}


def _floor_to_ceiling(price: Optional[float], max_price: float) -> Optional[float]:
    # Seller utterances re-emit prices rounded to whole dollars (e.g. "$2739"),
    # so for normal-priced items the buyer must offer at floor(ceiling) in
    # whole dollars; for very small ceilings keep cent precision so we do not
    # collapse a $1.52 ceiling to $1.
    if price is None:
        return None
    capped = min(float(price), float(max_price))
    if float(max_price) >= 10:
        return float(math.floor(capped))
    return math.floor(capped * 100) / 100.0


def _format_price(value: Optional[float]) -> str:
    if value is None:
        return "0"
    rounded = round(float(value), 2)
    if abs(rounded - round(rounded)) < 1e-9:
        return f"{int(round(rounded))}"
    return f"{rounded:.2f}"


def normalize_weight(weight: List[float]) -> List[float]:
    # Collapse to the active 3-D objective space and renormalise (legacy 4-D
    # vectors drop their trailing avg_turn term).
    weight = [max(0.0, float(x)) for x in weight]
    return coerce_objective_weight(weight)


class RuleMetaController:
    """Deterministic stand-in for the H-MOD buyer LLM meta-controller."""

    def __init__(
        self,
        objective_library: Optional[BuyerObjectiveLibrary] = None,
        objective_id: Optional[str] = None,
        reflection_horizon: int = 3,
    ):
        if reflection_horizon <= 0:
            raise ValueError("reflection_horizon must be a positive integer")
        self.objective_library = objective_library
        self.objective_id = objective_id
        self.reflection_horizon = reflection_horizon

    def _objective_mapping(self, scenario: HMODScenario) -> ObjectiveMapping:
        if self.objective_library is None:
            return ObjectiveMapping(
                objective_id=scenario.buyer_intent_id,
                cluster=None,
                weight_vector=normalize_weight(scenario.static_w),
                source="scenario_static_w",
                match_reason="no_objective_file",
            )
        return self.objective_library.map_objective(
            macro_goal=scenario.macro_goal,
            scenario_objective_id=scenario.buyer_intent_id,
            override_objective_id=self.objective_id,
            fallback_weight=scenario.static_w,
        )

    def _should_reflect(self, turn: int, previous_weight: Optional[List[float]]) -> bool:
        return previous_weight is None or turn % self.reflection_horizon == 0

    def _latest_intent_state(self, scenario: HMODScenario, simulator_trace: Dict[str, Any]) -> str:
        intent_rows = simulator_trace.get("intent_state_by_turn", [])
        if not intent_rows:
            return scenario.initial_intent
        return intent_rows[-1]["intent_state"]

    def _latest_trace_value(self, simulator_trace: Dict[str, Any], key: str) -> Optional[Any]:
        rows = simulator_trace.get(key, [])
        if not rows:
            return None
        latest = rows[-1]
        if isinstance(latest, dict):
            return latest.get("price", latest.get("value"))
        return latest

    def _latest_seller_offer(self, simulator_trace: Dict[str, Any]) -> Optional[float]:
        value = self._latest_trace_value(simulator_trace, "seller_offer_by_turn")
        if value is None:
            return None
        return float(value)

    def _latest_round_without_deal(self, simulator_trace: Dict[str, Any]) -> int:
        rows = simulator_trace.get("round_without_deal_by_turn", [])
        if not rows:
            return 0
        latest = rows[-1]
        if isinstance(latest, dict):
            return int(latest.get("round_without_deal", latest.get("value", 0)))
        return int(latest)

    def _latest_frustration(self, simulator_trace: Dict[str, Any]) -> int:
        rows = simulator_trace.get("frustration_by_turn", [])
        if not rows:
            return 0
        latest = rows[-1]
        if isinstance(latest, dict):
            return int(latest.get("frustration_level", latest.get("value", 0)))
        return int(latest)

    def _rule_matches(
        self,
        rule: Dict[str, Any],
        intent_state: str,
        simulator_trace: Dict[str, Any],
        scenario: HMODScenario,
        turn: int,
    ) -> bool:
        when = rule.get("when", {})
        if not isinstance(when, dict):
            return False

        seller_intent = when.get("seller_intent")
        if seller_intent is not None:
            allowed = seller_intent if isinstance(seller_intent, list) else [seller_intent]
            if intent_state not in {str(item) for item in allowed}:
                return False

        if "turn_gte" in when and turn < int(when["turn_gte"]):
            return False
        if "turn_lte" in when and turn > int(when["turn_lte"]):
            return False
        if "turn_eq" in when and turn != int(when["turn_eq"]):
            return False

        last_offer = self._latest_seller_offer(simulator_trace)
        if when.get("seller_offer_above_ceiling") and not (
            last_offer is not None and last_offer > scenario.max_acceptable_price()
        ):
            return False
        if when.get("seller_offer_within_ceiling") and not (
            last_offer is not None and last_offer <= scenario.max_acceptable_price()
        ):
            return False

        if "round_without_deal_gte" in when and self._latest_round_without_deal(simulator_trace) < int(
            when["round_without_deal_gte"]
        ):
            return False
        if "frustration_gte" in when and self._latest_frustration(simulator_trace) < int(
            when["frustration_gte"]
        ):
            return False
        return True

    def _apply_objective_rule(
        self,
        weight: List[float],
        objective: Optional[BuyerStrategyObjective],
        intent_state: str,
        simulator_trace: Dict[str, Any],
        scenario: HMODScenario,
        turn: int,
    ) -> Dict[str, Any]:
        if objective is None:
            return {"weight_vector": weight, "adjustment": None}

        for rule in objective.adaptation_rules:
            if not self._rule_matches(rule, intent_state, simulator_trace, scenario, turn):
                continue

            if "target_stage" in rule:
                stage_id = str(rule["target_stage"])
                stage_weight = objective.stage_weights.get(stage_id)
                if stage_weight is None:
                    continue
                return {
                    "weight_vector": list(stage_weight),
                    "adjustment": rule.get("summary", f"objective stage rule -> {stage_id}"),
                }

            if "target_weight" in rule and isinstance(rule["target_weight"], list):
                return {
                    "weight_vector": list(rule["target_weight"]),
                    "adjustment": rule.get("summary", "objective-specific target weight applied"),
                }

            if "delta" in rule and isinstance(rule["delta"], list):
                delta = [float(x) for x in rule["delta"]]
                padded = list(weight)
                if len(delta) < len(padded):
                    delta.extend([0.0] * (len(padded) - len(delta)))
                return {
                    "weight_vector": [value + shift for value, shift in zip(padded, delta)],
                    "adjustment": rule.get("summary", "objective-specific delta applied"),
                }

        return {"weight_vector": weight, "adjustment": None}

    def _adapt_weight(
        self,
        base_weight: List[float],
        intent_state: str,
        simulator_trace: Dict[str, Any],
        scenario: HMODScenario,
        turn: int,
        objective: Optional[BuyerStrategyObjective],
    ) -> Dict[str, Any]:
        weight = list(base_weight)
        adjustments: List[str] = []
        drift_states = {"firm", "final_offer", "walkaway_risk", "frustrated"}

        # 3-D objective space [sl_ratio, fairness, deal_rate]. The former
        # avg_turn (urgency) adjustments are folded into deal_rate, since under
        # the merged pipeline "close faster" is expressed by raising deal_rate.
        if intent_state == "final_offer":
            weight[0] -= 0.22  # sl_ratio / buyer price gain
            weight[1] += 0.05  # fairness
            weight[2] += 0.30  # deal_rate (incl. folded urgency)
            adjustments.append("seller issued final offer, prioritize securing the item")
        elif intent_state in {"walkaway_risk", "frustrated"}:
            weight[0] -= 0.25
            weight[1] += 0.10
            weight[2] += 0.25
            adjustments.append("seller walkaway risk detected, reduce bargain aggression")
        elif intent_state == "firm":
            weight[0] -= 0.15
            weight[1] += 0.07
            weight[2] += 0.16
            adjustments.append("seller became firm, trade some price gain for deal probability")
        elif intent_state not in drift_states:
            adjustments.append("no seller drift signal, keep buyer objective-derived priority")

        last_offer = self._latest_seller_offer(simulator_trace)
        if last_offer is not None and last_offer > scenario.max_acceptable_price():
            weight[0] += 0.06
            weight[2] -= 0.03
            adjustments.append("latest seller offer is above buyer ceiling, preserve price constraint")

        objective_rule = self._apply_objective_rule(
            weight=weight,
            objective=objective,
            intent_state=intent_state,
            simulator_trace=simulator_trace,
            scenario=scenario,
            turn=turn,
        )
        weight = objective_rule["weight_vector"]
        if objective_rule["adjustment"]:
            adjustments.append(objective_rule["adjustment"])

        return {
            "weight_vector": normalize_weight(weight),
            "adjustments": adjustments,
        }

    def select_local_weight(
        self,
        scenario: HMODScenario,
        simulator_trace: Dict[str, Any],
        previous_weight: Optional[List[float]],
        mode: str,
        turn: int = 0,
        dialogue_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        mapping = self._objective_mapping(scenario)
        objective = self.objective_library.get(mapping.objective_id) if self.objective_library else None
        intent_state = self._latest_intent_state(scenario, simulator_trace)

        if mode == "padpp_static":
            return {
                "weight_vector": list(mapping.weight_vector),
                "intent_state": "static",
                "decision_summary": (
                    "PADPP static baseline uses the initial mapped buyer W for the full episode."
                ),
                "selected_objective_id": mapping.objective_id,
                "objective_cluster": mapping.cluster,
                "objective_source": mapping.source,
                "objective_match_reason": mapping.match_reason,
                "base_weight": list(mapping.weight_vector),
                "reflection_step": False,
                "reflection_horizon": self.reflection_horizon,
            }

        should_reflect = self._should_reflect(turn, previous_weight)
        if not should_reflect and previous_weight is not None:
            return {
                "weight_vector": list(previous_weight),
                "intent_state": intent_state,
                "decision_summary": (
                    f"Carry previous buyer W until next self-reflection step; seller_intent={intent_state}."
                ),
                "selected_objective_id": mapping.objective_id,
                "objective_cluster": mapping.cluster,
                "objective_source": mapping.source,
                "objective_match_reason": mapping.match_reason,
                "base_weight": list(mapping.weight_vector),
                "reflection_step": False,
                "reflection_horizon": self.reflection_horizon,
            }

        adapted = self._adapt_weight(
            base_weight=mapping.weight_vector,
            intent_state=intent_state,
            simulator_trace=simulator_trace,
            scenario=scenario,
            turn=turn,
            objective=objective,
        )
        summary = (
            f"Self-reflection at turn {turn}: buyer_objective={mapping.objective_id or 'static_w'}, "
            f"seller_intent={intent_state}; " + "; ".join(adapted["adjustments"])
        )
        return {
            "weight_vector": adapted["weight_vector"],
            "intent_state": intent_state,
            "decision_summary": summary,
            "selected_objective_id": mapping.objective_id,
            "objective_cluster": mapping.cluster,
            "objective_source": mapping.source,
            "objective_match_reason": mapping.match_reason,
            "base_weight": list(mapping.weight_vector),
            "reflection_step": True,
            "reflection_horizon": self.reflection_horizon,
        }


class RuleBuyerPolicy:
    """A small buyer worker used by eval_hmod.py.

    This is not a replacement for trained H-MOD; it gives the simulator/metrics
    path an executable harness and allows fair static-vs-dynamic comparisons.
    """

    def _price_from_ratio(self, scenario: HMODScenario, ratio: float) -> float:
        buyer = float(scenario.case["buyer_price"])
        seller = float(scenario.case["seller_price"])
        return round(buyer + ratio * (seller - buyer), 2)

    def _last_seller_offer(self, state: Dict[str, Any]) -> Optional[float]:
        for turn in reversed(state.get("dialogue_context", [])):
            if turn.get("role") != "user":
                continue
            from hmod.simulator import first_price

            price = first_price(turn.get("content", ""))
            if price is not None:
                return price
        return None

    def _turn_progress(self, state: Dict[str, Any], scenario: HMODScenario) -> float:
        turn = int(state.get("turn_id", 0))
        limit = int(state.get("effective_turn_limit") or scenario.turn_limit or 1)
        limit = max(limit, 1)
        return min(1.0, max(0.0, float(turn + 1) / float(limit)))

    def _should_agree_now(
        self,
        seller_offer: Optional[float],
        target_price: float,
        max_price: float,
        price_gain_w: float,
        deal_w: float,
        progress: float,
    ) -> bool:
        if seller_offer is None or seller_offer > max_price:
            return False
        # GSR-optimized rule: once the seller quotes any price within the
        # buyer's hard ceiling, close immediately.
        return True

    def _utterance(self, action: Dict[str, Any], item_name: str) -> str:
        strategy = action["strategy"]
        price = action.get("price")
        price_text = _format_price(price)
        if strategy == "agree":
            return f"Deal, I can buy the {item_name} for ${price_text}."
        if strategy == "final_offer":
            return f"My final offer for the {item_name} is ${price_text}."
        if strategy == "counter":
            return f"That is still high for me, but I can offer ${price_text}."
        if strategy == "walk_away":
            return "I cannot make the price work, so I will pass."
        if strategy == "reject":
            return "I cannot accept that price."
        return f"I can offer ${price_text} for the {item_name}."

    def select_action(
        self,
        scenario: HMODScenario,
        state: Dict[str, Any],
        weight: List[float],
        mode: str,
    ) -> Dict[str, Any]:
        max_price = scenario.max_acceptable_price()
        target_price = scenario.target_price()
        item_name = scenario.case["item_name"]
        seller_offer = self._last_seller_offer(state)
        progress = self._turn_progress(state, scenario)
        # 3-D weight [sl_ratio, fairness, deal_rate]; closing urgency (old
        # avg_turn) is now folded into deal_rate.
        price_gain_w, fairness_w, deal_w = (list(weight) + [0.0, 0.0, 0.0])[:3]
        turn_w = 0.0

        if self._should_agree_now(
            seller_offer=seller_offer,
            target_price=target_price,
            max_price=max_price,
            price_gain_w=price_gain_w,
            deal_w=deal_w,
            progress=progress,
        ):
            raw_action = {"strategy": "agree", "price": _floor_to_ceiling(seller_offer, max_price)}
        elif seller_offer is not None:
            if seller_offer > max_price:
                strategy = "final_offer" if progress >= 0.50 else "counter"
                raw_action = {"strategy": strategy, "price": _floor_to_ceiling(max_price, max_price)}
            else:
                concession = min(1.0, max(0.0, deal_w + fairness_w + turn_w - 0.5 * price_gain_w))
                counter_price = min(max_price, target_price + concession * (max_price - target_price))
                if seller_offer <= counter_price and (deal_w + turn_w > price_gain_w or progress >= 0.75):
                    raw_action = {"strategy": "agree", "price": _floor_to_ceiling(seller_offer, max_price)}
                else:
                    strategy = "final_offer" if (deal_w > 0.45 or progress >= 0.70) else "counter"
                    raw_action = {"strategy": strategy, "price": _floor_to_ceiling(counter_price, max_price)}
        else:
            # On later turns, start closer to the acceptable band so the
            # seller has a realistic path to accept before the deadline.
            offer_ratio = min(
                scenario.buyer_constraints.max_acceptable_price_ratio,
                max(
                    scenario.buyer_constraints.target_price_ratio,
                    0.20 + (1.0 - price_gain_w) * 0.25 + 0.20 * progress,
                ),
            )
            raw_action = {
                "strategy": "propose",
                "price": _floor_to_ceiling(self._price_from_ratio(scenario, offer_ratio), max_price),
            }

        blocked = False
        reason = None
        action = dict(raw_action)
        if mode != "hmod_no_mask" and raw_action["strategy"] in PRICE_ACTIONS:
            raw_price = float(raw_action["price"])
            if raw_price > max_price:
                blocked = True
                reason = "price_above_max_acceptable_ceiling"
                action = {"strategy": "counter", "price": _floor_to_ceiling(max_price, max_price)}

        actual_violation = (
            action["strategy"] in PRICE_ACTIONS
            and action.get("price") is not None
            and float(action["price"]) > max_price
        )
        return {
            "raw_action": raw_action,
            "action": action,
            "buyer_response": self._utterance(action, item_name),
            "blocked_violation": blocked,
            "actual_violation": bool(actual_violation),
            "violation_reason": reason,
            "max_acceptable_price": max_price,
        }


# Backward-compatible alias for older imports.
RuleSellerPolicy = RuleBuyerPolicy


class NeuralBuyerPolicy:
    """Buyer policy backed by the trained R-PADPP neural low policy.

    This is the merged-pipeline worker: the H-MOD meta-controller produces the
    dynamic 3-D weight w_t, and this policy delegates the per-turn action to the
    trained DMORL/R-PADPP low policy (w -> action) instead of the hand-written
    RuleBuyerPolicy.

    `act_fn(dmorl_state: dict, weight: List[float]) -> dict` must return:
        {"strategy": str, "price": Optional[float], "utterance": str}
    where the low policy has mapped (state, w) -> (strategy, bin) -> a buyer
    utterance and its committed price. The heavy DMORL wiring lives in
    `hmod.low_policy.NeuralLowPolicy` so this class stays import-light.
    """

    def __init__(self, act_fn):
        self.act_fn = act_fn

    @staticmethod
    def _build_dmorl_state(scenario: HMODScenario, state: Dict[str, Any]) -> Dict[str, Any]:
        case = scenario.case
        return {
            "task_background": {
                "item_name": case["item_name"],
                "buyer_price": case["buyer_price"],
                "seller_price": case["seller_price"],
                "buyer_item_description": case.get("buyer_item_description", ""),
                "seller_item_description": case.get("seller_item_description", ""),
            },
            "dialogue_context": list(state.get("dialogue_context", [])),
            "pre_goals": [""],
            "pre_topics": [""],
            "goal": "greet",
        }

    @staticmethod
    def _last_seller_offer(state: Dict[str, Any]) -> Optional[float]:
        for turn in reversed(state.get("dialogue_context", [])):
            if turn.get("role") != "user":
                continue
            from hmod.simulator import first_price

            price = first_price(turn.get("content", ""))
            if price is not None:
                return price
        return None

    @staticmethod
    def _turn_progress(state: Dict[str, Any], scenario: HMODScenario) -> float:
        turn = int(state.get("turn_id", 0))
        limit = int(state.get("effective_turn_limit") or scenario.turn_limit or 1)
        limit = max(limit, 1)
        return min(1.0, max(0.0, float(turn + 1) / float(limit)))

    def select_action(
        self,
        scenario: HMODScenario,
        state: Dict[str, Any],
        weight: List[float],
        mode: str,
    ) -> Dict[str, Any]:
        max_price = scenario.max_acceptable_price()
        item_name = scenario.case["item_name"]
        dmorl_state = self._build_dmorl_state(scenario, state)

        out = self.act_fn(dmorl_state, list(weight))
        strategy = out.get("strategy")
        price = out.get("price")
        utterance = out.get("utterance") or ""

        seller_offer = self._last_seller_offer(state)
        progress = self._turn_progress(state, scenario)
        price_gain_w, _, deal_w = (list(weight) + [0.0, 0.0, 0.0])[:3]

        if seller_offer is not None and seller_offer <= max_price:
            strategy = "agree"
            price = _floor_to_ceiling(seller_offer, max_price)
            utterance = f"Deal, I can buy the {item_name} for ${_format_price(price)}."
        elif seller_offer is not None and seller_offer > max_price:
            # Strict GSR objective: quote the highest valid buyer price to maximize
            # seller acceptance probability while remaining within constraints.
            strategy = "final_offer" if progress >= 0.50 else "counter"
            price = _floor_to_ceiling(max_price, max_price)
            utterance = f"My best possible offer is ${_format_price(price)} for the {item_name}."

        raw_action = {"strategy": strategy, "price": price}
        blocked = False
        reason = None
        action = dict(raw_action)
        if (
            mode != "hmod_no_mask"
            and strategy in PRICE_ACTIONS
            and price is not None
            and float(price) > max_price
        ):
            blocked = True
            reason = "price_above_max_acceptable_ceiling"
            safe_price = _floor_to_ceiling(max_price, max_price)
            action = {"strategy": "counter", "price": safe_price}
            utterance = f"That is above my budget; I can do ${_format_price(safe_price)} for the {item_name}."

        actual_violation = (
            action["strategy"] in PRICE_ACTIONS
            and action.get("price") is not None
            and float(action["price"]) > max_price
        )
        return {
            "raw_action": raw_action,
            "action": action,
            "buyer_response": utterance,
            "blocked_violation": blocked,
            "actual_violation": bool(actual_violation),
            "violation_reason": reason,
            "max_acceptable_price": max_price,
        }


class LLMReflectionMetaController:
    """Paper-path meta-controller: one NL objective -> LLM-reflected W_t."""

    def __init__(
        self,
        reflector: LLMWeightReflector,
        reflection_horizon: int = 3,
        fallback_controller: Optional[RuleMetaController] = None,
        fallback_to_rule: bool = False,
        experience_provider=None,
    ):
        if reflection_horizon <= 0:
            raise ValueError("reflection_horizon must be a positive integer")
        self.reflector = reflector
        self.reflection_horizon = reflection_horizon
        self.fallback_controller = fallback_controller
        self.fallback_to_rule = fallback_to_rule
        # Optional callable(macro_goal, drift_mode) -> Optional[str] returning a
        # short summary of past episode outcomes to ground the reflection.
        self.experience_provider = experience_provider

    def _should_reflect(self, turn: int, previous_weight: Optional[List[float]]) -> bool:
        return previous_weight is None or turn % self.reflection_horizon == 0

    def _latest_visible_seller_offer(
        self,
        dialogue_history: Optional[List[Dict[str, str]]],
    ) -> Optional[float]:
        if not dialogue_history:
            return None
        from hmod.simulator import first_price

        for row in reversed(dialogue_history):
            if row.get("role") != "user":
                continue
            price = first_price(row.get("content", ""))
            if price is not None:
                return price
        return None

    def _fallback(
        self,
        scenario: HMODScenario,
        simulator_trace: Dict[str, Any],
        previous_weight: Optional[List[float]],
        mode: str,
        turn: int,
        dialogue_history: Optional[List[Dict[str, str]]],
        error: Exception,
    ) -> Dict[str, Any]:
        if not self.fallback_to_rule or self.fallback_controller is None:
            raise error
        selected = self.fallback_controller.select_local_weight(
            scenario=scenario,
            simulator_trace=simulator_trace,
            previous_weight=previous_weight,
            mode=mode,
            turn=turn,
            dialogue_history=dialogue_history,
        )
        selected["controller_mode"] = "llm_reflection_fallback_rule"
        selected["llm_error"] = str(error)
        selected["decision_summary"] = (
            f"LLM reflection failed; fallback rule controller used. Error={error}"
        )
        return selected

    def select_local_weight(
        self,
        scenario: HMODScenario,
        simulator_trace: Dict[str, Any],
        previous_weight: Optional[List[float]],
        mode: str,
        turn: int = 0,
        dialogue_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        if mode == "padpp_static":
            if self.fallback_controller is None:
                return {
                    "weight_vector": normalize_weight(scenario.static_w),
                    "intent_state": "static",
                    "decision_summary": "PADPP static baseline uses scenario.static_w.",
                    "selected_objective_id": scenario.buyer_intent_id,
                    "objective_cluster": None,
                    "objective_source": "scenario_static_w",
                    "objective_match_reason": "padpp_static",
                    "base_weight": normalize_weight(scenario.static_w),
                    "reflection_step": False,
                    "reflection_horizon": self.reflection_horizon,
                    "controller_mode": "padpp_static",
                }
            return self.fallback_controller.select_local_weight(
                scenario=scenario,
                simulator_trace=simulator_trace,
                previous_weight=previous_weight,
                mode=mode,
                turn=turn,
                dialogue_history=dialogue_history,
            )

        if not self._should_reflect(turn, previous_weight) and previous_weight is not None:
            return {
                "weight_vector": list(previous_weight),
                "intent_state": "carried",
                "decision_summary": (
                    "Carry previous LLM-reflected buyer W until next self-reflection step."
                ),
                "selected_objective_id": None,
                "objective_cluster": None,
                "objective_source": "direct_nl_macro_goal",
                "objective_match_reason": "llm_reflection_carry",
                "base_weight": list(previous_weight),
                "reflection_step": False,
                "reflection_horizon": self.reflection_horizon,
                "controller_mode": "llm_reflection",
            }

        experience_text = None
        if self.experience_provider is not None:
            try:
                experience_text = self.experience_provider(
                    scenario.macro_goal, scenario.drift_mode)
            except Exception:
                experience_text = None

        try:
            reflection = self.reflector.reflect(
                macro_goal=scenario.macro_goal,
                dialogue_history=dialogue_history or [],
                previous_weight=previous_weight,
                turn=turn,
                buyer_constraints={
                    "max_acceptable_price": scenario.max_acceptable_price(),
                    "target_price": scenario.target_price(),
                    "turn_limit": scenario.turn_limit,
                },
                item_context=scenario.case,
                last_seller_offer=self._latest_visible_seller_offer(dialogue_history),
                experience=experience_text,
            )
        except Exception as exc:
            return self._fallback(
                scenario=scenario,
                simulator_trace=simulator_trace,
                previous_weight=previous_weight,
                mode=mode,
                turn=turn,
                dialogue_history=dialogue_history,
                error=exc,
            )

        reason = reflection.get("reason", "")
        local_objective = reflection.get("local_objective", "")
        intent_state = reflection.get("detected_seller_intent", "unknown")
        return {
            "weight_vector": reflection["weight_vector"],
            "intent_state": intent_state,
            "decision_summary": (
                f"LLM self-reflection at turn {turn}: seller_intent={intent_state}; "
                f"local_objective={local_objective}; {reason}"
            ),
            "selected_objective_id": None,
            "objective_cluster": None,
            "objective_source": "direct_nl_macro_goal",
            "objective_match_reason": "llm_reflection",
            "base_weight": list(previous_weight) if previous_weight is not None else None,
            "reflection_step": True,
            "reflection_horizon": self.reflection_horizon,
            "controller_mode": "llm_reflection",
            "llm_reflection": {
                key: value
                for key, value in reflection.items()
                if key not in {"raw_response"}
            },
        }
