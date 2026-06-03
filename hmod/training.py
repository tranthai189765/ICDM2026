"""Training-time H-MOD buyer controller plugged into the DMORL pipeline."""

import json
import os
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

from dmorl.llm_controller import HintManager, SkillLibrary
from hmod.llm_reflection import LLMWeightReflector
from hmod.objectives import BuyerObjectiveLibrary, normalize_weight


# Merged into the 3-objective R-PADPP low policy (no avg_turn).
HMOD_OBJECTIVE_ORDER = ["sl_ratio", "fairness", "deal_rate"]


class HMODController:
    """DMORL-compatible controller driven by buyer ambiguous objectives.

    It exposes the same minimal interface used by DMORLTrainer:
    - skill_library for Phase 1a/1b GPI training
    - initialize_skills()
    - get_dynamic_weight() for Phase 2 dynamic RLT/inference
    - refine_after_dialogue() for post-dialogue hints
    """

    def __init__(
        self,
        n_objectives: int,
        objective_names: List[str],
        scenario: str,
        objective_file: str,
        objective_id: Optional[str] = None,
        n_basic_skills: int = 5,
        n_advanced_skills: int = 5,
        dynamic_weight_horizon: int = 3,
        skills_file: str = "hmod_skills.json",
        hints_file: str = "hmod_hints.json",
        skill_log_file: Optional[str] = None,
        controller_mode: str = "rule_scaffold",
        macro_goal: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_api_key_env: str = "DEEPINFRA_API_KEY",
        llm_base_url: Optional[str] = None,
        llm_temperature: float = 0.0,
        llm_max_tokens: int = 500,
        llm_fallback_to_rule: bool = False,
    ):
        if not objective_file:
            raise ValueError("H-MOD training requires hmod_objective_file")
        self.n_objectives = n_objectives
        self.objective_names = objective_names
        self.scenario = scenario
        self.objective_file = objective_file
        self.objective_id = objective_id
        self.n_basic_skills = n_basic_skills
        self.n_advanced_skills = n_advanced_skills
        self.dynamic_weight_horizon = dynamic_weight_horizon
        self.skill_log_file = skill_log_file
        self.controller_mode = controller_mode
        self.macro_goal = macro_goal
        self.llm_fallback_to_rule = llm_fallback_to_rule
        self.objective_library = BuyerObjectiveLibrary.from_file(objective_file)
        self.skill_library = SkillLibrary(n_objectives, skills_file, skill_log_file)
        self.hint_manager = HintManager(hints_file)
        self._base_weight = self._initial_base_weight()
        self._last_dynamic_weight: Optional[List[float]] = None
        self.llm_reflector = None
        if self.controller_mode == "llm_reflection":
            self.llm_reflector = LLMWeightReflector(
                model=llm_model,
                api_key=llm_api_key,
                api_key_env=llm_api_key_env,
                base_url=llm_base_url,
                temperature=llm_temperature,
                max_tokens=llm_max_tokens,
            )

    def initialize_skills(self, force_rediscover: bool = False) -> None:
        if not force_rediscover and self.skill_library.load():
            return

        basic_skills = self._build_basic_skills()
        advanced_skills = self._build_advanced_skills()
        self.skill_library.basic_skills = basic_skills
        self.skill_library.advanced_skills = advanced_skills
        self.skill_library.save()
        self._write_skill_log(basic_skills, advanced_skills)
        logger.info(
            f"[H-MOD] Loaded {len(basic_skills)} basic + "
            f"{len(advanced_skills)} advanced buyer-objective skills."
        )

    def get_dynamic_weight(self, dialogue_history: List[Dict]) -> List[float]:
        if self.controller_mode == "llm_reflection":
            try:
                return self._get_llm_dynamic_weight(dialogue_history)
            except Exception as exc:
                if not self.llm_fallback_to_rule:
                    raise
                logger.warning(
                    f"[H-MOD Phase-2] LLM reflection failed; using rule fallback. {exc}"
                )

        intent_state = self._infer_intent_state(dialogue_history)
        weight = list(self._base_weight)
        objective = self.objective_library.get(self.objective_id)
        stage_id = self._stage_for_intent(intent_state)

        # 3-D objective space [sl_ratio, fairness, deal_rate]; the former
        # avg_turn urgency term is folded into deal_rate.
        if objective is not None and stage_id in objective.stage_weights:
            weight = list(objective.stage_weights[stage_id])
        elif intent_state in {"final_offer", "final_ultimatum"}:
            weight[0] -= 0.22
            weight[1] += 0.05
            weight[2] += 0.30
        elif intent_state == "walkaway_risk":
            weight[0] -= 0.25
            weight[1] += 0.10
            weight[2] += 0.25
        elif intent_state in {"firm", "hard_pressure"}:
            weight[0] -= 0.15
            weight[1] += 0.07
            weight[2] += 0.16

        hints = " ".join(self.hint_manager.get_hints()[-5:]).lower()
        if "price ceiling" in hints or "above budget" in hints:
            weight[0] += 0.05
            weight[2] -= 0.03

        w = self._coerce_weight(weight)
        logger.info(
            f"[H-MOD Phase-2] intent={intent_state}, "
            f"w={[round(x, 3) for x in w]}"
        )
        self._last_dynamic_weight = w
        return w

    def refine_after_dialogue(self, dialogue_history: List[Dict], outcome: str) -> List[str]:
        history = "\n".join(
            f"{turn.get('role', '').upper()}: {turn.get('content', '')}"
            for turn in dialogue_history[-12:]
        )
        hint = (
            f"Outcome={outcome}. If the seller shows walkaway or final-offer language, "
            "shift weight from sl_ratio to deal_rate/fairness but keep the buyer price ceiling."
        )
        if "final offer" in history.lower() or "walk away" in history.lower() or "sell elsewhere" in history.lower():
            self.hint_manager.hints.append(hint)
            self.hint_manager.hints = self.hint_manager.hints[-self.hint_manager.MAX_HINTS:]
            self.hint_manager.save()
            return [hint]
        return []

    def _initial_base_weight(self) -> List[float]:
        if self.objective_id:
            mapping = self.objective_library.map_objective(
                macro_goal="",
                scenario_objective_id=self.objective_id,
                fallback_weight=[0.4, 0.2, 0.3, 0.1],
            )
            return self._coerce_weight(mapping.weight_vector)

        if self.objective_library.intents:
            first_id = next(iter(self.objective_library.intents))
            mapping = self.objective_library.map_objective(
                macro_goal="",
                scenario_objective_id=first_id,
                fallback_weight=[0.4, 0.2, 0.3, 0.1],
            )
            return self._coerce_weight(mapping.weight_vector)

        return self._coerce_weight([0.4, 0.2, 0.3, 0.1])

    def _runtime_macro_goal(self) -> str:
        if self.macro_goal:
            return self.macro_goal
        objective = self.objective_library.get(self.objective_id)
        if objective is not None:
            return objective.natural_language_intent or objective.description
        if self.objective_library.intents:
            first = next(iter(self.objective_library.intents.values()))
            return first.natural_language_intent or first.description
        return "Negotiate as the buyer according to the current ambiguous objective."

    def _get_llm_dynamic_weight(self, dialogue_history: List[Dict]) -> List[float]:
        if self.llm_reflector is None:
            raise ValueError("controller_mode=llm_reflection requires an LLM reflector")
        if len(dialogue_history) <= 2:
            previous_weight = None
            self._last_dynamic_weight = None
        else:
            previous_weight = self._last_dynamic_weight

        reflection = self.llm_reflector.reflect(
            macro_goal=self._runtime_macro_goal(),
            dialogue_history=dialogue_history,
            previous_weight=previous_weight,
            turn=max(0, len(dialogue_history) // 2),
            buyer_constraints={},
            item_context={},
            last_seller_offer=None,
        )
        w = self._coerce_weight(reflection["weight_vector"])
        self._last_dynamic_weight = w
        logger.info(
            f"[H-MOD Phase-2 LLM] intent={reflection.get('detected_seller_intent')}, "
            f"w={[round(x, 3) for x in w]}"
        )
        return w

    def _objective_ids_for_basic_skills(self) -> List[str]:
        objective_ids = list(self.objective_library.intents.keys())
        if self.objective_id and self.objective_id in objective_ids:
            objective_ids.remove(self.objective_id)
            objective_ids.insert(0, self.objective_id)
        return objective_ids[: self.n_basic_skills]

    def _build_basic_skills(self) -> List[Dict]:
        skills = []
        for objective_id in self._objective_ids_for_basic_skills():
            objective = self.objective_library.get(objective_id)
            mapping = self.objective_library.map_objective(
                macro_goal="",
                scenario_objective_id=objective_id,
                fallback_weight=[0.4, 0.2, 0.3, 0.1],
            )
            skills.append(
                {
                    "name": objective_id,
                    "description": objective.description if objective else "",
                    "weight_vector": self._coerce_weight(mapping.weight_vector),
                    "type": "basic",
                    "source": "hmod_buyer_objective",
                    "cluster": mapping.cluster,
                }
            )
        return skills

    def _build_advanced_skills(self) -> List[Dict]:
        skills = []
        for cluster_id, objective_ids in self.objective_library.clusters.items():
            if len(skills) >= self.n_advanced_skills:
                break
            member_weights = []
            for objective_id in objective_ids:
                if self.objective_library.get(objective_id) is None:
                    continue
                mapping = self.objective_library.map_objective(
                    macro_goal="",
                    scenario_objective_id=objective_id,
                    fallback_weight=[0.4, 0.2, 0.3, 0.1],
                )
                member_weights.append(mapping.weight_vector)
            if not member_weights:
                continue
            avg_weight = np.mean(np.array(member_weights, dtype=float), axis=0).tolist()
            skills.append(
                {
                    "name": f"HMOD_{cluster_id}",
                    "description": (
                        f"Composite buyer skill for macro cluster {cluster_id}, "
                        "derived from objective-level W vectors."
                    ),
                    "weight_vector": self._coerce_weight(avg_weight),
                    "type": "advanced",
                    "source": "hmod_macro_cluster",
                    "cluster": cluster_id,
                }
            )
        return skills

    def _coerce_weight(self, weight: List[float]) -> List[float]:
        values = list(weight)
        if len(values) < self.n_objectives:
            values.extend([0.0] * (self.n_objectives - len(values)))
        values = values[: self.n_objectives]
        return normalize_weight(values)

    def _infer_intent_state(self, dialogue_history: List[Dict]) -> str:
        user_text = " ".join(
            turn.get("content", "")
            for turn in dialogue_history[-4:]
            if turn.get("role") == "user"
        ).lower()
        if any(marker in user_text for marker in ["final offer", "take it or leave", "last offer"]):
            return "final_offer"
        if any(marker in user_text for marker in ["walk away", "will pass", "sell elsewhere", "someone else"]):
            return "walkaway_risk"
        if any(marker in user_text for marker in ["too low", "firm", "cannot go lower", "can't go lower"]):
            return "firm"
        return "neutral"

    def _stage_for_intent(self, intent_state: str) -> str:
        if intent_state in {"final_offer", "final_ultimatum"}:
            return "final_offer_response"
        if intent_state == "walkaway_risk":
            return "walkaway_response"
        if intent_state in {"firm", "hard_pressure"}:
            return "firm_response"
        return "initial"

    def _write_skill_log(self, basic_skills: List[Dict], advanced_skills: List[Dict]) -> None:
        if not self.skill_log_file:
            return
        os.makedirs(os.path.dirname(self.skill_log_file) or ".", exist_ok=True)
        payload = {
            "mode": "hmod_buyer_training",
            "controller_mode": self.controller_mode,
            "objective_file": self.objective_file,
            "objective_id": self.objective_id,
            "macro_goal": self._runtime_macro_goal(),
            "objective_names": self.objective_names,
            "basic": basic_skills,
            "advanced": advanced_skills,
        }
        with open(self.skill_log_file, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
