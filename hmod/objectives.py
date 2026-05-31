"""Buyer objective loading and objective-to-weight mapping for H-MOD."""

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from hmod.scenario import OBJECTIVE_ORDER


DEFAULT_CLUSTER_PRIORS: Dict[str, List[float]] = {
    "PRICE_GAIN": [0.72, 0.10, 0.13, 0.05],
    "FAST_PURCHASE": [0.25, 0.15, 0.48, 0.12],
    "FAIR_VALUE": [0.28, 0.42, 0.20, 0.10],
    "RISK_AWARE_PURCHASE": [0.35, 0.30, 0.25, 0.10],
    "WALKAWAY_DISCIPLINE": [0.62, 0.12, 0.12, 0.14],
    "ADAPTIVE_BUYER_CONTROL": [0.42, 0.22, 0.28, 0.08],
    # Backward-compatible cluster names from old seller-objective files.
    "MARGIN_DEFENSE": [0.72, 0.10, 0.13, 0.05],
    "CONVERSION_OPTIMIZATION": [0.25, 0.15, 0.48, 0.12],
    "ADAPTIVE_BUYER_HANDLING": [0.42, 0.22, 0.28, 0.08],
    "VALUE_TRUST_AND_FAIRNESS": [0.28, 0.42, 0.20, 0.10],
}


KEYWORD_DELTAS: Tuple[Tuple[Tuple[str, ...], List[float]], ...] = (
    (
        (
            "cheap",
            "lowest",
            "price gain",
            "save",
            "saving",
            "discount",
            "bargain",
            "low price",
            "under budget",
        ),
        [0.12, -0.02, -0.07, -0.03],
    ),
    (
        (
            "urgent",
            "fast",
            "quick",
            "birthday",
            "must buy",
            "need it today",
            "close",
            "deal rate",
            "secure the item",
        ),
        [-0.08, 0.00, 0.13, 0.05],
    ),
    (
        (
            "fair",
            "fairness",
            "reasonable",
            "trust",
            "relationship",
            "mutual",
            "polite",
        ),
        [-0.04, 0.13, -0.02, -0.02],
    ),
    (
        (
            "risk",
            "warranty",
            "condition",
            "inspection",
            "quality",
            "safe",
            "assurance",
        ),
        [-0.02, 0.10, 0.00, -0.02],
    ),
    (
        (
            "walk away",
            "walkaway",
            "seller pressure",
            "firm seller",
            "final offer",
            "take it or leave",
        ),
        [0.02, 0.06, 0.08, 0.02],
    ),
)


def normalize_weight(weight: Iterable[float]) -> List[float]:
    values = [max(0.0, float(x)) for x in weight]
    total = sum(values)
    if total <= 0:
        return [1.0 / len(OBJECTIVE_ORDER)] * len(OBJECTIVE_ORDER)
    return [x / total for x in values]


@dataclass(frozen=True)
class BuyerStrategyObjective:
    id: str
    description: str
    natural_language_intent: str
    typical_steps: List[str] = field(default_factory=list)
    cluster: Optional[str] = None
    stage_weights: Dict[str, List[float]] = field(default_factory=dict)
    adaptation_rules: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(
            [
                self.id,
                self.description,
                self.natural_language_intent,
                " ".join(self.typical_steps),
                " ".join(self.stage_weights.keys()),
                " ".join(str(rule.get("summary", "")) for rule in self.adaptation_rules),
            ]
        )


@dataclass(frozen=True)
class ObjectiveMapping:
    objective_id: Optional[str]
    cluster: Optional[str]
    weight_vector: List[float]
    source: str
    match_reason: str


class BuyerObjectiveLibrary:
    """Loads buyer strategy intents and maps ambiguous text to W vectors."""

    def __init__(
        self,
        intents: Dict[str, BuyerStrategyObjective],
        clusters: Optional[Dict[str, List[str]]] = None,
    ):
        self.intents = dict(intents)
        self.clusters = {key: list(value) for key, value in (clusters or {}).items()}
        self._cluster_by_intent = self._build_cluster_reverse_index()

    @classmethod
    def from_file(cls, path: str) -> "BuyerObjectiveLibrary":
        payload = _read_assignment_file(path)
        raw_intents = payload.get("BUYER_STRATEGY_INTENTS", payload.get("SELLER_STRATEGY_INTENTS"))
        if not isinstance(raw_intents, dict):
            raise ValueError(
                "Objective file must define BUYER_STRATEGY_INTENTS as a dictionary"
            )
        raw_clusters = payload.get(
            "BUYER_STRATEGY_MACRO_CLUSTERS",
            payload.get("SELLER_STRATEGY_MACRO_CLUSTERS", {}),
        )
        if raw_clusters is None:
            raw_clusters = {}
        if not isinstance(raw_clusters, dict):
            raise ValueError(
                "BUYER_STRATEGY_MACRO_CLUSTERS must be a dictionary when provided"
            )

        clusters = {
            str(cluster_id): [str(item) for item in values]
            for cluster_id, values in raw_clusters.items()
            if isinstance(values, list)
        }
        reverse_clusters: Dict[str, str] = {}
        for cluster_id, objective_ids in clusters.items():
            for objective_id in objective_ids:
                reverse_clusters.setdefault(objective_id, cluster_id)

        intents: Dict[str, BuyerStrategyObjective] = {}
        for objective_id, raw in raw_intents.items():
            if not isinstance(raw, dict):
                raise ValueError(f"Objective {objective_id!r} must map to a dictionary")
            oid = str(objective_id)
            steps = raw.get("typical_steps", [])
            if not isinstance(steps, list):
                steps = []
            stage_weights = raw.get("stage_weights", {})
            if not isinstance(stage_weights, dict):
                stage_weights = {}
            parsed_stage_weights = {
                str(stage_id): normalize_weight(values)
                for stage_id, values in stage_weights.items()
                if isinstance(values, list)
            }
            adaptation_rules = raw.get("adaptation_rules", [])
            if not isinstance(adaptation_rules, list):
                adaptation_rules = []
            intents[oid] = BuyerStrategyObjective(
                id=oid,
                description=str(raw.get("description", "")),
                natural_language_intent=str(raw.get("natural_language_intent", "")),
                typical_steps=[str(step) for step in steps],
                cluster=reverse_clusters.get(oid),
                stage_weights=parsed_stage_weights,
                adaptation_rules=[dict(rule) for rule in adaptation_rules if isinstance(rule, dict)],
            )
        return cls(intents=intents, clusters=clusters)

    def _build_cluster_reverse_index(self) -> Dict[str, str]:
        reverse: Dict[str, str] = {}
        for cluster_id, objective_ids in self.clusters.items():
            for objective_id in objective_ids:
                reverse.setdefault(objective_id, cluster_id)
        return reverse

    def get(self, objective_id: Optional[str]) -> Optional[BuyerStrategyObjective]:
        if objective_id is None:
            return None
        return self.intents.get(objective_id)

    def select_objective(
        self,
        macro_goal: str,
        scenario_objective_id: Optional[str] = None,
        override_objective_id: Optional[str] = None,
    ) -> Tuple[Optional[BuyerStrategyObjective], str]:
        if override_objective_id:
            objective = self.get(override_objective_id)
            if objective is None:
                raise ValueError(f"Unknown objective_id {override_objective_id!r}")
            return objective, "cli_objective_id"

        if scenario_objective_id:
            objective = self.get(scenario_objective_id)
            if objective is None:
                raise ValueError(f"Unknown scenario buyer_intent_id {scenario_objective_id!r}")
            return objective, "scenario_buyer_intent_id"

        objective = self._best_macro_goal_match(macro_goal)
        if objective is None:
            return None, "no_objective_match"
        return objective, "macro_goal_keyword_match"

    def map_objective(
        self,
        macro_goal: str,
        scenario_objective_id: Optional[str] = None,
        override_objective_id: Optional[str] = None,
        fallback_weight: Optional[List[float]] = None,
    ) -> ObjectiveMapping:
        objective, match_reason = self.select_objective(
            macro_goal=macro_goal,
            scenario_objective_id=scenario_objective_id,
            override_objective_id=override_objective_id,
        )
        if objective is None:
            return ObjectiveMapping(
                objective_id=None,
                cluster=None,
                weight_vector=normalize_weight(fallback_weight or []),
                source="fallback_static_w",
                match_reason=match_reason,
            )

        cluster = objective.cluster or self._cluster_by_intent.get(objective.id)
        if "initial" in objective.stage_weights:
            weight = normalize_weight(objective.stage_weights["initial"])
        else:
            base = list(
                DEFAULT_CLUSTER_PRIORS.get(cluster or "", fallback_weight or [0.45, 0.25, 0.20, 0.10])
            )
            weight = _apply_keyword_deltas(base, objective.text)
        return ObjectiveMapping(
            objective_id=objective.id,
            cluster=cluster,
            weight_vector=weight,
            source="buyer_strategy_objective",
            match_reason=match_reason,
        )

    def _best_macro_goal_match(self, macro_goal: str) -> Optional[BuyerStrategyObjective]:
        goal_tokens = _tokens(macro_goal)
        if not goal_tokens:
            return None
        best_score = 0
        best: Optional[BuyerStrategyObjective] = None
        for objective in self.intents.values():
            objective_tokens = _tokens(objective.text)
            score = len(goal_tokens.intersection(objective_tokens))
            if score > best_score:
                best_score = score
                best = objective
        return best if best_score > 0 else None


def _apply_keyword_deltas(base: List[float], text: str) -> List[float]:
    weight = list(base)
    lowered = text.lower().replace("_", " ")
    for keywords, delta in KEYWORD_DELTAS:
        if any(keyword in lowered for keyword in keywords):
            weight = [value + shift for value, shift in zip(weight, delta)]
    return normalize_weight(weight)


def _tokens(text: str) -> set:
    normalized = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return {token for token in normalized.split() if len(token) >= 4}


def _read_assignment_file(path: str) -> Dict[str, Any]:
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=path)
    payload: Dict[str, Any] = {}
    allowed = {
        "BUYER_STRATEGY_INTENTS",
        "BUYER_STRATEGY_MACRO_CLUSTERS",
        "SELLER_STRATEGY_INTENTS",
        "SELLER_STRATEGY_MACRO_CLUSTERS",
    }
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id not in allowed:
                continue
            payload[target.id] = ast.literal_eval(node.value)
    return payload


# Backward-compatible aliases for older imports/tests.
SellerStrategyObjective = BuyerStrategyObjective
SellerObjectiveLibrary = BuyerObjectiveLibrary
