"""Scenario loading for H-MOD buyer-agent drift evaluation."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


ALLOWED_DRIFT_MODES = {
    "static_no_drift",
    "gradual_firming",
    "abrupt_final_offer",
    "frustrated_walkaway",
    # Backward-compatible names accepted by older scenario files.
    "gradual_pressure",
    "abrupt_ultimatum",
}

OBJECTIVE_ORDER = ["sl_ratio", "fairness", "deal_rate", "avg_turn"]


@dataclass
class BuyerConstraints:
    max_acceptable_price_ratio: float
    target_price_ratio: float
    turn_limit: int


@dataclass
class SellerPersona:
    type: str
    description: str
    initial_ask_ratio: float = 0.95
    neutral_accept_ratio: float = 0.58
    post_drift_ask_ratio: float = 0.72
    final_ask_ratio: float = 0.65


@dataclass
class DriftTrigger:
    round_without_deal: int = 3
    low_offer_streak: int = 2
    low_offer_multiplier: float = 0.85
    turn_id: int = 4
    frustration_threshold: int = 3
    walkaway_after_turns: int = 2


@dataclass
class HMODScenario:
    id: str
    macro_goal: str
    static_w: List[float]
    buyer_constraints: BuyerConstraints
    seller_persona: SellerPersona
    drift_mode: str
    drift_trigger: DriftTrigger
    turn_limit: int
    initial_intent: str = "neutral"
    buyer_intent_id: Optional[str] = None
    expected_weight_shift: Dict[str, str] = field(default_factory=dict)
    case: Dict[str, Any] = field(default_factory=dict)

    @property
    def seller_constraints(self) -> BuyerConstraints:
        """Backward-compatible alias for old seller-mode code paths."""
        return self.buyer_constraints

    @property
    def buyer_persona(self) -> SellerPersona:
        """Backward-compatible alias for old seller-mode code paths."""
        return self.seller_persona

    @property
    def seller_intent_id(self) -> Optional[str]:
        """Backward-compatible alias; H-MOD now uses buyer_intent_id."""
        return self.buyer_intent_id

    def price_span(self) -> float:
        return float(self.case["seller_price"]) - float(self.case["buyer_price"])

    def max_acceptable_price(self) -> float:
        return float(self.case["buyer_price"]) + (
            self.buyer_constraints.max_acceptable_price_ratio * self.price_span()
        )

    def min_acceptable_price(self) -> float:
        """Backward-compatible alias: buyer-side constraint is a price ceiling."""
        return self.max_acceptable_price()

    def target_price(self) -> float:
        return float(self.case["buyer_price"]) + (
            self.buyer_constraints.target_price_ratio * self.price_span()
        )


def _require(mapping: Dict[str, Any], key: str, scenario_id: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Scenario {scenario_id!r} missing required field {key!r}")
    return mapping[key]


def _default_case(raw: Dict[str, Any]) -> Dict[str, Any]:
    case = dict(raw or {})
    case.setdefault("item_name", "used laptop")
    case.setdefault("buyer_price", 100.0)
    case.setdefault("seller_price", 200.0)
    case.setdefault("seller_item_description", "A reliable used item in good condition.")
    case.setdefault("buyer_item_description", "A buyer looking for a fair deal.")
    if float(case["seller_price"]) <= float(case["buyer_price"]):
        raise ValueError("seller_price must be greater than buyer_price for H-MOD scenarios")
    return case


def _normalize_drift_mode(mode: str) -> str:
    aliases = {
        "gradual_pressure": "gradual_firming",
        "abrupt_ultimatum": "abrupt_final_offer",
    }
    return aliases.get(mode, mode)


def parse_scenario(raw: Dict[str, Any]) -> HMODScenario:
    scenario_id = str(_require(raw, "id", "<unknown>"))
    static_w = [float(x) for x in _require(raw, "static_w", scenario_id)]
    if len(static_w) != len(OBJECTIVE_ORDER):
        raise ValueError(
            f"Scenario {scenario_id!r} static_w must have {len(OBJECTIVE_ORDER)} entries"
        )
    if any(x < 0 for x in static_w) or sum(static_w) <= 0:
        raise ValueError(f"Scenario {scenario_id!r} static_w must be non-negative")
    total = sum(static_w)
    static_w = [x / total for x in static_w]

    drift_mode = _normalize_drift_mode(str(_require(raw, "drift_mode", scenario_id)))
    if drift_mode not in ALLOWED_DRIFT_MODES:
        raise ValueError(
            f"Scenario {scenario_id!r} drift_mode must be one of {sorted(ALLOWED_DRIFT_MODES)}"
        )

    constraints_raw = raw.get("buyer_constraints", raw.get("seller_constraints"))
    constraints_raw = _require({"constraints": constraints_raw}, "constraints", scenario_id)
    max_ratio = constraints_raw.get(
        "max_acceptable_price_ratio",
        constraints_raw.get("min_acceptable_price_ratio"),
    )
    constraints = BuyerConstraints(
        max_acceptable_price_ratio=float(
            _require({"max_acceptable_price_ratio": max_ratio}, "max_acceptable_price_ratio", scenario_id)
        ),
        target_price_ratio=float(_require(constraints_raw, "target_price_ratio", scenario_id)),
        turn_limit=int(_require(constraints_raw, "turn_limit", scenario_id)),
    )
    if not 0.0 <= constraints.max_acceptable_price_ratio <= 1.0:
        raise ValueError(f"Scenario {scenario_id!r} max_acceptable_price_ratio must be in [0, 1]")
    if not 0.0 <= constraints.target_price_ratio <= 1.0:
        raise ValueError(f"Scenario {scenario_id!r} target_price_ratio must be in [0, 1]")

    persona_raw = raw.get("seller_persona", raw.get("buyer_persona"))
    persona_raw = _require({"persona": persona_raw}, "persona", scenario_id)
    persona = SellerPersona(
        type=str(_require(persona_raw, "type", scenario_id)),
        description=str(_require(persona_raw, "description", scenario_id)),
        initial_ask_ratio=float(persona_raw.get("initial_ask_ratio", 0.95)),
        neutral_accept_ratio=float(
            persona_raw.get("neutral_accept_ratio", persona_raw.get("neutral_budget_ratio", 0.58))
        ),
        post_drift_ask_ratio=float(
            persona_raw.get("post_drift_ask_ratio", persona_raw.get("post_drift_offer_ratio", 0.72))
        ),
        final_ask_ratio=float(
            persona_raw.get("final_ask_ratio", persona_raw.get("final_offer_ratio", 0.65))
        ),
    )

    trigger = DriftTrigger(**raw.get("drift_trigger", {}))
    turn_limit = int(raw.get("turn_limit", constraints.turn_limit))
    case = _default_case(raw.get("case", {}))
    buyer_intent_id = raw.get("buyer_intent_id", raw.get("seller_intent_id"))

    return HMODScenario(
        id=scenario_id,
        macro_goal=str(_require(raw, "macro_goal", scenario_id)),
        static_w=static_w,
        buyer_constraints=constraints,
        seller_persona=persona,
        drift_mode=drift_mode,
        drift_trigger=trigger,
        turn_limit=turn_limit,
        initial_intent=str(raw.get("initial_intent", "neutral")),
        buyer_intent_id=str(buyer_intent_id) if buyer_intent_id is not None else None,
        expected_weight_shift=dict(raw.get("expected_weight_shift", {})),
        case=case,
    )


def load_scenarios(path: str, limit: Optional[int] = None) -> List[HMODScenario]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.load(fh, Loader=yaml.FullLoader)
    raw_scenarios = data.get("scenarios", data) if isinstance(data, dict) else data
    if not isinstance(raw_scenarios, list):
        raise ValueError("H-MOD scenario file must contain a list or a 'scenarios' list")
    scenarios = [parse_scenario(raw) for raw in raw_scenarios]
    return scenarios[:limit] if limit is not None else scenarios
