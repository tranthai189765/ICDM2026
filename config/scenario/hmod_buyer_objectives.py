BUYER_STRATEGY_INTENTS = {
    "AGGRESSIVE_SAVINGS_THEN_RECOVERY": {
        "description": "Push hard for savings early, then recover the deal if seller firmness or walkaway risk appears.",
        "natural_language_intent": "The buyer wants a very low price at first, but the policy must stop over-haggling once the seller signals frustration, firmness, or a credible final offer.",
        "typical_steps": [
            "Open with a low but defensible offer",
            "Use alternatives and budget pressure to justify the ask",
            "Detect seller firmness or frustration",
            "Shift to a serious counter inside the ceiling",
            "Close if the seller reaches a price that preserves enough buyer surplus",
        ],
        "stage_weights": {
            "initial": [0.72, 0.10, 0.13, 0.05],
            "firm_response": [0.42, 0.22, 0.28, 0.08],
            "final_offer_response": [0.30, 0.24, 0.36, 0.10],
            "walkaway_response": [0.26, 0.28, 0.36, 0.10],
            "above_ceiling_defense": [0.62, 0.15, 0.16, 0.07]
        },
        "adaptation_rules": [
            {
                "when": {"seller_offer_above_ceiling": True},
                "target_stage": "above_ceiling_defense",
                "summary": "seller price is above buyer ceiling, defend the constraint before chasing deal rate"
            },
            {
                "when": {"seller_intent": "walkaway_risk"},
                "target_stage": "walkaway_response",
                "summary": "walkaway risk detected, convert from lowballing to deal recovery"
            },
            {
                "when": {"seller_intent": "final_offer"},
                "target_stage": "final_offer_response",
                "summary": "final offer detected, make one serious close attempt inside ceiling"
            },
            {
                "when": {"seller_intent": "firm"},
                "target_stage": "firm_response",
                "summary": "seller became firm, reduce price aggression and raise deal probability"
            }
        ]
    },
    "URGENT_GIFT_WITH_HARD_CEILING": {
        "description": "Buy quickly for a time-sensitive need, but treat the maximum acceptable price as a hard constraint.",
        "natural_language_intent": "The buyer needs the item soon and should close fast when the seller is inside budget, yet must not exceed the ceiling even under deadline pressure.",
        "typical_steps": [
            "Signal serious intent and time pressure",
            "Move rapidly toward the acceptable range",
            "Accept a seller offer inside the ceiling",
            "If seller asks above ceiling, hold budget and ask for a small concession",
        ],
        "stage_weights": {
            "initial": [0.24, 0.16, 0.48, 0.12],
            "within_ceiling_close": [0.12, 0.20, 0.54, 0.14],
            "final_offer_response": [0.16, 0.22, 0.48, 0.14],
            "above_ceiling_defense": [0.58, 0.14, 0.20, 0.08]
        },
        "adaptation_rules": [
            {
                "when": {"seller_offer_within_ceiling": True},
                "target_stage": "within_ceiling_close",
                "summary": "seller offer is inside ceiling, prioritize immediate purchase"
            },
            {
                "when": {"seller_offer_above_ceiling": True},
                "target_stage": "above_ceiling_defense",
                "summary": "urgent buyer still refuses to cross the hard budget ceiling"
            },
            {
                "when": {"seller_intent": "final_offer"},
                "target_stage": "final_offer_response",
                "summary": "final offer creates urgency, but only close inside ceiling"
            }
        ]
    },
    "QUALITY_RISK_THEN_PRICE_PUSH": {
        "description": "Start by reducing product-quality risk, then use verified condition information to negotiate price.",
        "natural_language_intent": "The buyer should first ask about condition, warranty, inspection, and missing accessories. After enough information is gathered, it should push for price gain unless a final offer appears.",
        "typical_steps": [
            "Ask about condition, warranty, proof, and defects",
            "Avoid committing before uncertainty is reduced",
            "Use any risk signal to justify a lower offer",
            "If the seller gives a credible final offer within ceiling, close instead of over-checking",
        ],
        "stage_weights": {
            "initial": [0.28, 0.40, 0.22, 0.10],
            "late_price_push": [0.56, 0.20, 0.18, 0.06],
            "firm_response": [0.40, 0.28, 0.24, 0.08],
            "final_offer_response": [0.24, 0.28, 0.38, 0.10]
        },
        "adaptation_rules": [
            {
                "when": {"turn_gte": 3},
                "target_stage": "late_price_push",
                "summary": "after quality checks, shift from risk reduction to price leverage"
            },
            {
                "when": {"seller_intent": "firm"},
                "target_stage": "firm_response",
                "summary": "seller firmness requires a balanced risk-price counter"
            },
            {
                "when": {"seller_intent": "final_offer"},
                "target_stage": "final_offer_response",
                "summary": "final offer shifts priority from further checks to closing safely"
            }
        ]
    },
    "BACKUP_OPTION_PRESSURE_CONTROL": {
        "description": "Use a competing alternative to bargain, but avoid losing this seller if their price becomes competitive.",
        "natural_language_intent": "The buyer has another option and can pressure the seller early, but should close quickly if the seller reaches the buyer ceiling or starts to walk away.",
        "typical_steps": [
            "Mention a credible backup option",
            "Ask the seller to beat the alternative",
            "Keep a hard price ceiling",
            "When the seller is close enough, stop using pressure and secure the item",
        ],
        "stage_weights": {
            "initial": [0.64, 0.12, 0.16, 0.08],
            "within_ceiling_close": [0.18, 0.20, 0.50, 0.12],
            "walkaway_response": [0.22, 0.24, 0.42, 0.12],
            "above_ceiling_defense": [0.66, 0.12, 0.14, 0.08]
        },
        "adaptation_rules": [
            {
                "when": {"seller_offer_within_ceiling": True},
                "target_stage": "within_ceiling_close",
                "summary": "seller is competitive against backup option, close before losing availability"
            },
            {
                "when": {"seller_intent": "walkaway_risk"},
                "target_stage": "walkaway_response",
                "summary": "backup pressure is causing walkaway risk, soften and close if viable"
            },
            {
                "when": {"seller_offer_above_ceiling": True},
                "target_stage": "above_ceiling_defense",
                "summary": "backup option justifies holding the buyer ceiling"
            }
        ]
    },
    "SCARCITY_ADAPTIVE_BUYER": {
        "description": "Handle scarcity pressure from a seller who claims another buyer is ready.",
        "natural_language_intent": "The buyer should not panic-buy above budget, but if a scarce item reaches the ceiling it should move fast instead of continuing a low-price strategy.",
        "typical_steps": [
            "Start with a fair but conservative offer",
            "If seller introduces another buyer, assess whether the current price is inside ceiling",
            "Close rapidly inside ceiling",
            "Walk away if scarcity pressure asks for over-budget payment",
        ],
        "stage_weights": {
            "initial": [0.36, 0.24, 0.30, 0.10],
            "final_offer_response": [0.14, 0.22, 0.50, 0.14],
            "within_ceiling_close": [0.12, 0.18, 0.54, 0.16],
            "above_ceiling_defense": [0.60, 0.16, 0.16, 0.08]
        },
        "adaptation_rules": [
            {
                "when": {"seller_intent": "final_offer", "seller_offer_within_ceiling": True},
                "target_stage": "within_ceiling_close",
                "summary": "scarcity final offer is inside ceiling, close immediately"
            },
            {
                "when": {"seller_intent": "final_offer"},
                "target_stage": "final_offer_response",
                "summary": "scarcity pressure raises urgency but keeps budget discipline"
            },
            {
                "when": {"seller_offer_above_ceiling": True},
                "target_stage": "above_ceiling_defense",
                "summary": "scarcity pressure is above ceiling, resist overpaying"
            }
        ]
    },
    "FAIR_RELATIONSHIP_REPEAT_BUYER": {
        "description": "Prefer a fair relationship-preserving deal for repeat purchases, but still respond to seller deadlines.",
        "natural_language_intent": "The buyer values fairness and a smooth relationship because the seller may be useful later. It should avoid exploitative lowballing, but if the seller becomes final or impatient, it should close within budget.",
        "typical_steps": [
            "Acknowledge the seller's stated value",
            "Offer a fair midpoint rather than an aggressive lowball",
            "Maintain polite framing during disagreement",
            "Close within ceiling when the seller becomes impatient",
        ],
        "stage_weights": {
            "initial": [0.24, 0.46, 0.22, 0.08],
            "firm_response": [0.24, 0.34, 0.32, 0.10],
            "final_offer_response": [0.18, 0.30, 0.40, 0.12],
            "walkaway_response": [0.18, 0.34, 0.36, 0.12]
        },
        "adaptation_rules": [
            {
                "when": {"seller_intent": "firm"},
                "target_stage": "firm_response",
                "summary": "seller firmness calls for a fairer close-oriented counter"
            },
            {
                "when": {"seller_intent": "final_offer"},
                "target_stage": "final_offer_response",
                "summary": "relationship buyer should not prolong a credible final offer"
            },
            {
                "when": {"seller_intent": "walkaway_risk"},
                "target_stage": "walkaway_response",
                "summary": "walkaway risk threatens future relationship, prioritize respectful close"
            }
        ]
    },
    "BUDGET_LOCKED_FLEXIBLE_TIMING": {
        "description": "The buyer can wait or walk away, so budget discipline dominates unless the seller comes inside the ceiling.",
        "natural_language_intent": "The buyer has flexible timing and must never overpay. If seller remains above ceiling, the policy should preserve price gain and accept no deal; if seller comes under ceiling, close calmly.",
        "typical_steps": [
            "State a firm budget ceiling politely",
            "Reject over-ceiling offers without escalating",
            "Use time flexibility as leverage",
            "Accept only when the seller is inside the maximum acceptable price",
        ],
        "stage_weights": {
            "initial": [0.66, 0.16, 0.10, 0.08],
            "above_ceiling_defense": [0.74, 0.12, 0.08, 0.06],
            "within_ceiling_close": [0.20, 0.24, 0.42, 0.14],
            "walkaway_response": [0.58, 0.18, 0.14, 0.10]
        },
        "adaptation_rules": [
            {
                "when": {"seller_offer_above_ceiling": True},
                "target_stage": "above_ceiling_defense",
                "summary": "budget-locked buyer should prefer no deal over violating the ceiling"
            },
            {
                "when": {"seller_offer_within_ceiling": True},
                "target_stage": "within_ceiling_close",
                "summary": "seller finally entered budget, switch to closing"
            },
            {
                "when": {"seller_intent": "walkaway_risk"},
                "target_stage": "walkaway_response",
                "summary": "walkaway pressure is acceptable because timing is flexible"
            }
        ]
    },
    "LONG_HAGGLE_FATIGUE_CONTROL": {
        "description": "Bargain for savings early, but after a long negotiation reduce friction and finish efficiently.",
        "natural_language_intent": "The buyer can haggle for several rounds, but if the conversation drags on it should stop optimizing only price and move toward a bounded close.",
        "typical_steps": [
            "Start with a savings-oriented counter",
            "Make small concessions while seller remains neutral",
            "After enough unresolved turns, increase urgency and deal-rate priority",
            "If still above ceiling, make a final bounded offer and walk away",
        ],
        "stage_weights": {
            "initial": [0.60, 0.14, 0.18, 0.08],
            "fatigue_close": [0.26, 0.22, 0.36, 0.16],
            "firm_response": [0.34, 0.24, 0.30, 0.12],
            "final_offer_response": [0.20, 0.24, 0.42, 0.14]
        },
        "adaptation_rules": [
            {
                "when": {"turn_gte": 4},
                "target_stage": "fatigue_close",
                "summary": "long negotiation detected, reduce friction and move toward bounded close"
            },
            {
                "when": {"seller_intent": "firm"},
                "target_stage": "firm_response",
                "summary": "seller firmness during a long haggle requires faster convergence"
            },
            {
                "when": {"seller_intent": "final_offer"},
                "target_stage": "final_offer_response",
                "summary": "final offer after long haggle should trigger efficient closing behavior"
            }
        ]
    },
    "PRICE_GAIN_BUT_DEAL_AWARE": {
        "description": "Backward-compatible alias for price-gain-first buyer behavior.",
        "natural_language_intent": "The buyer wants the lowest possible price, but should adapt if deal-loss risk becomes high.",
        "typical_steps": ["Start low", "Observe seller firmness", "Concede if needed"],
        "stage_weights": {
            "initial": [0.70, 0.10, 0.15, 0.05],
            "firm_response": [0.42, 0.22, 0.28, 0.08],
            "walkaway_response": [0.26, 0.28, 0.36, 0.10]
        },
        "adaptation_rules": [
            {
                "when": {"seller_intent": "firm"},
                "target_stage": "firm_response",
                "summary": "backward-compatible firm seller adaptation"
            },
            {
                "when": {"seller_intent": "walkaway_risk"},
                "target_stage": "walkaway_response",
                "summary": "backward-compatible walkaway adaptation"
            }
        ]
    },
    "URGENT_PURCHASE_WITH_BUDGET_CEILING": {
        "description": "Backward-compatible alias for urgent purchase with a buyer price ceiling.",
        "natural_language_intent": "The buyer needs to close quickly but cannot exceed the maximum acceptable price.",
        "typical_steps": ["Signal urgency", "Move closer", "Stop at ceiling"],
        "stage_weights": {
            "initial": [0.24, 0.16, 0.48, 0.12],
            "final_offer_response": [0.16, 0.22, 0.48, 0.14],
            "above_ceiling_defense": [0.58, 0.14, 0.20, 0.08]
        },
        "adaptation_rules": [
            {
                "when": {"seller_offer_above_ceiling": True},
                "target_stage": "above_ceiling_defense",
                "summary": "urgent buyer still respects the ceiling"
            },
            {
                "when": {"seller_intent": "final_offer"},
                "target_stage": "final_offer_response",
                "summary": "urgent final-offer adaptation"
            }
        ]
    },
    "FAIR_VALUE_FAST_CLOSE": {
        "description": "Backward-compatible alias for fair fast closing.",
        "natural_language_intent": "The buyer wants a reasonable deal for both sides and efficient agreement.",
        "typical_steps": ["Acknowledge value", "Offer midpoint", "Close fairly"],
        "stage_weights": {
            "initial": [0.24, 0.42, 0.24, 0.10],
            "firm_response": [0.24, 0.34, 0.32, 0.10]
        },
        "adaptation_rules": [
            {
                "when": {"seller_intent": "firm"},
                "target_stage": "firm_response",
                "summary": "fair buyer becomes more close-oriented under firmness"
            }
        ]
    },
    "WALKAWAY_RISK_ADAPTIVE_BUYER": {
        "description": "Backward-compatible alias for walkaway-risk adaptation.",
        "natural_language_intent": "The buyer can push early but should recover when the seller threatens to leave.",
        "typical_steps": ["Start low", "Detect frustration", "Make a serious counter"],
        "stage_weights": {
            "initial": [0.68, 0.10, 0.15, 0.07],
            "walkaway_response": [0.24, 0.28, 0.38, 0.10]
        },
        "adaptation_rules": [
            {
                "when": {"seller_intent": "walkaway_risk"},
                "target_stage": "walkaway_response",
                "summary": "recover the deal when seller walkaway risk appears"
            }
        ]
    },
    "RISK_AWARE_QUALITY_CHECK": {
        "description": "Backward-compatible alias for quality-risk-aware buying.",
        "natural_language_intent": "The buyer wants to reduce uncertainty through inspection or proof before committing.",
        "typical_steps": ["Ask condition", "Request proof", "Use risk to negotiate"],
        "stage_weights": {
            "initial": [0.28, 0.40, 0.22, 0.10],
            "late_price_push": [0.56, 0.20, 0.18, 0.06]
        },
        "adaptation_rules": [
            {
                "when": {"turn_gte": 3},
                "target_stage": "late_price_push",
                "summary": "after quality checks, move to price negotiation"
            }
        ]
    }
}

BUYER_STRATEGY_MACRO_CLUSTERS = {
    "PRICE_GAIN": [
        "AGGRESSIVE_SAVINGS_THEN_RECOVERY",
        "PRICE_GAIN_BUT_DEAL_AWARE"
    ],
    "FAST_PURCHASE": [
        "URGENT_GIFT_WITH_HARD_CEILING",
        "URGENT_PURCHASE_WITH_BUDGET_CEILING"
    ],
    "FAIR_VALUE": [
        "FAIR_RELATIONSHIP_REPEAT_BUYER",
        "FAIR_VALUE_FAST_CLOSE"
    ],
    "RISK_AWARE_PURCHASE": [
        "QUALITY_RISK_THEN_PRICE_PUSH",
        "RISK_AWARE_QUALITY_CHECK"
    ],
    "WALKAWAY_DISCIPLINE": [
        "BACKUP_OPTION_PRESSURE_CONTROL",
        "BUDGET_LOCKED_FLEXIBLE_TIMING",
        "WALKAWAY_RISK_ADAPTIVE_BUYER"
    ],
    "SCARCITY_RESPONSE": [
        "SCARCITY_ADAPTIVE_BUYER"
    ],
    "ADAPTIVE_BUYER_CONTROL": [
        "LONG_HAGGLE_FATIGUE_CONTROL"
    ]
}
