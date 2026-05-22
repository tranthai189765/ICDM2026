"""
DMORL LLM Controller
Handles:
  1. DeepInfra API calls (Qwen3-32B)
  2. LLM-guided Skill Discovery (basic + advanced)
  3. Dynamic Weight Controller (per-turn intent detection)
  4. Hint Manager (post-dialogue refinement)
"""

import datetime
import json
import os
import re
import copy
from typing import List, Dict, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from loguru import logger

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# DeepInfra client
# ─────────────────────────────────────────────────────────────────────────────
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
DEEPINFRA_BASE_URL = os.getenv("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai")
DEEPINFRA_MODEL = os.getenv("DEEPINFRA_MODEL", "Qwen/Qwen3-32B")

# Set to True via enable_debug() to log all LLM prompts and responses
_DEBUG = False


def enable_debug(flag: bool = True) -> None:
    """Enable or disable verbose LLM prompt/response logging."""
    global _DEBUG
    _DEBUG = flag


def _call_llm(messages: List[Dict], temperature: float = 0.3, max_tokens: int = None) -> str:
    """Call DeepInfra's OpenAI-compatible API and return the text response."""
    if _DEBUG:
        logger.debug("[LLM PROMPT] ─────────────────────────────────────────")
        for m in messages:
            logger.debug(f"  [{m['role'].upper()}] {m['content']}")
        logger.debug("───────────────────────────────────────────────────────")

    client = OpenAI(api_key=DEEPINFRA_API_KEY, base_url=DEEPINFRA_BASE_URL)
    kwargs = dict(model=DEEPINFRA_MODEL, messages=messages, temperature=temperature)
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content.strip()

    if _DEBUG:
        logger.debug(f"[LLM RESPONSE] {text}")
        logger.debug("───────────────────────────────────────────────────────")

    return text


def _parse_json_block(text: str) -> any:
    """Extract and parse the first JSON block found in the LLM response."""
    # Strip <think>...</think> chain-of-thought preamble (Qwen3, DeepSeek-R1, etc.)
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    # Fallback: try parsing the entire text
    return json.loads(text)


def _format_objective_glossary(objective_names: List[str],
                                objective_descriptions: Dict[str, str] = None) -> str:
    """Build a short glossary block to ground the LLM on objective semantics.
    Returns an empty string when no descriptions are available, so the
    surrounding prompt remains clean."""
    if not objective_descriptions:
        return ""
    lines = ["Objective semantics:"]
    for name in objective_names:
        desc = objective_descriptions.get(name)
        if desc:
            lines.append(f"  - {name}: {desc}")
    return "\n".join(lines) + "\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# Skill Library
# ─────────────────────────────────────────────────────────────────────────────
class SkillLibrary:
    """
    Stores basic and advanced skills discovered by the LLM.
    Each skill = {name, description, weight_vector}.
    weight_vector lives on the n_objectives-simplex.
    """

    def __init__(self, n_objectives: int, skills_file: str = "dmorl_skills.json",
                 skill_log_file: str = None):
        self.n_objectives = n_objectives
        self.skills_file = skills_file
        self.skill_log_file = skill_log_file
        self._skill_log_initialized = False  # first write this run → "w", rest → "a"
        self.basic_skills: List[Dict] = []
        self.advanced_skills: List[Dict] = []

    # ── Internal logging helper ──────────────────────────────────────────────

    def _log_skill_discovery(self, phase: str, scenario: str,
                              objective_names: List[str], messages: List[Dict],
                              raw_response: str, skills: List[Dict],
                              fallback: bool = False) -> None:
        """Append a skill discovery record (prompt + response + parsed result) to the log file."""
        if not self.skill_log_file:
            return
        sep = "=" * 70
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        obj_str = ", ".join(objective_names)
        lines = [
            sep,
            f"[Phase 1{'a' if phase == 'basic' else 'b'}] {phase.upper()} SKILL DISCOVERY",
            f"Timestamp : {ts}",
            f"Scenario  : {scenario}",
            f"Objectives: {obj_str}",
            f"Status    : {'FALLBACK (LLM parse failed)' if fallback else 'OK'}",
            sep,
            "",
        ]
        for msg in messages:
            lines.append(f"[PROMPT – {msg['role'].upper()}]")
            lines.append(msg["content"])
            lines.append("")
        lines += [
            "[RAW LLM RESPONSE]",
            raw_response if raw_response else "(none – error before LLM call)",
            "",
            "[PARSED SKILLS]",
        ]
        for i, s in enumerate(skills, 1):
            wv = s.get("weight_vector", [])
            wv_str = "  ".join(f"{objective_names[j]}={wv[j]:.4f}"
                               for j in range(min(len(wv), len(objective_names))))
            lines.append(f"  {i}. {s.get('name', '?')}  [{s.get('type', phase)}]")
            lines.append(f"     {s.get('description', '')}")
            lines.append(f"     Weights: {wv_str}")
        lines += ["", sep, ""]

        try:
            os.makedirs(os.path.dirname(self.skill_log_file) or ".", exist_ok=True)
            mode = "w" if not self._skill_log_initialized else "a"
            with open(self.skill_log_file, mode, encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            self._skill_log_initialized = True
        except Exception as e:
            logger.warning(f"[SkillLibrary] Could not write skill log ({e})")

    # ── LLM-based discovery ──────────────────────────────────────────────────

    def discover_basic_skills(self, scenario: str, objective_names: List[str],
                               n: int = 5,
                               objective_descriptions: Dict[str, str] = None) -> List[Dict]:
        """
        Ask the LLM to propose N basic dialogue skills and map each to a
        weight vector over the objectives.
        """
        obj_str = ", ".join(objective_names)
        glossary = _format_objective_glossary(objective_names, objective_descriptions)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert in multi-objective dialogue policy design. "
                    "Your task is to propose foundational dialogue skills for an agent."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Scenario: {scenario}.\n"
                    f"Objectives (in order): [{obj_str}].\n"
                    f"{glossary}"
                    f"Please propose exactly {n} BASIC dialogue skills for this scenario. "
                    "These should be simple, foundational skills that cover diverse strategy profiles.\n\n"
                    "For each skill provide:\n"
                    "  - name: short skill name\n"
                    "  - description: one-sentence description\n"
                    "  - weight_vector: a list of non-negative floats (one per objective) that sum to 1.0, "
                    "reflecting how much this skill prioritises each objective.\n\n"
                    f"The weight_vector MUST have exactly {len(objective_names)} entries, one per objective in the order above.\n\n"
                    "Respond with ONLY a JSON array, no extra text. Example format:\n"
                    "```json\n"
                    "[\n"
                    '  {"name": "Active Listening", "description": "...", "weight_vector": [0.7, 0.2, 0.1]},\n'
                    "  ...\n"
                    "]\n"
                    "```"
                ),
            },
        ]
        raw = ""
        try:
            raw = _call_llm(messages, temperature=0.5)
            skills = _parse_json_block(raw)
            normalized = []
            for s in skills[:n]:
                wv = s.get("weight_vector", [1.0 / self.n_objectives] * self.n_objectives)
                total = sum(wv)
                wv = [x / total for x in wv]
                normalized.append({
                    "name": s.get("name", f"BasicSkill_{len(normalized)}"),
                    "description": s.get("description", ""),
                    "weight_vector": wv,
                    "type": "basic",
                })
            self.basic_skills = normalized
            self._log_skill_discovery("basic", scenario, objective_names,
                                      messages, raw, normalized, fallback=False)
            logger.info(f"[SkillLibrary] Discovered {len(normalized)} basic skills.")
            return normalized
        except Exception as e:
            logger.warning(f"[SkillLibrary] LLM skill discovery failed ({e}). Using uniform fallback.")
            fallback = self._fallback_skills(n, "basic")
            self._log_skill_discovery("basic", scenario, objective_names,
                                      messages, raw, fallback, fallback=True)
            return fallback

    def discover_advanced_skills(self, scenario: str, objective_names: List[str],
                                  m: int = 5,
                                  objective_descriptions: Dict[str, str] = None) -> List[Dict]:
        """
        Ask the LLM to propose M advanced skills, building on the basic ones.
        Advanced skills model more nuanced, context-dependent strategies.
        """
        obj_str = ", ".join(objective_names)
        glossary = _format_objective_glossary(objective_names, objective_descriptions)
        basic_summary = "\n".join(
            f"  - {s['name']}: {s['description']}" for s in self.basic_skills
        ) or "  (none yet)"
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert in multi-objective dialogue policy design. "
                    "Your task is to propose advanced, composite dialogue skills."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Scenario: {scenario}.\n"
                    f"Objectives (in order): [{obj_str}].\n"
                    f"{glossary}"
                    f"We already have these BASIC skills:\n{basic_summary}\n\n"
                    f"Please propose exactly {m} ADVANCED dialogue skills. "
                    "These should be more nuanced, mixing objectives in sophisticated ways "
                    "that go beyond the basic skills and help handle complex conversation dynamics.\n\n"
                    "For each skill provide:\n"
                    "  - name: short skill name\n"
                    "  - description: one-sentence description\n"
                    "  - weight_vector: a list of non-negative floats (one per objective) summing to 1.0.\n\n"
                    f"The weight_vector MUST have exactly {len(objective_names)} entries, one per objective in the order above.\n\n"
                    "Respond with ONLY a JSON array, no extra text. Example format:\n"
                    "```json\n"
                    "[\n"
                    '  {"name": "Adaptive Negotiation", "description": "...", "weight_vector": [0.4, 0.4, 0.2]},\n'
                    "  ...\n"
                    "]\n"
                    "```"
                ),
            },
        ]
        raw = ""
        try:
            raw = _call_llm(messages, temperature=0.5)
            skills = _parse_json_block(raw)
            normalized = []
            for s in skills[:m]:
                wv = s.get("weight_vector", [1.0 / self.n_objectives] * self.n_objectives)
                total = sum(wv)
                wv = [x / total for x in wv]
                normalized.append({
                    "name": s.get("name", f"AdvancedSkill_{len(normalized)}"),
                    "description": s.get("description", ""),
                    "weight_vector": wv,
                    "type": "advanced",
                })
            self.advanced_skills = normalized
            self._log_skill_discovery("advanced", scenario, objective_names,
                                      messages, raw, normalized, fallback=False)
            logger.info(f"[SkillLibrary] Discovered {len(normalized)} advanced skills.")
            return normalized
        except Exception as e:
            logger.warning(f"[SkillLibrary] Advanced skill discovery failed ({e}). Using uniform fallback.")
            fallback = self._fallback_skills(m, "advanced")
            self._log_skill_discovery("advanced", scenario, objective_names,
                                      messages, raw, fallback, fallback=True)
            return fallback

    def all_weight_vectors(self):
        """Return weight vectors for all skills (basic + advanced) as a list."""
        return [s["weight_vector"] for s in self.basic_skills + self.advanced_skills]

    def _fallback_skills(self, n: int, skill_type: str) -> List[Dict]:
        """Generate simple corner + midpoint weight vectors as fallback."""
        skills = []
        k = self.n_objectives
        for i in range(min(n, k)):
            wv = [0.0] * k
            wv[i] = 1.0
            skills.append({
                "name": f"{skill_type.capitalize()}Skill_Obj{i}",
                "description": f"Maximise objective {i}",
                "weight_vector": wv,
                "type": skill_type,
            })
        # Fill remaining with uniform
        while len(skills) < n:
            skills.append({
                "name": f"{skill_type.capitalize()}Skill_Uniform",
                "description": "Balance all objectives equally",
                "weight_vector": [1.0 / k] * k,
                "type": skill_type,
            })
        if skill_type == "basic":
            self.basic_skills = skills
        else:
            self.advanced_skills = skills
        return skills

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self):
        data = {"basic": self.basic_skills, "advanced": self.advanced_skills}
        with open(self.skills_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"[SkillLibrary] Skills saved to {self.skills_file}")

    def load(self) -> bool:
        if os.path.exists(self.skills_file):
            with open(self.skills_file, "r") as f:
                data = json.load(f)
            self.basic_skills = data.get("basic", [])
            self.advanced_skills = data.get("advanced", [])
            logger.info(
                f"[SkillLibrary] Loaded {len(self.basic_skills)} basic + "
                f"{len(self.advanced_skills)} advanced skills."
            )
            return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic Weight Controller
# ─────────────────────────────────────────────────────────────────────────────
class DynamicWeightController:
    """
    At inference time, reads the current dialogue context + accumulated hints
    and asks the LLM for an optimal objective weight vector for the next T steps.
    """

    def __init__(self, n_objectives: int, objective_names: List[str],
                 horizon: int = 3):
        self.n_objectives = n_objectives
        self.objective_names = objective_names
        self.horizon = horizon  # T steps before re-querying LLM

    def get_weight(self, dialogue_history: List[Dict], hints: List[str],
                   scenario: str) -> List[float]:
        """
        Returns a weight vector (list of floats summing to 1) for the next
        `self.horizon` dialogue turns based on current context and hints.
        """
        # Format dialogue history (keep last 10 turns to stay within token limit)
        recent = dialogue_history[-10:] if len(dialogue_history) > 10 else dialogue_history
        history_str = "\n".join(
            f"  {u['role'].upper()}: {u['content']}" for u in recent
        ) or "  (conversation just started)"

        hints_str = "\n".join(f"  - {h}" for h in hints[-5:]) if hints else "  (no hints yet)"
        obj_str = ", ".join(self.objective_names)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strategic dialogue controller for a multi-objective conversational agent. "
                    "Your job is to assign objective weights based on the current conversation state."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Scenario: {scenario}\n"
                    f"Objectives (in order): [{obj_str}]\n\n"
                    f"Recent conversation:\n{history_str}\n\n"
                    f"Accumulated strategic hints:\n{hints_str}\n\n"
                    f"Based on the conversation and hints, provide a weight vector of {self.n_objectives} "
                    "non-negative floats summing to 1.0 that the agent should use for the next "
                    f"{self.horizon} dialogue turns to best achieve the overall goal.\n\n"
                    "Respond with ONLY a JSON object, no extra text:\n"
                    '```json\n{"weight_vector": [<float>, ...]}\n```'
                ),
            },
        ]
        try:
            raw = _call_llm(messages, temperature=0.2, max_tokens=128)
            parsed = _parse_json_block(raw)
            wv = parsed["weight_vector"]
            total = sum(wv)
            if total <= 0:
                raise ValueError("Weight vector sums to 0")
            wv = [x / total for x in wv]
            logger.info(f"[DynamicWeight] LLM weight: {[round(x, 3) for x in wv]}")
            return wv
        except Exception as e:
            logger.warning(f"[DynamicWeight] Failed ({e}). Using uniform weight.")
            return [1.0 / self.n_objectives] * self.n_objectives


# ─────────────────────────────────────────────────────────────────────────────
# Hint Manager
# ─────────────────────────────────────────────────────────────────────────────
class HintManager:
    """
    After each dialogue, asks the LLM to reflect on what happened and generate
    tactical hints. These hints are persisted and fed back at the next inference.
    """

    MAX_HINTS = 20  # cap to avoid unbounded growth

    def __init__(self, hints_file: str = "dmorl_hints.json"):
        self.hints_file = hints_file
        self.hints: List[str] = []
        self.load()

    def generate_hints(self, dialogue_history: List[Dict], outcome: str,
                       scenario: str, objective_names: List[str]) -> List[str]:
        """
        Call LLM to derive tactical hints from a finished dialogue.
        `outcome`: e.g. "success", "failure", "partial"
        """
        history_str = "\n".join(
            f"  {u['role'].upper()}: {u['content']}" for u in dialogue_history
        )
        obj_str = ", ".join(objective_names)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a dialogue strategy analyst. "
                    "Your task is to extract actionable tactical hints from a completed conversation."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Scenario: {scenario}\n"
                    f"Objectives: [{obj_str}]\n"
                    f"Dialogue outcome: {outcome}\n\n"
                    f"Conversation:\n{history_str}\n\n"
                    "Based on this conversation and its outcome, generate up to 3 concise tactical hints "
                    "that an agent should follow in FUTURE similar conversations to improve outcomes. "
                    "Hints should be specific about WHEN to shift objective priorities.\n\n"
                    "Respond with ONLY a JSON array of hint strings, no extra text:\n"
                    '```json\n["hint 1", "hint 2", ...]\n```'
                ),
            },
        ]
        try:
            raw = _call_llm(messages, temperature=0.4, max_tokens=300)
            new_hints = _parse_json_block(raw)
            if isinstance(new_hints, list):
                new_hints = [h for h in new_hints if isinstance(h, str)]
                self.hints.extend(new_hints)
                # Keep only the most recent MAX_HINTS
                self.hints = self.hints[-self.MAX_HINTS:]
                self.save()
                logger.info(f"[HintManager] Added {len(new_hints)} hints. Total: {len(self.hints)}")
                return new_hints
        except Exception as e:
            logger.warning(f"[HintManager] Hint generation failed ({e}).")
        return []

    def get_hints(self) -> List[str]:
        return list(self.hints)

    def save(self):
        with open(self.hints_file, "w") as f:
            json.dump(self.hints, f, indent=2)

    def load(self):
        if os.path.exists(self.hints_file):
            with open(self.hints_file, "r") as f:
                self.hints = json.load(f)
            logger.info(f"[HintManager] Loaded {len(self.hints)} hints from {self.hints_file}")


# ─────────────────────────────────────────────────────────────────────────────
# Master Controller
# ─────────────────────────────────────────────────────────────────────────────
class DMORLController:
    """
    Aggregates SkillLibrary, DynamicWeightController, and HintManager
    into a single interface used by DMORLTrainer and DMORLPipeline.
    """

    def __init__(
        self,
        n_objectives: int,
        objective_names: List[str],
        scenario: str,
        n_basic_skills: int = 5,
        n_advanced_skills: int = 5,
        dynamic_weight_horizon: int = 3,
        skills_file: str = "dmorl_skills.json",
        hints_file: str = "dmorl_hints.json",
        skill_log_file: str = None,
        objective_descriptions: Dict[str, str] = None,
    ):
        self.n_objectives = n_objectives
        self.objective_names = objective_names
        self.objective_descriptions = objective_descriptions or {}
        self.scenario = scenario
        self.n_basic_skills = n_basic_skills
        self.n_advanced_skills = n_advanced_skills

        self.skill_library = SkillLibrary(n_objectives, skills_file, skill_log_file)
        self.weight_controller = DynamicWeightController(
            n_objectives, objective_names, dynamic_weight_horizon
        )
        self.hint_manager = HintManager(hints_file)

    # ── Phase 1: Skill Discovery ──────────────────────────────────────────────

    def initialize_skills(self, force_rediscover: bool = False) -> None:
        """Load skills from file; if missing or forced, discover via LLM."""
        if not force_rediscover and self.skill_library.load():
            return
        logger.info("[DMORLController] Discovering basic skills via LLM ...")
        self.skill_library.discover_basic_skills(
            self.scenario, self.objective_names, self.n_basic_skills,
            objective_descriptions=self.objective_descriptions,
        )
        logger.info("[DMORLController] Discovering advanced skills via LLM ...")
        self.skill_library.discover_advanced_skills(
            self.scenario, self.objective_names, self.n_advanced_skills,
            objective_descriptions=self.objective_descriptions,
        )
        self.skill_library.save()

    # ── Phase 2: Dynamic Inference ─────────────────────────────────────────────

    def get_dynamic_weight(self, dialogue_history: List[Dict]) -> List[float]:
        hints = self.hint_manager.get_hints()
        return self.weight_controller.get_weight(dialogue_history, hints, self.scenario)

    # ── Phase 3: Post-Dialogue Refinement ─────────────────────────────────────

    def refine_after_dialogue(self, dialogue_history: List[Dict], outcome: str) -> List[str]:
        return self.hint_manager.generate_hints(
            dialogue_history, outcome, self.scenario, self.objective_names
        )
