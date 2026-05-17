"""
Unit tests for dmorl/llm_controller.py

All LLM API calls are mocked — no network access or API keys required.
"""
import json
import os
import pytest
from unittest.mock import patch, call

from dmorl.llm_controller import (
    _parse_json_block,
    SkillLibrary,
    DynamicWeightController,
    HintManager,
    DMORLController,
)

LLM_PATCH = "dmorl.llm_controller._call_llm"


# ─────────────────────────────────────────────────────────────────────────────
# _parse_json_block
# ─────────────────────────────────────────────────────────────────────────────

class TestParseJsonBlock:
    def test_fenced_with_json_label(self):
        text = '```json\n{"key": "value"}\n```'
        assert _parse_json_block(text) == {"key": "value"}

    def test_fenced_without_label(self):
        text = '```\n{"key": 42}\n```'
        assert _parse_json_block(text) == {"key": 42}

    def test_unfenced_object(self):
        assert _parse_json_block('{"a": 1}') == {"a": 1}

    def test_unfenced_array(self):
        assert _parse_json_block('[1, 2, 3]') == [1, 2, 3]

    def test_fenced_array(self):
        text = '```json\n["h1", "h2"]\n```'
        assert _parse_json_block(text) == ["h1", "h2"]

    def test_invalid_json_raises(self):
        with pytest.raises(Exception):
            _parse_json_block("not json at all !!!")

    def test_nested_json(self):
        data = {"skills": [{"name": "S", "weight_vector": [0.5, 0.5]}]}
        text = f'```json\n{json.dumps(data)}\n```'
        assert _parse_json_block(text) == data


# ─────────────────────────────────────────────────────────────────────────────
# SkillLibrary — fallback
# ─────────────────────────────────────────────────────────────────────────────

class TestSkillLibraryFallback:
    def test_corner_vectors_when_n_equals_n_objectives(self):
        lib = SkillLibrary(n_objectives=3)
        skills = lib._fallback_skills(3, "basic")
        assert len(skills) == 3
        for i, s in enumerate(skills):
            wv = s["weight_vector"]
            assert len(wv) == 3
            assert abs(sum(wv) - 1.0) < 1e-6
            assert abs(wv[i] - 1.0) < 1e-6, f"skill[{i}] should be a corner vector"

    def test_extra_slots_filled_with_uniform(self):
        lib = SkillLibrary(n_objectives=2)
        skills = lib._fallback_skills(5, "basic")
        assert len(skills) == 5
        for s in skills[2:]:
            wv = s["weight_vector"]
            assert all(abs(w - 0.5) < 1e-6 for w in wv), "extra skills must be uniform"

    def test_all_weights_sum_to_one(self):
        lib = SkillLibrary(n_objectives=3)
        for s in lib._fallback_skills(6, "advanced"):
            assert abs(sum(s["weight_vector"]) - 1.0) < 1e-6

    def test_sets_basic_skills_attribute(self):
        lib = SkillLibrary(n_objectives=2)
        skills = lib._fallback_skills(3, "basic")
        assert lib.basic_skills is skills

    def test_sets_advanced_skills_attribute(self):
        lib = SkillLibrary(n_objectives=2)
        skills = lib._fallback_skills(3, "advanced")
        assert lib.advanced_skills is skills

    def test_type_field_set_correctly(self):
        lib = SkillLibrary(n_objectives=2)
        for s in lib._fallback_skills(2, "basic"):
            assert s["type"] == "basic"
        for s in lib._fallback_skills(2, "advanced"):
            assert s["type"] == "advanced"


# ─────────────────────────────────────────────────────────────────────────────
# SkillLibrary — LLM-based discovery
# ─────────────────────────────────────────────────────────────────────────────

def _skill_json(n: int, n_obj: int, skill_type: str = "basic") -> str:
    skills = [
        {
            "name": f"Skill{i}",
            "description": f"desc {i}",
            "weight_vector": [1.0 / n_obj] * n_obj,
        }
        for i in range(n)
    ]
    return f"```json\n{json.dumps(skills)}\n```"


class TestSkillLibraryDiscover:
    @patch(LLM_PATCH)
    def test_basic_returns_n_skills(self, mock_llm):
        mock_llm.return_value = _skill_json(3, 2)
        lib = SkillLibrary(n_objectives=2)
        skills = lib.discover_basic_skills("recommendation", ["user_reward", "item_freq"], n=3)
        assert len(skills) == 3

    @patch(LLM_PATCH)
    def test_basic_normalizes_weights(self, mock_llm):
        # LLM returns un-normalized weights [2, 6] → should become [0.25, 0.75]
        raw = [{"name": "S", "description": "d", "weight_vector": [2.0, 6.0]}]
        mock_llm.return_value = f"```json\n{json.dumps(raw)}\n```"
        lib = SkillLibrary(n_objectives=2)
        skills = lib.discover_basic_skills("recommendation", ["user_reward", "item_freq"], n=1)
        wv = skills[0]["weight_vector"]
        assert abs(sum(wv) - 1.0) < 1e-6
        assert abs(wv[0] - 0.25) < 1e-6
        assert abs(wv[1] - 0.75) < 1e-6

    @patch(LLM_PATCH)
    def test_basic_truncates_extra_skills(self, mock_llm):
        mock_llm.return_value = _skill_json(10, 2)
        lib = SkillLibrary(n_objectives=2)
        skills = lib.discover_basic_skills("recommendation", ["a", "b"], n=3)
        assert len(skills) == 3

    @patch(LLM_PATCH)
    def test_basic_sets_type_field(self, mock_llm):
        mock_llm.return_value = _skill_json(2, 2)
        lib = SkillLibrary(n_objectives=2)
        skills = lib.discover_basic_skills("recommendation", ["a", "b"], n=2)
        assert all(s["type"] == "basic" for s in skills)

    @patch(LLM_PATCH)
    def test_basic_api_error_falls_back(self, mock_llm):
        mock_llm.side_effect = Exception("API timeout")
        lib = SkillLibrary(n_objectives=2)
        skills = lib.discover_basic_skills("recommendation", ["a", "b"], n=3)
        assert len(skills) == 3
        assert lib.basic_skills == skills

    @patch(LLM_PATCH)
    def test_basic_invalid_json_falls_back(self, mock_llm):
        mock_llm.return_value = "not json"
        lib = SkillLibrary(n_objectives=2)
        skills = lib.discover_basic_skills("recommendation", ["a", "b"], n=2)
        assert len(skills) == 2

    @patch(LLM_PATCH)
    def test_advanced_sets_type_field(self, mock_llm):
        raw = [{"name": "A", "description": "d", "weight_vector": [0.6, 0.4]}]
        mock_llm.return_value = f"```json\n{json.dumps(raw)}\n```"
        lib = SkillLibrary(n_objectives=2)
        skills = lib.discover_advanced_skills("recommendation", ["a", "b"], m=1)
        assert skills[0]["type"] == "advanced"

    @patch(LLM_PATCH)
    def test_advanced_includes_basic_context_in_prompt(self, mock_llm):
        raw = [{"name": "A", "description": "d", "weight_vector": [0.5, 0.5]}]
        mock_llm.return_value = f"```json\n{json.dumps(raw)}\n```"
        lib = SkillLibrary(n_objectives=2)
        lib.basic_skills = [{"name": "BasicOne", "description": "basic", "weight_vector": [1.0, 0.0]}]
        lib.discover_advanced_skills("recommendation", ["a", "b"], m=1)

        user_msg = next(m["content"] for m in mock_llm.call_args[0][0] if m["role"] == "user")
        assert "BasicOne" in user_msg

    @patch(LLM_PATCH)
    def test_advanced_api_error_falls_back(self, mock_llm):
        mock_llm.side_effect = RuntimeError("network error")
        lib = SkillLibrary(n_objectives=2)
        skills = lib.discover_advanced_skills("negotiation", ["sl", "fair", "deal"], m=3)
        assert len(skills) == 3


# ─────────────────────────────────────────────────────────────────────────────
# SkillLibrary — all_weight_vectors
# ─────────────────────────────────────────────────────────────────────────────

class TestAllWeightVectors:
    def test_empty_when_no_skills(self):
        lib = SkillLibrary(n_objectives=2)
        assert lib.all_weight_vectors() == []

    def test_basic_only(self):
        lib = SkillLibrary(n_objectives=2)
        lib.basic_skills = [{"weight_vector": [1.0, 0.0]}, {"weight_vector": [0.5, 0.5]}]
        assert lib.all_weight_vectors() == [[1.0, 0.0], [0.5, 0.5]]

    def test_combined_basic_plus_advanced(self):
        lib = SkillLibrary(n_objectives=2)
        lib.basic_skills = [{"weight_vector": [1.0, 0.0]}]
        lib.advanced_skills = [{"weight_vector": [0.3, 0.7]}]
        vectors = lib.all_weight_vectors()
        assert len(vectors) == 2
        assert [1.0, 0.0] in vectors
        assert [0.3, 0.7] in vectors


# ─────────────────────────────────────────────────────────────────────────────
# SkillLibrary — save / load
# ─────────────────────────────────────────────────────────────────────────────

class TestSkillLibrarySaveLoad:
    def test_roundtrip_preserves_skills(self, tmp_path):
        path = str(tmp_path / "skills.json")
        lib = SkillLibrary(n_objectives=2, skills_file=path)
        lib.basic_skills = [{"name": "B", "weight_vector": [1.0, 0.0], "type": "basic"}]
        lib.advanced_skills = [{"name": "A", "weight_vector": [0.5, 0.5], "type": "advanced"}]
        lib.save()

        lib2 = SkillLibrary(n_objectives=2, skills_file=path)
        assert lib2.load() is True
        assert lib2.basic_skills == lib.basic_skills
        assert lib2.advanced_skills == lib.advanced_skills

    def test_load_returns_false_for_missing_file(self, tmp_path):
        lib = SkillLibrary(n_objectives=2, skills_file=str(tmp_path / "no.json"))
        assert lib.load() is False

    def test_load_returns_empty_lists_on_missing_keys(self, tmp_path):
        path = str(tmp_path / "s.json")
        with open(path, "w") as f:
            json.dump({}, f)
        lib = SkillLibrary(n_objectives=2, skills_file=path)
        lib.load()
        assert lib.basic_skills == []
        assert lib.advanced_skills == []


# ─────────────────────────────────────────────────────────────────────────────
# DynamicWeightController
# ─────────────────────────────────────────────────────────────────────────────

class TestDynamicWeightController:
    @patch(LLM_PATCH)
    def test_returns_normalized_weight(self, mock_llm):
        mock_llm.return_value = '```json\n{"weight_vector": [3.0, 1.0]}\n```'
        ctrl = DynamicWeightController(n_objectives=2, objective_names=["a", "b"])
        w = ctrl.get_weight([], [], "recommendation")
        assert len(w) == 2
        assert abs(sum(w) - 1.0) < 1e-6
        assert abs(w[0] - 0.75) < 1e-6
        assert abs(w[1] - 0.25) < 1e-6

    @patch(LLM_PATCH)
    def test_api_failure_returns_uniform(self, mock_llm):
        mock_llm.side_effect = Exception("timeout")
        ctrl = DynamicWeightController(n_objectives=3, objective_names=["a", "b", "c"])
        w = ctrl.get_weight([], [], "negotiation")
        assert len(w) == 3
        assert all(abs(x - 1 / 3) < 1e-6 for x in w)

    @patch(LLM_PATCH)
    def test_zero_sum_weight_falls_back_to_uniform(self, mock_llm):
        mock_llm.return_value = '```json\n{"weight_vector": [0.0, 0.0]}\n```'
        ctrl = DynamicWeightController(n_objectives=2, objective_names=["a", "b"])
        w = ctrl.get_weight([], [], "recommendation")
        assert all(abs(x - 0.5) < 1e-6 for x in w)

    @patch(LLM_PATCH)
    def test_trims_history_to_last_10_turns(self, mock_llm):
        mock_llm.return_value = '```json\n{"weight_vector": [0.5, 0.5]}\n```'
        ctrl = DynamicWeightController(n_objectives=2, objective_names=["a", "b"])
        history = [{"role": "user", "content": f"msg{i}"} for i in range(20)]
        ctrl.get_weight(history, [], "recommendation")

        user_content = next(
            m["content"] for m in mock_llm.call_args[0][0] if m["role"] == "user"
        )
        assert "msg19" in user_content   # most recent → included
        assert "msg0" not in user_content  # oldest 10 → trimmed

    @patch(LLM_PATCH)
    def test_hints_appear_in_prompt(self, mock_llm):
        mock_llm.return_value = '```json\n{"weight_vector": [0.5, 0.5]}\n```'
        ctrl = DynamicWeightController(n_objectives=2, objective_names=["a", "b"])
        ctrl.get_weight([], ["use empathy early"], "recommendation")

        user_content = next(
            m["content"] for m in mock_llm.call_args[0][0] if m["role"] == "user"
        )
        assert "use empathy early" in user_content


# ─────────────────────────────────────────────────────────────────────────────
# HintManager
# ─────────────────────────────────────────────────────────────────────────────

class TestHintManager:
    @patch(LLM_PATCH)
    def test_generate_appends_hints(self, mock_llm, tmp_path):
        mock_llm.return_value = '```json\n["hint A", "hint B"]\n```'
        hm = HintManager(hints_file=str(tmp_path / "h.json"))
        new_hints = hm.generate_hints([], "success", "rec", ["a", "b"])
        assert new_hints == ["hint A", "hint B"]
        assert "hint A" in hm.hints
        assert "hint B" in hm.hints

    @patch(LLM_PATCH)
    def test_generate_enforces_max_hints_cap(self, mock_llm, tmp_path):
        mock_llm.return_value = '```json\n["n1", "n2", "n3"]\n```'
        hm = HintManager(hints_file=str(tmp_path / "h.json"))
        hm.hints = [f"old_{i}" for i in range(19)]  # 19 existing
        hm.generate_hints([], "success", "rec", ["a", "b"])
        assert len(hm.hints) == HintManager.MAX_HINTS  # 19 + 3 → capped at 20
        assert "n3" in hm.hints  # most recent preserved

    @patch(LLM_PATCH)
    def test_generate_skips_non_string_items(self, mock_llm, tmp_path):
        mock_llm.return_value = '```json\n["valid hint", 42, null, "another"]\n```'
        hm = HintManager(hints_file=str(tmp_path / "h.json"))
        result = hm.generate_hints([], "success", "rec", ["a", "b"])
        assert result == ["valid hint", "another"]

    @patch(LLM_PATCH)
    def test_generate_api_failure_returns_empty(self, mock_llm, tmp_path):
        mock_llm.side_effect = Exception("error")
        hm = HintManager(hints_file=str(tmp_path / "h.json"))
        hm.hints = ["existing"]
        result = hm.generate_hints([], "failure", "rec", ["a", "b"])
        assert result == []
        assert hm.hints == ["existing"]  # unchanged on failure

    def test_save_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "hints.json")
        hm = HintManager(hints_file=path)
        hm.hints = ["h1", "h2", "h3"]
        hm.save()

        hm2 = HintManager(hints_file=path)
        assert hm2.hints == ["h1", "h2", "h3"]

    def test_get_hints_returns_copy(self, tmp_path):
        hm = HintManager(hints_file=str(tmp_path / "h.json"))
        hm.hints = ["original"]
        copy = hm.get_hints()
        copy.append("injected")
        assert "injected" not in hm.hints

    def test_init_loads_existing_file(self, tmp_path):
        path = str(tmp_path / "hints.json")
        with open(path, "w") as f:
            json.dump(["pre-existing"], f)
        hm = HintManager(hints_file=path)
        assert hm.hints == ["pre-existing"]


# ─────────────────────────────────────────────────────────────────────────────
# DMORLController
# ─────────────────────────────────────────────────────────────────────────────

class TestDMORLController:
    def _make_ctrl(self, tmp_path, **kwargs):
        skills_file = str(tmp_path / "skills.json")
        hints_file = str(tmp_path / "hints.json")
        return DMORLController(
            n_objectives=2,
            objective_names=["a", "b"],
            scenario="recommendation",
            n_basic_skills=kwargs.get("n_basic_skills", 2),
            n_advanced_skills=kwargs.get("n_advanced_skills", 2),
            skills_file=skills_file,
            hints_file=hints_file,
        )

    @patch(LLM_PATCH)
    def test_initialize_loads_from_existing_file(self, mock_llm, tmp_path):
        ctrl = self._make_ctrl(tmp_path)
        # Pre-write a skills file
        with open(ctrl.skill_library.skills_file, "w") as f:
            json.dump({
                "basic": [{"name": "B", "weight_vector": [1.0, 0.0], "type": "basic"}],
                "advanced": [],
            }, f)
        ctrl.initialize_skills(force_rediscover=False)

        mock_llm.assert_not_called()
        assert len(ctrl.skill_library.basic_skills) == 1

    @patch(LLM_PATCH)
    def test_initialize_calls_llm_when_no_file(self, mock_llm, tmp_path):
        ctrl = self._make_ctrl(tmp_path, n_basic_skills=1, n_advanced_skills=1)
        skill_json = json.dumps([{"name": "S", "description": "d", "weight_vector": [0.5, 0.5]}])
        mock_llm.return_value = f"```json\n{skill_json}\n```"

        ctrl.initialize_skills(force_rediscover=False)
        assert mock_llm.call_count >= 2  # at least basic + advanced

    @patch(LLM_PATCH)
    def test_initialize_force_rediscover_ignores_existing_file(self, mock_llm, tmp_path):
        ctrl = self._make_ctrl(tmp_path, n_basic_skills=1, n_advanced_skills=1)
        # Pre-write a skills file
        with open(ctrl.skill_library.skills_file, "w") as f:
            json.dump({"basic": [{"name": "Old"}], "advanced": []}, f)

        skill_json = json.dumps([{"name": "New", "weight_vector": [0.5, 0.5]}])
        mock_llm.return_value = f"```json\n{skill_json}\n```"
        ctrl.initialize_skills(force_rediscover=True)

        assert mock_llm.call_count >= 2  # must re-discover even with existing file

    @patch(LLM_PATCH)
    def test_get_dynamic_weight_passes_accumulated_hints(self, mock_llm, tmp_path):
        mock_llm.return_value = '```json\n{"weight_vector": [0.5, 0.5]}\n```'
        ctrl = self._make_ctrl(tmp_path)
        ctrl.hint_manager.hints = ["be empathetic in the first 3 turns"]
        ctrl.get_dynamic_weight([])

        user_content = next(
            m["content"] for m in mock_llm.call_args[0][0] if m["role"] == "user"
        )
        assert "be empathetic in the first 3 turns" in user_content

    @patch(LLM_PATCH)
    def test_refine_after_dialogue_stores_hints(self, mock_llm, tmp_path):
        mock_llm.return_value = '```json\n["new tactical hint"]\n```'
        ctrl = self._make_ctrl(tmp_path)
        result = ctrl.refine_after_dialogue(
            [{"role": "user", "content": "hi"}], "success"
        )
        assert "new tactical hint" in result
        assert "new tactical hint" in ctrl.hint_manager.hints
