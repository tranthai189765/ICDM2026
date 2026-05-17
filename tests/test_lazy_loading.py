"""
Verifies that the lazy-loading fix in utils/prompt.py works correctly:
  - Importing the module must NOT load LLaMA3 or the sentiment model.
  - The private helpers _get_llama_pipeline / _get_sentiment_pipeline must
    initialize their globals on first call and cache them for subsequent calls.

NOTE: These tests require transformers.pipeline to be importable, which in
turn requires PyTorch >= 2.1.  On older local environments the whole file is
skipped; the tests still run on the training server.
"""
import sys
import importlib
import pytest
from unittest.mock import MagicMock, patch, call

# Skip the entire module if transformers.pipeline is not available
# (happens when PyTorch < 2.1 is installed locally)
try:
    from transformers import pipeline as _tp  # noqa: F401
    _PIPELINE_AVAILABLE = True
except Exception:
    _PIPELINE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _PIPELINE_AVAILABLE,
    reason="transformers.pipeline not available (requires PyTorch >= 2.1)",
)


def _fresh_import_prompt():
    """Force a clean re-import of utils.prompt by removing any cached module."""
    for key in list(sys.modules.keys()):
        if key in ("utils.prompt", "utils"):
            del sys.modules[key]
    # Also remove the dotenv load side-effect module if cached
    import utils.prompt  # noqa: F401
    return sys.modules["utils.prompt"]


# ─────────────────────────────────────────────────────────────────────────────
# Import-time behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestImportDoesNotLoadModels:
    def test_llama_pipeline_global_is_none_after_import(self):
        mod = _fresh_import_prompt()
        assert mod._llama_pipeline is None, \
            "LLaMA3 pipeline must NOT be loaded at import time"

    def test_terminators_global_is_none_after_import(self):
        mod = _fresh_import_prompt()
        assert mod._terminators is None

    def test_sentiment_pipeline_global_is_none_after_import(self):
        mod = _fresh_import_prompt()
        assert mod._sentiment_pipeline is None, \
            "Sentiment model must NOT be loaded at import time"

    def test_transformers_pipeline_not_called_on_import(self):
        """transformers.pipeline() must not be invoked during module import."""
        for key in list(sys.modules.keys()):
            if "utils.prompt" in key:
                del sys.modules[key]

        call_log = []

        def spy_pipeline(*args, **kwargs):
            call_log.append(args[0] if args else kwargs.get("model", "?"))
            m = MagicMock()
            m.tokenizer.eos_token_id = 2
            m.tokenizer.convert_tokens_to_ids.return_value = 3
            return m

        with patch("transformers.pipeline", side_effect=spy_pipeline):
            import utils.prompt  # noqa: F401

        assert len(call_log) == 0, (
            f"transformers.pipeline was called during import with args: {call_log}. "
            "Models should only be loaded on first use (lazy loading)."
        )


# ─────────────────────────────────────────────────────────────────────────────
# _get_llama_pipeline — lazy init and caching
# ─────────────────────────────────────────────────────────────────────────────

class TestGetLlamaPipeline:
    def test_initialises_on_first_call(self):
        mod = _fresh_import_prompt()
        assert mod._llama_pipeline is None

        dummy = MagicMock()
        dummy.tokenizer.eos_token_id = 2
        dummy.tokenizer.convert_tokens_to_ids.return_value = 3

        with patch("transformers.pipeline", return_value=dummy) as mock_pipe:
            pipeline, terminators = mod._get_llama_pipeline()

        mock_pipe.assert_called_once()
        assert pipeline is dummy
        assert terminators is not None
        assert len(terminators) == 2

    def test_caches_after_first_call(self):
        mod = _fresh_import_prompt()

        dummy = MagicMock()
        dummy.tokenizer.eos_token_id = 2
        dummy.tokenizer.convert_tokens_to_ids.return_value = 3

        with patch("transformers.pipeline", return_value=dummy) as mock_pipe:
            p1, t1 = mod._get_llama_pipeline()
            p2, t2 = mod._get_llama_pipeline()

        # Second call must NOT reinitialise
        assert mock_pipe.call_count == 1, "Pipeline should only be created once (cached)"
        assert p1 is p2
        assert t1 is t2

    def test_global_variable_set_after_call(self):
        mod = _fresh_import_prompt()
        dummy = MagicMock()
        dummy.tokenizer.eos_token_id = 2
        dummy.tokenizer.convert_tokens_to_ids.return_value = 3

        with patch("transformers.pipeline", return_value=dummy):
            mod._get_llama_pipeline()

        assert mod._llama_pipeline is dummy


# ─────────────────────────────────────────────────────────────────────────────
# _get_sentiment_pipeline — lazy init and caching
# ─────────────────────────────────────────────────────────────────────────────

class TestGetSentimentPipeline:
    def test_initialises_on_first_call(self):
        mod = _fresh_import_prompt()
        assert mod._sentiment_pipeline is None

        dummy_sentiment = MagicMock()
        with patch("transformers.pipeline", return_value=dummy_sentiment) as mock_pipe:
            p = mod._get_sentiment_pipeline()

        mock_pipe.assert_called_once()
        assert p is dummy_sentiment

    def test_caches_after_first_call(self):
        mod = _fresh_import_prompt()
        dummy_sentiment = MagicMock()

        with patch("transformers.pipeline", return_value=dummy_sentiment) as mock_pipe:
            p1 = mod._get_sentiment_pipeline()
            p2 = mod._get_sentiment_pipeline()

        assert mock_pipe.call_count == 1
        assert p1 is p2

    def test_global_variable_set_after_call(self):
        mod = _fresh_import_prompt()
        dummy = MagicMock()
        with patch("transformers.pipeline", return_value=dummy):
            mod._get_sentiment_pipeline()
        assert mod._sentiment_pipeline is dummy


# ─────────────────────────────────────────────────────────────────────────────
# call_llama3_model — uses lazy helper
# ─────────────────────────────────────────────────────────────────────────────

class TestCallLlama3Model:
    def test_uses_lazy_pipeline(self):
        mod = _fresh_import_prompt()

        dummy_pipeline = MagicMock()
        dummy_pipeline.return_value = [
            {"generated_text": [{"role": "assistant", "content": "test response"}]}
        ]
        dummy_pipeline.tokenizer.eos_token_id = 2
        dummy_pipeline.tokenizer.convert_tokens_to_ids.return_value = 3

        with patch("transformers.pipeline", return_value=dummy_pipeline):
            result = mod.call_llama3_model("hello", temperature=0.1, max_token=10)

        assert result == "test response"

    def test_pipeline_called_with_correct_kwargs(self):
        mod = _fresh_import_prompt()

        dummy_pipeline = MagicMock()
        dummy_pipeline.return_value = [
            {"generated_text": [{"role": "assistant", "content": "out"}]}
        ]
        dummy_pipeline.tokenizer.eos_token_id = 2
        dummy_pipeline.tokenizer.convert_tokens_to_ids.return_value = 3

        with patch("transformers.pipeline", return_value=dummy_pipeline):
            mod.call_llama3_model("prompt", temperature=0.5, max_token=20)

        call_kwargs = dummy_pipeline.call_args[1]
        assert call_kwargs["max_new_tokens"] == 20
        assert abs(call_kwargs["temperature"] - 0.5) < 1e-6
        assert call_kwargs["do_sample"] is True
