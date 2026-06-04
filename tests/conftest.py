"""
Shared fixtures and path setup for all DMORL tests.
"""
import sys
import os
import types
import importlib.machinery
import importlib.util
from unittest.mock import MagicMock

# ── Project root on sys.path (no setup.py) ────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Our top-level packages `utils`/`base` are namespace packages (no __init__.py),
# so a foreign *editable* install on the global interpreter that ships a regular
# `utils`/`base` package would shadow ours (regular package wins over namespace).
# Drop such unrelated editable entries so tests always import THIS project's code.
# No-op in a clean environment.
sys.path[:] = [p for p in sys.path if "legal_btc_graph" not in p.lower()]

# ── Stub heavy/absent packages before any project imports ────────────────────
# These are installed on the training server but not in the local test env.
# We use types.ModuleType (not MagicMock) so that importlib.util.find_spec
# works correctly — transformers checks __spec__ internally.

def _make_stub(name: str):
    """
    Create a module stub that satisfies both:
    - importlib.util.find_spec  (needs a real ModuleSpec on __spec__)
    - attribute access like `from nltk import ngrams` (MagicMock handles unknown attrs)
    """
    mod = MagicMock()
    mod.__name__ = name
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    mod.__path__ = []
    mod.__package__ = name
    mod.__loader__ = None
    mod.__file__ = None
    return mod

_LOCAL_STUBS = [
    # Logging / tracking
    "wandb",                       # padpp.trainer → logger.wandb_logger
    # NLP evaluation metrics
    "rouge",
    "rouge.rouge",
    "bert_score",
    "evaluate",
    "nltk",
    "nltk.translate",
    "nltk.translate.bleu_score",
    # Google APIs (used in utils/prompt.py toxicity eval)
    "googleapiclient",
    "googleapiclient.discovery",
    "google",
    "google.api_core",
    # FastChat (optional generation backend)
    "fastchat",
    "fastchat.conversation",
    # HuggingFace training utils (server only)
    "accelerate",
    "peft",
]
for _pkg in _LOCAL_STUBS:
    if _pkg not in sys.modules and importlib.util.find_spec(_pkg) is None:
        sys.modules[_pkg] = _make_stub(_pkg)

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


# ─── Tiny stand-in for roberta-large ─────────────────────────────────────────

class MockPLM(nn.Module):
    """Replaces roberta-large so tests never touch the network."""

    def __init__(self, lm_size: int = 16):
        super().__init__()
        self.lm_size = lm_size
        self._dummy = nn.Linear(1, 1)

    def forward(self, **kwargs):
        ids = kwargs.get("input_ids", torch.zeros(1, 4, dtype=torch.long))
        bs = ids.shape[0]
        return (torch.randn(bs, 4, self.lm_size),)

    def resize_token_embeddings(self, size: int):
        pass


# ─── Config fixture ───────────────────────────────────────────────────────────

@pytest.fixture
def fake_model_config():
    """Minimal model config for DMORLModel tests — no downloads, no data."""
    return SimpleNamespace(
        tokenizer="roberta-large",
        plm="roberta-large",
        lm_size=16,
        mlp_hidden_size=8,
        objective_embedding_size=4,
        n_objectives=2,
        n_goals=4,
        n_topics=3,
        combined_action=False,
        dropout=0.0,
        freeze_plm=True,
        cached_dir=None,
        special_tokens_dict={},
        num_workers=0,
    )


# ─── Model fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def dmorl_model(fake_model_config):
    """DMORLModel with mocked roberta backbone — instantiates in <1 s."""
    mock_tok = MagicMock()
    mock_tok.add_special_tokens.return_value = 0
    mock_plm = MockPLM(lm_size=fake_model_config.lm_size)

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok), \
         patch("transformers.AutoModel.from_pretrained", return_value=mock_plm):
        from dmorl.model import DMORLModel
        model = DMORLModel(fake_model_config)
    return model


# ─── Trainer factory (bypasses heavy PADPPTrainer.__init__) ──────────────────

def make_trainer(
    use_hints: bool = True,
    use_dynamic: bool = True,
    horizon: int = 3,
    n_obj: int = 2,
):
    """
    Create a DMORLTrainer without calling __init__.
    Only the attributes exercised by the tested methods are populated.
    """
    from dmorl.trainer import DMORLTrainer

    trainer = object.__new__(DMORLTrainer)
    trainer.model_config = SimpleNamespace(
        use_dynamic_weight=use_dynamic,
        dynamic_weight_horizon=horizon,
        n_objectives=n_obj,
        use_hints=use_hints,
        n_skill_train_epochs=2,
        n_advanced_train_epochs=2,
        num_train_rl_epochs=4,
        train_rl_batch_size=4,
        buffer_length=16,
        sampled_times=2,
        warmup_steps=0,
        actor_learning_rate=1e-3,
        preference_buffer_length=8,
        epsilon=0.1,
        num_workers=0,
    )
    trainer.device = torch.device("cpu")
    trainer.dmorl_controller = None
    trainer.model = MagicMock()
    trainer.game = MagicMock()
    trainer.memory_buffer = []
    trainer.save_model = MagicMock()
    trainer._eval_basic_skills_per_skill = MagicMock()
    return trainer
