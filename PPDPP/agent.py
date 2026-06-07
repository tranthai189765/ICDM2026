from torch.distributions import Categorical
import random
import numpy as np
import torch
from torch.optim import AdamW
from transformers import BertModel, RobertaModel
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from utils import *
from prompt import ESConvAct, CIMAAct, CBAct

model = {'bert': BertModel, 'roberta': RobertaModel}
act = {'esc': ESConvAct, 'cima': CIMAAct, 'cb': CBAct}
TMP_DIR = {
    'esc': './tmp/esc',
    'cima': './tmp/cima',
    'cb': './tmp/cb',
}

class PPDPP(nn.Module):
    def __init__(self, args, config, tokenizer):
        super().__init__()
        self.runtime_device = torch.device(getattr(args, "device", "cpu"))
        self.model_dtype = torch.float32
        load_kwargs = {
            'from_tf': bool('.ckpt' in args.model_name_or_path),
            'config': config,
            'cache_dir': args.cache_dir,
        }
        self.policy = model[args.model_name].from_pretrained(args.model_name_or_path, **load_kwargs)
        self.dropout = nn.Dropout(0.5)
        self.act = sorted(list(act[args.data_name].keys()))
        self.classifier = nn.Linear(config.hidden_size, len(self.act))
        self.tokenizer = tokenizer
        self.policy.to(device=self.runtime_device, dtype=self.model_dtype)
        self.classifier.to(device=self.runtime_device, dtype=self.model_dtype)
        self.dropout.to(device=self.runtime_device, dtype=self.model_dtype)
        self.to(device=self.runtime_device, dtype=self.model_dtype)
        self.optimizer = AdamW(
            self.parameters(), lr=args.learning_rate
        )
        self.eps = np.finfo(np.float32).eps.item()
        self.config = config
        self.args = args
        self.saved_log_probs = []
        self.rewards = []

    def build_input(self, state, max_seq_length=None):
        seq_limit = max_seq_length or self.args.max_seq_length
        dial_id = []
        for turn in state[::-1]:
            s = self.tokenizer.encode("%s: %s" % (turn['role'], turn['content']))
            if len(dial_id) + len(s) > seq_limit:
                break
            dial_id = s[1:] + dial_id
        inp = s[:1] + dial_id
        return [inp]

    def _policy_forward(self, inp):
        return self.policy(inp)

    def _stable_probs(self, logits):
        probs = nn.functional.softmax(logits.float(), dim=1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        sums = probs.sum(dim=1, keepdim=True)
        invalid = sums <= 0
        if invalid.any():
            probs = probs.clone()
            probs[invalid] = 1.0 / probs.shape[1]
            sums = probs.sum(dim=1, keepdim=True)
        return probs / sums

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.policy(input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, len(self.act)), labels.view(-1))
            return loss
        else:
            return F.softmax(logits, dim=-1)

    def select_action(self, state, is_test=False):
        seq_limits = [self.args.max_seq_length]
        if self.runtime_device.type == 'mps':
            for fallback_limit in (384, 256, 192, 128):
                if fallback_limit < self.args.max_seq_length:
                    seq_limits.append(fallback_limit)

        device = next(self.parameters()).device
        last_exc = None
        for seq_limit in seq_limits:
            try:
                inp = self.build_input(state, max_seq_length=seq_limit)
                inp = torch.tensor(inp).long().to(device)

                if is_test:
                    with torch.no_grad():
                        outputs = self._policy_forward(inp)
                        pooled_output = outputs[1]
                        pooled_output = self.dropout(pooled_output)
                        logits = self.classifier(pooled_output)
                        probs = self._stable_probs(logits)
                        action = probs.argmax().item()
                    return self.act[action]

                outputs = self._policy_forward(inp)
                pooled_output = outputs[1]
                pooled_output = self.dropout(pooled_output)
                logits = self.classifier(pooled_output)
                probs = self._stable_probs(logits)
                m = Categorical(probs)
                action = m.sample()
                self.saved_log_probs.append(m.log_prob(action))
                return self.act[action]
            except RuntimeError as exc:
                last_exc = exc
                is_mps_oom = self.runtime_device.type == 'mps' and 'out of memory' in str(exc).lower()
                if not is_mps_oom:
                    raise
                if hasattr(torch, 'mps'):
                    torch.mps.empty_cache()
                if seq_limit == seq_limits[-1]:
                    raise
        raise last_exc

    def optimize_model(self):
        R = 0
        policy_loss = []
        rewards = []
        for r in self.rewards[::-1]:
            R = r + self.args.gamma * R
            rewards.insert(0, R)
        rewards = torch.stack(rewards).view(-1)
        if rewards.shape[0] > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)
        for log_prob, reward in zip(self.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]
        if self.runtime_device.type == 'mps' and hasattr(torch, 'mps'):
            torch.mps.empty_cache()
        return policy_loss.data
    
    def save_model(self, data_name, filename, epoch_user):
        output_dir = TMP_DIR[data_name] + '/RL-agent/' + filename + '-epoch-{}'.format(epoch_user)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(self.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, 'optimizer.bin'))
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))

    def load_model(self, data_name, filename, epoch_user=None):
        if epoch_user: 
            output_dir = TMP_DIR[data_name] + '/RL-agent/' + filename + '-epoch-{}'.format(epoch_user)
        else:
            output_dir = filename
        if hasattr(self, 'module'):
            self.module.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
        else:
            self.load_state_dict(torch.load(
                os.path.join(output_dir, 'pytorch_model.bin'),
                map_location=self.args.device))
        optimizer_path = os.path.join(output_dir, 'optimizer.bin')
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.args.device))
