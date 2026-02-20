from torch import nn
from typing import Optional
from pathlib import Path
import os
import torch
from transformers import AutoTokenizer
from transformers import AutoModel, AutoConfig
from .focal_loss import FocalLoss

# default local cache directory (relative to repo); transformers will also use HF cache
DEFAULT_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "./models_cache")

_ENCODER_CACHE = {}
_TOKENIZER_CACHE = {}


def get_tokenizer(model_name: str, cache_dir: Optional[str] = None):
    cd = cache_dir or DEFAULT_CACHE_DIR
    if model_name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_name]

    # Prefer local cached files to avoid remote rate limits
    try:
        tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cd, local_files_only=True)
    except Exception:
        # fallback to normal download (may be rate-limited)
        tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cd)

    _TOKENIZER_CACHE[model_name] = tok
    return tok


def get_encoder(model_name: str, device: Optional[str] = None, cache_dir: Optional[str] = None):
    cd = cache_dir or DEFAULT_CACHE_DIR
    if model_name not in _ENCODER_CACHE:
        # Try to load config & model from local cache first
        try:
            cfg = AutoConfig.from_pretrained(model_name, cache_dir=cd, local_files_only=True)
            model = AutoModel.from_pretrained(model_name, config=cfg, cache_dir=cd, local_files_only=True)
        except Exception:
            # If local-only load fails, fall back to normal (may download)
            cfg = AutoConfig.from_pretrained(model_name, cache_dir=cd)
            model = AutoModel.from_pretrained(model_name, config=cfg, cache_dir=cd)

        _ENCODER_CACHE[model_name] = {"config": cfg, "state_dict": model.state_dict()}

    entry = _ENCODER_CACHE[model_name]
    model = AutoModel.from_config(entry["config"])
    # load a copy to ensure independent parameters per instance
    sd = {k: v.clone() for k, v in entry["state_dict"].items()}
    model.load_state_dict(sd)
    if device is not None:
        model.to(device)
    return model


def warmup_model(model_name: str, device: Optional[str] = None, cache_dir: Optional[str] = None):
    """Ensure model (encoder and tokenizers) downloaded (or present locally) and cached in-memory."""
    get_encoder(model_name, device=device, cache_dir=cache_dir)
    return get_tokenizer(model_name, cache_dir=cache_dir)


def clear_caches(model_name: Optional[str] = None):
    if model_name is None:
        _ENCODER_CACHE.clear()
        _TOKENIZER_CACHE.clear()
    else:
        _ENCODER_CACHE.pop(model_name, None)
        _TOKENIZER_CACHE.pop(model_name, None)


class PCLModel(nn.Module):
    def __init__(self, encoder_name, n_labels, aux_dim, dropout=0.1, device: str = None):
        super().__init__()
        self.encoder = get_encoder(encoder_name, device=device)
        hidden = self.encoder.config.hidden_size

        self.aux_dim = aux_dim
        if aux_dim > 0: 
            # project auxiliary features to smaller dimension 
            # before fusion to avoid grad explosion with small batch size + complex task
            self.aux_proj = nn.Sequential(
                nn.Linear(aux_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            fused_dim = hidden + 64
        else:
            self.aux_proj = None
            fused_dim = hidden


        self.dropout = nn.Dropout(dropout)
        self.bin_head = nn.Linear(fused_dim, 1)
        self.multi_head = nn.Linear(fused_dim, n_labels)
        self.bin_loss_fn = nn.BCEWithLogitsLoss()  # keep BCE for binary (it's balanced enough)
        self.multi_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)  # focal for sparse multilabel

    def forward(self, input_ids=None, attention_mask=None, labels=None, aux_features=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        cls = out.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)

        if self.aux_proj is not None and aux_features is not None:
            aux = self.aux_proj(aux_features)
            cls = torch.cat([cls, aux], dim=-1)

        logit_bin = self.bin_head(cls).squeeze(-1)
        logit_multi = self.multi_head(cls)

        outputs = {"logit_bin": logit_bin, "logit_multi": logit_multi}

        if labels is not None:
            labels = labels.to(logit_multi.device)
            bin_labels = labels[:, 0].float()
            multi_labels = labels[:, 1:].float()

            loss_bin = self.bin_loss_fn(logit_bin, bin_labels)
            loss_multi = self.multi_loss_fn(logit_multi, multi_labels)

            outputs["loss"] = loss_multi + loss_bin

        return outputs