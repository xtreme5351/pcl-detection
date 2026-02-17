from torch import nn
from typing import Optional
from transformers import AutoTokenizer
from transformers import AutoModel, AutoConfig

_ENCODER_CACHE = {}
_TOKENIZER_CACHE = {}


def get_tokenizer(model_name: str):
    if model_name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_name]
    tok = AutoTokenizer.from_pretrained(model_name)
    _TOKENIZER_CACHE[model_name] = tok
    return tok


def get_encoder(model_name: str, device: Optional[str] = None, cache_dir: Optional[str] = None):
    if model_name not in _ENCODER_CACHE:
        cfg = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModel.from_pretrained(model_name, config=cfg, cache_dir=cache_dir)
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
    return get_encoder(model_name, device=device, cache_dir=cache_dir)


def clear_caches(model_name: Optional[str] = None):
    if model_name is None:
        _ENCODER_CACHE.clear()
        _TOKENIZER_CACHE.clear()
    else:
        _ENCODER_CACHE.pop(model_name, None)
        _TOKENIZER_CACHE.pop(model_name, None)


__all__ = ["get_tokenizer", "get_encoder", "warmup_model", "clear_caches"]


class PCLModel(nn.Module):
    def __init__(self, encoder_name, n_labels, dropout=0.1, device: str = None):
        super().__init__()
        # Use cached encoder loader — returns a fresh model instance
        self.encoder = get_encoder(encoder_name, device=device)
        hidden = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.bin_head = nn.Linear(hidden, 1)
        self.multi_head = nn.Linear(hidden, n_labels)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        cls = out.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)

        logit_bin = self.bin_head(cls).squeeze(-1)
        logit_multi = self.multi_head(cls)
        outputs = {"logit_bin": logit_bin, "logit_multi": logit_multi}

        if labels is not None:
            labels = labels.to(logit_multi.device)
            bin_labels = labels[:, 0]
            multi_labels = labels[:, 1:]
            loss_bin = self.loss_fn(logit_bin, bin_labels)
            loss_multi = self.loss_fn(logit_multi, multi_labels)
            outputs["loss"] = loss_multi + loss_bin

        return outputs