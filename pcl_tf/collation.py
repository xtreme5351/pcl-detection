import torch
import numpy as np

DEFAULT_MAX_LEN = 128


def collate_fn(tokenizer, batch, max_len=None):
    if len(batch) == 0:
        return {}

    max_len = max_len or DEFAULT_MAX_LEN

    first = batch[0]
    # Support multiple batch element types: dicts (preferred) or raw strings
    if isinstance(first, str):
        texts = batch
        have_labels = False
    elif isinstance(first, dict):
        texts = [b["text"] for b in batch]
        have_labels = True
    else:
        # fallback: try to treat elements as sequences where first element is text
        try:
            texts = [b[0] for b in batch]
            have_labels = len(batch[0]) > 1
        except Exception:
            raise TypeError("Unsupported batch element type for collate_fn")

    enc = tokenizer(texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")

    if have_labels:
        labels_bin = torch.tensor([b["bin"] for b in batch], dtype=torch.float32).unsqueeze(1)
        labels_multi = torch.tensor(np.asarray([b["multi"] for b in batch]), dtype=torch.float32)
        enc["labels"] = torch.cat([labels_bin, labels_multi], dim=1)

    return enc