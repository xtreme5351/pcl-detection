import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class DatasetManager(Dataset):
    def __init__(self, path, texts_df, aux_features: "np.ndarray | None" = None):
        self.path = path
        self.texts_df = texts_df
        self.aux_features = aux_features

        labels_df, label_array = self._load_csv(path)

        # drop par_ids not present in texts_df
        valid_mask = np.isin(labels_df.index.values, texts_df.index.values)
        labels_df = labels_df[valid_mask]
        label_array = label_array[valid_mask]

        merged = self._join_texts(labels_df)

        # pick texts, drop empty
        texts = merged["text"].fillna("")
        keep_mask = texts.astype(bool).values
        texts = texts[keep_mask].astype(str).tolist()

        label_array = label_array[keep_mask]

        self.texts = texts
        self.bin = (label_array.sum(axis=1) > 0).astype(int).tolist()
        self.mult = label_array.astype(np.float32)

        # need to align features with filtered texts
        self.aux_features = self.aux_features[keep_mask] if self.aux_features is not None else None

    def _load_csv(self, path):
        df = pd.read_csv(path)
        df["par_id"] = df["par_id"].astype(int)

        # Remove surrounding brackets and split into columns (vectorized)
        s = df["label"].astype(str).str.replace(r"[\[\]]", "", regex=True)
        parts = s.str.split(",", expand=True)

        # strip spaces and convert to int
        parts = parts.apply(lambda col: col.str.strip())
        try:
            label_array = parts.values.astype(int)
        except Exception:
            # Fall back to safe parsing if unexpected formats appear
            import ast

            def safe_parse(x):
                try:
                    return np.array(ast.literal_eval(x), dtype=int)
                except Exception:
                    return np.zeros(parts.shape[1], dtype=int)

            label_array = np.vstack(df["label"].apply(safe_parse).values)

        # Build a labels DataFrame indexed by par_id for fast join
        cols = [f"c{i}" for i in range(label_array.shape[1])]
        labels_df = pd.DataFrame(label_array, columns=cols, index=df["par_id"].values)
        labels_df.index.name = "par_id"

        return labels_df, label_array

    def _join_texts(self, labels_df):
        # Join labels (index=par_id) with texts_df (index=par_id)
        texts_sub = self.texts_df[[col for col in ("text_clean","text") if col in self.texts_df.columns]].copy()
        merged = labels_df.join(texts_sub, how="left")

        missing = merged[merged["text_clean"].isna() & merged["text"].isna()].shape[0]
        if missing > 0:
            example_missing = list(merged[merged["text_clean"].isna() & merged["text"].isna()].index[:10])
            print(f"Warning: {missing} label par_id(s) missing from texts_df. Example missing ids: {example_missing}")

        # Prefer cleaned text when available
        if "text_clean" in merged.columns:
            merged["text"] = merged["text_clean"].fillna(merged.get("text"))
        merged["text"] = merged["text"].fillna("")

        return merged

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {
            "text": self.texts[idx],
            "bin": self.bin[idx],
            "multi": self.mult[idx]
        }
        if self.aux_features is not None:
            item["aux_features"] = self.aux_features[idx]
        return item

    def get_texts(self):
        return self.texts

    def print_stats(self):
        print(f"Total samples: {len(self)}")
        print(f"Binary distribution: {np.bincount(self.bin)}")
        print(f"Multilabel distribution: {np.sum(self.mult, axis=0)}")
