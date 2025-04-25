import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from tqdm import tqdm
import ast

class TextClassifierHelper:
    def __init__(self, model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device).eval()

    def chunk_text(self, text, max_length=512, stride=256):
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        chunks = []
        for i in range(0, len(tokens), stride):
            chunk = tokens[i:i + max_length]
            if len(chunk) < 10:
                break
            chunks.append(self.tokenizer.decode(chunk, skip_special_tokens=True))
        return chunks

    def classify_column(self, df, text_column, prefix, batch_size=32):
        print(f"\nApplying model: {prefix} | column: {text_column}")
        valid_mask = df[text_column].fillna("").str.strip().str.len() > 10
        texts = df.loc[valid_mask, text_column].astype(str).tolist()

        all_mean_logits, all_max_logits, all_min_logits = [], [], []
        all_mean_probs, all_max_probs, all_min_probs = [], [], []

        for text in tqdm(texts):
            chunks = self.chunk_text(text)
            chunk_logits = []
            with torch.no_grad():
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs)
                    chunk_logits.append(outputs.logits.cpu())

            logits = torch.cat(chunk_logits, dim=0)
            probs = softmax(logits, dim=1)

            all_mean_logits.append(logits.mean(dim=0).numpy())
            all_max_logits.append(logits.max(dim=0).values.numpy())
            all_min_logits.append(logits.min(dim=0).values.numpy())

            all_mean_probs.append(probs.mean(dim=0).numpy())
            all_max_probs.append(probs.max(dim=0).values.numpy())
            all_min_probs.append(probs.min(dim=0).values.numpy())

        df.loc[valid_mask, f"{prefix}_mean_logit_0"] = [x[0] for x in all_mean_logits]
        df.loc[valid_mask, f"{prefix}_mean_logit_1"] = [x[1] for x in all_mean_logits]
        df.loc[valid_mask, f"{prefix}_max_logit_0"] = [x[0] for x in all_max_logits]
        df.loc[valid_mask, f"{prefix}_max_logit_1"] = [x[1] for x in all_max_logits]
        df.loc[valid_mask, f"{prefix}_min_logit_0"] = [x[0] for x in all_min_logits]
        df.loc[valid_mask, f"{prefix}_min_logit_1"] = [x[1] for x in all_min_logits]

        df.loc[valid_mask, f"{prefix}_mean_prob_0"] = [x[0] for x in all_mean_probs]
        df.loc[valid_mask, f"{prefix}_mean_prob_1"] = [x[1] for x in all_mean_probs]
        df.loc[valid_mask, f"{prefix}_max_prob_0"] = [x[0] for x in all_max_probs]
        df.loc[valid_mask, f"{prefix}_max_prob_1"] = [x[1] for x in all_max_probs]
        df.loc[valid_mask, f"{prefix}_min_prob_0"] = [x[0] for x in all_min_probs]
        df.loc[valid_mask, f"{prefix}_min_prob_1"] = [x[1] for x in all_min_probs]

        return df

    def join_text_list(self, x):
        if isinstance(x, str):
            try:
                x_list = ast.literal_eval(x)
                if isinstance(x_list, list):
                    return "\n".join(x_list)
            except:
                return x
        elif isinstance(x, list):
            return "\n".join(x)
        return str(x)

    def join_changes(self, x):
        if isinstance(x, str):
            try:
                x_list = ast.literal_eval(x)
                if isinstance(x_list, list):
                    return [" [SEP] ".join(pair) for pair in x_list if isinstance(pair, (list, tuple)) and len(pair) == 2]
            except:
                return []
        return []

    def classify_texts(self, texts, prefix, batch_size=16):
        all_features = []

        for text in tqdm(texts, desc=f"Running classifier for {prefix}"):
            chunks = self.chunk_text(text)
            if not chunks:
                all_features.append(np.zeros(12))
                continue

            all_logits = []
            with torch.no_grad():
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                    outputs = self.model(**inputs)
                    all_logits.append(outputs.logits.cpu())

            logits = torch.cat(all_logits, dim=0)
            probs = softmax(logits, dim=1)

            mean_logits = logits.mean(dim=0).numpy()
            max_logits = logits.max(dim=0).values.numpy()
            min_logits = logits.min(dim=0).values.numpy()

            mean_probs = probs.mean(dim=0).numpy()
            max_probs = probs.max(dim=0).values.numpy()
            min_probs = probs.min(dim=0).values.numpy()

            combined = np.concatenate([mean_logits, max_logits, min_logits, mean_probs, max_probs, min_probs])
            all_features.append(combined)

        columns = (
            [f"{prefix}_mean_logit_0", f"{prefix}_mean_logit_1"] +
            [f"{prefix}_max_logit_0", f"{prefix}_max_logit_1"] +
            [f"{prefix}_min_logit_0", f"{prefix}_min_logit_1"] +
            [f"{prefix}_mean_prob_0", f"{prefix}_mean_prob_1"] +
            [f"{prefix}_max_prob_0", f"{prefix}_max_prob_1"] +
            [f"{prefix}_min_prob_0", f"{prefix}_min_prob_1"]
        )

        import pandas as pd
        return pd.DataFrame(all_features, columns=columns)
