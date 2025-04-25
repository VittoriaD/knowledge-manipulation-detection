from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset

class BertTrainerHelper:
    def __init__(self, model_name="bert-base-multilingual-cased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name

    def model_init(self):
        return AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, average="binary"),
            "recall": recall_score(labels, preds, average="binary"),
            "f1": f1_score(labels, preds, average="binary"),
        }

    def hp_space_optuna(self, trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64]),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 6),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.3),
        }

    def tokenize_dataset(self, dataset, text_col="joined_texts"):
        def tokenize(batch):
            return self.tokenizer(batch[text_col], truncation=True, padding="max_length", max_length=512)
        return dataset.map(tokenize, batched=True)
