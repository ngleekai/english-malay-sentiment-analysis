"""Train the corrected common baseline offline, without accessing test data."""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, disable_progress_bars
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback,
                          Trainer, TrainerCallback, TrainingArguments, set_seed)
from transformers.trainer_callback import PrinterCallback
from transformers.utils import logging as transformers_logging
from src.training_losses import sample_normalized_weighted_loss

disable_progress_bars()
transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models/A1_xlmr/best_model"
RUN_DIR = ROOT / "models/A1_xlmr/checkpoints"
RESULTS_DIR = ROOT / "results/A1"
SEED = 42
MAX_LENGTH = 96
OFFLINE_MODE = os.environ.get("HF_HUB_OFFLINE") == "1"


class CleanOutputCallback(TrainerCallback):
    """Show only the epoch information needed to judge training."""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metrics = metrics or {}
        train_loss = next(
            (row["loss"] for row in reversed(state.log_history) if "loss" in row),
            None,
        )
        loss_text = f"{train_loss:.4f}" if train_loss is not None else "not available"
        print(
            f"Epoch {state.epoch:.0f}/{args.num_train_epochs:.0f} | "
            f"Training loss: {loss_text} | "
            f"Validation Macro-F1: {metrics.get('eval_macro_f1', 0) * 100:.2f}% | "
            f"Validation accuracy: {metrics.get('eval_accuracy', 0) * 100:.2f}%",
            flush=True,
        )


def compute_metrics(output):
    logits, labels = output
    pred = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average="macro", zero_division=0)
    _, _, per_class, _ = precision_recall_fscore_support(labels, pred, labels=[0,1,2], average=None, zero_division=0)
    return dict(accuracy=float(accuracy_score(labels,pred)), macro_precision=float(precision),
                macro_recall=float(recall), macro_f1=float(f1),
                **{f"{n}_f1":float(v) for n,v in zip(["negative","neutral","positive"],per_class)})


def prepare_training(seed=SEED, max_length=MAX_LENGTH, learning_rate=2e-5,
                     trainable_layers=4, train_frame=None, validation_frame=None,
                     run_dir=RUN_DIR):
    print("Preparing A1 training...", flush=True)
    set_seed(seed)  # Before classifier initialization, independent of notebook history.
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("This 4 GB GPU configuration requires CUDA")
    train = pd.read_csv(ROOT / "data/processed/train.csv") if train_frame is None else train_frame.copy()
    validation = pd.read_csv(ROOT / "data/processed/validation.csv") if validation_frame is None else validation_frame.copy()
    labels = {0:"Negative",1:"Neutral",2:"Positive"}
    tokenizer = AutoTokenizer.from_pretrained(
        "FacebookAI/xlm-roberta-base", local_files_only=OFFLINE_MODE
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "FacebookAI/xlm-roberta-base", local_files_only=OFFLINE_MODE, num_labels=3,
        id2label=labels, label2id={v:k for k,v in labels.items()})
    for parameter in model.roberta.parameters():
        parameter.requires_grad = False
    for layer in model.roberta.encoder.layer[-trainable_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    counts = train.label.value_counts().reindex([0,1,2])
    if counts.isna().any() or counts.le(0).any():
        raise ValueError("All three training classes are required")
    weights = torch.tensor((len(train)/(3*counts)).to_numpy(),dtype=torch.float32)

    def loss_fn(outputs, labels, num_items_in_batch=None):
        return sample_normalized_weighted_loss(outputs.logits, labels, weights, num_items_in_batch)

    def encode(frame):
        ds = Dataset.from_pandas(frame[["tweet_text","label"]],preserve_index=False)
        return ds.map(lambda b:tokenizer(b["tweet_text"],truncation=True,max_length=max_length),
                      batched=True,remove_columns=["tweet_text"])

    args = TrainingArguments(
        output_dir=str(run_dir), num_train_epochs=10,
        per_device_train_batch_size=1, per_device_eval_batch_size=4,
        gradient_accumulation_steps=8, fp16=True, gradient_checkpointing=False,
        use_cache=False, optim="adamw_torch_fused", learning_rate=learning_rate,
        weight_decay=.01, warmup_steps=50, max_grad_norm=1.,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True,
        save_total_limit=2, save_only_model=True, torch_empty_cache_steps=50,
        logging_strategy="epoch", report_to="none",
        seed=seed, data_seed=seed, disable_tqdm=True)
    trainer = Trainer(model=model,args=args,train_dataset=encode(train),
                      eval_dataset=encode(validation),processing_class=tokenizer,
                      data_collator=DataCollatorWithPadding(tokenizer),compute_metrics=compute_metrics,
                      compute_loss_func=loss_fn,callbacks=[EarlyStoppingCallback(early_stopping_patience=2),
                                                          CleanOutputCallback()])
    trainer.remove_callback(PrinterCallback)
    # The callback normalizes over the whole accumulation window. Avoid dividing twice.
    trainer.loss_is_scaled_for_ga = True
    assert trainer.accelerator.gradient_accumulation_steps == 1
    metadata = {"seed":seed,"max_length":max_length,"learning_rate":learning_rate,
                "trainable_layers":trainable_layers,"train_rows":len(train),"selection_rows":len(validation),
                "class_weights":weights.tolist(),
                "loss":"sum(weight[label] * CE) / accumulated valid sample count",
                "official_test_used":False,"model_initialization":"original pretrained XLM-R, fresh classifier",
                "versions":{k:__import__(k).__version__ for k in ["torch","transformers","accelerate","numpy","pandas"]},
                "data_sha256":{n:hashlib.sha256((ROOT/f'data/processed/{n}.csv').read_bytes()).hexdigest() for n in ["train","validation"]}}
    print(f"Train rows: {len(train)} | Validation rows: {len(validation)}")
    print(f"Maximum epochs: 10 | Learning rate: {learning_rate} | Trainable XLM-R layers: {trainable_layers}")
    print("Starting training...\n", flush=True)
    return trainer, tokenizer, validation, metadata


def train_and_save(trainer, tokenizer, validation, metadata):
    if MODEL_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite completed model: {MODEL_DIR}")
    trainer.train()
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    output = trainer.predict(trainer.eval_dataset)
    RESULTS_DIR.mkdir(parents=True,exist_ok=True)
    frame = validation.copy()
    probabilities = torch.softmax(torch.tensor(output.predictions),dim=-1).numpy()
    for i,n in enumerate(["negative","neutral","positive"]):
        frame[f"xlmr_prob_{n}"] = probabilities[:,i]
    frame["A1_prediction"] = probabilities.argmax(1)
    frame.to_csv(RESULTS_DIR/"validation_predictions.csv",index=False)
    metadata.update(best_checkpoint=trainer.state.best_model_checkpoint,
                    best_metric=trainer.state.best_metric,completed_epochs=trainer.state.epoch,
                    validation_metrics=compute_metrics((output.predictions,output.label_ids)))
    (RESULTS_DIR/"validation_metrics.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    (MODEL_DIR/"experiment_manifest.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    metrics = metadata["validation_metrics"]
    print("\nA1 TRAINING COMPLETE")
    print(f"Completed epochs: {metadata['completed_epochs']:.0f}")
    print(f"Validation Macro-F1: {metrics['macro_f1'] * 100:.2f}%")
    print(f"Validation accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Negative F1: {metrics['negative_f1'] * 100:.2f}%")
    print(f"Neutral F1: {metrics['neutral_f1'] * 100:.2f}%")
    print(f"Positive F1: {metrics['positive_f1'] * 100:.2f}%")
    print("Official test used: No")
    print(f"Full results saved to: {RESULTS_DIR}")
    return metadata


if __name__ == "__main__":
    train_and_save(*prepare_training())

