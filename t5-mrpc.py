import logging
import random
import warnings
import os
import math
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Union, TypeVar
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Sampler, DataLoader
from torch.cuda.amp import autocast, GradScaler

from peft import LoraConfig, get_peft_model, TaskType
from transformers import T5ForConditionalGeneration
import torch.nn as nn
from datasets import load_metric
from torch.utils.data import DataLoader
import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from tabulate import tabulate

from datasets import Dataset, concatenate_datasets, load_dataset, load_metric
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AdamW,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    T5ForConditionalGeneration,
    EvalPrediction,
    GenerationConfig
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

import wandb

# Set a seed for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-base")
max_length = 512
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing functions for different tasks
def preprocess_cola(examples):
    return {
        "input_text": ["cola: " + sent for sent in examples["sentence"]],
        "target_text": [str(label) for label in examples["label"]],
        "task_type": ["cola"] * len(examples["sentence"])
    }

def preprocess_mnli(examples):
    return {
        "input_text": [f"mnli: premise: {premise} hypothesis: {hypothesis}"
                      for premise, hypothesis in zip(examples["premise"], examples["hypothesis"])],
        "target_text": [str(label) for label in examples["label"]],
        "task_type": ["mnli"] * len(examples["premise"])
    }

def preprocess_mnli_matched(examples):
    return {
        "input_text": [f"mnli_matched: premise: {premise} hypothesis: {hypothesis}"
                      for premise, hypothesis in zip(examples["premise"], examples["hypothesis"])],
        "target_text": [str(label) for label in examples["label"]],
        "task_type": ["mnli_matched"] * len(examples["premise"])
    }

def preprocess_mnli_mismatched(examples):
    return {
        "input_text": [f"mnli_mismatched: premise: {premise} hypothesis: {hypothesis}"
                      for premise, hypothesis in zip(examples["premise"], examples["hypothesis"])],
        "target_text": [str(label) for label in examples["label"]],
        "task_type": ["mnli_mismatched"] * len(examples["premise"])
    }

def preprocess_mrpc(examples):
    return {
        "input_text": [f"mrpc: sentence1: {s1} sentence2: {s2}" 
                      for s1, s2 in zip(examples["sentence1"], examples["sentence2"])],
        "target_text": [str(label) for label in examples["label"]],
        "task_type": ["mrpc"] * len(examples["sentence1"])
    }

def preprocess_qnli(examples):
    return {
        "input_text": [f"qnli: question: {question} sentence: {sentence}" 
                      for question, sentence in zip(examples["question"], examples["sentence"])],
        "target_text": [str(label) for label in examples["label"]],
        "task_type": ["qnli"] * len(examples["question"])
    }

def preprocess_qqp(examples):
    return {
        "input_text": [f"qqp: question1: {q1} question2: {q2}" 
                      for q1, q2 in zip(examples["question1"], examples["question2"])],
        "target_text": [str(label) for label in examples["label"]],
        "task_type": ["qqp"] * len(examples["question1"])
    }

def preprocess_rte(examples):
    return {
        "input_text": [f"rte: sentence1: {s1} sentence2: {s2}" 
                      for s1, s2 in zip(examples["sentence1"], examples["sentence2"])],
        "target_text": [str(label) for label in examples["label"]],
        "task_type": ["rte"] * len(examples["sentence1"])
    }

def preprocess_sst2(examples):
    return {
        "input_text": ["sst2: " + sent for sent in examples["sentence"]],
        "target_text": [str(label) for label in examples["label"]],
        "task_type": ["sst2"] * len(examples["sentence"])
    }

def preprocess_stsb(examples):
    def round_to_nearest(x):
        return str(round(x * 5) / 5)
    
    return {
        "input_text": [
            f"stsb sentence1: {s1} sentence2: {s2}"
            for s1, s2 in zip(examples["sentence1"], examples["sentence2"])
        ],
        "target_text": [round_to_nearest(score) for score in examples["label"]],
        "task_type": ["stsb"] * len(examples["sentence1"])
    }

def prepare_datasets():
    datasets_info = [
        # ("cola", preprocess_cola),
        # ("mnli", preprocess_mnli),
        ("mrpc", preprocess_mrpc),
        # ("qnli", preprocess_qnli),
        # ("qqp", preprocess_qqp),
        # ("rte", preprocess_rte),
        # ("sst2", preprocess_sst2),
        # ("stsb", preprocess_stsb),
    ]

    train_datasets = []
    validation_datasets = []
    test_datasets = []

    for task, preprocess_fn in datasets_info:
        train_dataset = load_dataset("glue", task, split="train")

        if task == "mnli":
            validation_matched = load_dataset("glue", task, split="validation_matched")
            validation_mismatched = load_dataset("glue", task, split="validation_mismatched")

            train_dataset = train_dataset.map(preprocess_mnli_mismatched, batched=True, batch_size=10000, num_proc=10, remove_columns=train_dataset.column_names)
            validation_matched = validation_matched.map(preprocess_mnli_matched, batched=True, batch_size=10000, num_proc=10, remove_columns=validation_matched.column_names)
            validation_mismatched = validation_mismatched.map(preprocess_mnli_mismatched, batched=True, batch_size=10000, num_proc=10, remove_columns=validation_mismatched.column_names)

            train_datasets.append(train_dataset)
            validation_datasets.append(validation_matched)
            validation_datasets.append(validation_mismatched)
            test_datasets.append(validation_matched)
            test_datasets.append(validation_mismatched)
        else:
            validation_dataset = load_dataset("glue", task, split="validation")

            if len(train_dataset) > 10000 or task == "stsb":
                train_indices, new_validation_indices = get_train_validation_split(train_dataset)
                new_validation = train_dataset.select(new_validation_indices)
                train_dataset = train_dataset.select(train_indices)
                test_dataset = validation_dataset
                validation_dataset = new_validation
            # elif task == "stsb":
            #     test_dataset = load_dataset("glue", task, split="test")
            else:
                validation_indices, test_indices = get_validation_test_split(validation_dataset)
                test_dataset = validation_dataset.select(test_indices)
                validation_dataset = validation_dataset.select(validation_indices)

            train_dataset = train_dataset.map(preprocess_fn, batched=True, batch_size=10000, num_proc=10, remove_columns=train_dataset.column_names)
            validation_dataset = validation_dataset.map(preprocess_fn, batched=True, batch_size=10000, num_proc=10, remove_columns=validation_dataset.column_names)
            test_dataset = test_dataset.map(preprocess_fn, batched=True, batch_size=10000, num_proc=10, remove_columns=test_dataset.column_names)

            train_datasets.append(train_dataset)
            validation_datasets.append(validation_dataset)
            test_datasets.append(test_dataset)

    return train_datasets, validation_datasets, test_datasets

def create_label_mappings(datasets):
    all_labels = [label for dataset in datasets for label in dataset['target_text']]
    unique_labels = sorted(set(all_labels))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label

def convert_labels_to_ids(dataset, label2id):
    def mapper(example):
        example['labels'] = label2id[example['target_text']]
        return example
    return dataset.map(mapper)

def tokenize_function(examples):
    model_name = "t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    inputs = examples["input_text"]
    targets = examples["target_text"]
    
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, padding="max_length", truncation=True, max_length=64)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def calculate_dataset_distribution(train_dataset, validation_dataset, test_dataset):
    distribution = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})
    
    for dataset, split_name in [(train_dataset, 'train'), (validation_dataset, 'val'), (test_dataset, 'test')]:
        task_counts = Counter(dataset['task_type'])
        for task, count in task_counts.items():
            distribution[task][split_name] = count
    
    return distribution

def get_train_validation_split(dataset, validation_size=1000, seed=42):
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_size = len(dataset)
    indices = torch.randperm(train_size, generator=generator).tolist()
    return indices[validation_size:], indices[:validation_size]

def get_validation_test_split(dataset, seed=42):
    generator = torch.Generator()
    generator.manual_seed(seed)
    validation_size = len(dataset)
    indices = torch.randperm(validation_size, generator=generator).tolist()
    split_point = validation_size // 2
    return indices[:split_point], indices[split_point:]

def prepare_full_data_pipeline():
    train_datasets, validation_datasets, test_datasets = prepare_datasets()
    
    combined_train_dataset = concatenate_datasets(train_datasets)
    combined_validation_dataset = concatenate_datasets(validation_datasets)
    combined_test_dataset = concatenate_datasets(test_datasets)

    # Tokenize datasets
    train_dataset = combined_train_dataset.map(tokenize_function, batched=True, num_proc=15, batch_size=40000, remove_columns=['input_text', 'target_text'])
    validation_dataset = combined_validation_dataset.map(tokenize_function, batched=True, num_proc=15, batch_size=40000, remove_columns=['input_text', 'target_text'])
    test_dataset = combined_test_dataset.map(tokenize_function, batched=True, num_proc=15, batch_size=40000, remove_columns=['input_text', 'target_text'])
    
    train_dataset = train_dataset.shuffle(seed=42)
    
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "task_type"])
    validation_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "task_type"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "task_type"])
    
    logger.info("Data preprocessing with task relationship analysis and visualization completed successfully.")
    
    return train_dataset, validation_dataset, test_dataset

from typing import Union

def prepare_dataloaders(train_dataset, validation_dataset, test_dataset, batch_size, num_workers=4):
    # Calculate distribution
    distribution = calculate_dataset_distribution(train_dataset, validation_dataset, test_dataset)
    
    # Print distribution table
    print("\nDataset Distribution:")
    print(f"{'Task Name':<20} {'Train Samples':<15} {'Val Samples':<15} {'Test Samples':<15} {'Total Samples':<15}")
    print("-" * 80)
    
    total_train, total_val, total_test = 0, 0, 0
    for task, counts in distribution.items():
        train_count = counts['train']
        val_count = counts['val']
        test_count = counts['test']
        total_count = train_count + val_count + test_count
        print(f"{task:<20} {train_count:<15} {val_count:<15} {test_count:<15} {total_count:<15}")
        total_train += train_count
        total_val += val_count
        total_test += test_count
    
    print("-" * 80)
    print(f"{'Total':<20} {total_train:<15} {total_val:<15} {total_test:<15} {total_train + total_val + total_test:<15}")
    
    # Initialize DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, task_to_id),
        num_workers=num_workers,
        pin_memory=True
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, task_to_id),
        num_workers=num_workers,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, task_to_id),
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, validation_dataloader, test_dataloader

tasks = [
    "cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2",
    "mnli_matched", "mnli_mismatched", "stsb"
]
task_to_id = {task: idx for idx, task in enumerate(tasks)}
id_to_task = {idx: task for task, idx in task_to_id.items()}

# Initialize the datasets
train_dataset, validation_dataset, test_dataset = prepare_full_data_pipeline()

print(len(train_dataset))
print(len(validation_dataset))
print(len(test_dataset))


warnings.filterwarnings("ignore")
tokenizer = tokenizer
model = T5ForConditionalGeneration.from_pretrained(
    "t5-base", device_map="auto"
)
# Define LoRA Config
lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.1, bias="none", task_type=TaskType.SEQ_2_SEQ_LM
)

# Get the PEFT mode
model = get_peft_model(model, lora_config)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
            print(f"Trainable: {name}, Shape: {param.shape}")
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}")

print_trainable_parameters(model)

wandb.init(project="YOUR-WANDB-PROJECT", entity="YOUR-WANDB-ENTITY", name="t5-separate-lora-mrpc-seed42", allow_val_change=True)
# File: trainer.py
import itertools
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score, matthews_corrcoef
from scipy.stats import spearmanr,pearsonr
from dataclasses import dataclass

@dataclass
class EvalLoopOutput:
    predictions: Optional[np.ndarray] = None
    label_ids: Optional[np.ndarray] = None
    metrics: Optional[Dict[str, float]] = None
    num_samples: Optional[int] = None

def evaluate_task(model, dataset, task_type, metric_name, task_to_id):
    print(f"Evaluating task: {task_type}")
    metric = load_metric("glue", task_type, trust_remote_code=True)
    model.eval()

    device = next(model.parameters()).device

    # Use a lambda function to pass task_to_id to collate_fn
    dataloader = DataLoader(dataset, batch_size=32,
                            collate_fn=lambda batch: collate_fn(batch, task_to_id))

    all_preds = []
    all_labels = []

    for batch in dataloader:
        task_types = batch.pop("task_type")  # Remove task_type from batch
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=64,
                early_stopping=True,
            )

            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

            all_preds.extend(predictions)
            all_labels.extend(labels)

    if task_type == "cola":
        mcc = matthews_corrcoef(all_labels, all_preds)
        print(f"Matthews Correlation Coefficient: {mcc}")
        return {"matthews_correlation": mcc}
    elif task_type == "mrpc":
        f1 = f1_score(all_labels, all_preds, average="micro")
        acc = accuracy_score(all_labels, all_preds)
        return {"f1": f1, "accuracy": acc}
    elif task_type == "qqp":
        f1 = f1_score(all_labels, all_preds, average="macro")
        acc = accuracy_score(all_labels, all_preds)
        return {"f1": f1, "accuracy": acc}
    elif task_type == "stsb":
        pearson, _ = pearsonr(all_labels, all_preds)
        spearman, _ = spearmanr(all_labels, all_preds)
        return {"pearson": pearson, "spearman": spearman}
    else:
        results = metric.compute(predictions=all_preds, references=all_labels)
        return results

# The main evaluation loop
def call():
    task_metrics = [
        # ("cola", "matthews_correlation"),
        # ("sst2", "accuracy"),
        ("mrpc", "f1"),
        ("mrpc","accuracy"),
        # ("qqp","f1"),
        # ("qqp","accuracy"),
        # ("stsb","spearmanr"),
        # ("stsb","pearson"),
        # ("mnli", "accuracy"),
        # ("qnli", "accuracy"),
        # ("rte","accuracy"),
    ]
    
    results_table = []
    
    for task, main_metric in task_metrics:
        print(f"Processing task: {task}")
        try:
            if task == "mnli":
                matched_results = evaluate_task(
                    model,
                    test_dataset.filter(lambda x: x["task_type"] == "mnli_matched"),
                    "mnli_matched",
                    task,
                    task_to_id
                )
                mismatched_results = evaluate_task(
                    model,
                    test_dataset.filter(lambda x: x["task_type"] == "mnli_mismatched"),
                    "mnli_mismatched",
                    task,
                    task_to_id
                )
                results_table.append([f"{task} matched", f"{matched_results[main_metric]:.4f}"])
                results_table.append([f"{task} mismatched", f"{mismatched_results[main_metric]:.4f}"])
                wandb.log({
                    f"final_{main_metric}_mnli_matched": matched_results[main_metric],
                    f"final_{main_metric}_mnli_mismatched": mismatched_results[main_metric]
                })
            elif task == "stsb":
                results = evaluate_task(
                    model,
                    test_dataset.filter(lambda x: x["task_type"] == task),
                    task,
                    task,
                    task_to_id
                )
                results_table.append([task, f"Pearson: {results['pearson']:.4f}, Spearman: {results['spearman']:.4f}"])
                wandb.log({
                    f"final_pearson_{task}": results["pearson"],
                    f"final_spearman_{task}": results["spearman"]
                })
            else:
                results = evaluate_task(
                    model, 
                    test_dataset.filter(lambda x: x["task_type"] == task), 
                    task, 
                    task,
                    task_to_id
                )
                results_table.append([task, f"{results[main_metric]:.4f}"])
                wandb.log({f"final_{main_metric}_{task}": results[main_metric]})
        except Exception as e:
            print(f"Error evaluating task {task}: {str(e)}")
            results_table.append([task, "Error"])
    
    print(tabulate(results_table, headers=["Task", "Score"], tablefmt="grid"))

class CustomTrainer(Trainer):
    def __init__(self, *args, train_dataloader=None, eval_dataloader=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = GradScaler()
        self.max_grad_norm = 1.0
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.task_losses = defaultdict(list)
        self.additional_eval_step = 65536

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=inputs['labels']
        )
        
        loss = outputs.loss
        
        # Compute task-wise losses without weighting
        task_types = inputs.get('task_type')
        if task_types is not None:
            for task in task_types:
                self.task_losses[task].append(loss.item())

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with autocast():
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        self.scaler.scale(loss).backward()

        if (self.state.global_step + 1) % self.args.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        ####
        if self.state.global_step == self.additional_eval_step:
            call()

        return loss.detach()

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        model = self._wrap_model(self.model, training=False)
        model.eval()
    
        eval_losses = defaultdict(list)
        task_metrics = defaultdict(lambda: {'predictions': [], 'labels': [], 'samples': 0})
    
        eval_bar = tqdm(total=len(dataloader), desc=description, leave=False)
    
        for step, inputs in enumerate(dataloader):
            if inputs is None:
                continue  # Skip invalid batches
            inputs = self._prepare_inputs(inputs)
            batch_task_types = inputs.pop('task_type', [])
    
            with torch.no_grad():
                outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss, _ = outputs
    
                generated_ids = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=64,
                    early_stopping=True
                )
    
                decoded_preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                decoded_labels = self.tokenizer.batch_decode(inputs['labels'], skip_special_tokens=True)
    
                for task in batch_task_types:
                    eval_losses[task].append(loss.item())
    
            for task, pred, label in zip(batch_task_types, decoded_preds, decoded_labels):
                task_metrics[task]['predictions'].append(pred)
                task_metrics[task]['labels'].append(label)
                task_metrics[task]['samples'] += 1
    
            eval_bar.update(1)
    
        eval_bar.close()
    
        # Compute final metrics
        metrics = {}
        for task, task_data in task_metrics.items():
            preds = task_data['predictions']
            labels = task_data['labels']
            
            if task == 'stsb':
                # Assuming predictions and labels are already floating-point numbers
                try:
                    preds = np.array([float(p) for p in preds])
                    labels = np.array([float(l) for l in labels])
                    pearson_corr, _ = pearsonr(labels, preds)
                    spearman_corr, _ = spearmanr(labels, preds)
                    metrics[f'{task}_pearson'] = pearson_corr
                    metrics[f'{task}_spearmanr'] = spearman_corr
                    metrics[f'{task}_corr'] = (pearson_corr + spearman_corr) / 2
                except ValueError:
                    # Handle cases where conversion to float fails
                    metrics[f'{task}_pearson'] = 0.0
                    metrics[f'{task}_spearmanr'] = 0.0
                    metrics[f'{task}_corr'] = 0.0
            elif task == 'cola':
                mcc = matthews_corrcoef(preds, labels)
                metrics['cola_mcc'] = mcc
            elif task in ["mrpc","qqp"]:
                if task == "mrpc":
                    f1 = f1_score(labels,preds,average="micro")
                else: 
                    f1 = f1_score(labels,preds,average="macro")
                    
                acc = accuracy_score(labels,preds)
                metrics[f'eval_{task}_f1'] = f1  # Note the 'eval_' prefix
                metrics[f'eval_{task}_accuracy'] = acc  # Note the 'eval_' prefix
            else:
                # For classification tasks
                correct = sum(
                    (p.strip() if isinstance(p, str) else str(p)) == 
                    (l.strip() if isinstance(l, str) else str(l))
                    for p, l in zip(preds, labels)
                )
                accuracy = correct / len(preds) if len(preds) > 0 else 0.0
                metrics[f'{task}_accuracy'] = accuracy
            
            metrics[f'{task}_samples'] = task_data['samples']
    
        # Compute overall accuracy for classification tasks
        classification_tasks = [task for task in task_metrics if task != 'stsb']
        overall_accuracy = sum(metrics.get(f'{task}_accuracy', 0.0) * metrics.get(f'{task}_samples', 0) for task in classification_tasks)
        overall_samples = sum(metrics.get(f'{task}_samples', 0) for task in classification_tasks)
        metrics['overall_accuracy'] = overall_accuracy / overall_samples if overall_samples > 0 else 0.0
    
        # Compute average loss per task
        for task, losses in eval_losses.items():
            metrics[f'{task}_loss'] = np.mean(losses) if len(losses) > 0 else 0.0
    
        # Log evaluation metrics to wandb
        wandb.log(metrics, step=self.state.global_step)
    
        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=sum(task_data['samples'] for task_data in task_metrics.values()))
    
    def _prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)
        return inputs

    def get_train_dataloader(self):
        return self.train_dataloader or super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None):
        return self.eval_dataloader or super().get_eval_dataloader(eval_dataset)

    def log(self, logs: Dict[str, float]) -> None:
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
        wandb.log(logs, step=self.state.global_step)

# Define the collate function with task_to_id mapping
def collate_fn(batch: list, task_to_id: Dict[str, int]) -> Optional[Dict[str, Union[torch.Tensor, Any]]]:
    valid_examples = [
        x for x in batch
        if all(key in x for key in ['input_ids', 'attention_mask', 'labels', 'task_type'])
    ]

    if not valid_examples:
        logger.warning("No valid examples in the batch. Returning None.")
        return None

    input_ids = torch.stack([torch.tensor(x['input_ids']) for x in valid_examples])
    attention_mask = torch.stack([torch.tensor(x['attention_mask']) for x in valid_examples])
    labels = torch.stack([torch.tensor(x['labels']) for x in valid_examples])
    # task_ids = torch.tensor([task_to_id[x['task_type']] for x in valid_examples])
    task_types = [x['task_type'] for x in valid_examples]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        # 'task_ids': task_ids,
        'task_type': task_types
    }


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=50,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=1000,
    weight_decay=0.01,
    learning_rate=1e-4,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    bf16=True,
    report_to="wandb",
    load_best_model_at_end=True,
    metric_for_best_model="eval_mrpc_f1",  # Using MRPC's F1 score
    greater_is_better=True,  # Higher F1 score is better
)

# Prepare dataloaders
train_dataloader, eval_dataloader, test_dataloader = prepare_dataloaders(
    train_dataset, 
    validation_dataset, 
    test_dataset, 
    batch_size=training_args.per_device_train_batch_size
)

# Calculate the number of training steps
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = int(training_args.num_train_epochs * num_update_steps_per_epoch)
num_warmup_steps = training_args.warmup_steps

optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    optimizers=(optimizer, lr_scheduler),
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader
)

# Start training
trainer.train()
call()
trainer.save_model("shared_mrpc_seed42")