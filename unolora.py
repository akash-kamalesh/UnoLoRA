import logging
import random
import warnings
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Union, TypeVar
from tabulate import tabulate
import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import Sampler, DataLoader
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, AdamW, get_cosine_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from dataclasses import dataclass
from tqdm.notebook import tqdm
import wandb
from datasets import load_metric
from torch.utils.data import DataLoader
import torch
import numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
import os
import math

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

T_co = TypeVar('T_co', covariant=True)

class MultiTaskBatchSampler(Sampler[T_co]):
    """Defines a sampler to sample multiple datasets with temperature sampling."""

    def __init__(self, dataset_sizes: List[int], batch_size: int, temperature: float,
                 num_replicas: Optional[int] = None, rank: Optional[int] = None,
                 seed: int = 0, shuffle: bool = True):
        if num_replicas is None:
            num_replicas = 1
        if rank is None:
            rank = 0
        
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.dataset_sizes = dataset_sizes
        self.rank_dataset_sizes = [dataset_size // self.num_replicas for dataset_size in self.dataset_sizes]
        self.dataset_offsets = torch.cumsum(torch.LongTensor([0] + dataset_sizes), 0)
        self.total_sizes = [(dataset_size // self.num_replicas) * self.num_replicas for dataset_size in self.dataset_sizes]
        self.temperature = temperature
        self.seed = seed
        self.epoch = 0  # Initialize epoch to 0
        self.num_batches_per_epoch = (sum(dataset_sizes) + self.batch_size - 1) // self.batch_size // self.num_replicas
        self.shuffle = shuffle
        self.sampled_task_distribution = Counter()  # Initialize the counter

    def generate_tasks_distribution(self):
        """Given the dataset sizes, computes the weights to sample each dataset
        according to the temperature sampling."""
        total_size = sum(self.dataset_sizes)
        weights = torch.tensor([(size / total_size) ** (1.0 / self.temperature) for size in self.dataset_sizes])
        return weights / weights.sum()

    def __iter__(self):
        generator = torch.Generator()
        # Ensure seed + epoch is always an integer
        current_epoch = self.epoch if self.epoch is not None else 0
        generator.manual_seed(self.seed + current_epoch)

        indices = [
            torch.randperm(dataset_size, generator=generator).tolist() if self.shuffle
            else list(range(dataset_size))
            for dataset_size in self.dataset_sizes
        ]

        self.rank_indices = [
            indices[i][self.rank:self.total_sizes[i]:self.num_replicas]
            for i in range(len(self.dataset_sizes))
        ]

        tasks_distribution = self.generate_tasks_distribution()
        batch_task_assignments = torch.multinomial(
            tasks_distribution,
            self.num_batches_per_epoch,
            replacement=True,
            generator=generator
        )

        self.sampled_task_distribution = Counter()  # Reset the counter at the start of __iter__

        for batch_task in batch_task_assignments:
            task_idx = batch_task.item()
            self.sampled_task_distribution[task_idx] += 1
            num_task_samples = self.rank_dataset_sizes[task_idx]
            if num_task_samples == 0:
                continue  # Skip if no samples are available for the selected task
            indices_sample = torch.randint(low=0, high=num_task_samples, size=(self.batch_size,), generator=generator).tolist()
            selected_indices = (self.dataset_offsets[batch_task] + torch.tensor(self.rank_indices[batch_task])[indices_sample]).tolist()
            yield selected_indices

    def __len__(self):
        return self.num_batches_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def get_sampled_task_distribution(self):
        """Returns the sampled task distribution as a dictionary."""
        return dict(self.sampled_task_distribution)

def prepare_datasets():
    datasets_info = [
        ("cola", preprocess_cola),
        ("mnli", preprocess_mnli),
        ("mrpc", preprocess_mrpc),
        ("qnli", preprocess_qnli),
        ("qqp", preprocess_qqp),
        ("rte", preprocess_rte),
        ("sst2", preprocess_sst2),
        ("stsb", preprocess_stsb),  # Ensure STS-B is included here
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
    
    # Calculate task distribution based on temperature and sampler
    task_counter = Counter(train_dataset['task_type'])
    tasks = [
        "cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2",
        "mnli_matched", "mnli_mismatched", "stsb"
    ]
    dataset_sizes = [task_counter.get(task, 0) for task in tasks]
    
    temperature = 10  # You can adjust this value to see its effect
    sampler = MultiTaskBatchSampler(
        dataset_sizes=dataset_sizes,
        batch_size=batch_size,
        temperature=temperature,
        num_replicas=1,
        rank=0,
        seed=42,
        shuffle=True
    )
    
    # Compute the normalized weights based on temperature
    task_distribution = sampler.generate_tasks_distribution()
    
    # Calculate expected number of samples per task
    total_train_samples = sampler.num_batches_per_epoch * batch_size
    sampled_task_distribution = {tasks[i]: int(task_distribution[i].item() * sampler.num_batches_per_epoch * batch_size) 
                                 for i in range(len(tasks))}
    
    # Print sampling distribution table
    print("\nSampling Distribution Based on Temperature:")
    print(f"{'Task Name':<20} {'Total Train Samples':<20} {'Sampled Train Samples':<25}")
    print("-" * 70)
    
    for task in tasks:
        total_samples = task_counter.get(task, 0)
        sampled_samples = sampled_task_distribution.get(task, 0)
        print(f"{task:<20} {total_samples:<20} {sampled_samples:<25}")
    
    print("-" * 70)
    print(f"{'Total':<20} {total_train:<20} {sum(sampled_task_distribution.values()):<25}")
    
    # Initialize DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
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


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union
import math
from transformers import T5ForConditionalGeneration, T5Config

wandb.init(project="YOUR-WANDB-PROJECT", entity="YOUR-WANDB-ENTITY", name=f"Unolora_t5_base_seed42", allow_val_change=True)
class SharedHypernetwork(nn.Module):
    def __init__(
        self, 
        num_tasks: int,
        hidden_dim: int,
        output_dim: int,
        sample_encoding_dim: int,
        bottleneck_dim: int = 32,
        num_layers: int = 3,
        max_position: int = 100
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.max_position = max_position
        self.bottleneck_dim = bottleneck_dim
        self.hidden_dim = hidden_dim
        
        self.bottleneck = nn.Sequential(
            nn.Linear(sample_encoding_dim, bottleneck_dim), 
            nn.ReLU(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.ReLU()
        )
        
        self.task_embedding = nn.Embedding(num_tasks, hidden_dim)
        self.position_embedding = nn.Embedding(max_position, hidden_dim)
        self.bottleneck_projection = nn.Linear(bottleneck_dim, hidden_dim)
        
        self.combined_network = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, task_ids: torch.Tensor, sample_encodings: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        bottleneck_features = self.bottleneck(sample_encodings)
        
        task_embeddings = self.task_embedding(task_ids).unsqueeze(1).expand(-1, positions.size(1), -1)
        position_embeddings = self.position_embedding(positions)
        bottleneck_embeddings = self.bottleneck_projection(bottleneck_features).unsqueeze(1).expand(-1, positions.size(1), -1)
        
        combined_embeddings = torch.cat([task_embeddings, position_embeddings, bottleneck_embeddings], dim=-1)
        
        task_embedding = self.combined_network(combined_embeddings)
        
        return task_embedding

class Unolora(nn.Module):
    def __init__(
        self, 
        linear: nn.Linear, 
        rank: int, 
        alpha: float, 
        task_embedding_dim: int, 
        layer_idx: int,
        is_query: bool,
        dropout: float = 0.1
    ):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.task_embedding_dim = task_embedding_dim
        self.layer_idx = layer_idx
        self.is_query = is_query

        self.lora_A = nn.Parameter(torch.zeros(linear.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, linear.out_features))

        self.task_scale = nn.Linear(task_embedding_dim, rank, bias=True)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.normal_(self.task_scale.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.task_scale.bias)

        self.dropout = nn.Dropout(dropout)

        self.current_task_embedding = None

    def forward(self, x):
        original = self.linear(x)

        if self.current_task_embedding is not None:
            task_scale = self.task_scale(self.current_task_embedding) 
            task_scale = task_scale.unsqueeze(1)
        else:
            task_scale = torch.ones((x.size(0), 1, self.rank), device=x.device)

        x = self.dropout(x)
        lora_A_out = x @ self.lora_A
        lora_scaled = lora_A_out * task_scale
        lora = lora_scaled @ self.lora_B

        result = original + self.scaling * lora
        return result

class EnhancedUnoloraWrapper(nn.Module):
    def __init__(
        self,
        model: T5ForConditionalGeneration,
        tasks: List[str],
        rank: int,
        alpha: float,
        task_feature_dim: int,
        sample_encoding_dim: int,
        bottleneck_dim: int = 32,
        max_position: int = 100
    ):
        super().__init__()
        self.model = model
        self.tasks = tasks
        self.rank = rank
        self.alpha = alpha
        
        self.shared_hypernetwork = SharedHypernetwork(
            num_tasks=len(tasks),
            hidden_dim=task_feature_dim,
            output_dim=task_feature_dim,
            sample_encoding_dim=sample_encoding_dim,
            bottleneck_dim=bottleneck_dim,
            num_layers=3,
            max_position=max_position
        )
        self.freeze_base_model()
        self.num_layers = self.replace_lora_layers(task_feature_dim)

    def freeze_base_model(self):
        for param in self.model.parameters():
            param.requires_grad = False
        print("Base model parameters frozen")

    def replace_lora_layers(self, task_feature_dim: int):
        layer_idx = 0
        for block in self.model.encoder.block:
            block.layer[0].SelfAttention.q = Unolora(
                block.layer[0].SelfAttention.q,
                self.rank,
                self.alpha,
                task_feature_dim,
                layer_idx,
                is_query=True
            )
            layer_idx += 1
            block.layer[0].SelfAttention.v = Unolora(
                block.layer[0].SelfAttention.v,
                self.rank,
                self.alpha,
                task_feature_dim,
                layer_idx,
                is_query=False
            )
            layer_idx += 1

        for block in self.model.decoder.block:
            block.layer[0].SelfAttention.q = Unolora(
                block.layer[0].SelfAttention.q,
                self.rank,
                self.alpha,
                task_feature_dim,
                layer_idx,
                is_query=True
            )
            layer_idx += 1
            block.layer[0].SelfAttention.v = Unolora(
                block.layer[0].SelfAttention.v,
                self.rank,
                self.alpha,
                task_feature_dim,
                layer_idx,
                is_query=False
            )
            layer_idx += 1
            block.layer[1].EncDecAttention.q = Unolora(
                block.layer[1].EncDecAttention.q,
                self.rank,
                self.alpha,
                task_feature_dim,
                layer_idx,
                is_query=True
            )
            layer_idx += 1
            block.layer[1].EncDecAttention.v = Unolora(
                block.layer[1].EncDecAttention.v,
                self.rank,
                self.alpha,
                task_feature_dim,
                layer_idx,
                is_query=False
            )
            layer_idx += 1

        return layer_idx

    def apply_task_embeddings(self, task_embeddings):
        lora_idx = 0
        for module in self.model.modules():
            if isinstance(module, Unolora):
                module.current_task_embedding = task_embeddings[:, lora_idx, :]
                lora_idx += 1

    def forward(self, input_ids, attention_mask, labels=None, task_ids=None):
        if task_ids is None:
            raise ValueError("task_ids must be provided")
        
        sample_encodings = self.model.encoder.embed_tokens(input_ids).mean(dim=1)
        positions = torch.arange(self.num_layers, device=input_ids.device).expand(task_ids.size(0), -1)
        positions = positions % self.shared_hypernetwork.max_position
        
        task_embeddings = self.shared_hypernetwork(task_ids, sample_encodings, positions)
        
        self.apply_task_embeddings(task_embeddings)
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def generate_custom(self, input_ids, attention_mask, task_ids, **kwargs):
        if task_ids is None:
            raise ValueError("task_ids must be provided")
        
        sample_encodings = self.model.encoder.embed_tokens(input_ids).mean(dim=1)
        
        positions = torch.arange(self.num_layers, device=input_ids.device).expand(task_ids.size(0), -1)
        positions = positions % self.shared_hypernetwork.max_position
        
        task_embeddings = self.shared_hypernetwork(task_ids, sample_encodings, positions)
        
        self.apply_task_embeddings(task_embeddings)
        
        max_length = kwargs.get('max_length', 64)
        max_length = min(max_length, 512)
        kwargs['max_length'] = max_length
        
        try:
            generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        except RuntimeError as e:
            print(f"Error in generate: {str(e)}")
            print(f"input_ids max: {input_ids.max().item()}, min: {input_ids.min().item()}")
            print(f"attention_mask sum: {attention_mask.sum().item()}")
            raise
        
        return generated_ids

    def print_trainable_parameters(self):
        trainable_params = 0
        all_params = 0
        for name, param in self.named_parameters():
            num_params = param.numel()
            all_params += num_params
            if param.requires_grad:
                trainable_params += num_params
                print(f"Trainable: {name}, Shape: {param.shape}")
        
        print(f"Trainable params: {trainable_params} || All params: {all_params} || Trainable%: {100 * trainable_params / all_params:.2f}")
        
def get_Unolora_model(
    model_name: str, 
    tasks: List[str], 
    rank: int, 
    alpha: float, 
    task_embedding_dim: int, 
    sample_encoding_dim: int, 
    bottleneck_dim: int = 32
) -> EnhancedUnoloraWrapper:
    """
    Initializes the EnhancedUnoloraWrapper model with the specified parameters.

    Args:
        model_name (str): Name of the pre-trained model (e.g., "t5-base").
        tasks (List[str]): List of task names.
        rank (int): Rank parameter for LoRA.
        alpha (float): Alpha parameter for LoRA scaling.
        task_embedding_dim (int): Dimension of the task embeddings.
        sample_encoding_dim (int): Dimension of the sample's encoded representation.
        bottleneck_dim (int): Dimension of the bottleneck layer.

    Returns:
        EnhancedUnoloraWrapper: The wrapped model with Unolora and Shared Hypernetwork.
    """
    # Enable hidden states output in the model configuration
    config = T5Config.from_pretrained(model_name)
    config.output_hidden_states = True  # Enable hidden states output

    model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)
    
    # Calculate the correct max_position
    num_layers = len(model.encoder.block) + len(model.decoder.block)
    max_position = num_layers * 2  # *2 for query and value
    
    wrapped_model = EnhancedUnoloraWrapper(
        model=model, 
        tasks=tasks, 
        rank=rank, 
        alpha=alpha, 
        task_feature_dim=task_embedding_dim, 
        sample_encoding_dim=sample_encoding_dim,
        bottleneck_dim=bottleneck_dim,
        max_position=max_position  # Pass the calculated max_position
    )
    return wrapped_model

# Usage Example
# Define task list and create task_to_id mapping
tasks = [
    "cola",
    "mnli",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
    "mnli_matched",
    "mnli_mismatched",
    "stsb"  # Added STS-B
]
task_to_id = {task: idx for idx, task in enumerate(tasks)}
id_to_task = {idx: task for task, idx in task_to_id.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define embedding dimensions
task_embedding_dim = 8
sample_encoding_dim = 768  # T5-base hidden size
bottleneck_dim = 16 # Bottleneck dimension

# Define LoRA parameters
rank = 8
alpha = 16

# Initialize the model
model = get_Unolora_model(
    model_name="t5-base", 
    tasks=tasks, 
    rank=rank, 
    alpha=alpha, 
    task_embedding_dim=task_embedding_dim, 
    sample_encoding_dim=sample_encoding_dim,
    bottleneck_dim=bottleneck_dim
).to(device)

print("Model initialized, printing trainable parameters")
model.print_trainable_parameters()

import itertools
from collections import Counter
from sklearn.metrics import accuracy_score, matthews_corrcoef
from scipy.stats import spearmanr,pearsonr
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
            # Generate predictions
            generated_ids = model.generate_custom(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                task_ids=batch['task_ids'],
                max_length=64,  # Adjust as needed
                early_stopping=True
            )

            # Decode predictions and labels
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            labels = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)

            all_preds.extend(predictions)
            all_labels.extend(labels)

    # Process predictions and labels based on the task
    # if task_type in ['cola', 'sst2', 'mrpc', 'qqp', 'qnli', 'rte']:
    #     all_preds = [labelmapper.get(pred.strip(), '0') for pred in all_preds]
    #     all_labels = [labelmapper.get(label.strip(), '0') for label in all_labels]
    # if task_type == 'stsb':
    #     all_preds = [float(pred) if pred.replace('.','',1).isdigit() else 0.0 for pred in all_preds]
    #     all_labels = [float(label) if label.replace('.','',1).isdigit() else 0.0 for label in all_labels]
    if task_type!="stsb":
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
    else:
        all_preds = np.array(all_preds,dtype=float)
        all_labels = np.array(all_labels,dtype=float)

    if task_type == "cola":
        mcc = matthews_corrcoef(all_labels, all_preds)
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
        pearson_corr, _ = pearsonr(all_labels, all_preds)
        spearman_corr, _ = spearmanr(all_labels, all_preds)
        return {"pearson": pearson_corr, "spearmanr": spearman_corr, "corr": (pearson_corr + spearman_corr) / 2}
    else: 
        acc = accuracy_score(all_labels, all_preds)
        return {"accuracy":acc}

def call():
    # The main evaluation loop
    task_metrics = [
        ("cola", "matthews_correlation"),
        ("sst2", "accuracy"),
        ("mrpc", "f1"),
        ("mrpc","accuracy"),
        ("qqp","f1"),
        ("qqp","accuracy"),
        ("stsb","spearmanr"),
        ("stsb","pearson"),
        ("mnli", "accuracy"),
        ("qnli", "accuracy"),
        ("rte","accuracy"),
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
    print(results_table)
    wandb.log({"final_results_table": wandb.Table(data=results_table, columns=["Task", "Score"])})
    print(tabulate(results_table, headers=["Task", "Score"], tablefmt="grid"))

class CustomTrainer(Trainer):
    def __init__(self, *args, train_dataloader=None, eval_dataloader=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = GradScaler()
        self.max_grad_norm = 1.0
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.task_losses = defaultdict(list)
        self.additional_eval_step = 65535
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            task_ids=inputs['task_ids'],
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
            task_ids = inputs['task_ids']
    
            try:
                with torch.no_grad():
                    # Compute loss
                    loss, _ = self.compute_loss(model, inputs, return_outputs=True)
    
                    # Generate predictions
                    generated_ids = model.generate_custom(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        task_ids=task_ids,
                        max_length=64,  # Adjust as needed
                        early_stopping=True
                    )
    
                    # Decode predictions and labels
                    decoded_preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    decoded_labels = self.tokenizer.batch_decode(inputs['labels'], skip_special_tokens=True)
    
                    # Compute task-wise evaluation losses without weighting
                    for task in batch_task_types:
                        eval_losses[task].append(loss.item())
    
                # Store predictions and labels for each task
                for task, pred, label in zip(batch_task_types, decoded_preds, decoded_labels):
                    task_metrics[task]['predictions'].append(pred)
                    task_metrics[task]['labels'].append(label)
                    task_metrics[task]['samples'] += 1
    
            except Exception as e:
                print(f"Error in evaluation step {step}: {str(e)}")
                print(f"Input shapes: input_ids {inputs['input_ids'].shape}, attention_mask {inputs['attention_mask'].shape}, task_ids {task_ids.shape}")
                continue
    
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
                metrics[f'{task}_f1'] = f1
                metrics[f'{task}_accuracy'] = acc
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

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        self.tokenizer.save_pretrained(output_dir)
        
        if hasattr(self.model, 'model'):
            self.model.model.config.save_pretrained(output_dir)
        else:
            self.logger.warning("Base model config not found. Skipping config saving.")
        
        Unolora_config = {
            "tasks": self.model.tasks,
            "rank": self.model.rank,
            "alpha": self.model.alpha,
        }
        torch.save(Unolora_config, os.path.join(output_dir, "Unolora_config.bin"))

    def _save_checkpoint(self, model, trial, metrics=None):
        PREFIX_CHECKPOINT_DIR = "checkpoint"
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self.args.output_dir
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir)

        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        torch.save(self.state.__dict__, os.path.join(output_dir, "trainer_state.json"))

        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics.get(metric_to_check, None)

            if metric_value is not None:
                operator = np.greater if self.args.greater_is_better else np.less
                if (
                    self.state.best_metric is None
                    or self.state.best_model_checkpoint is None
                    or operator(metric_value, self.state.best_metric)
                ):
                    self.state.best_metric = metric_value
                    self.state.best_model_checkpoint = output_dir

        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

    def log(self, logs: Dict[str, float]) -> None:
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
        wandb.log(logs, step=self.state.global_step)
        
    def train(self, *args, **kwargs):
        current_epoch = self.state.epoch if self.state.epoch is not None else 0
        if hasattr(self.train_dataloader, 'batch_sampler') and hasattr(self.train_dataloader.batch_sampler, 'set_epoch'):
            self.train_dataloader.batch_sampler.set_epoch(current_epoch)
        else:
            self.logger.warning("The train_dataloader's batch_sampler does not support epoch setting.")
        return super().train(*args, **kwargs)

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
    task_ids = torch.tensor([task_to_id[x['task_type']] for x in valid_examples])
    task_types = [x['task_type'] for x in valid_examples]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'task_ids': task_ids,
        'task_type': task_types
    }

training_args = TrainingArguments(
    output_dir="./results",
    max_steps = 262144,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=1000,
    weight_decay=0.01,
    learning_rate=1e-4,
    eval_strategy="steps",
    eval_steps=29535,
    save_strategy="steps",
    save_steps=29535,
    logging_dir="./logs",
    logging_steps=100,
    bf16=True,
    report_to="wandb",
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
wandb.finish()