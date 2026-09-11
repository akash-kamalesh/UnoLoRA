import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from transformers import T5ForConditionalGeneration, AutoTokenizer
from peft import PeftModel, PeftConfig
 
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
def load_model(adapter_path, base_model_name):
    peft_config = PeftConfig.from_pretrained(adapter_path)
    model = T5ForConditionalGeneration.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(model, adapter_path)
    return model
 
def extract_lora_matrices(model):
    lora_B, lora_A = {}, {}
    for name, param in model.named_parameters():
        if "lora_B" in name:
            lora_B[name] = param.detach().cpu().numpy()
        elif "lora_A" in name:
            lora_A[name] = param.detach().cpu().numpy()
    return lora_B, lora_A
 
def analyze_importance(single_task_model, multi_task_model):
    single_A, single_B = extract_lora_matrices(single_task_model)
    multi_A, multi_B = extract_lora_matrices(multi_task_model)
 
    results = {
        'magnitude': {
            'single_task': {
                'A_norm': np.mean([np.linalg.norm(a) for a in single_A.values()]),
                'B_norm': np.mean([np.linalg.norm(b) for b in single_B.values()])
            },
            'multi_task': {
                'A_norm': np.mean([np.linalg.norm(a) for a in multi_A.values()]),
                'B_norm': np.mean([np.linalg.norm(b) for b in multi_B.values()])
            }
        },
        'correlations': {
            'A_mean_correlation': np.mean([pearsonr(single_A[k].flatten(), multi_A[k].flatten())[0] for k in single_A.keys()]),
            'B_mean_correlation': np.mean([pearsonr(single_B[k].flatten(), multi_B[k].flatten())[0] for k in single_B.keys()])
        },
        'ranks': {
            'A_mean_rank': np.mean([np.linalg.matrix_rank(a) for a in multi_A.values()]),
            'B_mean_rank': np.mean([np.linalg.matrix_rank(b) for b in multi_B.values()])
        }
    }
    return results, single_A, single_B, multi_A, multi_B
 
 
def sparsity_analysis(single_A, single_B, multi_A, multi_B, threshold=1e-5):
    def compute_sparsity(matrix):
        return np.mean(np.abs(matrix) < threshold)
 
    return {
        'single_task': {
            'A_sparsity': np.mean([compute_sparsity(a) for a in single_A.values()]),
            'B_sparsity': np.mean([compute_sparsity(b) for b in single_B.values()])
        },
        'multi_task': {
            'A_sparsity': np.mean([compute_sparsity(a) for a in multi_A.values()]),
            'B_sparsity': np.mean([compute_sparsity(b) for b in multi_B.values()])
        }
    }
 
def singular_value_analysis(single_A, single_B, multi_A, multi_B, k=5):
    def get_top_singular_values(matrix):
        return np.linalg.svd(matrix, compute_uv=False)[:k]
 
    return {
        'single_task': {
            'A_sv': np.mean([get_top_singular_values(a) for a in single_A.values()], axis=0),
            'B_sv': np.mean([get_top_singular_values(b) for b in single_B.values()], axis=0)
        },
        'multi_task': {
            'A_sv': np.mean([get_top_singular_values(a) for a in multi_A.values()], axis=0),
            'B_sv': np.mean([get_top_singular_values(b) for b in multi_B.values()], axis=0)
        }
    }
 
 
def plot_lora_metrics_comparison(single_task_paths, multi_task_path, base_model_name, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    def compute_metrics(model):
        effective_ranks, singular_values, eigenvalues, frobenius_norms = [], [], [], []
        for name, param in model.named_parameters():
            if 'lora' in name.lower() and 'lora_a' in name.lower():
                U, S, V = torch.svd(param.data)
                effective_ranks.append((torch.sum(S) / torch.max(S)).item())
                singular_values.append(S[0].item())
                eigenvalues.append((S[0] ** 2).item())
                frobenius_norms.append(torch.norm(param.data, p='fro').item())
        return effective_ranks, singular_values, eigenvalues, frobenius_norms
 
    single_task_metrics = {task: compute_metrics(load_model(path, base_model_name).to(device))
                           for task, path in single_task_paths.items()}
    multi_task_metrics = compute_metrics(load_model(multi_task_path, base_model_name).to(device))
 
    # Plotting
    metrics = ['Effective Rank', 'Singular Values', 'Eigenvalues', 'Frobenius Norm']
    for i, metric in enumerate(metrics):
        plt.figure(figsize=(12, 6))
        for task, task_metrics in single_task_metrics.items():
            plt.plot(range(len(task_metrics[i])), task_metrics[i], '-o', label=f'{task} (Single-task)', alpha=0.7)
        plt.plot(range(len(multi_task_metrics[i])), multi_task_metrics[i], '-o', label='Multi-task', linewidth=2, alpha=0.7)
        plt.xlabel('LoRA Layer Index')
        plt.ylabel(metric)
        plt.title(f'{metric} Comparison: Single-task vs Multi-task')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric.lower().replace(" ", "_")}_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
 
    # Log average metrics
    for task, task_metrics in single_task_metrics.items():
        logger.info(f"Average metrics for {task}:")
        for i, metric in enumerate(metrics):
            logger.info(f"  {metric}: {np.mean(task_metrics[i]):.4f}")
 
    logger.info("Average metrics for Multi-task:")
    for i, metric in enumerate(metrics):
        logger.info(f"  {metric}: {np.mean(multi_task_metrics[i]):.4f}")
 
def main():
    base_model_name = "t5-base"
    single_task_paths = {
        'CoLA': "YOUR_COLA_CHECKPOINT",
        'MNLI': "YOUR_MNLI_CHECKPOINT",
    }
    multi_task_path = "YOUR_MULTITASK_CHECKPOINT"
 
    # Load models
    single_task_model = load_model(list(single_task_paths.values())[0], base_model_name)
    multi_task_model = load_model(multi_task_path, base_model_name)
 
    # Analyze importance
    importance_results, single_A, single_B, multi_A, multi_B = analyze_importance(single_task_model, multi_task_model)
    logger.info("Importance Analysis Results:")
    logger.info(importance_results)
 
 
    # Sparsity analysis
    sparsity_results = sparsity_analysis(single_A, single_B, multi_A, multi_B)
    logger.info("Sparsity Analysis Results:")
    logger.info(sparsity_results)
 
    # Singular value analysis
    sv_results = singular_value_analysis(single_A, single_B, multi_A, multi_B)
    logger.info("Singular Value Analysis Results:")
    logger.info(sv_results)
 
 
    # Plot LoRA metrics comparison
    plot_lora_metrics_comparison(single_task_paths, multi_task_path, base_model_name)
 
if __name__ == "__main__":
    main()