import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

def main():
    json_path = config.PROJECT_ROOT / 'docs' / 'smoothing_grid_search_results.json'
    output_dir = config.FIGURES_DIR / 'smoothing'
    output_dir.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        print(f"Error: Could not find {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    gammas = data['gammas']
    mse_raw = data['mse_raw']
    corr_raw = data['corr_raw']
    mse_pr = data['mse_pr']
    variance_ret = data['variance_ret']

    # 1. Plot MSE and Correlation (Raw)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Smoothing Level ($\\gamma$)')
    ax1.set_ylabel('Held-out MSE', color=color)
    ax1.plot(gammas, mse_raw, color=color, linewidth=2, marker='o', label='MSE (Held-out)')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Pearson Correlation', color=color)
    ax2.plot(gammas, corr_raw, color=color, linewidth=2, marker='s', label='Correlation')
    ax2.tick_params(axis='y', labelcolor=color)

    min_mse_idx = mse_raw.index(min(mse_raw))
    max_corr_idx = corr_raw.index(max(corr_raw))
    ax1.axvline(x=gammas[min_mse_idx], color='r', linestyle='--', alpha=0.5, label=f'Min MSE ($\\gamma$={gammas[min_mse_idx]})')
    ax2.axvline(x=gammas[max_corr_idx], color='b', linestyle='--', alpha=0.5, label=f'Max Corr ($\\gamma$={gammas[max_corr_idx]})')

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.9))
    plt.title('Cross-validated Predictive Error vs. Smoothing')
    fig.tight_layout()
    plt.savefig(output_dir / 'cv_error_tradeoff.png')
    plt.close()

    # 2. Downstream PageRank Error vs Variance Retained
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:green'
    ax1.set_xlabel('Smoothing Level ($\\gamma$)')
    ax1.set_ylabel('PageRank MSE', color=color)
    ax1.plot(gammas, mse_pr, color=color, linewidth=2, marker='^', label='Downstream PR MSE')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:purple'
    ax2.set_ylabel('Variance Retained (%)', color=color)
    ax2.plot(gammas, variance_ret, color=color, linewidth=2, marker='d', label='Variance Retained')
    ax2.tick_params(axis='y', labelcolor=color)

    min_pr_idx = mse_pr.index(min(mse_pr))
    ax1.axvline(x=gammas[min_pr_idx], color='g', linestyle='--', alpha=0.5, label=f'Min PR MSE ($\\gamma$={gammas[min_pr_idx]})')

    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    plt.title('Downstream Stability & Variance Retention')
    fig.tight_layout()
    plt.savefig(output_dir / 'downstream_variance.png')
    plt.close()

    # 3. Bias-Variance Combined View
    plt.figure(figsize=(10, 6))

    def norm(arr):
        mn, mx = min(arr), max(arr)
        return [(x - mn)/(mx - mn) for x in arr]

    plt.plot(gammas, norm(mse_raw), color='red', marker='o', linestyle='-', label='Predictive Error (Normalized MSE)')
    plt.plot(gammas, norm(variance_ret), color='purple', marker='d', linestyle='-', label='Signal Variance (Normalized)')

    plt.xlabel('Smoothing Level ($\\gamma$)')
    plt.ylabel('Normalized Scale')
    plt.title('Bias-Variance Tradeoff')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'bias_variance.png')
    plt.close()

    print(f"Successfully generated all justification graphs in: {output_dir}")

if __name__ == '__main__':
    main()
