import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

output_dir = config.FIGURES_DIR / 'project'
output_dir.mkdir(parents=True, exist_ok=True)

# Use matplotlib style for better aesthetics
plt.style.use('ggplot')

# 1. Bar Chart: Strategy MRE Comparison
strategies = ['Original Baseline\n(mu=0, beta=0.2)', 'Top-K Boost\n(K=100, boost=2.0)', 'Travel-Time Self-Loops\n(mu=20, beta=0.9)']
top_100_mre = [0.7331, 0.3304, 0.0317]
overall_mre = [0.1980, 0.1942, 0.0471]

x = np.arange(len(strategies))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, top_100_mre, width, label='Top-100 MRE', color='#e63946')
rects2 = ax.bar(x + width/2, overall_mre, width, label='Overall MRE', color='#457b9d')

ax.set_ylabel('Mean Relative Error (MRE)', fontsize=12)
ax.set_title('Improvement in Prediction Accuracy Across Strategies', fontsize=16, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(strategies, fontsize=11)
ax.legend(fontsize=12)

# Add text labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

autolabel(rects1)
autolabel(rects2)
plt.tight_layout()
plt.savefig(output_dir / '1_mre_comparison.png', dpi=300, bbox_inches='tight')
plt.close()


# 2. Self-Loop Probability vs. Travel Time
travel_times = np.linspace(1, 60, 100) # seconds
successors = 3
mu_values = [1, 5, 10, 20]

plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(mu_values)))

for idx, mu in enumerate(mu_values):
    probabilities = (mu * travel_times) / (mu * travel_times + successors) * 100
    plt.plot(travel_times, probabilities, label=r'$\mu$' + f' = {mu}', linewidth=3, color=colors[idx])

plt.axvline(5, color='#e76f51', linestyle='--', alpha=0.8, linewidth=2)
plt.text(6, 40, 'Intersection\nConnector (~5s)', color='#e76f51', fontsize=11, fontweight='bold')

plt.axvline(20, color='#2a9d8f', linestyle='--', alpha=0.8, linewidth=2)
plt.text(21, 40, 'Highway\nSegment (~20s)', color='#2a9d8f', fontsize=11, fontweight='bold')

plt.xlabel('Physical Travel Time on Link (seconds)', fontsize=12)
plt.ylabel('Self-Loop Retention Probability (%)', fontsize=12)
plt.title(r'Effect of $\mu$ on Mass Retention (Assuming 3 successors)', fontsize=16, pad=15)
plt.legend(title=r'Dwell-time scale ($\mu$)', fontsize=11, title_fontsize=12, loc='lower right')
plt.tight_layout()
plt.savefig(output_dir / '2_self_loop_probability.png', dpi=300, bbox_inches='tight')
plt.close()


# 3. 2N x 2N Matrix Block Visualization
plt.figure(figsize=(8, 8))
# Create a conceptual matrix
matrix = np.zeros((100, 100))
# Top-Left: (1-beta) P
matrix[:50, :50] = 0.2
# Top-Right: beta I
for i in range(50):
    matrix[i, 50+i] = 0.8
# Bottom-Left: 0
matrix[50:, :50] = 0.0
# Bottom-Right: P
matrix[50:, 50:] = 0.5

ax = plt.gca()
im = ax.imshow(matrix, cmap='Blues')
ax.set_xticks([])
ax.set_yticks([])

plt.axhline(49.5, color='#1d3557', linewidth=3)
plt.axvline(49.5, color='#1d3557', linewidth=3)

plt.text(25, 25, r'$(1 - \beta) \cdot P$' + '\n\nUp Phase (Active Driving)', ha='center', va='center', fontsize=14, color='black')
plt.text(75, 25, r'$\beta \cdot I$' + '\n\nTransition to Down Phase', ha='center', va='center', fontsize=14, color='white')
plt.text(25, 75, r'$0$' + '\n\n(Cannot return Up)', ha='center', va='center', fontsize=14, color='black')
plt.text(75, 75, r'$P$' + '\n\nDown Phase (Local Dispersal)', ha='center', va='center', fontsize=14, color='white')

plt.title(r'Two-Phase Markov Chain Transition Matrix ($M_{2N}$)', fontsize=18, pad=20)
plt.tight_layout()
plt.savefig(output_dir / '3_block_matrix_visualization.png', dpi=300, bbox_inches='tight')
plt.close()


# 4. Impact of Mu Grid Search
mu_grid = [0, 1, 2, 3, 5, 8, 10, 12, 15, 20]
top100_grid = [0.5320, 0.2400, 0.1654, 0.1285, 0.0908, 0.0646, 0.0546, 0.0475, 0.0399, 0.0317]
overall_grid = [0.1390, 0.2081, 0.1731, 0.1478, 0.1156, 0.0883, 0.0767, 0.0679, 0.0582, 0.0471]

fig, ax1 = plt.subplots(figsize=(10, 6))

color = '#e63946'
ax1.set_xlabel(r'Self-Loop Parameter ($\mu$)', fontsize=12)
ax1.set_ylabel('Top-100 MRE', color=color, fontsize=12, fontweight='bold')
line1 = ax1.plot(mu_grid, top100_grid, marker='o', markersize=8, color=color, linewidth=3, label='Top-100 MRE')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = '#457b9d'
ax2.set_ylabel('Overall MRE', color=color, fontsize=12, fontweight='bold')  
line2 = ax2.plot(mu_grid, overall_grid, marker='s', markersize=8, color=color, linewidth=3, linestyle='--', label='Overall MRE')
ax2.tick_params(axis='y', labelcolor=color)
ax2.grid(False) # avoid overlapping grids

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right', fontsize=11)

plt.title(r'Grid Search: Effect of $\mu$ on Error Rates' + '\n(at damping=0.80, beta=0.9)', fontsize=16, pad=15)
fig.tight_layout()  
plt.savefig(output_dir / '4_mu_grid_search.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Visualizations successfully generated in '{output_dir}'.")