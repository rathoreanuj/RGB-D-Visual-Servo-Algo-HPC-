
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import cv2

frame_output = Path(r"output\berkeley_frames\NP1_270")
results_file = frame_output / 'data' / 'benchmark_results.json'

with open(results_file, 'r') as f:
    data = json.load(f)

frame_data = data['NP1_270']
threads = [1, 2, 4, 8]

# 1. Histogram Equalization Plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Load original grayscale image
img_orig = cv2.imread(str(frame_output / 'images' / 'original_grayscale.png'), cv2.IMREAD_GRAYSCALE)
img_eq = cv2.imread(str(frame_output / 'images' / 'equalized_grayscale.png'), cv2.IMREAD_GRAYSCALE)

if img_orig is not None and img_eq is not None:
    # (a) Original grayscale
    axes[0, 0].imshow(img_orig, cmap='gray')
    axes[0, 0].set_title('(a) Original Grayscale Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # (b) Original histogram
    hist_orig, bins = np.histogram(img_orig.flatten(), bins=256, range=[0, 256])
    prob_orig = hist_orig / hist_orig.sum()
    gray_values = np.arange(256)
    
    axes[0, 1].bar(gray_values, prob_orig, width=1.0, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('(b) Original Histogram', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Gray Value', fontsize=12)
    axes[0, 1].set_ylabel('Probability p(r)', fontsize=12)
    axes[0, 1].set_xlim(0, 255)
    axes[0, 1].grid(True, alpha=0.3)
    
    # (c) Equalized grayscale
    axes[1, 0].imshow(img_eq, cmap='gray')
    axes[1, 0].set_title('(c) Equalized Grayscale Image', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # (d) Equalized histogram
    hist_eq, _ = np.histogram(img_eq.flatten(), bins=256, range=[0, 256])
    prob_eq = hist_eq / hist_eq.sum()
    
    axes[1, 1].bar(gray_values, prob_eq, width=1.0, color='green', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('(d) Equalized Histogram', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Gray Value', fontsize=12)
    axes[1, 1].set_ylabel('Probability p(s)', fontsize=12)
    axes[1, 1].set_xlim(0, 255)
    axes[1, 1].grid(True, alpha=0.3)

fig.suptitle(f'Histogram Equalization - Frame: NP1_270', fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(frame_output / 'images' / 'histogram_equalization.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Performance Comparison Plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Execution Time
avg_times = [frame_data[str(t)]['avg_time'] for t in threads]
std_times = [frame_data[str(t)]['std_time'] for t in threads]

bars = axes[0, 0].bar(range(len(threads)), avg_times, yerr=std_times, 
               color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'], 
               edgecolor='black', linewidth=1.5, alpha=0.8)
axes[0, 0].set_xticks(range(len(threads)))
axes[0, 0].set_xticklabels([f'{t} thread' + ('s' if t>1 else '') for t in threads])
axes[0, 0].set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
axes[0, 0].set_title('(a) Execution Time vs Thread Count', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, avg_times)):
    height = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f} ms', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Speedup
baseline = avg_times[0]
speedups = [baseline / t for t in avg_times]
ideal_speedup = threads

axes[0, 1].plot(threads, speedups, 'o-', linewidth=3, markersize=10, 
                color='#2ecc71', label='Actual Speedup')
axes[0, 1].plot(threads, ideal_speedup, '--', linewidth=2, 
                color='#e74c3c', label='Ideal Speedup', alpha=0.7)

# Add value labels on points
for i, (t, s) in enumerate(zip(threads, speedups)):
    axes[0, 1].text(t, s, f'{s:.2f}x', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')

axes[0, 1].set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Speedup', fontsize=12, fontweight='bold')
axes[0, 1].set_title('(b) Parallel Speedup', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(True, alpha=0.3)

# Efficiency
efficiency = [s/t*100 for s, t in zip(speedups, threads)]

bars = axes[1, 0].bar(range(len(threads)), efficiency, 
               color=['#2ecc71' if e>=80 else '#f39c12' if e>=60 else '#e74c3c' for e in efficiency],
               edgecolor='black', linewidth=1.5, alpha=0.8)
axes[1, 0].axhline(y=100, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Ideal (100%)')
axes[1, 0].set_xticks(range(len(threads)))
axes[1, 0].set_xticklabels([f'{t} thread' + ('s' if t>1 else '') for t in threads])
axes[1, 0].set_ylabel('Parallel Efficiency (%)', fontsize=12, fontweight='bold')
axes[1, 0].set_title('(c) Parallel Efficiency', fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, eff) in enumerate(zip(bars, efficiency)):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{eff:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Convergence Quality
initial_mhd = frame_data['1']['avg_initial_mhd']
final_mhds = [frame_data[str(t)]['avg_final_mhd'] for t in threads]

axes[1, 1].axhline(y=initial_mhd, color='red', linestyle='--', linewidth=2, label='Initial MHD', alpha=0.7)
axes[1, 1].plot(threads, final_mhds, 'o-', linewidth=3, markersize=10, 
                color='#2ecc71', label='Final MHD')

# Add value labels
for i, (t, mhd) in enumerate(zip(threads, final_mhds)):
    axes[1, 1].text(t, mhd, f'{mhd:.1f}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')

axes[1, 1].set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('MHD Score', fontsize=12, fontweight='bold')
axes[1, 1].set_title('(d) Convergence Quality (MHD Score)', fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=11)
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle(f'Performance Analysis - Frame: NP1_270', fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(frame_output / 'images' / 'performance_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Convergence Comparison Plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for idx, t in enumerate(threads):
    ax = axes[idx // 2, idx % 2]
    
    # Load convergence logs for this thread count
    conv_files = sorted((frame_output / 'data').glob(f'convergence_{t}threads_*.csv'))
    
    if len(conv_files) > 0:
        for i, conv_file in enumerate(conv_files, 1):
            df = pd.read_csv(conv_file)
            if len(df) > 0:
                # Parse iteration and MHD from the CSV
                if 'iteration' in df.columns and 'mhd_score' in df.columns:
                    ax.plot(df['iteration'], df['mhd_score'], linewidth=2, alpha=0.7, label=f'Run {i}')
                elif 'Iteration' in df.columns and 'MHD' in df.columns:
                    ax.plot(df['Iteration'], df['MHD'], linewidth=2, alpha=0.7, label=f'Run {i}')
                elif 'iteration' in df.columns and 'mhd' in df.columns:
                    ax.plot(df['iteration'], df['mhd'], linewidth=2, alpha=0.7, label=f'Run {i}')
                # Handle the case where CSV might have different format
                elif len(df.columns) >= 2:
                    # Assume first col is iteration, second is MHD
                    ax.plot(df.iloc[:, 0], df.iloc[:, 1], linewidth=2, alpha=0.7, label=f'Run {i}')
    
    ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax.set_ylabel('MHD Score', fontsize=11, fontweight='bold')
    ax.set_title(f'{t} Thread' + ('s' if t>1 else ''), fontsize=13, fontweight='bold')
    if len(conv_files) > 0:
        ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

fig.suptitle(f'Convergence Trajectories - Frame: NP1_270', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig(frame_output / 'images' / 'convergence_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print(f'Generated plots for NP1_270')
