import json
import subprocess
from pathlib import Path
import shutil
import os
def run_single_frame_benchmark(frame_name, rgb_path, depth_path, model_path, 
                                thread_counts, runs_per_config, output_base):
    frame_output = output_base / frame_name
    frame_output.mkdir(parents=True, exist_ok=True)
    (frame_output / 'data').mkdir(exist_ok=True)
    (frame_output / 'images').mkdir(exist_ok=True)
    print(f"\n{'='*70}")
    print(f"Processing Frame: {frame_name}")
    print(f"Output folder: {frame_output}")
    print(f"{'='*70}\n")
    results = {}
    for num_threads in thread_counts:
        print(f"Testing with {num_threads} thread(s):")
        thread_results = []
        for run in range(1, runs_per_config + 1):
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = str(num_threads)
            cmd = [
                "bin/rgbd_pose_estimation.exe",
                "--rgb", rgb_path,
                "--depth", depth_path,
                "--model", model_path,
                "--output", str(frame_output)
            ]
            print(f"  Run {run}/{runs_per_config}...", end=" ", flush=True)
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=300,
                    encoding='utf-8',
                    errors='replace'
                )
                output = (result.stdout or '') + (result.stderr or '')
                import re
                time_match = re.search(r'TOTAL EXECUTION TIME:\s+([\d.]+)\s*ms', output, re.IGNORECASE)
                iter_match = re.search(r'Total iterations:\s+(\d+)', output, re.IGNORECASE)
                initial_mhd_match = re.search(r'Initial MHD [Ss]core:\s+([\d.]+)', output)
                final_mhd_match = re.search(r'Final MHD [Ss]core:\s+([\d.]+)', output)
                exec_time = float(time_match.group(1)) if time_match else None
                iterations = int(iter_match.group(1)) if iter_match else None
                initial_mhd = float(initial_mhd_match.group(1)) if initial_mhd_match else None
                final_mhd = float(final_mhd_match.group(1)) if final_mhd_match else None
                improvement = initial_mhd - final_mhd if (initial_mhd and final_mhd) else None
                if exec_time and iterations and initial_mhd and final_mhd:
                    print(f"{exec_time:.1f} ms | Iters: {iterations} | MHD: {initial_mhd:.1f}→{final_mhd:.1f} (Δ{improvement:.1f})")
                    thread_results.append({
                        'time_ms': exec_time,
                        'iterations': iterations,
                        'initial_mhd': initial_mhd,
                        'final_mhd': final_mhd,
                        'improvement': improvement
                    })
                    conv_src = frame_output / 'data' / 'convergence_log.csv'
                    if conv_src.exists():
                        conv_dst = frame_output / 'data' / f'convergence_{num_threads}threads_run{run}.csv'
                        shutil.copy(conv_src, conv_dst)
                else:
                    print("FAILED")
            except subprocess.TimeoutExpired:
                print("TIMEOUT")
            except Exception as e:
                print(f"ERROR: {e}")
        if thread_results:
            times = [r['time_ms'] for r in thread_results]
            results[str(num_threads)] = {
                'times': times,
                'avg_time': sum(times) / len(times),
                'std_time': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5,
                'avg_iterations': sum(r['iterations'] for r in thread_results) / len(thread_results),
                'avg_initial_mhd': sum(r['initial_mhd'] for r in thread_results) / len(thread_results),
                'avg_final_mhd': sum(r['final_mhd'] for r in thread_results) / len(thread_results),
                'avg_improvement': sum(r['improvement'] for r in thread_results) / len(thread_results),
                'successful_runs': len(thread_results)
            }
        print()
    results_file = frame_output / 'data' / 'benchmark_results.json'
    with open(results_file, 'w') as f:
        json.dump({frame_name: results}, f, indent=2)
    print(f" Frame {frame_name} results saved to: {results_file}\n")
    return results
def generate_frame_plots(frame_name, frame_output):
    print(f"Generating plots for {frame_name}...")
    plot_script = f
    plot_script_path = frame_output / 'generate_plots.py'
    with open(plot_script_path, 'w', encoding='utf-8') as f:
        f.write(plot_script)
    import sys
    python_exe = sys.executable
    subprocess.run([python_exe, str(plot_script_path)], check=True)
def main():
    thread_counts = [1, 2, 4, 8]
    runs_per_config = 5
    data_dir = Path("data/berkeley_diverse")
    output_base = Path("output/berkeley_frames")
    summary_path = data_dir / "dataset_summary.json"
    if not summary_path.exists():
        print(f"Error: Dataset summary not found at {summary_path}")
        print("Please run: python scripts/convert_berkeley_data.py --input data/004_sugar_box_berkeley_rgbd/004_sugar_box --output data/berkeley_diverse")
        return
    with open(summary_path, 'r') as f:
        dataset_info = json.load(f)
    print("="*70)
    print("PER-FRAME ANALYSIS - BERKELEY REAL DATASET")
    print("="*70)
    print(f"Dataset: {dataset_info['dataset']}")
    print(f"Frames to process: {dataset_info['num_frames']}")
    print(f"Thread counts: {thread_counts}")
    print(f"Runs per configuration: {runs_per_config}")
    print(f"Output base directory: {output_base}")
    print("="*70)
    for frame_info in dataset_info['frames']:
        frame_name = frame_info['frame_name']
        rgb_path = str(data_dir / frame_info['rgb_image'])
        depth_path = str(data_dir / frame_info['depth_map'])
        model_path = str(data_dir / frame_info['wireframe'])
        results = run_single_frame_benchmark(
            frame_name, rgb_path, depth_path, model_path,
            thread_counts, runs_per_config, output_base
        )
        frame_output = output_base / frame_name
        generate_frame_plots(frame_name, frame_output)
    print("\n" + "="*70)
    print(" ALL FRAMES PROCESSED SUCCESSFULLY!")
    print("="*70)
    print(f"\nResults saved in: {output_base}/")
    print("\nGenerated folders:")
    for frame_info in dataset_info['frames']:
        frame_name = frame_info['frame_name']
        print(f"  {frame_name}/")
        print(f"    ├── data/")
        print(f"    │   ├── benchmark_results.json")
        print(f"    │   ├── convergence_*threads_run*.csv")
        print(f"    │   └── performance_metrics.json")
        print(f"    └── images/")
        print(f"        ├── performance_analysis.png")
        print(f"        └── convergence_comparison.png")
    print("="*70)
if __name__ == '__main__':
    main()
