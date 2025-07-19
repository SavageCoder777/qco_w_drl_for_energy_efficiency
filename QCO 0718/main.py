import numpy as np
import pandas as pd
import pickle
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns 
import logging
import sys
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from quantum_rl_envs import *

# ---------------------
# SAVE RUNTIME DATA
# ---------------------

def get_true_env(env):
    while hasattr(env, 'env'):
        env = env.env
    return env

# Used to log and save the runtime data in seperate files for further reference
class SaveMetricsAndRolloutsCallback(BaseCallback):
    def __init__(self, save_freq=32, save_dir="outputs", verbose=1):
    # Checked
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.full_metrics_file = os.path.join(save_dir, "full_metrics_log.csv")

        self.rollout_dir = os.path.join(save_dir, "rollouts")
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.rollout_dir, exist_ok=True)

        # Create or overwrite the CSV file and write header once
        with open(self.full_metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'depth', 'gate_count', 'energy', 'qubit_count', 'r'])

    def _on_step(self) -> bool:
        step = self.num_timesteps
        if step % self.save_freq == 0 and step > 0:
            batch_env_instance = self.training_env.get_attr('env', indices=0)[0]
            quantum_prune_env_instance = batch_env_instance.env
            evals = evaluate_circuit(quantum_prune_env_instance, quantum_prune_env_instance.circuit)

            # Append metrics to CSV file
            with open(self.full_metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    step,
                    evals['depth'],
                    evals['gate_count'],
                    evals['energy'],
                    evals['qubit_count'],
                    evals.get('r', 'N/A'),
                ])

            if self.verbose > 0:
                print(f"[SaveMetrics] Step {step} - Depth: {evals['depth']}, Gates: {evals['gate_count']}, Energy: {evals['energy']:.2f}, r: {evals.get('r', 'N/A')}")

            rollout_path = os.path.join(self.rollout_dir, f"rollout_step_{step}.pkl")
            with open(rollout_path, 'wb') as f:
                pickle.dump({
                    'circuit_metrics': evals, 
                    'circuit': quantum_prune_env_instance.circuit,
                    'original_circuit': quantum_prune_env_instance.original_circuit,
                    'observation': quantum_prune_env_instance._encode_circuit(),
                    'last_reward': getattr(quantum_prune_env_instance, 'last_reward', None),
                }, f)
            if self.verbose > 0:
                print(f"[SaveMetrics] Saved rollout to {rollout_path}")

        return True

# ------------------------------
# 6. EVALUATION + VISUALIZATION
# ------------------------------

# Applies the model to each of the test circuits and works to optimize them
def evaluate_on_dataset_with_model(circuits, model):
# Checked
    results = []
    i = 0
    for c in circuits:
        logging.info(f"[EVAL] Optimizing test circuit {i + 1}/{len(circuits)}...")
        env = QuantumPruneEnv(c)
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    
        before = evaluate_circuit(env, env.original_circuit)
        after = evaluate_circuit(env, env.circuit)

        results.append((before, after))
        logging.info(f"[EVAL] Circuit {i + 1} done: depth {before['depth']} → {after['depth']}")
        i += 1
    return results

# Many different types of plots/graphs showing the results of the training and optimization on
# the test circuits
def plot_depth_comparison(results):
    before_energies = [r[0]['energy'] for r in results]
    after_energies = [r[1]['energy'] for r in results]
    plt.figure(figsize=(10, 5))
    plt.plot(before_energies, label='Before Pruning')
    plt.plot(after_energies, label='After Pruning')
    plt.xlabel("Circuit Index")
    plt.ylabel("Circuit Energy")
    plt.title("Circuit Energy Before and After RL Optimization")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("energy_comparison.png")
    plt.close()

def plot_rl_training_scatter(depths, gates, energies, qubit_counts):
    x = np.arange(0, len(depths)) * 10  # assuming logging every 10 epochs

    def plot_scatter(y, ylabel, filename):
        plt.figure(figsize=(8, 4))
        plt.scatter(x, y, s=8, alpha=0.6, c='brown', edgecolor='yellow')
        avg_y = np.mean(y)
        err_y = np.std(y) / np.sqrt(len(y))
        plt.axhline(avg_y, color='blue', label=f'⟨{ylabel}⟩ ≈ {avg_y:.2f} ± {err_y:.2f}')
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} During RL Training")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    plot_scatter(depths, "d", "train_depth_scatter.png")
    plot_scatter(gates, "n", "train_gate_scatter.png")
    plot_scatter(energies, "Energy", "train_energy_scatter.png")
    plot_scatter(qubit_counts, "qubit count", "train_qubit_count_scatter.png")

def plot_test_examples_5(test_logs, rounds, ylabel, filename):
    plt.figure(figsize=(8, 4))
    for y in test_logs:
        plt.plot(rounds, y, color='orange', linewidth=1.5, alpha=0.7)
    avg_curve = np.mean(test_logs, axis=0)
    plt.plot(rounds, avg_curve, color='blue', label=f"avg ≈ {avg_curve[-1]:.2f} ± {np.std(avg_curve):.2f}", linewidth=2.5)
    plt.xlabel("Transformation Round")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} During Testing (End of Training)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_percent_change_scatter(before_list, after_list, metric1, metric2, filename):
    x_change = []
    y_change = []

    for before, after in zip(before_list, after_list):
        x = (after[metric1] - before[metric1]) / before[metric1] * 100
        y = (after[metric2] - before[metric2]) / before[metric2] * 100
        x_change.append(x)
        y_change.append(y)

    plt.figure(figsize=(6, 6))
    plt.scatter(x_change, y_change, s=8, alpha=0.6, c='brown', edgecolor='yellow')
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel(f"Change in {metric1} (%)")
    plt.ylabel(f"Change in {metric2} (%)")
    plt.title(f"% Change: {metric1} vs {metric2}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def get_training_stats(model):
    depth_vals = []
    gate_vals = []
    energy_vals = []
    qubit_count_vals = []

    circuit_paths = get_circuit_paths("train_set/", limit=MAX_TRAIN_CIRCUITS)
    env = BatchEnv(circuit_paths)
    obs, _ = env.reset()

    for step in range(MAX_TRAIN_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)

        if step % 10 == 0:
            evals = evals = evaluate_circuit(env.env, env.env.circuit)
            depth_vals.append((step, evals['depth']))
            gate_vals.append((step, evals['gate_count']))
            energy_vals.append((step, evals['energy']))
            qubit_count_vals.append((step, evals['qubit_count']))

    return depth_vals, gate_vals, energy_vals, qubit_count_vals

def plot_training_stats(stats, title, ylabel, filename):
    steps, values = zip(*stats)
    avg = np.mean(values)
    std = np.std(values) / np.sqrt(len(values))

    plt.figure(figsize=(6, 4))
    plt.scatter(steps, values, s=1, alpha=0.6, c='brown')
    plt.plot(steps, [np.mean(values[:i+1]) for i in range(len(values))], label='moving average', color='blue')
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.annotate(f"⟨{ylabel}⟩ ≈ {avg:.2f} ± {std:.2f}", xy=(steps[-1], avg), xytext=(steps[-1] * 0.6, avg + 10), color='blue')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_in_game_progress(circuits, model, metric='depth', filename=''):
    curves = []
    for i, circ in enumerate(circuits[:5]):
        env = QuantumPruneEnv(circ)
        obs, _ = env.reset()
        values = []
        done = False
        while not done:
            val = evaluate_circuit(env, env.circuit)[metric]
            values.append(val)
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
        curves.append(values)

    max_len = max(len(c) for c in curves)
    avg_curve = np.mean([np.pad(c, (0, max_len - len(c)), constant_values=c[-1]) for c in curves], axis=0)

    plt.figure(figsize=(6, 4))
    for c in curves:
        plt.plot(c, color='orange', alpha=0.8)
    plt.plot(avg_curve, label='average', color='blue')
    plt.xlabel("transformation round")
    plt.ylabel(metric)
    plt.title(f"In-Game Progress of 5 Circuits ({metric})")
    avg_val = np.mean([c[-1] for c in curves])
    std_val = np.std([c[-1] for c in curves]) / np.sqrt(5)
    plt.annotate(f"⟨{metric}⟩ = {avg_val:.2f} ± {std_val:.2f}", xy=(len(avg_curve), avg_val), xytext=(0.6*len(avg_curve), avg_val + 5), color='blue')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def percent_delta(before, after, key):
    return 100 * (after[key] - before[key]) / before[key]

def plot_percent_scatter(results, key_x, key_y, label, filename):
    x = [percent_delta(b, a, key_x) for b, a in results]
    y = [percent_delta(b, a, key_y) for b, a in results]

    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, alpha=0.6, c='yellow', edgecolor='brown', s=8)
    plt.axvline(0, color='gray')
    plt.axhline(0, color='gray')
    plt.xlabel(f"Δ {key_x} (%)")
    plt.ylabel(f"Δ {key_y} (%)")
    plt.title(label)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_training_depth_with_scatter(depth_vals):
    steps, depths = zip(*depth_vals)
    steps = np.array(steps)
    depths = np.array(depths)

    # Compute moving average
    window_size = 5  # or 10 depending on how smooth you want it
    moving_avg = np.convolve(depths, np.ones(window_size)/window_size, mode='valid')

    # Adjust x values to align with convolution result
    avg_steps = steps[window_size-1:]

    plt.figure(figsize=(8, 4))
    plt.plot(avg_steps, moving_avg, label="Moving Avg", color='blue')
    plt.scatter(steps, depths, color='brown', edgecolor='yellow', alpha=0.5, s=10, label="Per-circuit depth")
    plt.xlabel("Epoch")
    plt.ylabel("Depth")
    plt.title("Circuit Depth During RL Training")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("train_depth_with_scatter.png")
    plt.close()

def plot_metric_from_log(file_path, metric, color='brown', save_as=None):
    """
    Plot a metric (depth, gate_count, or energy) from a CSV log file.

    Args:
        file_path (str): Path to the CSV log file.
        metric (str): Metric column name (e.g., 'depth', 'gate_count', 'energy', 'r').
        color (str): Color for scatter points.
        save_as (str): Filename to save the figure. Defaults to f"{metric}_log_scatter.png".
    """
    df = pd.read_csv(file_path)

    if metric not in df.columns:
        print(f"[!] Metric '{metric}' not found in log file columns: {df.columns}")
        return

    # Convert step and metric columns to numeric, coerce errors
    df['step'] = pd.to_numeric(df['step'], errors='coerce')
    df[metric] = pd.to_numeric(df[metric], errors='coerce')

    # Drop rows with NaNs in step or metric
    df = df.dropna(subset=['step', metric])

    if len(df) < 2:
        print(f"[!] Not enough data points to plot for metric '{metric}'.")
        return

    x = df['step'].values
    y = df[metric].values

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label=f"{metric} (line)", color='blue', linewidth=1.5)
    plt.scatter(x, y, s=20, c=color, edgecolor='yellow', alpha=0.7, label=f"{metric} (scatter)")
    plt.xlabel("Training Step")
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f"{metric.replace('_', ' ').title()} During Training")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_as is None:
        save_as = f"{metric}_log_scatter.png"
    plt.savefig(save_as)
    plt.close()

    print(f"[✓] Saved {metric} training plot to {save_as}")

def plot_r_values(r_values, filename="train_r_scatter.png"):
    steps = np.arange(len(r_values))
    r_values = np.array(r_values)

    window_size = 5
    moving_avg = np.convolve(r_values, np.ones(window_size)/window_size, mode='valid')
    avg_steps = steps[window_size-1:]

    plt.figure(figsize=(8, 4))
    plt.plot(avg_steps, moving_avg, label="Moving Avg of r", color='blue')
    plt.scatter(steps, r_values, color='brown', edgecolor='yellow', alpha=0.5, s=10, label="Per-step r")
    plt.xlabel("Step")
    plt.ylabel("Repetition Count (r)")
    plt.title("Repetition Count (r) During Training")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[✓] Saved r scatter plot to {filename}")

def plot_training_percent_change_from_log(file_path, metric_x, metric_y, save_path, LOG_FREQ=1024, num_circuits=5000):
    df = pd.read_csv(file_path)

    # Identify which circuit each log entry belongs to
    df['circuit_id'] = (df['step'] // LOG_FREQ) % num_circuits

    # Get first and last entry per circuit
    first = df.groupby('circuit_id').first()
    last = df.groupby('circuit_id').last()

    # Calculate percent change
    dx = 100 * (last[metric_x] - first[metric_x]) / first[metric_x]
    dy = 100 * (last[metric_y] - first[metric_y]) / first[metric_y]

    plt.figure(figsize=(6, 6))
    plt.scatter(dx, dy, alpha=0.6, c='brown', edgecolor='yellow', s=10)
    plt.axvline(0, color='gray')
    plt.axhline(0, color='gray')
    plt.xlabel(f"Δ {metric_x} (%)")
    plt.ylabel(f"Δ {metric_y} (%)")
    plt.title(f"{metric_x} vs {metric_y} Change in Training")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ----------------
# 7. SETUP LOGGER
# ----------------

# Logging configuration to 'print' to both terminal and logfile.log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logfile.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# ---------------
# 8. MAIN SCRIPT
# ---------------
sns.set(style="whitegrid")

# Part of the callback to print variable values every X number of times, X=100
class PrintStepsCallback(BaseCallback):
    def __init__(self, log_every=100, verbose=1):
        super().__init__(verbose)
        self.log_every = log_every

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_every == 0:
            env = self.training_env.get_attr('env', indices=0)[0]
            true_env = get_true_env(env)

            circuit = true_env.circuit
            reward = true_env.last_reward if hasattr(true_env, 'last_reward') else None
            evals = evaluate_circuit(true_env, circuit)

            reward_str = f"{reward:.4f}" if reward is not None else "N/A"
            logging.info(
                f"[Step {self.num_timesteps}] "
                f"Reward: {reward_str} | "
                f"Depth: {evals['depth']} | "
                f"Gate Count: {evals['gate_count']} | "
                f"Energy: {evals['energy']:.2f} | "
                f"r: {getattr(true_env, 'r', 'N/A')}"
            )
        return True

# Part of the callback to save the model at checkpoints at a frequency of save_freq
class TrueTimestepCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, name_prefix="model", verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            save_path = os.path.join(
                self.save_path,
                f"{self.name_prefix}_{self.num_timesteps}_steps.zip"
            )
            self.model.save(save_path)
            if self.verbose:
                logging.info(f"[✓] Model checkpoint saved at: {save_path}")
        return True

# The main running calls to run the above defined functions, print out steps along the way
# and successfully train a model and optimize a circuit for energy efficiency.
if __name__ == '__main__':
    logging.info("[1/4] Generating training circuits...")
    #generate_dataset(MAX_TRAIN_CIRCUITS, path='train_set/')

    logging.info("[2/4] Loading training circuits...")
    circuit_paths = get_circuit_paths("train_set/", limit=MAX_TRAIN_CIRCUITS)

    logging.info("[3/4] Training PPO model")

    vec_env = make_vec_env(
        lambda: BatchEnv(circuit_paths),
        n_envs=16,
        vec_env_cls=SubprocVecEnv
    )

    model = PPO("MlpPolicy", vec_env, verbose=1, n_steps=256)
    print(f"Using vectorized environment type: {type(vec_env)}") # Add this for verification

    checkpoint_callback = TrueTimestepCheckpointCallback(
    save_freq=1024,
    save_path="outputs/checkpoints/",
    name_prefix="ppo_quantum_prune")

    callback = CallbackList([
    SaveMetricsAndRolloutsCallback(save_freq=1024, save_dir="outputs"),
    PrintStepsCallback(),
    checkpoint_callback])
    try:
        model.learn(total_timesteps=MAX_TRAIN_STEPS, callback=callback)
    except KeyboardInterrupt:
        print("\n[!] Interrupted — saving model before exit...")
        os.makedirs("outputs", exist_ok=True)
        model.save("outputs/ppo_quantum_prune_model_interrupted")
        print("✅ Model saved to outputs/ppo_quantum_prune_model_interrupted.zip")

    os.makedirs("outputs", exist_ok=True)
    model.save("outputs/ppo_quantum_prune_model")
    logging.info("✅ PPO model saved to outputs/ppo_quantum_prune_model.zip")

    # PLOTTING

    depths, gates, energies, qubit_counts = get_training_stats(model)
    plot_training_stats(depths, "RL Training Progress (Depth)", "d", "train_depth.png")
    logging.info("✅ Training graph 1/11")
    plot_training_stats(gates, "RL Training Progress (Gate Count)", "n", "train_gate.png")
    logging.info("✅ Training graph 2/11")
    plot_training_stats(energies, "RL Training Progress (Energy)", "E", "train_energy.png")
    logging.info("✅ Training graph 3/11")
    plot_training_stats(qubit_counts, "RL Training Progress (Qubit Count)", "qubit count", "train_qubit_count.png")
    logging.info("✅ Training graph 4/11")
    plot_metric_from_log("outputs/full_metrics_log.csv", "depth")
    logging.info("✅ Training graph 5/11")
    plot_metric_from_log("outputs/full_metrics_log.csv", "gate_count")
    logging.info("✅ Training graph 6/11")
    plot_metric_from_log("outputs/full_metrics_log.csv", "energy")
    logging.info("✅ Training graph 7/11")
    plot_metric_from_log("outputs/full_metrics_log.csv", "r", color="purple", save_as="train_r_from_log.png")
    logging.info("✅ Training graph 8/11")

    plot_training_percent_change_from_log(
        "outputs/full_metrics_log.csv",
        metric_x='depth',
        metric_y='energy',
        save_path='train_percent_change_energy_vs_depth.png',
        LOG_FREQ=LOG_FREQ,
        num_circuits=MAX_TRAIN_CIRCUITS
    )
    logging.info("✅ Training graph 9/11")
    plot_training_percent_change_from_log(
        "outputs/full_metrics_log.csv",
        metric_x='depth',
        metric_y='gate_count',
        save_path='train_percent_change_gate_vs_depth.png',
        LOG_FREQ=LOG_FREQ,
        num_circuits=MAX_TRAIN_CIRCUITS
    )
    logging.info("✅ Training graph 10/11")
    plot_training_percent_change_from_log(
        "outputs/full_metrics_log.csv",
        metric_x='gate_count',
        metric_y='energy',
        save_path='train_percent_change_energy_vs_gate.png',
        LOG_FREQ=LOG_FREQ,
        num_circuits=MAX_TRAIN_CIRCUITS
    )
    logging.info("✅ Training graph 11/11")

    logging.info("[4/4] Generating and evaluating test circuits...")
    generate_dataset(MAX_TEST_CIRCUITS, path='test_set/')
    eval_circuits = load_dataset('test_set/', limit=MAX_TEST_CIRCUITS)
    results = evaluate_on_dataset_with_model(eval_circuits, model)

    plot_in_game_progress(eval_circuits, model, metric='depth', filename='test_depth.png')
    logging.info("✅ Testing graph 1/4")
    plot_in_game_progress(eval_circuits, model, metric='gate_count', filename='test_gate.png')
    logging.info("✅ Testing graph 2/4")
    plot_in_game_progress(eval_circuits, model, metric='energy', filename='test_energy.png')
    logging.info("✅ Testing graph 3/4")
    plot_in_game_progress(eval_circuits, model, metric='qubit_count', filename='test_qubit_count.png')
    logging.info("✅ Testing graph 4/4")

    before_avg_depth = np.mean([r[0]['depth'] for r in results])
    after_avg_depth = np.mean([r[1]['depth'] for r in results])
    logging.info(f"✔ Evaluation complete. Avg depth before: {before_avg_depth:.2f}, after: {after_avg_depth:.2f}")

    before_avg_qubit_count = np.mean([r[0]['qubit_count'] for r in results])
    after_avg_qubit_count = np.mean([r[1]['qubit_count'] for r in results])
    logging.info(f"✔ Evaluation complete. Avg qubit_count before: {before_avg_qubit_count:.2f}, after: {after_avg_qubit_count:.2f}")    

    before_avg_energy = np.mean([r[0]['energy'] for r in results])
    after_avg_energy = np.mean([r[1]['energy'] for r in results])
    logging.info(f"✔ Evaluation complete. Avg energy before: {before_avg_energy:.2f}, after: {after_avg_energy:.2f}")

    logging.info(f"Depth decreased by {(before_avg_depth - after_avg_depth) / before_avg_depth * 100:.2f}%")
    logging.info(f"Qubit count decreased by {(before_avg_qubit_count - after_avg_qubit_count) / before_avg_qubit_count * 100:.2f}%")
    logging.info(f"Energy decreased by {(before_avg_energy - after_avg_energy) / before_avg_energy * 100:.2f}%")
    logging.info("📊 Saving depth comparison plot to depth_comparison.png...")

    plot_percent_scatter(results, 'depth', 'energy', "RL Agent (600 rounds)", "percent_change_energy_vs_depth_rl.png")
    logging.info("✅ Final graphs 1/3")
    plot_percent_scatter(results, 'gate_count', 'depth', "RL Agent (600 rounds)", "percent_change_depth_vs_gate_count.png")
    logging.info("✅ Final graphs 2/3")
    plot_percent_scatter(results, 'gate_count', 'energy', "RL Agent (600 rounds)", "percent_change_energy_vs_gate_count_rl.png")
    logging.info("✅ Final graphs 3/3")
    plot_depth_comparison(results)
    logging.info("✅ All done!")

## CALL MODEL AFTER RUN (final model \n checkpoints respectively)
# model = PPO.load("outputs/ppo_quantum_prune_model")
# model = PPO.load("outputs/checkpoints/ppo_quantum_prune_50000_steps")
