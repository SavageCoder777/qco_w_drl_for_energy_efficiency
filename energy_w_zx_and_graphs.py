import cirq
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pickle
import os
import matplotlib.pyplot as plt
import pyzx as zx
from pyzx import extract
import pyzx.simplify as simplify
from fractions import Fraction
from scipy.stats import sem  # for standard error
import seaborn as sns  # for better scatter visuals

MAX_QUBITS = 12
MAX_DEPTH = 50
MAX_GATES = MAX_QUBITS * MAX_DEPTH
MAX_ENERGY = MAX_GATES * 0.18
MAX_TEST_STEPS = 20
MAX_TRAIN_STEPS = 50_000
MAX_TRAIN_CIRCUITS = 2000
MAX_TEST_CIRCUITS = 100

# -----------------------------
# 1. RANDOM CIRCUIT GENERATION
# -----------------------------


def generate_random_superconducting_circuit(n_qubits=MAX_QUBITS, depth=MAX_DEPTH):
# Verified
    qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
    circuit = cirq.Circuit()
    for _ in range(depth):
        layer = []
        i = 0
        while i < n_qubits: # changed to ensure two gates aren't applied to a qubit in one layer after CNOT
            op_type = np.random.choice(['rx', 'rz', 'cx'])
            if op_type == 'rx':
                theta = np.random.uniform(0, 2*np.pi)
                layer.append(cirq.rx(theta)(qubits[i]))
            elif op_type == 'rz':
                phi = np.random.uniform(0, 2*np.pi)
                layer.append(cirq.rz(phi)(qubits[i]))
            elif op_type == 'cx' and i < len(qubits) - 1:
                if np.random.rand() < 0.5:
                    layer.append(cirq.CNOT(qubits[i], qubits[i + 1]))
                    i += 1
        circuit.append(layer)
    return circuit

def generate_dataset(num_circuits, path='dataset/'):
# L!?: The function generates a number of circuits dataset; does not separate it into training and test sets
    os.makedirs(path, exist_ok=True)
    for i in range(num_circuits):
        c = generate_random_superconducting_circuit()
        with open(os.path.join(path, f'circuit_{i}.pkl'), 'wb') as f:
            pickle.dump(c, f)
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_circuits} circuits...")

def load_dataset(path='dataset/', limit=None):
# Verified
    files = sorted([f for f in os.listdir(path) if f.endswith('.pkl')])[:limit]
    return [pickle.load(open(os.path.join(path, f), 'rb')) for f in files]

# ----------------------
# 2. CIRCUIT EVALUATION
# ----------------------

def evaluate_circuit(circuit):
# B!: The energy is calculated as (qubit # * gate depth * coefficient)
# verified

# B!: qubit count calculation must be incorporated

    # Estimate depth as the number of non-empty moments (layers)
    depth = sum(1 for m in circuit if m.operations)
    # Count total number of gates
    gate_count = len([op for op in circuit.all_operations()])
    energy = 12 * depth * 0.18
    return {
        'depth': depth,
        'gate_count': gate_count,
        'energy': energy
    }

# ------------------------
# 3. GATE TRANSFORMATIONS
# ------------------------

def apply_gate_cancellation(circuit: cirq.Circuit) -> cirq.Circuit:
    new_ops = []
    prev_op = None
    for op in circuit.all_operations():
        if prev_op and isinstance(op.gate, cirq.EigenGate) and isinstance(prev_op.gate, cirq.EigenGate):
            if op.qubits == prev_op.qubits and type(op.gate) == type(prev_op.gate):
                if np.isclose(op.gate.exponent + prev_op.gate.exponent, 0.0, atol=1e-2):
                    prev_op = None
                    continue
        if prev_op:
            new_ops.append(prev_op)
        prev_op = op
    if prev_op:
        new_ops.append(prev_op)
    return cirq.Circuit(new_ops)

def apply_gate_merging(circuit: cirq.Circuit) -> cirq.Circuit:
    new_ops = []
    pending = {}

    rx_gate_type = type(cirq.rx(0))
    rz_gate_type = type(cirq.rz(0))

    for op in circuit.all_operations():
        if isinstance(op.gate, cirq.EigenGate) and len(op.qubits) == 1:
            q = op.qubits[0]
            gate_type = type(op.gate)
            key = (q, gate_type)

            if gate_type in [rx_gate_type, rz_gate_type]:
                current_angle = op.gate._rads  # radians angle
                if key in pending:
                    old_op = pending[key]
                    old_angle = old_op.gate._rads
                    combined_angle = old_angle + current_angle
                    if np.isclose(combined_angle, 0.0, atol=1e-6):
                        del pending[key]
                    else:
                        if gate_type == rx_gate_type:
                            new_gate = cirq.rx(combined_angle)
                        else:
                            new_gate = cirq.rz(combined_angle)
                        pending[key] = new_gate.on(q)
                else:
                    pending[key] = op

            elif hasattr(op.gate, 'exponent'):
                current_exp = op.gate.exponent
                if key in pending:
                    old_op = pending[key]
                    old_exp = old_op.gate.exponent
                    combined_exp = old_exp + current_exp
                    if np.isclose(combined_exp, 0.0, atol=1e-6):
                        del pending[key]
                    else:
                        new_gate = old_op.gate.__class__(exponent=combined_exp)
                        pending[key] = new_gate.on(q)
                else:
                    pending[key] = op

            else:
                new_ops.extend(pending.values())
                pending = {}
                new_ops.append(op)

        else:
            new_ops.extend(pending.values())
            pending = {}
            new_ops.append(op)

    new_ops.extend(pending.values())
    return cirq.Circuit(new_ops)

def apply_commutation(circuit: cirq.Circuit) -> cirq.Circuit:
    moments = list(circuit)
    new_moments = []

    i = 0
    while i < len(moments) - 1:
        m1, m2 = moments[i], moments[i + 1]
        # Ops from each moment
        m1_ops = list(m1.operations)
        m2_ops = list(m2.operations)

        # Ops to swap
        to_move_to_m2 = []
        to_move_to_m1 = []

        # Track which ops in m2 have been swapped
        swapped_m2_indices = set()

        for idx1, op1 in enumerate(m1_ops):
            for idx2, op2 in enumerate(m2_ops):
                if idx2 in swapped_m2_indices:
                    continue
                if cirq.commutes(op1, op2, atol=1e-6):
                    to_move_to_m2.append(op1)
                    to_move_to_m1.append(op2)
                    swapped_m2_indices.add(idx2)
                    break

        # Remaining ops after swapping
        new_m1_ops = [op for op in m1_ops if op not in to_move_to_m2] + to_move_to_m1
        new_m2_ops = [op for idx, op in enumerate(m2_ops) if idx not in swapped_m2_indices] + to_move_to_m2

        # Check for overlapping ops in new moments and split if needed
        def split_ops_into_moments(ops):
            moments_list = []
            used_qubits = set()
            current_ops = []
            for op in ops:
                if any(q in used_qubits for q in op.qubits):
                    moments_list.append(cirq.Moment(current_ops))
                    current_ops = [op]
                    used_qubits = set(op.qubits)
                else:
                    current_ops.append(op)
                    used_qubits.update(op.qubits)
            if current_ops:
                moments_list.append(cirq.Moment(current_ops))
            return moments_list

        # Split moments if needed to avoid overlap
        new_m1_moments = split_ops_into_moments(new_m1_ops)
        new_m2_moments = split_ops_into_moments(new_m2_ops)

        # Add the new moments in place of m1 and m2
        new_moments.extend(new_m1_moments)
        new_moments.extend(new_m2_moments)

        i += 2

    # If odd number of moments, append the last one
    if i == len(moments) - 1:
        new_moments.append(moments[-1])

    return cirq.Circuit(new_moments)
    
# ---------------------------------
# 3.5. ZX-CALCULUS TRANSFORMATIONS
# ---------------------------------

def circuit_from_cirq(circuit: cirq.Circuit) -> zx.Circuit:
    zx_circ = zx.Circuit(MAX_QUBITS)
    for moment in circuit:
        for op in moment:
            gate = op.gate
            q = [q.x for q in op.qubits]
            if isinstance(gate, cirq.HPowGate) and np.isclose(gate.exponent, 1):
                zx_circ.add_gate("H", q[0])
            elif isinstance(gate, cirq.XPowGate):
                zx_circ.add_gate("XPhase", q[0], phase=Fraction(gate.exponent).limit_denominator(1000))
            elif isinstance(gate, cirq.YPowGate):
                pass  # PyZX does not support Y rotations directly
            elif isinstance(gate, cirq.CNotPowGate) and np.isclose(gate.exponent, 1):
                zx_circ.add_gate("CNOT", q[0], q[1])
    return zx_circ

def pyzx_to_cirq(pyzx_circuit):
    ops = []
    def get_qubits_used():
        max_index = 0
        for gate in pyzx_circuit.gates:
            targets = []
            if hasattr(gate, 'target'):
                t = gate.target
                if isinstance(t, (list, tuple)):
                    targets.extend(t)
                else:
                    targets.append(t)
            if hasattr(gate, 'control'):
                targets.append(gate.control)
            if targets:
                max_index = max(max_index, max(targets))
        return max_index + 1
    num_qubits = pyzx_circuit.n_qubits if hasattr(pyzx_circuit, 'n_qubits') else get_qubits_used()
    cirq_qubits = [cirq.LineQubit(i) for i in range(num_qubits)]
    for gate in pyzx_circuit.gates:
        name = gate.name.upper()
        if name == "ZPHASE":
            q = gate.target if isinstance(gate.target, int) else gate.target[0]
            ops.append(cirq.rz(gate.phase).on(cirq_qubits[q]))
        elif name == "XPHASE":
            q = gate.target if isinstance(gate.target, int) else gate.target[0]
            ops.append(cirq.rx(gate.phase).on(cirq_qubits[q]))
        elif name == "HAD":
            q = gate.target if isinstance(gate.target, int) else gate.target[0]
            ops.append(cirq.H(cirq_qubits[q]))
        elif name == "CNOT":
            ctrl = gate.control
            tgt = gate.target if isinstance(gate.target, int) else gate.target[0]
            ops.append(cirq.CNOT(cirq_qubits[ctrl], cirq_qubits[tgt]))
        elif name == "CZ":
            ctrl = gate.control
            tgt = gate.target if isinstance(gate.target, int) else gate.target[0]
            ops.append(cirq.CZ(cirq_qubits[ctrl], cirq_qubits[tgt]))
        else:
            pass  # or: print(f"Unrecognized gate {name}, skipping.")
    return cirq.Circuit(ops)

def apply_zx_simplification(circuit: cirq.Circuit) -> cirq.Circuit:
    zx_circ = circuit_from_cirq(circuit)
    g = zx_circ.to_graph()
    simplify.match_w_fusion_parallel(g)  # fuse some gates
    matches = simplify.match_ids(g)      # find identity pairs
    simplify.remove_ids(g, matches)      # remove them
    simplified = extract.extract_circuit(g)
    cirq_circ = pyzx_to_cirq(simplified)
    if len(list(cirq_circ.all_operations())) == 0:
        return circuit  # fallback if too much was removed
    return cirq_circ

# ------------------
# 4. CUSTOM GYM ENV
# ------------------

class QuantumPruneEnv(gym.Env):
    def __init__(self, base_circuit):
        super().__init__()
        self.original_circuit = base_circuit
        self.circuit = base_circuit.copy()
        self.action_space = spaces.Discrete(4)  # merge, cancel, commute, ZX
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)
        self.max_steps = MAX_TEST_STEPS
        self.steps_taken = 0

    def reset(self, seed=None, options=None):
        self.circuit = self.original_circuit.copy()
        self.steps_taken = 0
        return self._encode_circuit(), {}

    def step(self, action):
        before = evaluate_circuit(self.circuit)
        if action == 0:
            self.circuit = apply_gate_merging(self.circuit)
        elif action == 1:
            self.circuit = apply_gate_cancellation(self.circuit)
        elif action == 2:
            self.circuit = apply_commutation(self.circuit)
        elif action == 3:
            self.circuit = apply_zx_simplification(self.circuit)
        after = evaluate_circuit(self.circuit)

        depth_delta = before['depth'] - after['depth']
        gate_delta = before['gate_count'] - after['gate_count']
        energy_delta = before['energy'] - after['energy']
        reward = depth_delta + 0.8 * gate_delta +  energy_delta
        if after['gate_count'] == 0: reward = -1
        self.steps_taken += 1
        done = self.steps_taken >= self.max_steps

        # Track metrics during training for Plot 1
        training_history['depth'].append(after['depth'])
        training_history['gate_count'].append(after['gate_count'])
        training_history['energy'].append(after['energy'])

        return self._encode_circuit(), reward, done, False, {}

    def _encode_circuit(self):
        stats = evaluate_circuit(self.circuit)
        return np.array([
            stats['depth'] / 100,
            stats['gate_count'] / 100,
            stats['energy'] / 100,
            len(self.circuit) / 100,
            self.steps_taken / self.max_steps
        ] + [0]*6, dtype=np.float32)

# ------------------------------
# 5. EVALUATION + VISUALIZATION
# ------------------------------

def evaluate_on_dataset_with_model(circuits, model):
    results = []
    for c in circuits:
        env = QuantumPruneEnv(c)
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
        before = evaluate_circuit(env.original_circuit)
        after = evaluate_circuit(env.circuit)
        results.append((before, after))
    return results

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

def plot_rl_training_scatter(depths, gates, energies):
    import matplotlib.pyplot as plt
    import numpy as np

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

def plot_test_examples_5(test_logs, rounds, ylabel, filename):
    import matplotlib.pyplot as plt
    import numpy as np
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
    import matplotlib.pyplot as plt
    import numpy as np

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

    env = BatchEnv(circuits)
    obs, _ = env.reset()

    for step in range(MAX_TRAIN_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)

        if step % 10 == 0:
            evals = evaluate_circuit(env.env.circuit)
            depth_vals.append((step, evals['depth']))
            gate_vals.append((step, evals['gate_count']))
            energy_vals.append((step, evals['energy']))

    return depth_vals, gate_vals, energy_vals

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
            val = evaluate_circuit(env.circuit)[metric]
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


# ---------------
# 6. MAIN SCRIPT
# ---------------
sns.set(style="whitegrid")

if __name__ == '__main__':
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    print("[1/4] Generating training circuits (small run)...")
    generate_dataset(MAX_TRAIN_CIRCUITS, path='train_set/')

    print("[2/4] Loading training circuits...")
    circuits = load_dataset('train_set/', limit=MAX_TRAIN_CIRCUITS)

    class BatchEnv(gym.Env):
        def __init__(self, circuits):
            self.circuits = circuits
            self.index = 0
            self.env = QuantumPruneEnv(self.circuits[self.index])
            self.action_space = self.env.action_space
            self.observation_space = self.env.observation_space

        def reset(self, seed=None, options=None):
            self.index = (self.index + 1) % len(self.circuits)
            self.env = QuantumPruneEnv(self.circuits[self.index])
            return self.env.reset()

        def step(self, action):
            return self.env.step(action)

    training_history = {
        'depth': [],
        'gate_count': [],
        'energy': []
    }
    print("[3/4] Training PPO model (short run: 50k steps)...")
    vec_env = make_vec_env(lambda: BatchEnv(circuits), n_envs=4)
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=MAX_TRAIN_STEPS)

    depths, gates, energies = get_training_stats(model)
    plot_training_stats(depths, "RL Training Progress (Depth)", "d", "train_depth.png")
    plot_training_stats(gates, "RL Training Progress (Gate Count)", "n", "train_gate.png")
    plot_training_stats(energies, "RL Training Progress (Energy)", "E", "train_energy.png")

    print("[4/4] Generating and evaluating test circuits...")
    generate_dataset(MAX_TEST_CIRCUITS, path='test_set/')
    eval_circuits = load_dataset('test_set/', limit=MAX_TEST_CIRCUITS)
    results = evaluate_on_dataset_with_model(eval_circuits, model)

    plot_in_game_progress(eval_circuits, model, metric='depth', filename='test_depth.png')
    plot_in_game_progress(eval_circuits, model, metric='gate_count', filename='test_gate.png')
    plot_in_game_progress(eval_circuits, model, metric='energy', filename='test_energy.png')

    before_avg_depth = np.mean([r[0]['depth'] for r in results])
    after_avg_depth = np.mean([r[1]['depth'] for r in results])
    print(f"✔ Evaluation complete. Avg depth before: {before_avg_depth:.2f}, after: {after_avg_depth:.2f}")
    before_avg_energy = np.mean([r[0]['energy'] for r in results])
    after_avg_energy = np.mean([r[1]['energy'] for r in results])
    print(f"✔ Evaluation complete. Avg energy before: {before_avg_energy:.2f}, after: {after_avg_energy:.2f}")
    print(f"Depth decreased by {(before_avg_depth - after_avg_depth) / before_avg_depth * 100:.2f}%")
    print(f"Energy decreased by {(before_avg_energy - after_avg_energy) / before_avg_energy * 100:.2f}%")
    print("📊 Saving depth comparison plot to depth_comparison.png...")

    plot_percent_scatter(results, 'depth', 'energy', "RL Agent (600 rounds)", "percent_change_rl.png")
    plot_depth_comparison(results)
    print("✅ All done!")