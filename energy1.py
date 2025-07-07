import cirq
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pickle
import os
import matplotlib.pyplot as plt
from typing import List, Tuple

# -----------------------------
# 1. RANDOM CIRCUIT GENERATION
# -----------------------------

def generate_random_superconducting_circuit(n_qubits=12, depth=150):
    qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
    circuit = cirq.Circuit()
    for _ in range(depth):
        layer = []
        for i in range(n_qubits):
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
        circuit.append(layer)
    return circuit

# Dataset generation and loading

def generate_dataset(num_circuits, path='dataset/'):
    os.makedirs(path, exist_ok=True)
    for i in range(num_circuits):
        c = generate_random_superconducting_circuit()
        with open(os.path.join(path, f'circuit_{i}.pkl'), 'wb') as f:
            pickle.dump(c, f)
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_circuits} circuits...")

def load_dataset(path='dataset/', limit=None):
    files = sorted([f for f in os.listdir(path) if f.endswith('.pkl')])[:limit]
    return [pickle.load(open(os.path.join(path, f), 'rb')) for f in files]

# ----------------------
# 2. CIRCUIT EVALUATION
# ----------------------

def evaluate_circuit(circuit):
    # Estimate depth as the number of non-empty moments (layers)
    depth = sum(1 for m in circuit if m.operations)
    # Count total number of gates
    gate_count = len([op for op in circuit.all_operations()])
    energy = 12 * gate_count * 0.18
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

# ------------------
# 4. CUSTOM GYM ENV
# ------------------

class QuantumPruneEnv(gym.Env):
    def __init__(self, base_circuit):
        super().__init__()
        self.original_circuit = base_circuit
        self.circuit = base_circuit.copy()
        self.action_space = spaces.Discrete(3)  # merge, cancel, commute
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)
        self.max_steps = 20
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
        after = evaluate_circuit(self.circuit)

        depth_delta = before['depth'] - after['depth']
        gate_delta = before['gate_count'] - after['gate_count']
        energy_delta = before['energy'] - after['energy']
        reward = depth_delta + 0.8 * gate_delta +  energy_delta

        self.steps_taken += 1
        done = self.steps_taken >= self.max_steps
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

# ---------------
# 6. MAIN SCRIPT
# ---------------

if __name__ == '__main__':
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    print("[1/4] Generating training circuits (small run)...")
    generate_dataset(5000, path='train_set/')

    print("[2/4] Loading training circuits...")
    circuits = load_dataset('train_set/', limit=5000)

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

    print("[3/4] Training PPO model (short run: 50k steps)...")
    vec_env = make_vec_env(lambda: BatchEnv(circuits), n_envs=4)
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=200_000)

    print("[4/4] Generating and evaluating test circuits...")
    generate_dataset(500, path='test_set/')
    eval_circuits = load_dataset('test_set/', limit=500)
    results = evaluate_on_dataset_with_model(eval_circuits, model)
    before_avg_depth = np.mean([r[0]['depth'] for r in results])
    after_avg_depth = np.mean([r[1]['depth'] for r in results])
    print(f"✔ Evaluation complete. Avg depth before: {before_avg_depth:.2f}, after: {after_avg_depth:.2f}")
    before_avg_energy = np.mean([r[0]['energy'] for r in results])
    after_avg_energy = np.mean([r[1]['energy'] for r in results])
    print(f"✔ Evaluation complete. Avg energy before: {before_avg_energy:.2f}, after: {after_avg_energy:.2f}")

    print("📊 Saving depth comparison plot to depth_comparison.png...")
    plot_depth_comparison(results)
    print("✅ All done!")