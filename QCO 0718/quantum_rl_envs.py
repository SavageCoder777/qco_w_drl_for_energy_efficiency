import cirq
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import pickle
import os
import glob
import csv
import matplotlib.pyplot as plt
import seaborn as sns  # for better scatter visuals
from collections import defaultdict
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from qsimcirq import QSimSimulator
import logging
import sys
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Setting Hyperparameters
# Checked
MAX_QUBITS = 12
MAX_DEPTH = 150
MAX_TEST_STEPS = 20 # maximum steps per episode during training
MAX_TRAIN_STEPS = 350_000 # time steps the model takes during training
MAX_TRAIN_CIRCUITS = 17_500 # how many circuits to be created and train the model on
MAX_TEST_CIRCUITS = 500 # how many circuits the trained model optimizes
P_SYSTEM = 15_000 # specific for superconducting quantum computer
W_FREQUENCY = 83_333_333.33 # specific for superconducting quantum computer
R_START = (2**MAX_QUBITS) * 50 # change depending on how many steps per gate are needed
LOG_FREQ = 512 # how frequently the parameter changes are logged

# -----------------------------
# 1. RANDOM CIRCUIT GENERATION
# -----------------------------

# Generate a random circuit with Rx, Rz, and CNOT quantum gates
import numpy as np
def generate_random_superconducting_circuit(n_qubits=12, depth=150):
    qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
    circuit = cirq.Circuit()
    for _ in range(depth):
        layer = []
        for i in range(n_qubits):
            op_type = np.random.choice(['phased_x', 'rz', 'cx'])

            if op_type == 'phased_x':
                phase_exp = np.random.uniform(0, 1)
                exp = np.random.uniform(0, 1)
                gate = cirq.PhasedXPowGate(phase_exponent=phase_exp, exponent=exp)
                layer.append(gate(qubits[i]))

            elif op_type == 'rz':
                phi = np.random.choice([
                    -7*np.pi/4, -3*np.pi/2, -5*np.pi/4, -np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0,
                    7*np.pi/4, 3*np.pi/2, 5*np.pi/4, np.pi, 3*np.pi/4, np.pi/2, np.pi/4
                ])
                layer.append(cirq.rz(phi)(qubits[i]))

            elif op_type == 'cx' and i < n_qubits - 1:
                if np.random.rand() < 0.5:
                    layer.append(cirq.CNOT(qubits[i], qubits[i + 1]))

        circuit.append(layer)

    return circuit

# The function generates a number of circuits dataset; does not separate it into training and test sets
def generate_dataset(num_circuits, path='dataset/'):
# Checked
    os.makedirs(path, exist_ok=True)
    for i in range(num_circuits):
        c = generate_random_superconducting_circuit()
        with open(os.path.join(path, f'circuit_{i}.pkl'), 'wb') as f:
            pickle.dump(c, f)
        if (i + 1) % 100 == 0:
            logging.info(f"Generated {i + 1}/{num_circuits} circuits...")

# Loads a generated dataset for model training
def load_dataset(path='dataset/', limit=None):
# Verified 
    files = sorted([f for f in os.listdir(path) if f.endswith('.pkl')])[:limit]
    return [pickle.load(open(os.path.join(path, f), 'rb')) for f in files]

def get_circuit_paths(dir_path, limit=None):
    paths = sorted(glob.glob(os.path.join(dir_path, "*.pkl")))
    return paths[:limit] if limit else paths


# ----------------------
# 2. CIRCUIT EVALUATION
# ----------------------
def evaluate_circuit(env, circuit):
# Checked
    # Estimate depth as the number of non-empty moments (layers)
    depth = sum(1 for m in circuit if m.operations)
    # Count total number of gates
    gate_count = len([op for op in circuit.all_operations()])
    qubit_count = len(circuit.all_qubits())
    energy = (1 * env.r * P_SYSTEM / W_FREQUENCY) * qubit_count * depth # the gate count in the formula is g=1, see paper
    return {
        'depth': depth,
        'gate_count': gate_count,
        'energy': energy,
        'qubit_count': qubit_count,
        'r': env.r
    }

# ------------------------
# 3. GATE TRANSFORMATIONS
# ------------------------
import cirq

def apply_gate_merging(circuit: cirq.Circuit) -> cirq.Circuit:
    """Merges single-qubit gates into Z rotations and PhasedX gates."""
    return cirq.merge_single_qubit_gates_to_phased_x_and_z(circuit)

def apply_gate_cancellation(circuit: cirq.Circuit) -> cirq.Circuit:
    """Cancels redundant Phased-Pauli and Z rotations, pushing them for simplification."""
    circuit = cirq.eject_phased_paulis(circuit)
    circuit = cirq.eject_z(circuit)
    return circuit

# Cancels gates if they're 1) on the same qubit 2) the same gate 3) exponents adds to 0
# def apply_gate_cancellation(circuit: cirq.Circuit) -> cirq.Circuit:
#     from collections import defaultdict

#     qubit_ops = defaultdict(list)
#     for moment_index, moment in enumerate(circuit):
#         for op in moment.operations:
#             for qubit in op.qubits: 
#                 qubit_ops[qubit].append((moment_index, op))

#     kept_ops = set()
    
#     for qubit, ops in qubit_ops.items():
#         skip_indices = set()
#         for idx in range(len(ops)):
#             if idx in skip_indices:
#                 continue  # Don't add this op; it was cancelled with the previous one
    
#             moment_index, op = ops[idx]
    
#             if idx < len(ops) - 1:
#                 next_moment_idx, next_op = ops[idx + 1]
    
#                 if (type(op.gate) == type(next_op.gate) and
#                     isinstance(op.gate, cirq.EigenGate) and
#                     isinstance(next_op.gate, cirq.EigenGate) and
#                     np.isclose(op.gate.exponent + next_op.gate.exponent, 0.0, atol=1e-2)):
    
#                     # These two cancel → skip both
#                     skip_indices.update([idx, idx+1])
#                     continue
    
#                 elif moment_index + 1 != next_moment_idx:
#                     # TODO: handle non-adjacent case
#                     next_cancellable_idx = check_rest_ops(ops, idx)
                    
#                     if next_cancellable_idx is not None:
#                         skip_indices.update([idx, next_cancellable_idx])
#                     else:
#                         kept_ops.add((moment_index, op))

#                 else:
#                     # Not cancelling but still worth keeping this op
#                     kept_ops.add((moment_index, op))
#             else:
#                 # Last operation; not part of a pair
#                 kept_ops.add((moment_index, op))
                
#     new_circuit = cirq.Circuit()
#     for moment_index, moment in enumerate(circuit):
#         new_moment_ops = []
#         for op in moment.operations:
#             if (moment_index, op) in kept_ops:
#                 new_moment_ops.append(op)
#         new_circuit.append(cirq.Moment(new_moment_ops))

#     return new_circuit
                
def check_rest_ops(ops, idx): 
    """
    Check whether the operation at ops[idx] has a canceling pair later in ops.
    Return the index of the canceling op if found, otherwise return None.
    """
    moment_idx, op = ops[idx]

    if not isinstance(op.gate, cirq.EigenGate):
        return None

    for j in range(idx + 1, len(ops)):
        next_moment_idx, next_op = ops[j]

        # Skip if gate types don't match or not EigenGate
        if not isinstance(next_op.gate, cirq.EigenGate):
            continue

        if type(op.gate) != type(next_op.gate):
            continue

        # Check for canceling exponents
        if np.isclose(op.gate.exponent + next_op.gate.exponent, 0.0, atol=1e-2):
            return j  # Found canceling op

    return None  # No canceling op found

# Merges neighboring, congruent gates on the same qubit by adding exponents
# def apply_gate_merging(circuit: cirq.Circuit) -> cirq.Circuit:
#     import numpy as np
#     from collections import defaultdict

#     qubit_ops = defaultdict(list)
#     for moment_index, moment in enumerate(circuit):
#         for op in moment.operations:
#             for qubit in op.qubits:
#                 qubit_ops[qubit].append((moment_index, op))

#     kept_ops = set()

#     for qubit, ops in qubit_ops.items():
#         i = 0
#         while i < len(ops):
#             moment_index, op = ops[i]

#             # Check if it's a mergeable gate
#             if (isinstance(op.gate, cirq.EigenGate) and hasattr(op.gate, '_rads')):
#                 gate_type = type(op.gate)
#                 total_angle = op.gate._rads
#                 j = i + 1

#                 # Accumulate consecutive mergeable gates of same type on same qubit
#                 while j < len(ops):
#                     _, next_op = ops[j]
#                     if (type(next_op.gate) == gate_type and
#                         isinstance(next_op.gate, cirq.EigenGate) and
#                         hasattr(next_op.gate, '_rads')):
#                         total_angle += next_op.gate._rads
#                         j += 1
#                     else:
#                         break

#                 # Normalize the angle
#                 normalized_angle = total_angle % (2 * np.pi)
#                 if normalized_angle > np.pi:
#                     normalized_angle -= 2 * np.pi

#                 if not np.isclose(normalized_angle, 0.0, atol=1e-6):
#                     if gate_type == type(cirq.rx(0)):
#                         merged_op = cirq.rx(normalized_angle).on(qubit)
#                     elif gate_type == type(cirq.ry(0)):
#                         merged_op = cirq.ry(normalized_angle).on(qubit)
#                     elif gate_type == type(cirq.rz(0)):
#                         merged_op = cirq.rz(normalized_angle).on(qubit)
#                     else:
#                         merged_op = op  # fallback
#                     kept_ops.add((moment_index, merged_op))
#                 # Skip to first non-mergeable
#                 i = j
#             else:
#                 kept_ops.add((moment_index, op))
#                 i += 1

#     # Rebuild circuit from kept_ops
#     ops_by_moment = defaultdict(list)
#     for moment_index, op in kept_ops:
#         ops_by_moment[moment_index].append(op)

#     new_circuit = cirq.Circuit()
#     for moment_index in range(len(circuit)):
#         new_circuit.append(cirq.Moment(ops_by_moment.get(moment_index, [])))

#     return new_circuit

def isolate_first_gate_after_cnot(moments):
# Verified
    new_moments = moments[:]
    i = 0
    while i < len(new_moments):
        moment = new_moments[i]
        cnot_ops = [op for op in moment.operations if isinstance(op.gate, cirq.CNotPowGate)]
        for cnot in cnot_ops:
            cnot_qubits = set(cnot.qubits)
            j = i + 1
            while j < len(new_moments):
                next_moment = new_moments[j]
                overlapping_ops = [op for op in next_moment.operations if not set(op.qubits).isdisjoint(cnot_qubits)]
                if overlapping_ops:
                    op_to_isolate = overlapping_ops[0]
                    if len(next_moment.operations) > 1:
                        remaining_ops = [op for op in next_moment.operations if op != op_to_isolate]
                        new_moments[j] = cirq.Moment(remaining_ops)
                        new_moments.insert(j, cirq.Moment([op_to_isolate]))
                    break
                j += 1
        i += 1
    return new_moments

def qubits_overlap(ops):
# Verified
    seen = set()
    for op in ops:
        for q in op.qubits:
            if q in seen:
                return True
            seen.add(q)
    return False

# def apply_commutation(circuit: cirq.Circuit) -> cirq.Circuit:
# # Verified
#     moments = list(circuit)

#     # Step 1: isolate first gate after each CNOT
#     moments = isolate_first_gate_after_cnot(moments)

#     # Step 2: single pass over adjacent moments to swap commuting ops
#     i = 0
#     while i < len(moments) - 1:
#         m1_ops = list(moments[i].operations)
#         m2_ops = list(moments[i + 1].operations)

#         to_move_to_m2 = []
#         to_move_to_m1 = []

#         for op1 in m1_ops:
#             for op2 in m2_ops:
#                 # Commute and overlapping qubits => candidates for swapping
#                 if cirq.commutes(op1, op2, atol=1e-10) and (not set(op1.qubits).isdisjoint(op2.qubits)):
#                     to_move_to_m2.append(op1)
#                     to_move_to_m1.append(op2)
#                     break

#         new_m1_ops = [op for op in m1_ops if op not in to_move_to_m2] + to_move_to_m1
#         new_m2_ops = [op for op in m2_ops if op not in to_move_to_m1] + to_move_to_m2

#         if not qubits_overlap(new_m1_ops) and not qubits_overlap(new_m2_ops):
#             moments[i] = cirq.Moment(new_m1_ops)
#             moments[i + 1] = cirq.Moment(new_m2_ops)

#         i += 1

#     return cirq.Circuit(moments)


def apply_commutation(circuit: cirq.Circuit) -> cirq.Circuit:
    return cirq.eject_z(circuit, eject_parameterized=False)

def rebuild_circuit(circuit: cirq.Circuit,) -> cirq.Circuit:
    all_operations = list(circuit.all_operations())
    rebuilt_circuit = cirq.Circuit(all_operations)
    return rebuilt_circuit

# -------------------------
# 3.75. HELLINGER FIDELITY
# -------------------------

def cirq_counts_faster(circuit: cirq.Circuit, shots=1024):
# Checked
    qubits = list(circuit.all_qubits())
    width = len(qubits)

    circuit = circuit.copy()
    circuit.append(cirq.measure(*qubits, key='m'))

    simulator = QSimSimulator()
    result = simulator.run(circuit, repetitions=int(shots))
    bitstrings = result.measurements['m']  # dimension of matrix is (shots, width)

    # fast histogram using vectorized bitstring to integer conversion
    powers = 1 << np.arange(width)[::-1]  # bit shift -> [2^(n-1), ..., 1], len(powers) equals width
    keys = np.dot(bitstrings, powers) # binary to integer
    counts_array = np.bincount(keys, minlength=2 ** width) # count up occurance of integer

    return {k: v for k, v in enumerate(counts_array) if v > 0}

def hellinger_fidelity(counts1, counts2):
# Verified and debugged. See code_verification\hellinger_fidelity.ipynb
    """Compute Hellinger fidelity between two count dictionaries."""
    if not counts1 or not counts2:
        return 0.0  # fallback if any input is empty

    all_keys = set(counts1.keys()).union(counts2.keys())
    shots1 = sum(counts1.values())
    shots2 = sum(counts2.values())

    if shots1 == 0 or shots2 == 0:
        return 0.0  # avoid divide by zero

    p = np.array([np.sqrt(counts1.get(k, 0) / shots1) for k in all_keys])
    q = np.array([np.sqrt(counts2.get(k, 0) / shots2) for k in all_keys])

    fidelity = np.sum(p * q) ** 2
    return fidelity

def _adjust_r_based_on_fidelity(circuit, baseline_counts, r_initial=(2**MAX_QUBITS), min_r=1, max_r=(2**MAX_QUBITS)*50, step=(2**MAX_QUBITS)):
# Checked
    r = r_initial
    fidelity = 0.0
    while r < max_r:

        agent_counts = cirq_counts_faster(circuit, shots=r)
        fidelity = hellinger_fidelity(agent_counts, baseline_counts)
        # print(r)
        # print(fidelity)
        if fidelity > 0.707: # see https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.110501
            break
        r += step
    return min(r, max_r), fidelity

# ------------------
# 4. CUSTOM GYM ENV
# ------------------

# Creates the gymnasium environment to run the DRL model
class QuantumPruneEnv(gym.Env):
    # Initializes necessary parameters and variables for further use and update
    def __init__(self, base_circuit):
    # Checked
        super().__init__()
        self.original_circuit = base_circuit
        self.circuit = base_circuit.copy()
        self.action_space = spaces.Discrete(4)  # merge, cancel, commute, remove negligible gates
        self.observation_space = spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32) # See _encode_circuit()
        self.max_steps = MAX_TEST_STEPS
        self.last_reward = 0
        self.steps_taken = 0
        self.r = R_START
        self.baseline_counts = cirq_counts_faster(self.original_circuit, shots=MAX_QUBITS**2 * 50)

    def reset(self, seed=None, options=None):
    # Checked
        self.circuit = self.original_circuit.copy()
        self.steps_taken = 0
        return self._encode_circuit(), {}

    # Defines what should be done each step the DRL model takes (chooses actions 0, 1, 2, 3, 4, or 5)
    def step(self, action):
    # Checked
    
        b4 = self.circuit.copy()
        before = evaluate_circuit(self, self.circuit)
        if action == 0:
            self.circuit = apply_gate_merging(self.circuit)
        elif action == 1:
            self.circuit = apply_gate_cancellation(self.circuit)
        elif action == 2:
            self.circuit = apply_commutation(self.circuit)
        elif action == 3:
            self.circuit = cirq.drop_negligible_operations(self.circuit)
        self.circuit = rebuild_circuit(self.circuit)
        
        
        # Prevent bugged function mesing with code
        # sim = cirq.Simulator()
        # state1 = sim.simulate(b4).final_state_vector
        # state2 = sim.simulate(self.circuit).final_state_vector
        # phase = np.vdot(state1, state2) / np.vdot(state1, state1)
        # if not np.allclose(state1, phase * state2, atol=1e-8):
        #     self.circuit = b4

        if self.r < 1:
            self.r = 1

        # Delete: For debugging
        # print(action)
        # out1 = cirq_counts_faster(self.original_circuit, 100000)
        # out2 = cirq_counts_faster(self.circuit, 100000)
        # print(hellinger_fidelity(out1, out2))

        after = evaluate_circuit(self, self.circuit)

        energy_delta = (after['energy'] - before['energy'])

        # Reward Calculation
        self.r, fidelity = _adjust_r_based_on_fidelity(self.circuit, self.baseline_counts)

        reward = -1 * energy_delta # scaled for stability
        terminated = False
        if after['gate_count'] == 0:
            terminated = True
            reward = -10
        self.last_reward = reward

        self.steps_taken += 1
        truncated = self.steps_taken >= self.max_steps
        
        # Track metrics during training for Plot 1
        # training_history['depth'].append(after['depth'])
        # training_history['gate_count'].append(after['gate_count'])
        # training_history['energy'].append(after['energy'])
        # training_history['qubit_count'].append(after['qubit_count'])
        # training_history['r'].append(self.r)
        # training_history['fidelity'].append(fidelity)

        info = {
            'depth': after['depth'],
            'gate_count': after['gate_count'],
            'energy': after['energy'],
            'qubit_count': after['qubit_count'],
            'r': self.r,
            'fidelity': fidelity
        }

        return self._encode_circuit(), reward, terminated, truncated, info
        # See step() in https://gymnasium.farama.org/api/env/

    def _encode_circuit(self):
    # Checked
        circuit_eval = evaluate_circuit(self, self.circuit)
        return np.array([
            circuit_eval['depth'] / 100,
            circuit_eval['gate_count'] / 100,
            circuit_eval['energy'] / 100,
            circuit_eval['qubit_count'] / 100,
            circuit_eval['r'] / 100,
            len(self.circuit) / 100,
            self.steps_taken / self.max_steps
        ], dtype=np.float32)

# Parallel Processing
class BatchEnv(gym.Env):
        def __init__(self, circuit_paths):
            self.circuit_paths = circuit_paths
            self.index = 0
            self.env = self._load_env()
            self.action_space = self.env.action_space
            self.observation_space = self.env.observation_space
        
        def _load_env(self):
            import pickle
            with open(self.circuit_paths[self.index], "rb") as f:
                circuit = pickle.load(f)
            return QuantumPruneEnv(circuit)


        def reset(self, seed=None, options=None):
            self.index = (self.index + 1) % len(self.circuit_paths)
            self.env = self._load_env()
            return self.env.reset()

        def step(self, action):
            return self.env.step(action)