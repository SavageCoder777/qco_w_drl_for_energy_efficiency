### Quantum Circuit Optimization (QCO) using Deep Reinforcement Learning (DRL) Model
#
# - Creates arbritrary qubit circuits and attempts to increase energy efficiency through
#   training the DRL model--specifically PPO--to apply gate transformations reducing the
#   depth/gate count/energy of the circuit while maintaining overall logic.
# - Generative AI was used to help with part of this code; the 'check' emojis from the
#   AI responses were liked by the researchers and incorporated into the final logs in
#   the code.
#
# Copyright 2025 UCSB SRA Track 7 Team 6

# Initial necessary imports; find required libraries in requirements.txt
import cirq
from cirq import ops
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import pickle
import os
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyzx as zx
from pyzx import extract
import pyzx.simplify as simplify
from fractions import Fraction
from scipy.stats import sem  # for standard error
import seaborn as sns  # for better scatter visuals
from collections import defaultdict
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from qsimcirq import QSimSimulator
import logging
import sys
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import CheckpointCallback

# Setting Hyperparameters
MAX_QUBITS = 2
MAX_DEPTH = 150
MAX_GATES = MAX_QUBITS * MAX_DEPTH
MAX_TEST_STEPS = 20
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
def generate_random_superconducting_circuit(n_qubits=MAX_QUBITS, depth=MAX_DEPTH):
# Verified 
    qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
    circuit = cirq.Circuit()
    for _ in range(depth):
        layer = []
        i = 0
        for i in range(n_qubits):
            op_type = np.random.choice(['rx', 'rz', 'cx'])
            if op_type == 'rx':
                theta = np.random.choice([(-7*np.pi/4), (-3*np.pi/2), (-5*np.pi/4), (-np.pi), (-3*np.pi/4), (-np.pi/2), (-np.pi/4), 0,
                                        (7*np.pi/4), (3*np.pi/2), (5*np.pi/4), (np.pi), (3*np.pi/4), (np.pi/2), (np.pi/4)])
                layer.append(cirq.rx(theta)(qubits[i]))
            elif op_type == 'rz':
                phi = np.random.choice([(-7*np.pi/4), (-3*np.pi/2), (-5*np.pi/4), (-np.pi), (-3*np.pi/4), (-np.pi/2), (-np.pi/4), 0,
                                        (7*np.pi/4), (3*np.pi/2), (5*np.pi/4), (np.pi), (3*np.pi/4), (np.pi/2), (np.pi/4)])
                layer.append(cirq.rz(phi)(qubits[i]))
            elif op_type == 'cx' and i < len(qubits) - 1:
                if np.random.rand() < 0.5:
                    layer.append(cirq.CNOT(qubits[i], qubits[i + 1]))
        circuit.append(layer)
    return circuit

# The function generates a number of circuits dataset; does not separate it into training and test sets
def generate_dataset(num_circuits, path='dataset/'):
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

# ----------------------
# 2. CIRCUIT EVALUATION
# ----------------------

# Evaluating circuit variables to measure reward and graph
def evaluate_circuit(env, circuit):
# Semantic error. The energy is calculated as (qubit # * gate depth * coefficient)
# Verified 

# B!: qubit count calculation must be incorporated
# verified 

    # Estimate depth as the number of non-empty moments (layers)
    depth = sum(1 for m in circuit if m.operations)
    # Count total number of gates
    gate_count = len([op for op in circuit.all_operations()])
    qubit_count = len(circuit.all_qubits())
    energy = (gate_count * env.r * P_SYSTEM / W_FREQUENCY) * qubit_count * depth
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

# Cancels gates if they're 1) on the same qubit 2) the same gate 3) exponents adds to 0
def apply_gate_cancellation(circuit: cirq.Circuit) -> cirq.Circuit:
    from collections import defaultdict

    qubit_ops = defaultdict(list)
    for moment_index, moment in enumerate(circuit):
        for op in moment.operations:
            for qubit in op.qubits:
                qubit_ops[qubit].append((moment_index, op))

    kept_ops = set()

    for qubit, ops in qubit_ops.items():
        prev_op = None
        for idx, (moment_index, op) in enumerate(ops):
            if (prev_op and isinstance(op.gate, cirq.EigenGate) and
                isinstance(prev_op[1].gate, cirq.EigenGate) and
                type(op.gate) == type(prev_op[1].gate) and
                np.isclose(op.gate.exponent + prev_op[1].gate.exponent, 0.0, atol=1e-2)):
                prev_op = None
                continue
            if prev_op:
                kept_ops.add(prev_op[1])
            prev_op = (moment_index, op)
        if prev_op:
            kept_ops.add(prev_op[1])

    new_circuit = cirq.Circuit()
    for moment in circuit:
        new_moment_ops = []
        for op in moment.operations:
            if op in kept_ops:
                new_moment_ops.append(op)
        new_circuit.append(cirq.Moment(new_moment_ops))

    return new_circuit

# Merges neighboring, congruent gates on the same qubit by adding exponents
def apply_gate_merging(circuit: cirq.Circuit) -> cirq.Circuit:
    merged_ops_with_moments = []
    pending_rotations = defaultdict(list)  # qubit -> list of (moment_idx, op)
    
    def flush(qubit, pending_list):
        if not pending_list:
            return
        gate_type = type(pending_list[0][1].gate)
        total_angle = sum(op.gate._rads for _, op in pending_list)
        norm_angle = total_angle % (2 * np.pi)
        if norm_angle > np.pi:
            norm_angle -= 2 * np.pi
        if not np.isclose(norm_angle, 0.0, atol=1e-6):
            moment_idx = pending_list[0][0]
            if gate_type == type(cirq.rx(0)):
                merged_op = cirq.rx(norm_angle).on(qubit)
            elif gate_type == type(cirq.ry(0)):
                merged_op = cirq.ry(norm_angle).on(qubit)
            elif gate_type == type(cirq.rz(0)):
                merged_op = cirq.rz(norm_angle).on(qubit)
            else:
                merged_op = pending_list[0][1]
            merged_ops_with_moments.append((moment_idx, merged_op))
        # else: effectively zero, skip
        pending_list.clear()

    for moment_idx, moment in enumerate(circuit):
        used_qubits = set()

        for op in moment.operations:
            qubits = op.qubits

            # Multi-qubit op: flush all qubit buffers it's involved in
            if len(qubits) > 1:
                for q in qubits:
                    flush(q, pending_rotations[q])
                merged_ops_with_moments.append((moment_idx, op))
                used_qubits.update(qubits)

            else:  # single-qubit op
                q = qubits[0]
                gate = op.gate

                # If it's a rotation gate (Rx, Rz)
                if hasattr(gate, "_rads"):
                    if pending_rotations[q]:
                        last_gate = pending_rotations[q][-1][1].gate
                        if type(last_gate) == type(gate):
                            # Still consecutive of same type
                            pending_rotations[q].append((moment_idx, op))
                        else:
                            flush(q, pending_rotations[q])
                            pending_rotations[q].append((moment_idx, op))
                    else:
                        pending_rotations[q].append((moment_idx, op))
                else:
                    # Non-rotation gate -> flush before it
                    flush(q, pending_rotations[q])
                    merged_ops_with_moments.append((moment_idx, op))

                used_qubits.add(q)

        # For any qubit not touched this moment, flush its buffer
        for q in pending_rotations:
            if q not in used_qubits:
                flush(q, pending_rotations[q])

    # Flush any remaining at the end
    for q in pending_rotations:
        flush(q, pending_rotations[q])

    # Group by moment
    ops_by_moment = defaultdict(list)
    for moment_idx, op in merged_ops_with_moments:
        ops_by_moment[moment_idx].append(op)

    max_moment = max(ops_by_moment.keys(), default=-1)
    moments = [cirq.Moment(ops_by_moment.get(i, [])) for i in range(max_moment + 1)]
    return cirq.Circuit(moments)

def remove_overlapping(ops_list):
    used_qubits = set()
    filtered_ops = []
    for op in ops_list:
        if not used_qubits.intersection(op.qubits):
            filtered_ops.append(op)
            used_qubits.update(op.qubits)
    return filtered_ops

def spacer_around_CNOT(circuit: cirq.Circuit) -> cirq.Circuit:
# Verified. See code_verification\apply_commutation.ipynb for test cases.
    """
    This function is a helper function for apply_commutation.

    This function first adds a identity gate around the CNOT gate along the target qubit to spread the 
    CNOT gates apart from the rest. Then the identity gates are swapped in the direction away from the 
    CNOT gates with a single qubit gate.
    """
    moments = list(circuit)
    # spread
    i = 0
    while i < len(moments):
        moment = moments[i]
        cnot_ops = [op for op in moment.operations if isinstance(op.gate, cirq.CNotPowGate)]
        for cnot in cnot_ops:
            q0, q1 = cnot.qubits
            if i > 0:
                left = moments[i - 1]
                if q0 in left.qubits and q1 in left.qubits:
                    moments.insert(i, cirq.Moment([cirq.I(q1)]))
                    i += 1
                    break
            if i + 1 < len(moments):
                right = moments[i + 1]
                if q0 in right.qubits and q1 in right.qubits:
                    moments.insert(i + 1, cirq.Moment([cirq.I(q1)]))
                    break
        i += 1
    
    circuit = cirq.Circuit(moments)

    # shift
    new_moments = list(circuit)
    num_moments = len(new_moments)

    shifted_ids = set()  # (moment index, qubit tuple)

    for i in range(num_moments):
        moment = new_moments[i]
        ops = list(moment.operations)

        for op_idx, op in enumerate(ops):
            if isinstance(op.gate, cirq.IdentityGate):
                id_key = (i, tuple(op.qubits))
                if id_key in shifted_ids:
                    continue

                id_qubits = set(op.qubits)

                # try swap with i+1 moment
                if i + 1 < num_moments:
                    next_ops = list(new_moments[i + 1].operations)
                    for j, next_op in enumerate(next_ops):
                        if not isinstance(next_op.gate, cirq.CNotPowGate) and set(next_op.qubits) == id_qubits:
                            ops[op_idx], next_ops[j] = next_ops[j], ops[op_idx]
                            new_moments[i] = cirq.Moment(ops)
                            new_moments[i + 1] = cirq.Moment(next_ops)
                            shifted_ids.discard((i, tuple(op.qubits)))
                            shifted_ids.add((i + 1, tuple(op.qubits)))  # mark new location
                            break

                # try swap with i-1 moment
                elif i - 1 >= 0:
                    prev_ops = list(new_moments[i - 1].operations)
                    for j, prev_op in enumerate(prev_ops):
                        if not isinstance(prev_op.gate, cirq.CNotPowGate) and set(prev_op.qubits) == id_qubits:
                            ops[op_idx], prev_ops[j] = prev_ops[j], ops[op_idx]
                            new_moments[i] = cirq.Moment(ops)
                            new_moments[i - 1] = cirq.Moment(prev_ops)
                            shifted_ids.discard((i, tuple(op.qubits)))
                            shifted_ids.add((i - 1, tuple(op.qubits)))
                            break
    return cirq.Circuit(new_moments)

# Commutes gates that satisfy AB=BA, where A and B are unitary operators
def apply_commutation(circuit: cirq.Circuit) -> cirq.Circuit:
# Verified and debugged. See code_verification\apply_commutation.ipynb for test cases.
    """
    This function applies commutation by first applyings spacers around the CNOT gates, 
    swapping the gates, then removing the identity spacers. The addition of Identity spacers
    around the CNOT is because CNOT is a multi-qubit gate and may overlap with other single 
    qubit gates. As a result, the swaps that occur may not be apparent because it was swapped 
    with an Identity gate, which is later removed.
    """
    # Add spacers around CNOT gate
    circuit = spacer_around_CNOT(circuit)

    # swap/commutation
    moments = list(circuit)

    i = 0
    while i < len(moments) - 1:
        m1, m2 = moments[i], moments[i + 1]
        m1_ops = list(m1.operations)
        m2_ops = list(m2.operations)

        to_move_to_m2 = []
        to_move_to_m1 = []

        for op1 in m1_ops:
            for op2 in m2_ops:
                if cirq.commutes(op1, op2, atol=1e-10) and (not set(op1.qubits).isdisjoint(op2.qubits)):
                    to_move_to_m2.append(op1)
                    to_move_to_m1.append(op2)
                    break

        new_m1_ops = [op for op in m1_ops if op not in to_move_to_m2] + to_move_to_m1
        new_m2_ops = [op for op in m2_ops if op not in to_move_to_m1] + to_move_to_m2

        def qubits_overlap(ops):
            qubits_seen = set()
            for op in ops:
                for q in op.qubits:
                    if q in qubits_seen:
                        return True
                    qubits_seen.add(q)
            return False

        # Before setting moments[i] and moments[i+1]:
        if not qubits_overlap(new_m1_ops) and not qubits_overlap(new_m2_ops):
            moments[i] = cirq.Moment(new_m1_ops)
            moments[i + 1] = cirq.Moment(new_m2_ops)
        else:
            # Skip swap or handle differently (e.g., don't swap)
            pass


        i += 1
    circuit = cirq.Circuit(moments)

    # remove the identity spacers
    ops_in_order = []

    for moment in circuit:
        for op in moment.operations:
            if not isinstance(op.gate, cirq.IdentityGate):
                ops_in_order.append(op)
                
    return cirq.Circuit(ops_in_order)


# ---------------------------------
# 3.5. ZX-CALCULUS TRANSFORMATIONS
# ---------------------------------

# Converts cirq.Circuit to a QuantumCircuit
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

# Converts QuantumCircuit to a cirq.Circuit
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
            pass
    return cirq.Circuit(ops)

# Uses built in ZX-Calculus simplification function to act as another action to optimize the circuit
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

# -------------------------
# 3.75. HELLINGER FIDELITY
# -------------------------

def cirq_counts_faster(circuit: cirq.Circuit, shots=1024):
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

def _adjust_r_based_on_fidelity(circuit, r_initial=(2**MAX_QUBITS), min_r=1, max_r=(2**MAX_QUBITS)*50, step=(2**MAX_QUBITS)*1):
    r = r_initial
    fidelity = 0.0
    baseline_counts = cirq_counts_faster(circuit, shots=max_r)
    while r < max_r:
        agent_counts = cirq_counts_faster(circuit, shots=r)
        fidelity = hellinger_fidelity(agent_counts, baseline_counts)
        if fidelity > 0.90:
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
        super().__init__()
        self.original_circuit = base_circuit
        self.circuit = base_circuit.copy()
        self.action_space = spaces.Discrete(4)  # merge, cancel, commute, ZX
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)
        self.max_steps = MAX_TEST_STEPS
        self.steps_taken = 0
        self.r = R_START # about 2^12 * 50, 50 shots per outcome on average
        self._gate_count = 0
        self._gate_depth = 0
        self._energy = 0.0
        self._r = R_START
        self._circuit = None

    def reset(self, seed=None, options=None):
        self.circuit = self.original_circuit.copy()
        self.steps_taken = 0
        return self._encode_circuit(), {}

    # Defines what should be done each step the DRL model takes (chooses actions 0, 1, 2, 3, 4, or 5)
    def step(self, action):
        before = evaluate_circuit(self, self.circuit)
        if action == 0:
            self.circuit = apply_gate_merging(self.circuit)
        elif action == 1:
            self.circuit = apply_gate_cancellation(self.circuit)
        elif action == 2:
            self.circuit = apply_commutation(self.circuit)
        elif action == 3:
            self.circuit = apply_zx_simplification(self.circuit)

        if self.r < 1:
            self.r = 1

        after = evaluate_circuit(self, self.circuit)
        self._update_metrics()
        depth_delta = before['depth'] - after['depth']
        gate_delta = before['gate_count'] - after['gate_count']
        energy_delta = (before['energy'] - after['energy'])

        # Reward Calculation
        self.r, fidelity = _adjust_r_based_on_fidelity(self.circuit)
        reward = energy_delta * 1E-7 # scaled for stability

        if after['gate_count'] == 0: reward = -10
        self.steps_taken += 1
        done = self.steps_taken >= self.max_steps
        # Track metrics during training for Plot 1
        training_history['depth'].append(after['depth'])
        training_history['gate_count'].append(after['gate_count'])
        training_history['energy'].append(after['energy'])
        training_history['qubit_count'].append(after['qubit_count'])
        training_history['r'].append(self.r)
        training_history['fidelity'].append(fidelity)
        self.last_reward = reward
        return self._encode_circuit(), reward, done, False, {}

    def _encode_circuit(self):
        stats = evaluate_circuit(self, self.circuit)
        return np.array([
            stats['depth'] / 100,
            stats['gate_count'] / 100,
            stats['energy'] / 100,
            stats['qubit_count'] / 100,
            stats['r'] / 100,
            len(self.circuit) / 100,
            self.steps_taken / self.max_steps
        ] + [0]*4, dtype=np.float32)
    def _update_metrics(self):
        self._gate_count = len(list(self.circuit.all_operations()))
        self._gate_depth = len(self.circuit)
        self._energy = self.circuit_energy()
        self._circuit = self.circuit  # just for access elsewhere
        self._r = self.r
    @property
    def gate_count(self):
        return self._gate_count
    @property
    def gate_depth(self):
        return self._gate_depth
    @property
    def energy(self):
        return self._energy
    @property
    def repetition_count(self):
        return self._r
    @property
    def current_circuit(self):
        return self._circuit  # Or serialize to QASM if needed
    def circuit_energy(self):
        g = len(list(self.circuit.all_operations()))
        r = self.r  # your dynamic repetition count
        Nq = len(self.circuit.all_qubits())
        Nd = len(self.circuit)  # depth as number of Moments
        return (g * r * P_SYSTEM / W_FREQUENCY) * Nq * Nd

# ---------------------
# 5. SAVE RUNTIME DATA
# ---------------------

def get_true_env(env):
    while hasattr(env, 'env'):
        env = env.env
    return env

# Used to log and save the runtime data in seperate files for further reference
class SaveMetricsAndRolloutsCallback(BaseCallback):
    def __init__(self, save_freq=256, save_dir="outputs", verbose=1):
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
            env = get_true_env(self.training_env.envs[0])

            # Evaluate metrics on current circuit state
            evals = evaluate_circuit(env, env.circuit)

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

            # Save rollout (full environment state) to a file for later analysis
            rollout_path = os.path.join(self.rollout_dir, f"rollout_step_{step}.pkl")
            with open(rollout_path, 'wb') as f:
                # Save whatever info you want — here saving environment state and circuit for replay
                pickle.dump({
                    'step': step,
                    'circuit': env.circuit,
                    'original_circuit': env.original_circuit,
                    'r': getattr(env, 'r', None),
                }, f)
            if self.verbose > 0:
                print(f"[SaveMetrics] Saved rollout to {rollout_path}")

        return True

# ------------------------------
# 6. EVALUATION + VISUALIZATION
# ------------------------------

# Applies the model to each of the test circuits and works to optimize them
def evaluate_on_dataset_with_model(circuits, model):
    results = []
    i = 0
    for c in circuits:
        logging.info(f"[EVAL] Optimizing test circuit {i + 1}/{len(circuits)}...")
        env = QuantumPruneEnv(c)
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
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

    env = BatchEnv(circuits)
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

def plot_training_percent_change_from_log(file_path, metric_x, metric_y, save_path, LOG_FREQ=512, num_circuits=5000):
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
            env = self.training_env.envs[0]
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
    logging.info("[1/4] Generating training circuits (small run)...")
    generate_dataset(MAX_TRAIN_CIRCUITS, path='train_set/')

    logging.info("[2/4] Loading training circuits...")
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
        'energy': [],
        'qubit_count': [],
        'fidelity': [],
        'reward': [],
        'r': []
    }

    logging.info("[3/4] Training PPO model (short run: 50k steps)...")
    vec_env = make_vec_env(lambda: BatchEnv(circuits), n_envs=4)
    model = PPO("MlpPolicy", vec_env, verbose=1, n_steps=256)

    checkpoint_callback = TrueTimestepCheckpointCallback(
    save_freq=512,
    save_path="outputs/checkpoints/",
    name_prefix="ppo_quantum_prune")

    callback = CallbackList([
    SaveMetricsAndRolloutsCallback(save_freq=512, save_dir="outputs"),
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

    depths, gates, energies, qubit_counts = get_training_stats(model)
    plot_training_stats(depths, "RL Training Progress (Depth)", "d", "train_depth.png")
    logging.info("✅ Training graph 1/12")
    plot_training_stats(gates, "RL Training Progress (Gate Count)", "n", "train_gate.png")
    logging.info("✅ Training graph 2/12")
    plot_training_stats(energies, "RL Training Progress (Energy)", "E", "train_energy.png")
    logging.info("✅ Training graph 3/12")
    plot_training_stats(qubit_counts, "RL Training Progress (Qubit Count)", "qubit count", "train_qubit_count.png")
    logging.info("✅ Training graph 4/12")
    plot_metric_from_log("outputs/full_metrics_log.csv", "depth")
    logging.info("✅ Training graph 5/12")
    plot_metric_from_log("outputs/full_metrics_log.csv", "gate_count")
    logging.info("✅ Training graph 6/12")
    plot_metric_from_log("outputs/full_metrics_log.csv", "energy")
    logging.info("✅ Training graph 7/12")
    plot_metric_from_log("outputs/full_metrics_log.csv", "r", color="purple", save_as="train_r_from_log.png")
    logging.info("✅ Training graph 8/12")
    plot_r_values(training_history['r'])
    logging.info("✅ Training graph 9/12")

    plot_training_percent_change_from_log(
        "outputs/full_metrics_log.csv",
        metric_x='depth',
        metric_y='energy',
        save_path='train_percent_change_energy_vs_depth.png',
        LOG_FREQ=LOG_FREQ,
        num_circuits=MAX_TRAIN_CIRCUITS
    )
    logging.info("✅ Training graph 10/12")
    plot_training_percent_change_from_log(
        "outputs/full_metrics_log.csv",
        metric_x='depth',
        metric_y='gate_count',
        save_path='train_percent_change_gate_vs_depth.png',
        LOG_FREQ=LOG_FREQ,
        num_circuits=MAX_TRAIN_CIRCUITS
    )
    logging.info("✅ Training graph 11/12")
    plot_training_percent_change_from_log(
        "outputs/full_metrics_log.csv",
        metric_x='gate_count',
        metric_y='energy',
        save_path='train_percent_change_energy_vs_gate.png',
        LOG_FREQ=LOG_FREQ,
        num_circuits=MAX_TRAIN_CIRCUITS
    )
    logging.info("✅ Training graph 12/12")

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
