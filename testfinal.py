import cirq
from cirq import ops
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
from collections import defaultdict
from qsimcirq import QSimSimulator
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

MAX_QUBITS = 12
MAX_DEPTH = 150
MAX_GATES = MAX_QUBITS * MAX_DEPTH
MAX_TEST_STEPS = 20
MAX_TRAIN_STEPS = 50_000
MAX_TRAIN_CIRCUITS = 17_500
MAX_TEST_CIRCUITS = 100
P_SYSTEM = 15_000
W_FREQUENCY = 83_333_333.33

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

def generate_dataset(num_circuits, path='dataset/'):
# The function generates a number of circuits dataset; does not separate it into training and test sets
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

def evaluate_circuit(self, circuit):
# Verified
    depth = sum(1 for m in circuit if m.operations)
    gate_count = len([op for op in circuit.all_operations()])
    qubit_count = len(circuit.all_qubits())
    r_value = getattr(self, 'r')  # fallback if not set
    energy = (1 * r_value * P_SYSTEM / W_FREQUENCY) * qubit_count * depth
    return {
        'depth': depth,
        'gate_count': gate_count,
        'energy': energy,
        'r_value': r_value,
        'qubit_count': qubit_count,
    }

# ------------------------
# 3. GATE TRANSFORMATIONS
# ------------------------

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

                # If it's a rotation gate (Rx, Ry, Rz)
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
                    #print(f"Swapping {op1} and {op2}")
                    to_move_to_m2.append(op1)
                    to_move_to_m1.append(op2)
                    break

        new_m1_ops = [op for op in m1_ops if op not in to_move_to_m2] + to_move_to_m1
        new_m2_ops = [op for op in m2_ops if op not in to_move_to_m1] + to_move_to_m2

        moments[i] = cirq.Moment(remove_overlapping(new_m1_ops))
        moments[i + 1] = cirq.Moment(remove_overlapping(new_m2_ops))


        i += 1
    circuit = cirq.Circuit(moments)

    # remove the identity spacers
    ops_in_order = []

    for moment in circuit:
        for op in moment.operations:
            if not isinstance(op.gate, cirq.IdentityGate):
                ops_in_order.append(op)
                
    return cirq.Circuit(ops_in_order)

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
    
# ---------------------------------
# 3.5. ZX-CALCULUS TRANSFORMATIONS
# ---------------------------------

def circuit_from_cirq(circuit: cirq.Circuit) -> zx.Circuit:
# Verified and debugged. See code_verification\circuit_from_cirq.ipynb
    '''
    This function converts a Cirq circuit to a PyZX circuit.
    It supports the following gates:
    - H (Hadamard)
    - X, Z (with phase)
    - CNOT (controlled-NOT)
    - CZ (controlled-Z)
    '''
    zx_circ = zx.Circuit(MAX_QUBITS)
    for i, moment in enumerate(circuit):
        for op in moment:
            gate = op.gate
            q = [q.x for q in op.qubits]
            if isinstance(gate, cirq.HPowGate) and np.isclose(gate.exponent, 1):
                zx_circ.add_gate("H", q[0])
            elif isinstance(gate, cirq.XPowGate):
                phase = Fraction(gate.exponent).limit_denominator(1000)
                zx_circ.add_gate("XPhase", q[0], phase=phase)
            elif isinstance(gate, cirq.ZPowGate):
                phase = Fraction(gate.exponent).limit_denominator(1000)
                zx_circ.add_gate("ZPhase", q[0], phase=phase)
            elif isinstance(gate, cirq.YPowGate):
                pass
                # We avoid using YPhase
                # phase = Fraction(gate.exponent).limit_denominator(1000)
                # zx_circ.add_gate("YPhase", q[0], phase=phase)
                # displays YPhase gate via combination of X and Z phases
            elif isinstance(gate, cirq.CNotPowGate) and np.isclose(gate.exponent, 1):
                zx_circ.add_gate("CNOT", q[0], q[1])
            elif isinstance(gate, cirq.CZPowGate) and np.isclose(gate.exponent, 1):
                zx_circ.add_gate("CZ", q[0], q[1])
            else:
                print(f"    -> Unsupported or unknown gate: {type(gate)}")
    return zx_circ

def pyzx_to_cirq(pyzx_circuit):
# Verified and debugged. See code_verification\pyzx_to_cirq.ipynb
    '''
    Converts a PyZX circuit to a Cirq circuit.
    Supports the following gates:
    - XPhase (X with phase)
    - ZPhase (Z with phase)
    - Hadamard (H)
    - CNOT (controlled-NOT)
    - CZ (controlled-Z)
    '''
    pi = 3.141592653589793
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
            ops.append(cirq.rz(gate.phase * pi).on(cirq_qubits[q]))
        elif name == "XPHASE":
            q = gate.target if isinstance(gate.target, int) else gate.target[0]
            ops.append(cirq.rx(gate.phase * pi).on(cirq_qubits[q]))
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
# Verified. See code_verification/apply_zx_simplification.ipynb
# The functionality or method for simplification is based on 
# https://arxiv.org/abs/2003.01664
    '''
    Applies simplification to the Cirq circuit using PyZX.
    These simplifications however tends to increase gate depth and gate count 
    so during the deep RL training, these functions may not be used as much.
    Uses the pyzx methods
    - match_w_fusion_parallel:  wire fusion.
                                See https://pyzx.readthedocs.io/en/latest/api.html#pyzx.rules.match_w_fusion_parallel

    - match_ids, remove_ids:    to find identity pairs and remove them.
                                See https://pyzx.readthedocs.io/en/latest/api.html#pyzx.rules.remove_ids

    - extract_circuit:          to extract the simplified circuit.
                                See https://arxiv.org/abs/2003.01664
    '''
    zx_circ = circuit_from_cirq(circuit)
    g = zx_circ.to_graph()
    simplify.match_w_fusion_parallel(g)  # fuse some gates

    matches = simplify.match_ids(g)      # find identity pairs and remove
    simplify.remove_ids(g, matches)      

    simplified = extract.extract_circuit(g) # simplify

    cirq_circ = pyzx_to_cirq(simplified)
    if len(list(cirq_circ.all_operations())) == 0:
        return circuit  # fallback if too much was removed
    return cirq_circ

# -------------------------
# 3.75. HELLINGER FIDELITY
# -------------------------

def cirq_counts(circuit: cirq.Circuit, shots=1024):
# Verified. See code_verification\cirq_counts.ipynb
    qubits = list(circuit.all_qubits())
    circuit = circuit.copy()
    circuit.append(cirq.measure(*qubits, key='m'))

    simulator = QSimSimulator()
    result = simulator.run(circuit, repetitions=shots)
    hist = result.histogram(key='m')

    counts = {format(k, f'0{len(qubits)}b'): v for k, v in hist.items()}
    return counts

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

# ------------------
# 4. CUSTOM GYM ENV
# ------------------

class QuantumPruneEnv(gym.Env):
    def __init__(self, base_circuit):
        super().__init__()
        self.original_circuit = base_circuit
        self.circuit = base_circuit.copy()

        self.action_space = spaces.Discrete(6)  # merge, cancel, commute, ZX, increase r, decrease r
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32) #? 1D array with 11 elements
        self.max_steps = MAX_TEST_STEPS
        self.steps_taken = 0

        self.baseline_counts = cirq_counts(self.original_circuit, shots=204800)
        self.r = 204800 # about 2^12 * 50, 50 shots per outcome on average

    def reset(self, seed=None, options=None):
        self.circuit = self.original_circuit.copy()
        self.steps_taken = 0
        return self._encode_circuit(), {}

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
        # the below is reasonable because on average, it increases by one shot number for every outcome
        elif action == 4: # increase r
            self.r += 2**MAX_QUBITS 
        elif action == 5: # decrease r
            self.r -= 2**MAX_QUBITS

        if self.r < 1:
            self.r = 1;
        
        after = evaluate_circuit(self, self.circuit)          

        # Reward
        agent_counts = cirq_counts(self.circuit, shots=self.r)
        fidelity = hellinger_fidelity(agent_counts, self.baseline_counts)
        energy_delta = before['energy'] - after['energy']

        reward = -energy_delta * fidelity # will need studying for design

        if after['gate_count'] == 0:
            reward = -1

        self.steps_taken += 1

        print(self.steps_taken)
        done = self.steps_taken >= self.max_steps

        # Track metrics during training for Plot 1
        training_history['depth'].append(after['depth'])
        training_history['gate_count'].append(after['gate_count'])
        training_history['energy'].append(after['energy'])
        training_history['qubit_count'].append(after['qubit_count'])
        training_history['fidelity'].append(fidelity)
        training_history['reward'].append(reward)

        return self._encode_circuit(), reward, done, False, {}

    def _encode_circuit(self):
        stats = evaluate_circuit(self, self.circuit)
        return np.array([
            stats['depth'] / 100,
            stats['gate_count'] / 100,
            stats['energy'] / 100,
            stats['qubit_count'] / 100,
            len(self.circuit) / 100,
            self.steps_taken / self.max_steps
        ] + [0]*5, dtype=np.float32)

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

def plot_rl_training_scatter(depths, gates, energies, qubit_counts):
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
    plot_scatter(qubit_counts, "qubit count", "train_qubit_count_scatter.png")

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
    qubit_count_vals = []

    env = BatchEnv(circuits)
    obs, _ = env.reset()

    for step in range(MAX_TRAIN_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)

        if step % 10 == 0:
            evals = evals = evaluate_circuit(env.env, env.env.circuit, step)
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

class PrintStepsCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        if self.num_timesteps % 25 == 0:
            print(f"[Training] Step {self.num_timesteps}")
        return True

if __name__ == '__main__':
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
        'energy': [],
        'qubit_count': [],
        'fidelity': [],
        'reward': []
    }

    print("[3/4] Training PPO model (short run: 50k steps)...")

    # Not parallized
    vec_env = make_vec_env(lambda: BatchEnv(circuits), n_envs=4)
    model = PPO("MlpPolicy", vec_env, verbose=1, n_steps=512)
    callback = PrintStepsCallback()
    model.learn(total_timesteps=MAX_TRAIN_STEPS, callback=callback)



    ### PLOTTING
    depths, gates, energies, qubit_counts = get_training_stats(model)
    plot_training_stats(depths, "RL Training Progress (Depth)", "d", "train_depth.png")
    plot_training_stats(gates, "RL Training Progress (Gate Count)", "n", "train_gate.png")
    plot_training_stats(energies, "RL Training Progress (Energy)", "E", "train_energy.png")
    plot_training_stats(qubit_counts, "RL Training Progress (Qubit Count)", "qubit count", "train_qubit_count.png")

    print("[4/4] Generating and evaluating test circuits...")
    generate_dataset(MAX_TEST_CIRCUITS, path='test_set/')
    eval_circuits = load_dataset('test_set/', limit=MAX_TEST_CIRCUITS)
    results = evaluate_on_dataset_with_model(eval_circuits, model)

    plot_in_game_progress(eval_circuits, model, metric='depth', filename='test_depth.png')
    plot_in_game_progress(eval_circuits, model, metric='gate_count', filename='test_gate.png')
    plot_in_game_progress(eval_circuits, model, metric='energy', filename='test_energy.png')
    plot_in_game_progress(eval_circuits, model, metric='qubit_count', filename='test_qubit_count.png')

    before_avg_depth = np.mean([r[0]['depth'] for r in results])
    after_avg_depth = np.mean([r[1]['depth'] for r in results])
    print(f"✔ Evaluation complete. Avg depth before: {before_avg_depth:.2f}, after: {after_avg_depth:.2f}")

    before_avg_qubit_count = np.mean([r[0]['qubit_count'] for r in results])
    after_avg_qubit_count = np.mean([r[1]['qubit_count'] for r in results])
    print(f"✔ Evaluation complete. Avg qubit_count before: {before_avg_qubit_count:.2f}, after: {after_avg_qubit_count:.2f}")    

    before_avg_energy = np.mean([r[0]['energy'] for r in results])
    after_avg_energy = np.mean([r[1]['energy'] for r in results])
    print(f"✔ Evaluation complete. Avg energy before: {before_avg_energy:.2f}, after: {after_avg_energy:.2f}")

    print(f"Depth decreased by {(before_avg_depth - after_avg_depth) / before_avg_depth * 100:.2f}%")
    print(f"Qubit count decreased by {(before_avg_qubit_count - after_avg_qubit_count) / before_avg_qubit_count * 100:.2f}%")
    print(f"Energy decreased by {(before_avg_energy - after_avg_energy) / before_avg_energy * 100:.2f}%")
    print("📊 Saving depth comparison plot to depth_comparison.png...")

    plot_percent_scatter(results, 'depth', 'energy', "RL Agent (600 rounds)", "percent_change_energy_vs_depth_rl.png")
    plot_percent_scatter(results, 'qubit_count', 'energy', "RL Agent (600 rounds)", "percent_change_energy_vs_qubit_count_rl.png")
    plot_depth_comparison(results)
    print("✅ All done!")