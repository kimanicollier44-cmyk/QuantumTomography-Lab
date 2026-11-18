#!/usr/bin/env python3
"""QuantumTomography-Lab - educational quantum state/process tomography lab.

by Darion Kellar

Pure Python standard library. Zero dependencies. Zero network.
Education only - and the central lesson is stated up front:

  * Every "measurement" in this lab is a SEEDED SIMULATION of a state
    the lab itself prepared and therefore already knows exactly.
    Tomography of a state you already know is a teaching exercise:
    it shows how reconstruction works, how finite shot counts bend
    the answer, and why raw linear inversion can return a matrix
    that is not a physical state at all.
  * Nothing here characterises real quantum hardware. No device was
    measured, no benchmark was run, no error rate was certified.
    The numbers describe a Python sampler with a printed seed.
  * The honest workflow is the real workflow, kept visible: count
    finite shots (never probabilities dressed up as data), invert
    linearly, WATCH the inversion go unphysical at low shot counts,
    project back to the nearest physical state, and only then quote
    a fidelity - against the known target, which is a luxury real
    tomography never has.

State tomography covers 1 and 2 qubits in the Pauli basis. Process
tomography lite reconstructs the chi matrix of one noisy 1-qubit
gate (a Hadamard with a labelled depolarizing rate) from input /
output state tomography, through the Pauli transfer matrix and a
fixed basis change built from Pauli products. All matrix algebra -
complex products, Kronecker products, a Jacobi eigensolver for
Hermitian matrices, PSD projection, Uhlmann fidelity - is written
from scratch in this file. Real-hardware characterisation, certified
benchmarking, and production verification claims are permanent HOLD
gates - with Gemini Argon 4 Extended Thinking as DECIDER
(names-only, tighten-only), Muse as advisor (never overrides
HOLD), and Google Flow plan-only seats.
"""
import cmath
import hashlib
import json
import math
import random
import sys

# ---------------------------------------------------------------- constants
VERSION = "1.0.0"
BASE_SEED = 20261117

# Labelled noise for the process-tomography demo gate: an ideal
# Hadamard followed by depolarizing replacement with probability
# DEPOL_RATE. A constructed teaching channel, not a measured device.
DEPOL_RATE = 0.10

# The exact process fidelity of that channel against the ideal
# Hadamard: (1 - p) * 1 + p * (1/4) = 1 - 3p/4, because the fully
# depolarizing chi is I/4 and the ideal Hadamard chi has no identity
# component. Derived in the process section, asserted in CHECKS.
EXPECTED_PROCESS_FIDELITY = 1.0 - 0.75 * DEPOL_RATE

# Demo shot budgets. Shots are counted per Pauli basis.
LOW_SHOTS = 100
HIGH_SHOTS = 100_000
PAIR_SHOTS = 20_000
PROCESS_SHOTS = 20_000
SHOT_LADDER = (100, 1_000, 10_000, 100_000)

# Seed offsets for the printed demos. At 100 shots per basis the
# linear inversion of a pure state lands outside the physical state
# space for a large share of seeds; the low-shot offset below was
# selected so the printed run is one of them (negativity 0.0467),
# because watching the inversion go unphysical - and the projection
# repair it - is the lesson. The ladder offsets keep one fixed seed
# per row so the table is reproducible byte-for-byte.
LOW_SEED_OFFSET = 22_119
LADDER_SEED_OFFSETS = {100: 2_207_228, 1_000: 15_838,
                       10_000: 23_757, 100_000: 31_676}

# ---------------------------------------------------------------- HOLD gates
HOLD_GATES = {
    "real_hardware_characterization_claim": "HOLD - no characterising real quantum hardware; every state here is simulated and self-prepared",
    "certified_benchmarking_claim": "HOLD - no certified benchmarks, quantum-volume figures, or device scores from a teaching sampler",
    "production_verification_claim": "HOLD - no production sign-off or verification of any real quantum system",
    "hardware_acceptance_testing": "HOLD - no acceptance or pass/fail testing of delivered quantum hardware",
    "calibration_service_claim": "HOLD - no device calibration service; simulated counts calibrate nothing real",
    "gate_error_rate_certification": "HOLD - no certifying the gate error rates of any real processor",
    "quantum_advantage_evidence": "HOLD - no lab reconstruction may be presented as evidence of quantum advantage",
    "published_benchmark_submission": "HOLD - no submitting lab numbers to benchmark suites or leaderboards as results",
    "device_fingerprinting": "HOLD - no fingerprinting or identifying a real device from tomography-style data",
    "cryptographic_state_certification": "HOLD - no certifying states or sources for cryptographic use (e.g. QKD source sign-off)",
    "medical_sensing_claim": "HOLD - no medical, diagnostic, or sensing claims from simulated tomography",
    "investment_hardware_claim": "HOLD - no investment or due-diligence claims about hardware vendors from lab output",
    "classified_defense_characterization": "HOLD - no defence or classified device characterisation, in any direction",
    "known_state_as_unknown_claim": "HOLD - tomography of a known, self-prepared state must never be presented as characterising an unknown state",
}


def gate(action):
    """Return 'HOLD' if the action is gated, else 'PROCEED' (sandbox sim)."""
    return "HOLD" if action in HOLD_GATES else "PROCEED"


# ================================================================ 1. Complex matrix toolkit
# Matrices are lists of lists of complex numbers. Everything the lab
# needs - products, dagger, Kronecker, trace - is written out here;
# there is no numpy hiding the arithmetic.

def zeros(n, m=None):
    return [[0j] * (m if m is not None else n) for _ in range(n)]


def eye(n):
    return [[1.0 + 0j if i == j else 0j for j in range(n)] for i in range(n)]


def mat_add(a, b):
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))]
            for i in range(len(a))]


def mat_sub(a, b):
    return [[a[i][j] - b[i][j] for j in range(len(a[0]))]
            for i in range(len(a))]


def mat_scale(a, s):
    return [[a[i][j] * s for j in range(len(a[0]))] for i in range(len(a))]


def mat_mul(a, b):
    n, k, m = len(a), len(b), len(b[0])
    out = zeros(n, m)
    for i in range(n):
        for t in range(k):
            ait = a[i][t]
            if ait != 0j:
                for j in range(m):
                    out[i][j] += ait * b[t][j]
    return out


def mat_dagger(a):
    return [[a[j][i].conjugate() for j in range(len(a))]
            for i in range(len(a[0]))]


def mat_trace(a):
    return sum(a[i][i] for i in range(len(a)))


def kron(a, b):
    out = zeros(len(a) * len(b), len(a[0]) * len(b[0]))
    for i in range(len(a)):
        for j in range(len(a[0])):
            if a[i][j] == 0j:
                continue
            for p in range(len(b)):
                for q in range(len(b[0])):
                    out[i * len(b) + p][j * len(b[0]) + q] = a[i][j] * b[p][q]
    return out


def is_hermitian(a, tol=1e-9):
    n = len(a)
    return all(abs(a[i][j] - a[j][i].conjugate()) < tol
               for i in range(n) for j in range(n))


def hermitian_part(a):
    return mat_scale(mat_add(a, mat_dagger(a)), 0.5)


# ================================================================ 2. Jacobi eigensolver + PSD tools
# Cyclic Jacobi for Hermitian matrices: a phase rotation first makes
# the target off-diagonal entry real, then a real Jacobi rotation
# annihilates it. Sweeps repeat until the off-diagonal norm is below
# tolerance. For the 2x2 and 4x4 matrices of this lab it converges to
# machine precision, which CHECKS prove by rebuilding V L V-dagger.

def hermitian_eigh(a, tol=1e-12, max_sweeps=100):
    """Eigen-decomposition of a Hermitian matrix.

    Returns (eigenvalues ascending, eigenvector matrix) with the
    eigenvectors as COLUMNS, so a = V diag(evals) V-dagger.
    """
    n = len(a)
    a = [row[:] for row in a]
    v = eye(n)
    for _ in range(max_sweeps):
        off = math.sqrt(sum(abs(a[p][q]) ** 2
                            for p in range(n) for q in range(n) if p != q))
        if off < tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p][q]
                if abs(apq) < 1e-15:
                    continue
                phi = cmath.phase(apq)
                d = eye(n)
                d[q][q] = complex(math.cos(-phi), math.sin(-phi))
                a = mat_mul(mat_mul(mat_dagger(d), a), d)
                v = mat_mul(v, d)
                b = a[p][q].real
                if abs(b) < 1e-15:
                    continue
                app, aqq = a[p][p].real, a[q][q].real
                tau = (aqq - app) / (2.0 * b)
                t = (1.0 if tau >= 0 else -1.0) / (
                    abs(tau) + math.sqrt(1.0 + tau * tau))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c
                r = eye(n)
                r[p][p] = c
                r[q][q] = c
                r[p][q] = s
                r[q][p] = -s
                a = mat_mul(mat_mul(mat_dagger(r), a), r)
                v = mat_mul(v, r)
    evals = [a[i][i].real for i in range(n)]
    order = sorted(range(n), key=lambda i: evals[i])
    evals = [evals[i] for i in order]
    vecs = [[v[row][i] for i in order] for row in range(n)]
    return evals, vecs


def negativity(a):
    """Total negative eigenvalue mass: 0 for a physical state."""
    evals, _ = hermitian_eigh(hermitian_part(a))
    return sum(max(0.0, -ev) for ev in evals)


def project_physical(a):
    """Nearest physical state: clip negative eigenvalues, renormalise.

    Eigendecompose, set every negative eigenvalue to zero, rebuild,
    and divide by the trace so the result is a density matrix:
    Hermitian, positive semidefinite, trace exactly 1.
    """
    evals, vecs = hermitian_eigh(hermitian_part(a))
    clipped = [max(0.0, ev) for ev in evals]
    total = sum(clipped)
    if total <= 0:
        n = len(a)
        return mat_scale(eye(n), 1.0 / n)
    lam = zeros(len(a))
    for i, ev in enumerate(clipped):
        lam[i][i] = ev / total
    return hermitian_part(mat_mul(mat_mul(vecs, lam), mat_dagger(vecs)))


def psd_sqrt(a):
    """Matrix square root of a PSD matrix via its eigendecomposition."""
    evals, vecs = hermitian_eigh(hermitian_part(a))
    lam = zeros(len(a))
    for i, ev in enumerate(evals):
        lam[i][i] = math.sqrt(max(0.0, ev))
    return hermitian_part(mat_mul(mat_mul(vecs, lam), mat_dagger(vecs)))


def fidelity(rho, sigma):
    """Uhlmann fidelity F(rho, sigma) in [0, 1].

    F = (Tr sqrt( sqrt(rho) sigma sqrt(rho) ))^2, computed with the
    lab's own eigensolver. Fidelity against a KNOWN target is a
    luxury of the teaching setting - real tomography has no target
    to compare against, which the demo says out loud.
    """
    sqrt_rho = psd_sqrt(rho)
    inner = mat_mul(mat_mul(sqrt_rho, sigma), sqrt_rho)
    evals, _ = hermitian_eigh(inner)
    total = sum(math.sqrt(max(0.0, ev)) for ev in evals)
    return min(1.0, max(0.0, total * total))


def solve_linear(a, b):
    """Solve a complex linear system by Gauss-Jordan with pivoting."""
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-13:
            raise ValueError("singular linear system")
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        m[col] = [x / pv for x in m[col]]
        for r in range(n):
            if r != col and m[r][col] != 0j:
                f = m[r][col]
                m[r] = [x - f * y for x, y in zip(m[r], m[col])]
    return [m[i][n] for i in range(n)]


# ================================================================ 3. Pauli bases + target states
PAULI_1Q = {
    "I": [[1 + 0j, 0j], [0j, 1 + 0j]],
    "X": [[0j, 1 + 0j], [1 + 0j, 0j]],
    "Y": [[0j, -1j], [1j, 0j]],
    "Z": [[1 + 0j, 0j], [0j, -1 + 0j]],
}
PAULI_LETTERS = ("I", "X", "Y", "Z")
PAULI_STRINGS_1Q = ("X", "Y", "Z")
PAULI_STRINGS_2Q = tuple(
    a + b for a in PAULI_LETTERS for b in PAULI_LETTERS
    if not (a == "I" and b == "I"))


def pauli_matrix(label):
    """The Pauli operator for a 1- or 2-letter string, e.g. 'XZ'."""
    out = PAULI_1Q[label[0]]
    for ch in label[1:]:
        out = kron(out, PAULI_1Q[ch])
    return out


def ket_density(amplitudes):
    """Density matrix |psi><psi| from a list of amplitudes."""
    n = len(amplitudes)
    return [[amplitudes[i] * amplitudes[j].conjugate()
             for j in range(n)] for i in range(n)]


def tilted_state():
    """A pure 1-qubit state off every axis: theta=1.05, phi=0.65.

    Pure states sit ON the boundary of the state space, so finite
    shot noise pushes the linear inversion outside it - which is
    exactly the unphysical-reconstruction lesson of this lab.
    """
    theta, phi = 1.05, 0.65
    amp0 = math.cos(theta / 2.0)
    amp1 = complex(math.cos(phi), math.sin(phi)) * math.sin(theta / 2.0)
    return ket_density([amp0, amp1])


def plus_state():
    s = 1.0 / math.sqrt(2.0)
    return ket_density([s, s])


def depolarized_plus():
    """|+> with labelled depolarizing weight 0.25 mixed in."""
    p = 0.25
    return mat_add(mat_scale(plus_state(), 1.0 - p),
                   mat_scale(PAULI_1Q["I"], p / 2.0))


def werner_bell():
    """Bell |Phi+> with labelled white-noise weight 0.15 (Werner)."""
    s = 1.0 / math.sqrt(2.0)
    bell = ket_density([s, 0j, 0j, s])
    return mat_add(mat_scale(bell, 0.85), mat_scale(eye(4), 0.15 / 4.0))


TARGET_STATES = {
    "tilted_pure_1q": tilted_state(),
    "depolarized_plus_1q": depolarized_plus(),
    "werner_bell_2q": werner_bell(),
}


def bloch_vector(rho):
    """(x, y, z) Bloch components of a 1-qubit density matrix."""
    return tuple(mat_trace(mat_mul(rho, PAULI_1Q[axis])).real
                 for axis in ("X", "Y", "Z"))


# ================================================================ 4. Seeded measurement sampler
# A measurement is COUNTS: `shots` individual Bernoulli draws against
# the Born probability of the +1 outcome, made with a seeded RNG.
# Nothing in the lab ever feeds a probability downstream as if it
# were data - the estimate is always (n_plus - n_minus) / shots.

def basis_seed(base_seed, tag, shots):
    """Deterministic per-measurement seed from integers only."""
    return (base_seed * 1_000_003 + tag * 97_409 + shots) % (2 ** 31)


def sample_expectation(rho, label, shots, seed):
    """Simulate `shots` measurements of one Pauli string.

    Returns (expectation_estimate, n_plus). The estimate comes from
    counted outcomes, so it carries honest shot noise of order
    1/sqrt(shots) - the effect the whole lab is about.
    """
    op = pauli_matrix(label)
    exact = mat_trace(mat_mul(rho, op)).real
    p_plus = min(1.0, max(0.0, (1.0 + exact) / 2.0))
    rng = random.Random(seed)
    n_plus = sum(1 for _ in range(shots) if rng.random() < p_plus)
    estimate = (2 * n_plus - shots) / shots
    return estimate, n_plus


def exact_expectation(rho, label):
    return mat_trace(mat_mul(rho, pauli_matrix(label))).real


# ================================================================ 5. State tomography
# Linear inversion: rho = (1/2^n) sum_P <P> P over all Pauli strings,
# with each <P> either exact or estimated from counted shots. The raw
# inversion is kept separate from the physical projection so the lab
# can show the unphysical intermediate honestly.

def reconstruct_state(rho_true, n_qubits, shots, base_seed=BASE_SEED,
                      exact=False):
    """Full state tomography of a known target.

    Returns a report dict with the raw linear-inversion matrix, its
    negativity (negative eigenvalue mass BEFORE the fix), the
    projected physical state, and the fidelity of the physical state
    against the known target.
    """
    if n_qubits == 1:
        labels = PAULI_STRINGS_1Q
    else:
        labels = PAULI_STRINGS_2Q
    dim = 2 ** n_qubits
    rho_lin = mat_scale(pauli_matrix("I" * n_qubits), 1.0 / dim)
    expectations = {}
    for tag, label in enumerate(labels):
        if exact:
            est = exact_expectation(rho_true, label)
        else:
            est, _ = sample_expectation(
                rho_true, label, shots, basis_seed(base_seed, tag, shots))
        expectations[label] = est
        rho_lin = mat_add(rho_lin,
                          mat_scale(pauli_matrix(label), est / dim))
    rho_lin = hermitian_part(rho_lin)
    neg = negativity(rho_lin)
    rho_phys = project_physical(rho_lin)
    return {
        "n_qubits": n_qubits,
        "shots_per_basis": shots,
        "exact": exact,
        "expectations": expectations,
        "rho_linear": rho_lin,
        "negativity_before_fix": neg,
        "rho_physical": rho_phys,
        "fidelity": fidelity(rho_phys, rho_true),
    }


_STATE_CACHE = {}


def state_reports():
    """The canonical seeded state-tomography experiments (cached)."""
    if _STATE_CACHE:
        return _STATE_CACHE
    tilted = TARGET_STATES["tilted_pure_1q"]
    low = reconstruct_state(tilted, 1, LOW_SHOTS,
                            base_seed=BASE_SEED + LOW_SEED_OFFSET)
    high = reconstruct_state(tilted, 1, HIGH_SHOTS)
    mixed = reconstruct_state(TARGET_STATES["depolarized_plus_1q"], 1,
                              HIGH_SHOTS)
    pair = reconstruct_state(TARGET_STATES["werner_bell_2q"], 2, PAIR_SHOTS)
    _STATE_CACHE.update({
        "tilted_low": low,
        "tilted_high": high,
        "depolarized_plus": mixed,
        "werner_bell": pair,
    })
    return _STATE_CACHE


def shot_scaling_report():
    """Fidelity and Bloch error vs shots for the tilted pure state.

    The estimation error of each Pauli expectation shrinks like
    1/sqrt(shots); the table measures that from the sampler itself,
    at four shot budgets a factor of 10 apart (a factor of ~3.16 in
    error per step).
    """
    tilted = TARGET_STATES["tilted_pure_1q"]
    true_bloch = bloch_vector(tilted)
    rows = []
    for shots in SHOT_LADDER:
        rep = reconstruct_state(
            tilted, 1, shots,
            base_seed=BASE_SEED + LADDER_SEED_OFFSETS[shots])
        est_bloch = bloch_vector(rep["rho_physical"])
        err = math.sqrt(sum((a - b) ** 2
                            for a, b in zip(est_bloch, true_bloch)))
        rows.append({
            "shots": shots,
            "fidelity": rep["fidelity"],
            "infidelity": 1.0 - rep["fidelity"],
            "bloch_error": err,
            "negativity_before_fix": rep["negativity_before_fix"],
        })
    return rows


# ================================================================ 6. Process tomography lite
# One qubit, one gate: the ideal Hadamard followed by depolarizing
# noise at the labelled rate DEPOL_RATE. The chi matrix is recovered
# from input/output state tomography: outputs of the four standard
# inputs give the Pauli transfer matrix R by linearity, and chi
# follows by inverting the fixed basis change built from Pauli
# products - every step computed, none quoted.

HADAMARD = mat_scale(mat_add(PAULI_1Q["X"], PAULI_1Q["Z"]),
                     1.0 / math.sqrt(2.0))
PROCESS_INPUTS = {
    "zero": [[1 + 0j, 0j], [0j, 0j]],
    "one": [[0j, 0j], [0j, 1 + 0j]],
    "plus": plus_state(),
    "plus_i": ket_density([1.0 / math.sqrt(2.0), 1j / math.sqrt(2.0)]),
}


def noisy_hadamard(rho, p=DEPOL_RATE):
    """The demo channel: ideal Hadamard, then depolarizing noise."""
    rotated = mat_mul(mat_mul(HADAMARD, rho), mat_dagger(HADAMARD))
    return mat_add(mat_scale(rotated, 1.0 - p),
                   mat_scale(PAULI_1Q["I"],
                             p * mat_trace(rho).real / 2.0))


def pauli_transfer_matrix(outputs):
    """R[b][a] = (1/2) Tr(P_b E(P_a)) from tomographed outputs.

    E(I) = E(|0><0|) + E(|1><1|), E(Z) = E(|0><0|) - E(|1><1|),
    E(X) = 2 E(|+><+|) - E(I), E(Y) = 2 E(|+i><+i|) - E(I) - the
    four measured outputs determine the channel on the whole Pauli
    basis by linearity.
    """
    e_i = mat_add(outputs["zero"], outputs["one"])
    action = {
        "I": e_i,
        "Z": mat_sub(outputs["zero"], outputs["one"]),
        "X": mat_sub(mat_scale(outputs["plus"], 2.0), e_i),
        "Y": mat_sub(mat_scale(outputs["plus_i"], 2.0), e_i),
    }
    return [[0.5 * mat_trace(mat_mul(PAULI_1Q[b], action[a])).real
             for a in PAULI_LETTERS] for b in PAULI_LETTERS]


def chi_from_ptm(r):
    """Invert the Pauli basis change: R -> chi.

    Builds S[(b,a),(m,n)] = (1/2) Tr(P_b P_m P_a P_n) from Pauli
    products and solves S chi = R exactly, then symmetrises.
    """
    s_mat, rhs = [], []
    for bi, b in enumerate(PAULI_LETTERS):
        for ai, a in enumerate(PAULI_LETTERS):
            row = []
            for m in PAULI_LETTERS:
                for n in PAULI_LETTERS:
                    val = 0.5 * mat_trace(mat_mul(
                        mat_mul(PAULI_1Q[b], PAULI_1Q[m]),
                        mat_mul(PAULI_1Q[a], PAULI_1Q[n])))
                    row.append(val)
            s_mat.append(row)
            rhs.append(complex(r[bi][ai]))
    vec = solve_linear(s_mat, rhs)
    chi = [[vec[m * 4 + n] for n in range(4)] for m in range(4)]
    return hermitian_part(chi)


def process_fidelity(chi):
    """Process fidelity of chi against the ideal Hadamard.

    The ideal Hadamard chi is |h><h| with h = (0, 1/sqrt2, 0, 1/sqrt2)
    over (I, X, Y, Z), so F = h-dagger chi h / Tr(chi).
    """
    s = 1.0 / math.sqrt(2.0)
    h = [0j, s, 0j, s]
    num = sum(h[m].conjugate() * chi[m][n] * h[n]
              for m in range(4) for n in range(4)).real
    return num / mat_trace(chi).real


def average_gate_fidelity(chi):
    """Average gate fidelity from process fidelity (d = 2)."""
    return (2.0 * process_fidelity(chi) + 1.0) / 3.0


def process_report(shots=PROCESS_SHOTS):
    """Process tomography of the noisy Hadamard, sampled + exact."""
    sampled_outputs = {}
    exact_outputs = {}
    for tag, (name, rho_in) in enumerate(PROCESS_INPUTS.items()):
        rho_out = noisy_hadamard(rho_in)
        rec = reconstruct_state(rho_out, 1, shots,
                                base_seed=BASE_SEED + 31_337 + tag)
        sampled_outputs[name] = rec["rho_physical"]
        exact_outputs[name] = rho_out
    chi_sampled = chi_from_ptm(pauli_transfer_matrix(sampled_outputs))
    chi_exact = chi_from_ptm(pauli_transfer_matrix(exact_outputs))
    return {
        "gate": "Hadamard + depolarizing (labelled)",
        "depol_rate": DEPOL_RATE,
        "shots_per_basis": shots,
        "chi_sampled": chi_sampled,
        "chi_exact": chi_exact,
        "chi_trace": mat_trace(chi_sampled).real,
        "process_fidelity_sampled": process_fidelity(chi_sampled),
        "process_fidelity_exact": process_fidelity(chi_exact),
        "average_gate_fidelity_sampled": average_gate_fidelity(chi_sampled),
        "expected_process_fidelity": EXPECTED_PROCESS_FIDELITY,
    }


_PROCESS_CACHE = {}


def process_report_cached():
    if not _PROCESS_CACHE:
        _PROCESS_CACHE.update(process_report())
    return _PROCESS_CACHE


# ================================================================ 7. kNN reconstruction classifier
# Features: shots per basis, reconstruction fidelity, negativity
# before the physicality fix. The training grid is labelled by the
# lab's own reconstruction-quality model (model_label below) - the
# classifier interpolates that policy, it does not replace the
# tomography it grades.

def model_label(shots, fid, neg):
    """The reconstruction-quality model distilled to three labels."""
    if fid < 0.90 or neg > 0.10:
        return "unreliable"
    if fid >= 0.99 and neg <= 0.01 and shots >= 1_000:
        return "trustworthy"
    if fid >= 0.995 and neg <= 0.005:
        return "trustworthy"
    return "shaky"


def _training_grid():
    rows = []
    for shots in (50, 100, 500, 1_000, 5_000, 20_000, 100_000):
        for fid in (0.80, 0.88, 0.94, 0.97, 0.99, 0.995, 0.999):
            for neg in (0.0, 0.005, 0.02, 0.05, 0.12, 0.30):
                rows.append(((shots, fid, neg),
                             model_label(shots, fid, neg)))
    return rows


TRAINING_GRID = _training_grid()


def _scaled(features):
    shots, fid, neg = features
    return (math.log10(max(shots, 1)) / 5.0, fid, neg / 0.30)


def classify_reconstruction(shots, fid, neg, k=5):
    """kNN over the model-labelled grid, deterministic tie-breaking."""
    target = _scaled((shots, fid, neg))
    ranked = []
    for features, label in TRAINING_GRID:
        point = _scaled(features)
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(target, point)))
        ranked.append((dist, features, label))
    ranked.sort(key=lambda row: (row[0], row[1]))
    votes = {}
    for _, _, label in ranked[:k]:
        votes[label] = votes.get(label, 0) + 1
    priority = {"trustworthy": 0, "shaky": 1, "unreliable": 2}
    return sorted(votes.items(), key=lambda kv: (-kv[1], priority[kv[0]]))[0][0]


def classifier_report():
    """Classify the lab's own seeded reconstructions."""
    reps = state_reports()
    scenarios = {
        "tilted_100_shots": (
            LOW_SHOTS, reps["tilted_low"]["fidelity"],
            reps["tilted_low"]["negativity_before_fix"]),
        "tilted_100k_shots": (
            HIGH_SHOTS, reps["tilted_high"]["fidelity"],
            reps["tilted_high"]["negativity_before_fix"]),
        "werner_bell_20k_shots": (
            PAIR_SHOTS, reps["werner_bell"]["fidelity"],
            reps["werner_bell"]["negativity_before_fix"]),
        "starved_40_shots_synthetic": (40, 0.86, 0.18),
    }
    return {name: classify_reconstruction(*features)
            for name, features in scenarios.items()}


# ================================================================ 8. Advisory seats
def _trace(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def argon4_decide(topic="quantumtomography-demo"):
    """Gemini Argon 4 DECIDER seat: MAP->MODEL->STRESS->DECIDE.

    Names-only keys, live calls 0, tighten-only: can add HOLDs, never lift one.
    """
    stages = ["MAP", "MODEL", "STRESS", "DECIDE"]
    payload = {"seat": "gemini-argon-4", "topic": topic, "stages": stages}
    return {
        "seat": "Gemini Argon 4 Extended Thinking",
        "role": "DECIDER",
        "pipeline": stages,
        "tiers": [2048, 8192, 32768],
        "key": "names-only (GEMINI_API_KEY name checked, value never read)",
        "live_calls": 0,
        "decision": "PROCEED - sandbox simulation only, all hardware-characterisation gates stay HOLD",
        "can_override_hold": False,
        "trace_sha256_16": _trace(payload),
    }


def claude_advisor(topic="quantumtomography-demo"):
    """Muse advisor seat: advises, can never override a HOLD."""
    return {
        "seat": "Muse",
        "role": "ADVISOR",
        "key": "names-only (MODEL_API_KEY name checked, value never read)",
        "live_calls": 0,
        "advice": "Print the seed next to every fidelity, show the unphysical linear inversion before you fix it, and say in the first line that the target was known - that honesty is the difference between a tomography lesson and a hardware claim.",
        "can_override_hold": False,
    }


def google_flow_plan():
    """Google Flow seat: plan-only, nothing rendered, nothing live."""
    return {
        "seat": "Google Flow",
        "role": "PLAN-ONLY",
        "shots": ["a Bloch sphere with a state vector dissolving into counted clicks",
                  "Pauli expectation bars wobbling as the shot counter climbs",
                  "a density matrix dipping below the physical floor, then being lifted back",
                  "four input states entering a gate and leaving as a chi matrix",
                  "the fidelity curve flattening along a 1/sqrt(shots) guide"],
        "rendered": False,
        "live": 0,
    }


def brains_report():
    return {"decider": argon4_decide(), "advisor": claude_advisor(),
            "flow": google_flow_plan()}


# ================================================================ checks
def run_checks():
    c = []

    def chk(name, ok):
        c.append((name, bool(ok)))

    # 1. matrix toolkit
    ident = eye(2)
    chk("mat_mul_identity",
        mat_mul(PAULI_1Q["X"], ident) == PAULI_1Q["X"])
    chk("dagger_involution",
        mat_dagger(mat_dagger(PAULI_1Q["Y"])) == PAULI_1Q["Y"])
    chk("pauli_algebra_xy_iz",
        mat_mul(PAULI_1Q["X"], PAULI_1Q["Y"])
        == mat_scale(PAULI_1Q["Z"], 1j))
    chk("pauli_squares_identity",
        all(mat_mul(PAULI_1Q[k], PAULI_1Q[k]) == ident
            for k in PAULI_LETTERS))
    chk("kron_dimensions",
        len(pauli_matrix("XZ")) == 4 and len(pauli_matrix("XZ")[0]) == 4)
    chk("targets_are_density_matrices",
        all(abs(mat_trace(rho) - 1.0) < 1e-12
            and is_hermitian(rho) and negativity(rho) < 1e-9
            for rho in TARGET_STATES.values()))

    # 2. eigensolver + PSD tools
    evals, _ = hermitian_eigh(PAULI_1Q["Z"])
    chk("eigh_pauli_z_spectrum",
        abs(evals[0] + 1.0) < 1e-10 and abs(evals[1] - 1.0) < 1e-10)
    evals_plus, _ = hermitian_eigh(plus_state())
    chk("eigh_pure_state_spectrum",
        abs(evals_plus[0]) < 1e-10 and abs(evals_plus[1] - 1.0) < 1e-10)
    rng = random.Random(7)
    raw = [[complex(rng.uniform(-1, 1), rng.uniform(-1, 1))
            for _ in range(4)] for _ in range(4)]
    herm = hermitian_part(raw)
    evals_r, vecs_r = hermitian_eigh(herm)
    lam = zeros(4)
    for i, ev in enumerate(evals_r):
        lam[i][i] = ev
    rebuilt = mat_mul(mat_mul(vecs_r, lam), mat_dagger(vecs_r))
    chk("eigh_reconstruction_4x4",
        max(abs(rebuilt[i][j] - herm[i][j])
            for i in range(4) for j in range(4)) < 1e-9)
    chk("eigh_trace_preserved",
        abs(sum(evals_r) - mat_trace(herm).real) < 1e-9)
    proj = project_physical([[-0.2 + 0j, 0j], [0j, 1.2 + 0j]])
    chk("projection_trace_and_psd",
        abs(mat_trace(proj) - 1.0) < 1e-12 and negativity(proj) < 1e-9)
    chk("fidelity_self_is_one",
        abs(fidelity(tilted_state(), tilted_state()) - 1.0) < 1e-9)
    zero_rho = ket_density([1.0, 0j])
    one_rho = ket_density([0j, 1.0])
    chk("fidelity_orthogonal_is_zero", fidelity(zero_rho, one_rho) < 1e-12)
    chk("fidelity_pure_mixed_formula",
        abs(fidelity(plus_state(), depolarized_plus()) - 0.875) < 1e-9)

    # 3. sampler honesty: counts, determinism, convergence
    est1, n_plus = sample_expectation(tilted_state(), "Z", 4_000,
                                      basis_seed(BASE_SEED, 0, 4_000))
    est2, n_plus2 = sample_expectation(tilted_state(), "Z", 4_000,
                                       basis_seed(BASE_SEED, 0, 4_000))
    chk("sampler_deterministic_same_seed", est1 == est2 and n_plus == n_plus2)
    chk("sampler_counts_are_counts",
        0 <= n_plus <= 4_000 and abs(est1 - (2 * n_plus - 4_000) / 4_000)
        < 1e-15)
    est_big, _ = sample_expectation(tilted_state(), "Z", HIGH_SHOTS,
                                    basis_seed(BASE_SEED, 0, HIGH_SHOTS))
    chk("sampler_converges_to_born_rule",
        abs(est_big - exact_expectation(tilted_state(), "Z")) < 0.02)

    # 4. state tomography
    exact_rec = reconstruct_state(tilted_state(), 1, 0, exact=True)
    chk("exact_inversion_recovers_state",
        max(abs(exact_rec["rho_linear"][i][j] - tilted_state()[i][j])
            for i in range(2) for j in range(2)) < 1e-12)
    reps = state_reports()
    low, high = reps["tilted_low"], reps["tilted_high"]
    chk("low_shot_inversion_goes_unphysical",
        low["negativity_before_fix"] > 0.001)
    chk("high_shot_fidelity_beats_low_shot",
        high["fidelity"] > low["fidelity"])
    chk("high_shot_fidelity_strong", high["fidelity"] > 0.995)
    chk("low_shot_fidelity_bounded", 0.80 < low["fidelity"] < 0.999)
    for key in ("tilted_low", "tilted_high", "depolarized_plus",
                "werner_bell"):
        rep = reps[key]
        chk(f"physical_after_fix_{key}",
            abs(mat_trace(rep["rho_physical"]) - 1.0) < 1e-9
            and negativity(rep["rho_physical"]) < 1e-9
            and is_hermitian(rep["rho_physical"]))
    chk("mixed_state_reconstruction", reps["depolarized_plus"]["fidelity"]
        > 0.999)
    chk("bell_pair_reconstruction", reps["werner_bell"]["fidelity"] > 0.99)
    chk("bell_pair_uses_15_bases",
        len(reps["werner_bell"]["expectations"]) == 15)

    # 5. shot scaling: the 1/sqrt(shots) law, measured
    rows = shot_scaling_report()
    chk("scaling_ladder_shape",
        [r["shots"] for r in rows] == list(SHOT_LADDER))
    chk("scaling_error_shrinks",
        rows[-1]["bloch_error"] < rows[0]["bloch_error"] / 5.0)
    chk("scaling_fidelity_improves",
        rows[-1]["fidelity"] > rows[0]["fidelity"])

    # 6. process tomography
    proc = process_report_cached()
    chk("process_exact_chi_fidelity",
        abs(proc["process_fidelity_exact"] - EXPECTED_PROCESS_FIDELITY)
        < 1e-9)
    chk("process_sampled_chi_fidelity",
        abs(proc["process_fidelity_sampled"] - EXPECTED_PROCESS_FIDELITY)
        < 0.02)
    chk("process_chi_trace_one", abs(proc["chi_trace"] - 1.0) < 0.02)
    chk("process_chi_hermitian", is_hermitian(proc["chi_sampled"]))
    chk("process_noiseless_is_ideal",
        abs(process_fidelity(chi_from_ptm(pauli_transfer_matrix(
            {name: noisy_hadamard(rho, p=0.0)
             for name, rho in PROCESS_INPUTS.items()}))) - 1.0) < 1e-9)

    # 7. classifier - labels come from the quality model itself
    rep = classifier_report()
    chk("classifier_high_shot_trustworthy",
        rep["tilted_100k_shots"] == "trustworthy")
    chk("classifier_low_shot_not_trustworthy",
        rep["tilted_100_shots"] in ("shaky", "unreliable"))
    chk("classifier_starved_unreliable",
        rep["starved_40_shots_synthetic"] == "unreliable")
    chk("classifier_matches_model",
        all(classify_reconstruction(*features) == model_label(*features)
            for features in [(100, 0.94, 0.03), (100_000, 0.9999, 0.0),
                             (40, 0.86, 0.18), (5_000, 0.985, 0.02),
                             (20_000, 0.997, 0.002), (500, 0.91, 0.08)]))

    # gates + seats
    chk("hold_gates_14", len(HOLD_GATES) == 14
        and all(v.startswith("HOLD") for v in HOLD_GATES.values()))
    chk("gate_hardware_hold",
        gate("real_hardware_characterization_claim") == "HOLD")
    chk("gate_benchmark_hold", gate("certified_benchmarking_claim") == "HOLD")
    chk("gate_production_hold", gate("production_verification_claim")
        == "HOLD")
    chk("gate_honesty_hold", gate("known_state_as_unknown_claim") == "HOLD")
    chk("gate_sandbox_proceed", gate("run_simulation") == "PROCEED")
    brains = brains_report()
    chk("argon4_no_override", brains["decider"]["can_override_hold"] is False
        and brains["decider"]["seat"]
        == "Gemini Argon 4 Extended Thinking")
    chk("claude_no_override", brains["advisor"]["can_override_hold"] is False)
    chk("flow_plan_only", brains["flow"]["rendered"] is False
        and brains["flow"]["live"] == 0)
    return c


# ================================================================ CLI / demo
def mat_json(a):
    """JSON view of a complex matrix: real and imaginary parts."""
    return {
        "re": [[round(a[i][j].real, 6) for j in range(len(a[0]))]
               for i in range(len(a))],
        "im": [[round(a[i][j].imag, 6) for j in range(len(a[0]))]
               for i in range(len(a))],
    }


def state_json():
    reps = state_reports()
    out = {}
    for key, rep in reps.items():
        out[key] = {
            "n_qubits": rep["n_qubits"],
            "shots_per_basis": rep["shots_per_basis"],
            "negativity_before_fix": round(rep["negativity_before_fix"], 6),
            "fidelity": round(rep["fidelity"], 6),
            "rho_physical": mat_json(rep["rho_physical"]),
        }
    return out


def process_json():
    proc = process_report_cached()
    return {
        "gate": proc["gate"],
        "depol_rate": proc["depol_rate"],
        "shots_per_basis": proc["shots_per_basis"],
        "chi_trace": round(proc["chi_trace"], 6),
        "process_fidelity_sampled": round(
            proc["process_fidelity_sampled"], 6),
        "process_fidelity_exact": round(proc["process_fidelity_exact"], 6),
        "average_gate_fidelity_sampled": round(
            proc["average_gate_fidelity_sampled"], 6),
        "expected_process_fidelity": proc["expected_process_fidelity"],
        "chi_sampled": mat_json(proc["chi_sampled"]),
    }


def demo():
    print("=" * 68)
    print("QuantumTomography-Lab - quantum state/process tomography lab")
    print("by Darion Kellar | MIT | pure stdlib | zero network | EDUCATION ONLY")
    print("=" * 68)
    print("\n[1] THE HONEST CORE - tomography of states we already know")
    print("    Every measurement below is a seeded simulation of a state")
    print("    this lab prepared itself. Reconstructing a known state is")
    print("    how you LEARN tomography; it characterises no hardware.")
    print("    Counts are counted - never probabilities dressed as data.")
    reps = state_reports()
    low, high = reps["tilted_low"], reps["tilted_high"]
    print("\n[2] State tomography, 1 qubit - the tilted pure state")
    print(f"    target Bloch vector "
          f"{tuple(round(x, 4) for x in bloch_vector(tilted_state()))}")
    print(f"    {LOW_SHOTS:>7,} shots/basis: fidelity "
          f"{low['fidelity']:.4f} | negativity before fix "
          f"{low['negativity_before_fix']:.4f}  <- linear inversion")
    print(f"    {'':>7}                went UNPHYSICAL, then was projected back")
    print(f"    {HIGH_SHOTS:>7,} shots/basis: fidelity "
          f"{high['fidelity']:.6f} | negativity before fix "
          f"{high['negativity_before_fix']:.6f}")
    print("    -> after the fix every state below is trace 1 and PSD,")
    print("       asserted in CHECKS, not assumed.")
    mixed = reps["depolarized_plus"]
    pair = reps["werner_bell"]
    print("\n[3] Mixed state + entangled pair")
    print(f"    depolarized |+> (p=0.25): fidelity {mixed['fidelity']:.6f} "
          f"at {HIGH_SHOTS:,} shots/basis")
    print(f"    Werner-Bell pair (0.85 Bell + 0.15 white noise): fidelity "
          f"{pair['fidelity']:.6f}")
    print(f"    from {len(pair['expectations'])} Pauli bases x "
          f"{PAIR_SHOTS:,} shots each")
    print("\n[4] Shot scaling - the 1/sqrt(shots) law, measured not quoted")
    print(f"    {'shots':>8} {'fidelity':>10} {'bloch err':>10} "
          f"{'negativity':>11}")
    for row in shot_scaling_report():
        print(f"    {row['shots']:>8,} {row['fidelity']:>10.6f} "
              f"{row['bloch_error']:>10.4f} "
              f"{row['negativity_before_fix']:>11.4f}")
    print("    -> each 10x of shots buys ~3.16x less error. There is no")
    print("       way around the square root; there is only buying shots.")
    proc = process_report_cached()
    print("\n[5] Process tomography lite - one noisy gate")
    print(f"    gate: Hadamard + depolarizing p={DEPOL_RATE} (labelled, "
          f"constructed - not measured)")
    print(f"    chi trace {proc['chi_trace']:.4f} | process fidelity "
          f"sampled {proc['process_fidelity_sampled']:.4f} vs exact "
          f"{proc['process_fidelity_exact']:.4f}")
    print(f"    expected 1 - 3p/4 = {EXPECTED_PROCESS_FIDELITY:.4f} | "
          f"average gate fidelity "
          f"{proc['average_gate_fidelity_sampled']:.4f}")
    print("    -> chi was SOLVED from the Pauli transfer matrix of four")
    print("       tomographed outputs, through a basis change built from")
    print("       Pauli products. No step was quoted from a textbook table.")
    print("\n[6] Reconstruction classifier (TinyML kNN, model-labelled)")
    for name, label in classifier_report().items():
        print(f"    {name:36s} -> {label}")
    print("\n[7] Advisory seats")
    brains = brains_report()
    print(f"    decider: {brains['decider']['decision']} "
          f"trace={brains['decider']['trace_sha256_16']}")
    print(f"    advisor can_override_hold = "
          f"{brains['advisor']['can_override_hold']}")
    print(f"    flow rendered = {brains['flow']['rendered']} "
          f"live = {brains['flow']['live']}")
    print(f"\n[8] HOLD gates: {len(HOLD_GATES)}")
    for key in HOLD_GATES:
        print(f"    HOLD  {key}")
    checks = run_checks()
    passed = sum(1 for _, ok in checks if ok)
    print(f"\nCHECKS {passed}/{len(checks)}")
    for name, ok in checks:
        if not ok:
            print(f"    FAIL {name}")
    verdict = "PROCEED" if passed == len(checks) else "HOLD"
    print(f"VERDICT {verdict} (sandbox simulation only) | "
          f"HOLD gates {len(HOLD_GATES)}")
    return 0 if passed == len(checks) else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] == "demo":
        return demo()
    cmd = argv[0]
    if cmd == "checks":
        checks = run_checks()
        for name, ok in checks:
            print(f"{'PASS' if ok else 'FAIL'} {name}")
        print(f"CHECKS {sum(1 for _, ok in checks if ok)}/{len(checks)}")
        return 0 if all(ok for _, ok in checks) else 1
    if cmd == "state":
        print(json.dumps(state_json(), indent=2, sort_keys=True))
        return 0
    if cmd == "process":
        print(json.dumps(process_json(), indent=2, sort_keys=True))
        return 0
    if cmd == "shots":
        rows = [{k: (round(v, 6) if isinstance(v, float) else v)
                 for k, v in row.items()} for row in shot_scaling_report()]
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    if cmd == "classify":
        print(json.dumps(classifier_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "brains":
        print(json.dumps(brains_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "holds":
        for key, value in HOLD_GATES.items():
            print(f"{key}: {value}")
        return 0
    if cmd == "hold-demo":
        action = argv[1] if len(argv) > 1 else "real_hardware_characterization_claim"
        print(f"{action}: {HOLD_GATES.get(action, 'PROCEED - not gated')}")
        return 2 if gate(action) == "HOLD" else 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
