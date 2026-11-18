"""Tests for QuantumTomography-Lab: matrices, eigensolver, sampler,
state/process tomography, classifier, gates, seats, determinism."""
import math
import subprocess
import sys
from pathlib import Path

import pytest

import main as m

ROOT = Path(__file__).resolve().parent.parent


# ---- complex matrix toolkit
def test_mat_mul_identity():
    assert m.mat_mul(m.PAULI_1Q["X"], m.eye(2)) == m.PAULI_1Q["X"]


def test_dagger_involution():
    assert m.mat_dagger(m.mat_dagger(m.PAULI_1Q["Y"])) == m.PAULI_1Q["Y"]


def test_pauli_algebra():
    assert m.mat_mul(m.PAULI_1Q["X"], m.PAULI_1Q["Y"]) == \
        m.mat_scale(m.PAULI_1Q["Z"], 1j)
    for k in m.PAULI_LETTERS:
        assert m.mat_mul(m.PAULI_1Q[k], m.PAULI_1Q[k]) == m.eye(2)


def test_kron_dimensions():
    op = m.pauli_matrix("XZ")
    assert len(op) == 4 and len(op[0]) == 4


def test_targets_are_density_matrices():
    for rho in m.TARGET_STATES.values():
        assert abs(m.mat_trace(rho) - 1.0) < 1e-12
        assert m.is_hermitian(rho)
        assert m.negativity(rho) < 1e-9


# ---- Jacobi eigensolver + PSD tools
def test_eigh_pauli_z():
    evals, _ = m.hermitian_eigh(m.PAULI_1Q["Z"])
    assert abs(evals[0] + 1.0) < 1e-10 and abs(evals[1] - 1.0) < 1e-10


def test_eigh_pure_state():
    evals, _ = m.hermitian_eigh(m.plus_state())
    assert abs(evals[0]) < 1e-10 and abs(evals[1] - 1.0) < 1e-10


def test_eigh_reconstruction_random_hermitian():
    import random
    rng = random.Random(7)
    raw = [[complex(rng.uniform(-1, 1), rng.uniform(-1, 1))
            for _ in range(4)] for _ in range(4)]
    herm = m.hermitian_part(raw)
    evals, vecs = m.hermitian_eigh(herm)
    lam = m.zeros(4)
    for i, ev in enumerate(evals):
        lam[i][i] = ev
    rebuilt = m.mat_mul(m.mat_mul(vecs, lam), m.mat_dagger(vecs))
    err = max(abs(rebuilt[i][j] - herm[i][j]) for i in range(4)
              for j in range(4))
    assert err < 1e-9
    assert abs(sum(evals) - m.mat_trace(herm).real) < 1e-9


def test_projection_trace_and_psd():
    proj = m.project_physical([[-0.2 + 0j, 0j], [0j, 1.2 + 0j]])
    assert abs(m.mat_trace(proj) - 1.0) < 1e-12
    assert m.negativity(proj) < 1e-9


def test_fidelity_self_and_orthogonal():
    assert abs(m.fidelity(m.tilted_state(), m.tilted_state()) - 1.0) < 1e-9
    zero_rho = m.ket_density([1.0, 0j])
    one_rho = m.ket_density([0j, 1.0])
    assert m.fidelity(zero_rho, one_rho) < 1e-12


def test_fidelity_pure_vs_depolarized():
    # F(|+>, 0.75|+><+| + 0.25 I/2) = <+|rho|+> = 0.875 exactly.
    assert abs(m.fidelity(m.plus_state(), m.depolarized_plus())
               - 0.875) < 1e-9


# ---- sampler honesty
def test_sampler_deterministic():
    args = (m.tilted_state(), "Z", 4_000, m.basis_seed(m.BASE_SEED, 0, 4_000))
    assert m.sample_expectation(*args) == m.sample_expectation(*args)


def test_sampler_counts_consistent():
    est, n_plus = m.sample_expectation(m.tilted_state(), "X", 2_000,
                                       m.basis_seed(m.BASE_SEED, 1, 2_000))
    assert 0 <= n_plus <= 2_000
    assert abs(est - (2 * n_plus - 2_000) / 2_000) < 1e-15


def test_sampler_converges():
    est, _ = m.sample_expectation(m.tilted_state(), "Z", m.HIGH_SHOTS,
                                  m.basis_seed(m.BASE_SEED, 0, m.HIGH_SHOTS))
    assert abs(est - m.exact_expectation(m.tilted_state(), "Z")) < 0.02


# ---- state tomography
def test_exact_inversion_recovers_state():
    rep = m.reconstruct_state(m.tilted_state(), 1, 0, exact=True)
    target = m.tilted_state()
    err = max(abs(rep["rho_linear"][i][j] - target[i][j])
              for i in range(2) for j in range(2))
    assert err < 1e-12


def test_low_shot_inversion_unphysical():
    rep = m.state_reports()["tilted_low"]
    assert rep["negativity_before_fix"] > 0.001


def test_high_shot_beats_low_shot():
    reps = m.state_reports()
    assert reps["tilted_high"]["fidelity"] > reps["tilted_low"]["fidelity"]
    assert reps["tilted_high"]["fidelity"] > 0.995
    assert 0.80 < reps["tilted_low"]["fidelity"] < 0.999


def test_physical_after_fix_all_states():
    for rep in m.state_reports().values():
        rho = rep["rho_physical"]
        assert abs(m.mat_trace(rho) - 1.0) < 1e-9
        assert m.negativity(rho) < 1e-9
        assert m.is_hermitian(rho)


def test_mixed_and_bell_reconstruction():
    reps = m.state_reports()
    assert reps["depolarized_plus"]["fidelity"] > 0.999
    assert reps["werner_bell"]["fidelity"] > 0.99
    assert len(reps["werner_bell"]["expectations"]) == 15


# ---- shot scaling
def test_shot_scaling_law():
    rows = m.shot_scaling_report()
    assert [r["shots"] for r in rows] == list(m.SHOT_LADDER)
    assert rows[-1]["bloch_error"] < rows[0]["bloch_error"] / 5.0
    assert rows[-1]["fidelity"] > rows[0]["fidelity"]


# ---- process tomography
def test_process_exact_fidelity_formula():
    proc = m.process_report_cached()
    assert abs(proc["process_fidelity_exact"]
               - m.EXPECTED_PROCESS_FIDELITY) < 1e-9
    # 1 - 3p/4 at p = 0.10 is 0.925: the I/4 chi of full
    # depolarization still overlaps the ideal gate by 1/4.
    assert abs(m.EXPECTED_PROCESS_FIDELITY - 0.925) < 1e-12


def test_process_sampled_fidelity():
    proc = m.process_report_cached()
    assert abs(proc["process_fidelity_sampled"]
               - m.EXPECTED_PROCESS_FIDELITY) < 0.02
    assert abs(proc["chi_trace"] - 1.0) < 0.02
    assert m.is_hermitian(proc["chi_sampled"])


def test_process_noiseless_is_ideal():
    outputs = {name: m.noisy_hadamard(rho, p=0.0)
               for name, rho in m.PROCESS_INPUTS.items()}
    chi = m.chi_from_ptm(m.pauli_transfer_matrix(outputs))
    assert abs(m.process_fidelity(chi) - 1.0) < 1e-9


def test_average_gate_fidelity_relation():
    proc = m.process_report_cached()
    expect = (2.0 * proc["process_fidelity_sampled"] + 1.0) / 3.0
    assert abs(proc["average_gate_fidelity_sampled"] - expect) < 1e-12


# ---- classifier
def test_classifier_scenarios():
    rep = m.classifier_report()
    assert rep["tilted_100k_shots"] == "trustworthy"
    assert rep["tilted_100_shots"] in ("shaky", "unreliable")
    assert rep["starved_40_shots_synthetic"] == "unreliable"


def test_classifier_matches_model():
    for features in [(100, 0.94, 0.03), (100_000, 0.9999, 0.0),
                     (40, 0.86, 0.18), (5_000, 0.985, 0.02),
                     (20_000, 0.997, 0.002), (500, 0.91, 0.08)]:
        assert m.classify_reconstruction(*features) == \
            m.model_label(*features)


# ---- HOLD gates + council seats
def test_hold_gates_14():
    assert len(m.HOLD_GATES) == 14
    assert all(v.startswith("HOLD") for v in m.HOLD_GATES.values())


def test_specific_gates_hold():
    for action in ("real_hardware_characterization_claim",
                   "certified_benchmarking_claim",
                   "production_verification_claim",
                   "gate_error_rate_certification",
                   "known_state_as_unknown_claim"):
        assert m.gate(action) == "HOLD"
    assert m.gate("run_simulation") == "PROCEED"


def test_council_seats_cannot_override():
    brains = m.brains_report()
    assert brains["decider"]["seat"] == "Gemini Argon 4 Extended Thinking"
    assert brains["decider"]["can_override_hold"] is False
    assert brains["decider"]["live_calls"] == 0
    assert len(brains["decider"]["trace_sha256_16"]) == 16
    assert brains["advisor"]["can_override_hold"] is False
    assert brains["flow"]["rendered"] is False
    assert brains["flow"]["live"] == 0


# ---- CLI behaviour
def test_checks_all_pass():
    checks = m.run_checks()
    assert all(ok for _, ok in checks), \
        [name for name, ok in checks if not ok]


def test_hold_demo_exits_2():
    proc = subprocess.run(
        [sys.executable, "main.py", "hold-demo"],
        cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 2
    assert "HOLD" in proc.stdout


def test_demo_deterministic_byte_identical():
    outs = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "main.py"], cwd=ROOT,
            capture_output=True, text=True)
        assert proc.returncode == 0
        outs.append(proc.stdout)
    assert outs[0] == outs[1]
