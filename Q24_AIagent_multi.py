#!/usr/bin/env python3
"""
Q24 AI Agents — tehnika: Quantum Multi-Tool Agent (QMTA)
(čisto kvantno: heterogeni kvantni eksperti + data-driven router preko aux-registra).

Koncept (kvantni analog AI Agent-a „choose tool → apply tool → observe"):
  1) 4 HETEROGENA kvantna eksperta (strukturno različita sub-kola) — svaki „tool"
     iz |0⟩ proizvodi svoj izlazni state |ψ_t⟩:
        Tool A (HOT)      : StatePreparation(amp_hot)  — čist amp-encoding CELOG CSV-a.
        Tool B (PAIR-PQC) : StatePreparation(amp_hot) + Ry-sloj(pair_angles) + ring-CNOT
                           — PQC refinement nad HOT stanjem (strukturno proširenje).
        Tool C (COLD)     : StatePreparation(amp_cold) — inverzni freq signal (1/(f+ε)).
        Tool D (COND/TOP-1): StatePreparation(amp_conditional) — freq nad redovima
                             koji sadrže TOP-1 broj iz CELOG CSV-a.
  2) Router (aux, m=2 qubit-a) — NON-UNIFORM StatePreparation sa DATA-DEPENDENT težinama
     izvedenim iz CSV statistika:
        w_A — HOT peakedness     : max(f_hot) / mean(f_hot)
        w_B — PAIR signal        : max_{i≠j} P[i,j] / mean(P[i≠j])
        w_C — RECENT/OLD ratio   : ‖freq_recent‖ / ‖freq_old‖
        w_D — CONDITIONAL strength: max(f_cond) / mean(f_cond)
        Router state: |R⟩ = Σ_t √(w_t/Σw) · |t⟩_tool  (NE Hadamard-uniform).
  3) Dispatch: za svako t, multi-ctrl (ctrl_state=t) Tool-t sub-kolo na state registar.
  4) Ukupno stanje: |Ψ⟩ = Σ_t √(w_t/Σw) · |t⟩_tool ⊗ |ψ_t⟩_state.
  5) Marginalizacija tool-a → p = Σ_t (w_t/Σw)|ψ_t|² → bias_39 → TOP-7 = NEXT.

Razlika u odnosu na slične fajlove:
  Q20 (QPTM): HOMOGENA template-a (sve SP), UNIFORMNI Hadamard router.
  Q22 (QRAG): homogeni documents + similarity-weights (retrieval), NE multi-tool dispatch.
  Q23 (QVDB): jedan tip embedding-a u entanglovanom DB-u, NN-lookup.
  Q21 (QCoT): sekvencijalni lanac istog tipa unitara (Ry+CNOT po feature-u).
  QMTA:       paralelni dispatch HETEROGENIH ekspert-kola kroz NON-UNIFORM aux router
              sa DATA-DEPENDENT težinama.

Sve deterministički: seed=39; sve tool-izlaze i težine izvedene iz CELOG CSV-a.
Deterministička grid-optimizacija (nq, K) po cos(bias_39, freq_csv).

Okruženje: Python 3.11.13, qiskit 1.4.4, qiskit-machine-learning 0.8.3, macOS M1 (vidi README.md).
"""

from __future__ import annotations

import csv
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except ImportError:
    pass

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector

# =========================
# Seed
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
try:
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = SEED
except ImportError:
    pass

# =========================
# Konfiguracija
# =========================
CSV_PATH = Path("/Users/4c/Desktop/GHQ/data/loto7hh_4600_k31.csv")
N_NUMBERS = 7
N_MAX = 39

GRID_NQ = (5, 6)
GRID_K = (50, 200, 500, 1000, 2000)

NUM_TOOLS = 4
M_TOOL = 2
EPS_COLD = 1e-3


# =========================
# CSV
# =========================
def load_rows(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
        for row in r:
            if not row or row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(N_NUMBERS)])
    return np.array(rows, dtype=int)


def freq_vector(H: np.ndarray) -> np.ndarray:
    c = np.zeros(N_MAX, dtype=np.float64)
    for v in H.ravel():
        if 1 <= v <= N_MAX:
            c[int(v) - 1] += 1.0
    return c


def amp_from_freq(f: np.ndarray, nq: int) -> np.ndarray:
    dim = 2 ** nq
    edges = np.linspace(0, N_MAX, dim + 1, dtype=int)
    amp = np.array(
        [float(f[edges[i] : edges[i + 1]].mean()) if edges[i + 1] > edges[i] else 0.0 for i in range(dim)],
        dtype=np.float64,
    )
    amp = np.maximum(amp, 0.0)
    n2 = float(np.linalg.norm(amp))
    if n2 < 1e-18:
        amp = np.ones(dim, dtype=np.float64) / np.sqrt(dim)
    else:
        amp = amp / n2
    return amp


# =========================
# Pomoćni feature-i (iz CELOG CSV-a)
# =========================
def pair_matrix(H: np.ndarray) -> np.ndarray:
    P = np.zeros((N_MAX, N_MAX), dtype=np.float64)
    for row in H:
        for a in row:
            for b in row:
                if a != b and 1 <= a <= N_MAX and 1 <= b <= N_MAX:
                    P[a - 1, b - 1] += 1.0
    return P


def feature_pair(H: np.ndarray) -> np.ndarray:
    P = pair_matrix(H)
    f = np.zeros(N_MAX, dtype=np.float64)
    for i in range(N_MAX):
        row = P[i].copy()
        row[i] = 0.0
        f[i] = float(row.max()) if row.size else 0.0
    return f


def feature_to_angles(f: np.ndarray, nq: int) -> np.ndarray:
    m = float(f.max())
    if m < 1e-18:
        return np.zeros(nq, dtype=np.float64)
    edges = np.linspace(0, N_MAX, nq + 1, dtype=int)
    angles = np.zeros(nq, dtype=np.float64)
    for k in range(nq):
        lo, hi = int(edges[k]), int(edges[k + 1])
        seg = f[lo:hi] if hi > lo else np.array([0.0])
        angles[k] = float(np.pi * (float(seg.mean()) / m))
    return angles


def inverse_freq(f: np.ndarray, eps: float = EPS_COLD) -> np.ndarray:
    return 1.0 / (f + eps)


def conditional_top1_freq(H: np.ndarray) -> np.ndarray:
    f = freq_vector(H)
    top1 = int(np.argmax(f)) + 1
    mask = np.any(H == top1, axis=1)
    sel = H[mask]
    if sel.shape[0] < 1:
        return f
    return freq_vector(sel)


# =========================
# Router: data-dependent težine w_t, sqrt-normalizovan aux-vector
# =========================
def router_weights(H: np.ndarray, K: int) -> np.ndarray:
    n = H.shape[0]
    K_eff = max(N_NUMBERS, min(n - N_NUMBERS, int(K)))

    f_hot = freq_vector(H)
    P = pair_matrix(H)
    f_rec = freq_vector(H[-K_eff:])
    f_old = freq_vector(H[: n - K_eff])
    f_cond = conditional_top1_freq(H)

    mean_hot = float(f_hot.mean()) + 1e-18
    w_A = float(f_hot.max()) / mean_hot

    P_off = P.copy()
    np.fill_diagonal(P_off, 0.0)
    mean_off = float(P_off[P_off > 0].mean()) + 1e-18 if np.any(P_off > 0) else 1e-18
    w_B = float(P_off.max()) / mean_off

    w_C = float(np.linalg.norm(f_rec)) / (float(np.linalg.norm(f_old)) + 1e-18)

    w_D = float(f_cond.max()) / (float(f_cond.mean()) + 1e-18)

    w = np.array([w_A, w_B, w_C, w_D], dtype=np.float64)
    w = np.maximum(w, 0.0)
    s = float(w.sum())
    if s < 1e-18:
        return np.ones(NUM_TOOLS, dtype=np.float64) / NUM_TOOLS
    return w / s


def router_amps(w: np.ndarray) -> np.ndarray:
    a = np.sqrt(np.maximum(w, 0.0))
    n2 = float(np.linalg.norm(a))
    if n2 < 1e-18:
        return np.ones(NUM_TOOLS, dtype=np.float64) / np.sqrt(NUM_TOOLS)
    return a / n2


# =========================
# 4 heterogena tool sub-kola (svako iz |0⟩ daje svoje |ψ_t⟩)
# =========================
def tool_a_subcircuit(nq: int, amp_hot: np.ndarray) -> QuantumCircuit:
    qc = QuantumCircuit(nq, name="TA_HOT")
    qc.append(StatePreparation(amp_hot.tolist()), range(nq))
    return qc


def tool_b_subcircuit(nq: int, amp_hot: np.ndarray, pair_angles: np.ndarray) -> QuantumCircuit:
    qc = QuantumCircuit(nq, name="TB_PAIR")
    qc.append(StatePreparation(amp_hot.tolist()), range(nq))
    for k in range(nq):
        qc.ry(float(pair_angles[k]), k)
    for k in range(nq):
        qc.cx(k, (k + 1) % nq)
    return qc


def tool_c_subcircuit(nq: int, amp_cold: np.ndarray) -> QuantumCircuit:
    qc = QuantumCircuit(nq, name="TC_COLD")
    qc.append(StatePreparation(amp_cold.tolist()), range(nq))
    return qc


def tool_d_subcircuit(nq: int, amp_cond: np.ndarray) -> QuantumCircuit:
    qc = QuantumCircuit(nq, name="TD_COND")
    qc.append(StatePreparation(amp_cond.tolist()), range(nq))
    return qc


# =========================
# QMTA kolo: router + tool dispatch
# =========================
def build_qmta_state(H: np.ndarray, nq: int, K: int) -> Tuple[Statevector, np.ndarray]:
    amp_hot = amp_from_freq(freq_vector(H), nq)
    amp_cold = amp_from_freq(inverse_freq(freq_vector(H)), nq)
    amp_cond = amp_from_freq(conditional_top1_freq(H), nq)
    pair_angles = feature_to_angles(feature_pair(H), nq)

    w = router_weights(H, K)
    aux_vec = router_amps(w)

    state = QuantumRegister(nq, name="s")
    tool = QuantumRegister(M_TOOL, name="t")
    qc = QuantumCircuit(state, tool)

    qc.append(StatePreparation(aux_vec.tolist()), tool)

    subs = [
        tool_a_subcircuit(nq, amp_hot),
        tool_b_subcircuit(nq, amp_hot, pair_angles),
        tool_c_subcircuit(nq, amp_cold),
        tool_d_subcircuit(nq, amp_cond),
    ]
    for t, sub in enumerate(subs):
        U = sub.to_gate(label=f"T{t}")
        U_ctrl = U.control(num_ctrl_qubits=M_TOOL, ctrl_state=t)
        qc.append(U_ctrl, list(tool) + list(state))

    return Statevector(qc), w


def qmta_state_probs(H: np.ndarray, nq: int, K: int) -> Tuple[np.ndarray, np.ndarray]:
    sv, w = build_qmta_state(H, nq, K)
    p = np.abs(sv.data) ** 2
    dim_s = 2 ** nq
    dim_t = 2 ** M_TOOL
    mat = p.reshape(dim_t, dim_s)
    p_s = mat.sum(axis=0)
    s_tot = float(p_s.sum())
    return (p_s / s_tot if s_tot > 0 else p_s), w


# =========================
# Readout
# =========================
def bias_39(probs: np.ndarray, n_max: int = N_MAX) -> np.ndarray:
    b = np.zeros(n_max, dtype=np.float64)
    for idx, p in enumerate(probs):
        b[idx % n_max] += float(p)
    s = float(b.sum())
    return b / s if s > 0 else b


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-18 or nb < 1e-18:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pick_next_combination(probs: np.ndarray, k: int = N_NUMBERS, n_max: int = N_MAX) -> Tuple[int, ...]:
    b = bias_39(probs, n_max)
    order = np.argsort(-b, kind="stable")
    return tuple(sorted(int(o + 1) for o in order[:k]))


# =========================
# Determ. grid-optimizacija (nq, K)
# =========================
def optimize_hparams(H: np.ndarray):
    f_csv = freq_vector(H)
    s_tot = float(f_csv.sum())
    f_csv_n = f_csv / s_tot if s_tot > 0 else np.ones(N_MAX) / N_MAX
    best = None
    for nq in GRID_NQ:
        for K in GRID_K:
            try:
                p, w = qmta_state_probs(H, nq, int(K))
                bi = bias_39(p)
                score = cosine(bi, f_csv_n)
            except Exception:
                continue
            key = (score, nq, -int(K))
            if best is None or key > best[0]:
                best = (key, dict(nq=nq, K=int(K), score=float(score), w=w.tolist()))
    return best[1] if best else None


def main() -> int:
    H = load_rows(CSV_PATH)
    if H.shape[0] < 1:
        print("premalo redova")
        return 1

    print("Q24 AI Agent (QMTA — heterogeni eksperti + data-driven router): CSV:", CSV_PATH)
    print("redova:", H.shape[0], "| seed:", SEED, "| alata:", NUM_TOOLS)

    best = optimize_hparams(H)
    if best is None:
        print("grid optimizacija nije uspela")
        return 2
    print(
        "BEST hparam:",
        "nq=", best["nq"],
        "| K (recent/old split):", best["K"],
        "| cos(bias, freq_csv):", round(float(best["score"]), 6),
    )

    nq_best = int(best["nq"])
    K_best = int(best["K"])

    w_best = np.array(best["w"], dtype=np.float64)
    names = ("Tool A HOT      ", "Tool B PAIR-PQC ", "Tool C COLD     ", "Tool D COND/TOP1")
    print("--- router težine (normalizovane) ---")
    for name, wi in zip(names, w_best):
        print(f"  {name}  w={float(wi):.6f}")

    f_csv = freq_vector(H)
    s_tot = float(f_csv.sum())
    f_csv_n = f_csv / s_tot if s_tot > 0 else np.ones(N_MAX) / N_MAX

    amp_hot = amp_from_freq(freq_vector(H), nq_best)
    amp_cold = amp_from_freq(inverse_freq(freq_vector(H)), nq_best)
    amp_cond = amp_from_freq(conditional_top1_freq(H), nq_best)
    pair_angles = feature_to_angles(feature_pair(H), nq_best)

    def sub_probs(sub_qc: QuantumCircuit) -> np.ndarray:
        sv = Statevector(sub_qc)
        p = np.abs(sv.data) ** 2
        s = float(p.sum())
        return p / s if s > 0 else p

    subs = [
        tool_a_subcircuit(nq_best, amp_hot),
        tool_b_subcircuit(nq_best, amp_hot, pair_angles),
        tool_c_subcircuit(nq_best, amp_cold),
        tool_d_subcircuit(nq_best, amp_cond),
    ]
    print("--- predikcije pojedinačnih alata (samostalno) ---")
    for name, sub in zip(names, subs):
        p_t = sub_probs(sub)
        pred_t = pick_next_combination(p_t)
        cos_t = cosine(bias_39(p_t), f_csv_n)
        print(f"  {name}  cos={cos_t:.6f}  NEXT={pred_t}")

    p_mix, _ = qmta_state_probs(H, nq_best, K_best)
    pred = pick_next_combination(p_mix)
    print("--- glavna predikcija (QMTA multi-tool dispatch) ---")
    print("predikcija NEXT:", pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
Q24 AI Agent (QMTA — heterogeni eksperti + data-driven router): CSV: /data/loto7hh_4600_k31.csv
redova: 4600 | seed: 39 | alata: 4
BEST hparam: nq= 6 | K (recent/old split): 2000 | cos(bias, freq_csv): 0.424073
--- router težine (normalizovane) ---
  Tool A HOT        w=0.125275
  Tool B PAIR-PQC   w=0.149203
  Tool C COLD       w=0.088059
  Tool D COND/TOP1  w=0.637462
--- predikcije pojedinačnih alata (samostalno) ---
  Tool A HOT        cos=0.809889  NEXT=(4, 9, 12, 14, 17, 19, 22)
  Tool B PAIR-PQC   cos=0.809987  NEXT=(2, 9, 10, 12, 19, 21, 28)
  Tool C COLD       cos=0.828120  NEXT=(4, 7, 9, 12, 19, 22, 25)
  Tool D COND/TOP1  cos=0.295196  NEXT=(4, 7, 9, 14, 17, 19, 22)
--- glavna predikcija (QMTA multi-tool dispatch) ---
predikcija NEXT: (7, 9, 12, 14, 17, 19, 22)
"""



"""
Q24_AIagent_multi.py — tehnika: Quantum Multi-Tool Agent (QMTA).

Koncept:
AI Agent kao kvantni dispatcher heterogenih eksperata. 4 alata (strukturno različita
sub-kola) svaki proizvodi svoj |ψ_t⟩ iz |0⟩; router (aux m=2 qubit-a) je pripremljen
NON-UNIFORM preko StatePreparation sa težinama izvedenim iz CSV statistika, pa svaki
alat radi preko multi-ctrl dispatch sa ctrl_state=t.

Kolo (nq + 2 aux qubit-a):
  StatePreparation(√(w_t/Σw)) na tool-registar → NON-UNIFORM router.
  Za t = 0..3: multi-ctrl apply Tool-t sub-gate na state registar sa ctrl_state=t.
  |Ψ⟩ = Σ_t √(w_t/Σw) |t⟩_tool ⊗ |ψ_t⟩_state.
Readout:
  Marginala tool registra → p = Σ_t (w_t/Σw)|ψ_t|² → bias_39 → TOP-7 = NEXT.

Tehnike:
Amplitude encoding + heterogeni sub-kolo eksperti (od čiste SP do SP+PQC).
NON-UNIFORMNI router preko StatePreparation sa CSV-izvedenim težinama.
Multi-controlled gate-level dispatch (sub.to_gate().control sa ctrl_state).
Egzaktni Statevector (bez uzorkovanja).
Deterministička grid-optimizacija (nq, K).

Prednosti:
Direktan kvantni analog multi-tool agenta: „izaberi alat prema svetu → primeni alat".
Heterogeni eksperti (strukturno, ne samo po podacima) razlikuju QMTA od Q20.
NON-UNIFORMNI router razlikuje QMTA od Q13/Q20 (Hadamard) i od Q22 (retrieval-only).
Sve težine deterministički iz CELOG CSV-a (pravilo 10).
Čisto kvantno: bez klasičnog treninga, bez softmax-a, bez hibrida.

Nedostaci:
Router težine su deterministička heuristika (izbor statistika menja orkestrator ponašanje).
Marginala je linearna mešavina alata (tool-registar je ortogonalan — bez interferencije
između alata; to je po dizajnu, jer agent „pita sve alate odjednom" kroz kvantnu
superpoziciju, ne sinergijski).
Multi-ctrl sub.gate je skupo — NUM_TOOLS = 4 je praktičan plafon sa nq ≤ 6.
mod-39 readout meša stanja (dim 2^nq ≠ 39).
"""
