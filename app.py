import numpy as np
import streamlit as st

from core.fock import hom_interference
from core.physics import check_unitarity, mzi_unitary
from core.qiskit_mapper import (
    exact_probabilities,
    mzi_circuit,
    run_shots,
    sweep_phase,
    visibility,
)
from utils.visuals import (
    plot_benchmark,
    plot_hom_dip,
    plot_hom_histogram,
    plot_noise_comparison,
    plot_phase_sweep,
)

st.set_page_config(page_title="QUANTA-SIM", layout="wide")

st.title("QUANTA-SIM: Photonic Quantum Circuit Simulator")
st.write("Real-time photonic interference & quantum logic simulation.")

tab1, tab2 = st.tabs(["Single-Photon MZI", "Two-Photon HOM Effect"])

with tab1:
    st.header("Mach-Zehnder Interferometer (MZI)")
    phi = st.slider("Phase Shift (rad)", 0.0, 2 * np.pi, np.pi / 2, step=0.01)

    U = mzi_unitary(phi)
    st.subheader("Unitary Matrix (U)")
    st.write(U)
    st.success(f"Unitarity Verified: {check_unitarity(U)}")

    probs = exact_probabilities(U)
    c1, c2 = st.columns(2)
    c1.metric("P(Mode 0)", f"{probs[0]:.4f}")
    c2.metric("P(Mode 1)", f"{probs[1]:.4f}")

    st.subheader("Qiskit shot-based execution")
    shots = st.select_slider("Shots", [128, 512, 1024, 4096, 8192], value=1024)
    decompose = st.checkbox("Decompose into Rz/Ry gates", value=True)
    qc = mzi_circuit(phi, decompose=decompose)
    st.code(str(qc.draw("text")))
    res = run_shots(qc, shots)
    st.write(
        f"Sampled: Mode 0 = {res['mode0']:.4f}, Mode 1 = {res['mode1']:.4f} "
        f"(exact: {probs[0]:.4f}, {probs[1]:.4f})"
    )

    st.subheader("Phase sweep and noise")
    noisy = st.toggle("Inject physical imperfections")
    sigma = bs = 0.0
    if noisy:
        sigma = st.slider("Phase jitter std (rad)", 0.0, 1.0, 0.2, 0.01)
        bs = st.slider("Beam-splitter angle drift std (rad)", 0.0, 0.3, 0.05, 0.01)

    phis = np.linspace(0, 2 * np.pi, 60)
    sweep = sweep_phase(phis, shots=shots, phase_noise_std=sigma, bs_drift_std=bs, seed=1)

    if noisy:
        v_i, v_n = visibility(sweep["ideal"]), visibility(sweep["exact"])
        st.pyplot(plot_noise_comparison(phis, sweep["ideal"], sweep["exact"], v_i, v_n))
    else:
        st.pyplot(plot_phase_sweep(phis, sweep["ideal"]))
    st.pyplot(plot_benchmark(phis, sweep["exact"], sweep["sampled"], shots))

with tab2:
    st.header("Hong-Ou-Mandel (HOM) Interference")
    bs_angle = st.slider(
        "Beam Splitter Angle (theta)",
        0.0,
        np.pi / 2,
        np.pi / 4,
        step=0.01,
        help="pi/4 = 50:50 Beam Splitter",
    )

    hom_probs = hom_interference(bs_angle)
    st.subheader("Fock State Outcome Probabilities")
    st.json(hom_probs)

    if np.isclose(bs_angle, np.pi / 4, atol=0.02):
        st.info("HOM Dip active! State |1,1> cancels out due to quantum interference.")

    # Distinguishable particles: R = sin^2, T = cos^2
    T, R = np.cos(bs_angle) ** 2, np.sin(bs_angle) ** 2
    classical = [T * R, T**2 + R**2, T * R]
    quantum = [hom_probs["|2,0>"], hom_probs["|1,1>"], hom_probs["|0,2>"]]
    st.pyplot(plot_hom_histogram(quantum=quantum, classical=classical))

    st.subheader("HOM dip vs photon delay")
    vis = st.slider("Photon indistinguishability (visibility)", 0.0, 1.0, 1.0, 0.01)
    st.pyplot(plot_hom_dip(np.linspace(-3, 3, 300), visibility=vis))
