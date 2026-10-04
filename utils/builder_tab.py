"""Streamlit tab for the Optical Builder. In app.py call: render_builder_tab()"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from core.builder import OpticalCircuit, mzi_circuit, pattern_label
from core.qiskit_mapper import run_shots, unitary_to_circuit
from utils.schematic import draw_circuit


def _get_circuit(n_modes):
    """Keep one circuit in session state; start fresh if the mode count changes."""
    circ = st.session_state.get("circuit")
    if circ is None or circ.n_modes != n_modes:
        circ = OpticalCircuit(n_modes)
        st.session_state["circuit"] = circ
    return circ


def _heatmap(U):
    fig, ax = plt.subplots(figsize=(4, 3.4))
    im = ax.imshow(np.abs(U) ** 2, cmap="viridis", vmin=0, vmax=1)
    n = U.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("Input mode")
    ax.set_ylabel("Output mode")
    ax.set_title(r"$|U_{ij}|^2$")
    for i in range(n):
        for j in range(n):
            val = abs(U[i, j]) ** 2
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val < 0.6 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def _prob_bars(probs, labels, title):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(labels, probs)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Probability")
    ax.set_title(title)
    for k, p in enumerate(probs):
        ax.text(k, p + 0.02, f"{p:.3f}", ha="center")
    fig.tight_layout()
    return fig


def render_builder_tab():
    st.header("Optical Circuit Builder")
    st.write("Build a linear optical circuit from beam splitters and phase shifters.")

    n_modes = st.selectbox("Number of optical modes", [2, 3, 4], index=0)
    circ = _get_circuit(n_modes)
    modes = list(range(n_modes))

    # ---- add components
    st.subheader("Add a component")
    kind = st.radio("Component type", ["Beam splitter", "Phase shifter"], horizontal=True)

    if kind == "Beam splitter":
        c1, c2 = st.columns(2)
        mode_a = c1.selectbox("Mode A", modes, index=0, key="bs_a")
        mode_b = c2.selectbox("Mode B", modes, index=1, key="bs_b")
        theta = st.slider("Splitting angle theta (pi/4 = 50:50)",
                          0.0, float(np.pi / 2), float(np.pi / 4), 0.01, key="bs_theta")
        if st.button("Add beam splitter"):
            if mode_a == mode_b:
                st.error("Choose two different modes.")
            else:
                circ.add_beam_splitter(mode_a, mode_b, theta)
    else:
        mode = st.selectbox("Mode", modes, key="ps_mode")
        phi = st.slider("Phase phi (rad)", 0.0, float(2 * np.pi),
                        float(np.pi / 2), 0.01, key="ps_phi")
        if st.button("Add phase shifter"):
            circ.add_phase_shifter(mode, phi)

    b1, b2, b3 = st.columns(3)
    if b1.button("Undo last"):
        circ.remove_last()
    if b2.button("Clear all"):
        circ.clear()
    if b3.button("Load MZI preset", disabled=(n_modes != 2)):
        st.session_state["circuit"] = circ = mzi_circuit(np.pi / 2)

    # ---- component list
    st.subheader("Circuit (applied top to bottom)")
    if not circ.components:
        st.info("No components yet. Add a beam splitter or phase shifter above.")
        return
    for k, comp in enumerate(circ.components, 1):
        st.write(f"{k}. {comp.label()}")
    s = circ.depth_summary()
    st.caption(f"{s['components']} components: {s['beam_splitters']} beam splitters, "
               f"{s['phase_shifters']} phase shifters")
    st.pyplot(draw_circuit(circ))

    # ---- total unitary
    U = circ.total_unitary()
    st.subheader("Total unitary U_total")
    if circ.is_unitary():
        st.success("Unitarity verified: U†U = I")
    else:
        st.error("Matrix is not unitary!")
    col_m, col_h = st.columns(2)
    col_m.dataframe(pd.DataFrame(np.round(U, 4)).astype(str))
    col_h.pyplot(_heatmap(U))

    # ---- single photon
    st.subheader("Single-photon output")
    in_mode = st.selectbox("Photon input mode", modes, key="sp_in")
    p1 = circ.single_photon_probabilities(in_mode)
    st.pyplot(_prob_bars(p1, [f"Mode {m}" for m in modes], "Output probabilities"))

    # ---- Qiskit mapping (2 modes only)
    if n_modes == 2:
        st.subheader("Send to Qiskit (2 modes)")
        shots = st.select_slider("Shots", [128, 512, 1024, 4096, 8192],
                                 value=1024, key="bld_shots")
        qc = unitary_to_circuit(U, input_mode=in_mode, decompose=True)
        st.code(str(qc.draw("text")))
        res = run_shots(qc, shots)
        st.write(f"Sampled: Mode 0 = {res['mode0']:.4f}, Mode 1 = {res['mode1']:.4f} "
                 f"(exact: {p1[0]:.4f}, {p1[1]:.4f})")
    else:
        st.caption("Qiskit mapping is available for 2-mode circuits.")

    # ---- two photons
    st.subheader("Two-photon output (Fock states)")
    c1, c2 = st.columns(2)
    in_a = c1.selectbox("Photon 1 input mode", modes, index=0, key="tp_a")
    in_b = c2.selectbox("Photon 2 input mode", modes, index=1, key="tp_b")
    p2 = circ.two_photon_probabilities(in_a, in_b)
    labels = [pattern_label(p) for p in p2]
    st.pyplot(_prob_bars(list(p2.values()), labels, "Output configurations"))
    st.caption(f"Total probability: {sum(p2.values()):.4f}")
