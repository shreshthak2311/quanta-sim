"""Optical Builder: build a list of optical components and cascade them into
one N-mode unitary  U_total = U_n ... U_2 U_1  (2-4 modes supported).

Uses beam_splitter / phase_shifter from core.physics.
"""
from dataclasses import dataclass
from itertools import combinations_with_replacement
from math import factorial

import numpy as np

from core.physics import beam_splitter, check_unitarity, phase_shifter


def embed_2mode(U2, i, j, n_modes):
    """Embed a 2x2 unitary acting on modes (i, j) into an NxN identity."""
    if i == j or not (0 <= i < n_modes and 0 <= j < n_modes):
        raise ValueError(f"Invalid modes ({i}, {j}) for {n_modes}-mode circuit")
    U = np.eye(n_modes, dtype=complex)
    U[i, i], U[i, j] = U2[0, 0], U2[0, 1]
    U[j, i], U[j, j] = U2[1, 0], U2[1, 1]
    return U


@dataclass
class Component:
    kind: str        # "BS" or "PS"
    modes: tuple     # (i, j) for BS, (k,) for PS
    param: float     # theta (BS) or phi (PS), radians

    def label(self):
        if self.kind == "BS":
            return f"BS(modes {self.modes[0]},{self.modes[1]}; theta={self.param:.2f})"
        return f"PS(mode {self.modes[0]}; phi={self.param:.2f})"


class OpticalCircuit:
    def __init__(self, n_modes=2):
        if not 2 <= n_modes <= 4:
            raise ValueError("n_modes must be between 2 and 4")
        self.n_modes = n_modes
        self.components = []

    # ---- building
    def add_beam_splitter(self, mode_a, mode_b, theta=np.pi / 4):
        self.components.append(Component("BS", (mode_a, mode_b), theta))
        return self

    def add_phase_shifter(self, mode, phi):
        if not 0 <= mode < self.n_modes:
            raise ValueError(f"Mode {mode} out of range")
        self.components.append(Component("PS", (mode,), phi))
        return self

    def remove_last(self):
        if self.components:
            self.components.pop()

    def clear(self):
        self.components.clear()

    # ---- math
    def component_unitary(self, comp):
        n = self.n_modes
        if comp.kind == "BS":
            return embed_2mode(beam_splitter(comp.param), *comp.modes, n)
        U = np.eye(n, dtype=complex)
        U[comp.modes[0], comp.modes[0]] = phase_shifter(comp.param)[0, 0]
        return U

    def total_unitary(self):
        U = np.eye(self.n_modes, dtype=complex)
        for comp in self.components:  # first component acts first
            U = self.component_unitary(comp) @ U
        return U

    def is_unitary(self):
        return check_unitarity(self.total_unitary())

    def depth_summary(self):
        return {
            "components": len(self.components),
            "beam_splitters": sum(c.kind == "BS" for c in self.components),
            "phase_shifters": sum(c.kind == "PS" for c in self.components),
        }

    # ---- simulation
    def single_photon_probabilities(self, input_mode=0):
        return np.abs(self.total_unitary()[:, input_mode]) ** 2

    def two_photon_probabilities(self, in_a=0, in_b=1):
        """Probabilities of output occupation patterns for two photons entering
        modes in_a and in_b. Returns {(n_0, ..., n_{N-1}): probability}."""
        U = self.total_unitary()
        out = {}
        for i, j in combinations_with_replacement(range(self.n_modes), 2):
            M = U[np.ix_([i, j], [in_a, in_b])]
            perm = M[0, 0] * M[1, 1] + M[0, 1] * M[1, 0]
            norm = (2 if i == j else 1) * (2 if in_a == in_b else 1)
            pattern = [0] * self.n_modes
            pattern[i] += 1
            pattern[j] += 1
            out[tuple(pattern)] = float(np.abs(perm) ** 2 / norm)
        return out


def pattern_label(pattern):
    return "|" + ",".join(str(n) for n in pattern) + ">"


def mzi_circuit(phi, theta1=np.pi / 4, theta2=np.pi / 4):
    """Preset: Mach-Zehnder interferometer (same order as physics.mzi_unitary)."""
    return (OpticalCircuit(2)
            .add_beam_splitter(0, 1, theta1)
            .add_phase_shifter(0, phi)
            .add_beam_splitter(0, 1, theta2))
