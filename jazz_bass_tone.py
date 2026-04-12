#!/usr/bin/env python3
"""
Jazz Bass Passive Electronics -- Frequency Response Simulator

Models the complete passive signal chain from pickup coil to interface input.
Pure math (numpy/scipy/matplotlib), no SPICE needed.

Usage:
    python3 jazz_bass_tone.py              # Full analysis + plots
    python3 jazz_bass_tone.py --summary    # Numbers only, no plots

What it answers:
    - 47nF vs 33nF vs 22nF vs 20nF tone cap comparison
    - With/without treble bleed at different volume positions
    - 250K vs 500K pot comparison
    - NoLoad vs standard pot at position 10
    - Parallel vs series pickup wiring
    - Effect of cable capacitance

Companion article: https://toyrobot.studio/series/jazz-bass-special/
"""

import os
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# =============================================================================
# PICKUP MODELS -- RLC equivalent circuits
# =============================================================================
# A passive magnetic pickup is: voltage source in series with R + L,
# with parasitic capacitance C across the coil. This creates a resonant peak.

@dataclass
class Pickup:
    """Lumped RLC model of a bass pickup."""
    name: str
    R: float       # DC resistance (ohms)
    L: float       # Inductance (henries)
    C: float       # Distributed capacitance (farads)

    @property
    def f_res(self) -> float:
        """Resonant frequency (Hz) -- the 'presence peak'."""
        return 1.0 / (2 * np.pi * np.sqrt(self.L * self.C))

    @property
    def Q_factor(self) -> float:
        """Quality factor -- how sharp the resonant peak is."""
        return (1.0 / self.R) * np.sqrt(self.L / self.C)

    def Z_series(self, f: np.ndarray) -> np.ndarray:
        """Series impedance of the pickup source (R + jomegaL)."""
        w = 2 * np.pi * f
        return self.R + 1j * w * self.L

    def Z_C(self, f: np.ndarray) -> np.ndarray:
        """Impedance of the pickup's distributed capacitance (1/jomegaC).

        This capacitance is physically across the output terminals of the coil,
        so it belongs on the LOAD side of any voltage divider -- not lumped into
        a Thevenin impedance with R+L.
        """
        w = 2 * np.pi * f
        return 1.0 / (1j * w * self.C)


# Pickup definitions -- values from specs + typical measurements
PICKUPS = {
    "stock_fender": Pickup(
        name="Stock Fender Vintage-Style",
        R=7400,        # ~7.3k neck / 7.5k bridge, averaged
        L=2.5,         # ~2.5H typical single-coil J-bass
        C=120e-12,     # ~120pF distributed capacitance
    ),
    "aguilar_4jhc": Pickup(
        name="Aguilar 4J-HC",
        R=9300,        # ~9k neck / 9.6k bridge, averaged
        L=3.5,         # Higher inductance from split-coil design
        C=130e-12,     # Slightly higher parasitic C
    ),
    "fralin_split": Pickup(
        name="Fralin Split Jazz",
        R=11750,       # ~11.5k neck / 12k bridge, averaged
        L=4.5,         # Highest inductance -- wound hot
        C=110e-12,     # Fralin's tight winding keeps C lower
    ),
}


# =============================================================================
# CIRCUIT COMPONENTS
# =============================================================================

@dataclass
class ToneCircuit:
    """Complete passive tone circuit parameters."""
    # Pots
    R_volume: float = 250e3      # Volume pot value (ohms)
    R_tone: float = 250e3        # Tone pot value (ohms)
    R_blend: float = 250e3       # Blend pot value per gang (ohms)

    # Tone cap
    C_tone: float = 47e-9        # Tone capacitor (farads)

    # Treble bleed (series network across volume pot)
    treble_bleed: bool = False
    C_bleed: float = 1e-9        # Treble bleed cap
    R_bleed: float = 150e3       # Treble bleed resistor

    # Load pad (series mode, auto-engaged)
    R_pad: float = 220e3         # Load pad resistor

    # Cable + interface
    C_cable: float = 500e-12     # ~500pF for 3m cable (typical 100pF/m)
    R_load: float = 1e6          # Interface input impedance (1M Hi-Z)

    # NoLoad pot behavior
    noload: bool = False         # True = pot disconnects at position 10


# =============================================================================
# TRANSFER FUNCTION ENGINE
# =============================================================================

def parallel(*impedances):
    """Parallel combination of complex impedances."""
    Y = sum(1.0 / (Z + 1e-30) for Z in impedances)
    return 1.0 / Y


# =============================================================================
# STRING POSITION WEIGHTING
# =============================================================================
# A pickup at position x from the bridge senses the nth harmonic with
# amplitude sin(n * pi * x / L). The bridge pickup (small x) emphasises
# upper harmonics; the neck pickup (larger x) suppresses them.

# Standard J-Bass pickup positions (fraction of scale length from bridge)
POSITION_60S_NECK = 0.28
POSITION_60S_BRIDGE = 0.13
POSITION_70S_NECK = 0.26
POSITION_70S_BRIDGE = 0.11


def string_position_weight(f, x_rel):
    """
    Harmonic amplitude envelope from pickup position on the string.

    x_rel: pickup center distance from bridge as fraction of scale length
           (e.g., 0.13 for 60s bridge, 0.28 for 60s neck)

    Physics: the nth harmonic has amplitude sin(n * pi * x / L) at pickup
    position x. This creates a comb filter whose envelope shapes the
    tonal balance -- bridge pickups get relatively more upper harmonics,
    neck pickups get a stronger fundamental.

    Averages |sin(n*pi*x)| across all 4 open strings (standard tuning),
    then smooths in log-frequency space to extract the tonal envelope
    rather than individual comb-filter teeth. Normalized to 1.0 at 200Hz.
    """
    fundamentals = np.array([41.2, 55.0, 73.4, 98.0])  # E1 A1 D2 G2

    weights = np.zeros_like(f, dtype=float)
    for f0 in fundamentals:
        n = f / f0
        weights += np.abs(np.sin(n * np.pi * x_rel))

    weights /= len(fundamentals)

    # Smooth in log-frequency space (~1/3 octave window)
    window = max(3, len(f) // 10)
    kernel = np.ones(window) / window
    weights = np.convolve(weights, kernel, mode='same')

    idx_ref = np.argmin(np.abs(f - 200))
    ref = weights[idx_ref]
    if ref > 1e-10:
        weights /= ref

    return weights


def tone_pot_impedance(R_pot: float, C_tone: float, f: np.ndarray,
                       position: float, noload: bool = False) -> np.ndarray:
    """
    Impedance of tone pot + cap network as seen from the signal node.

    position: 0.0 = full bass (cap fully engaged) to 1.0 = full treble
    noload: if True and position >= 0.99, pot disconnects (infinite Z)
    """
    w = 2 * np.pi * f

    if noload and position >= 0.99:
        return np.full_like(f, 1e12, dtype=complex)

    R_upper = max(R_pot * position, 0.1)
    R_lower = max(R_pot * (1 - position), 0.1)

    Z_cap = 1.0 / (1j * w * C_tone)
    Z_lower = parallel(R_lower + 0j, Z_cap)

    return R_upper + Z_lower


def system_response(pickup: Pickup, circuit: ToneCircuit, f: np.ndarray,
                    vol_position: float = 1.0,
                    tone_position: float = 1.0,
                    blend_position: float = 0.5,
                    series_mode: bool = False,
                    pickup_neck: Optional[Pickup] = None,
                    pickup_bridge: Optional[Pickup] = None,
                    neck_position: Optional[float] = None,
                    bridge_position: Optional[float] = None):
    """
    Full system transfer function from string vibration to interface input.

    Signal chain:
        Pickup(s) -> Blend -> Volume (+ treble bleed) -> Tone (+ cap) -> Cable -> Load

    The pickup is modelled as V_emf in series with R + L. The pickup's
    distributed capacitance C is placed on the LOAD side (across the output
    terminals), not lumped into a Thevenin impedance. This correctly produces
    a 2-pole resonant lowpass that rolls off monotonically after the peak.

    Returns dict with 'f', 'H', 'dB', 'phase'.
    """
    w = 2 * np.pi * f

    pu_n = pickup_neck if pickup_neck else pickup
    pu_b = pickup_bridge if pickup_bridge else pickup

    # Build load chain (from output backwards to pickup terminals)
    Z_cable_cap = 1.0 / (1j * w * circuit.C_cable)
    Z_interface = circuit.R_load + 0j
    Z_cable_load = parallel(Z_cable_cap, Z_interface)

    Z_tone = tone_pot_impedance(
        circuit.R_tone, circuit.C_tone, f,
        tone_position, circuit.noload
    )

    Z_at_wiper = parallel(Z_tone, Z_cable_load)

    # Volume pot
    R_upper = max(circuit.R_volume * (1 - vol_position), 0.1)
    R_lower = max(circuit.R_volume * vol_position, 0.1)

    if circuit.treble_bleed:
        Z_bleed = circuit.R_bleed + 1.0 / (1j * w * circuit.C_bleed)
        Z_input_to_wiper = parallel(R_upper + 0j * np.ones_like(f), Z_bleed)
    else:
        Z_input_to_wiper = R_upper + 0j * np.ones_like(f)

    Z_wiper_to_gnd = parallel(R_lower + 0j * np.ones_like(f), Z_at_wiper)

    H_vol = Z_wiper_to_gnd / (Z_input_to_wiper + Z_wiper_to_gnd)
    Z_vol_input = Z_input_to_wiper + Z_wiper_to_gnd

    # Pickup -> summing node transfer
    if series_mode:
        R_total = pu_n.R + pu_b.R
        L_total = pu_n.L + pu_b.L
        C_total = (pu_n.C * pu_b.C) / (pu_n.C + pu_b.C)

        Z_series = R_total + 1j * w * L_total
        Z_C_pu = 1.0 / (1j * w * C_total)

        Z_pad = circuit.R_pad + 0j
        Z_vol_padded = parallel(Z_pad, Z_vol_input)

        Z_load_at_pu = parallel(Z_C_pu, Z_vol_padded)
        H_pickup = Z_load_at_pu / (Z_series + Z_load_at_pu)

        H_total = H_pickup * H_vol

        if neck_position is not None and bridge_position is not None:
            w_n = string_position_weight(f, neck_position)
            w_b = string_position_weight(f, bridge_position)
            H_total = H_total * (w_n + w_b)

    else:
        # Parallel mode -- standard J-Bass
        blend_n = 1.0 - blend_position
        blend_b = blend_position

        R_bn = max(circuit.R_blend * (1 - blend_n), 0.1)
        R_bb = max(circuit.R_blend * (1 - blend_b), 0.1)

        both_active = blend_n >= 0.01 and blend_b >= 0.01

        if both_active:
            # Cross-loaded nodal analysis
            Z_src_n = pu_n.Z_series(f)
            Z_src_b = pu_b.Z_series(f)
            Z_Cn = pu_n.Z_C(f)
            Z_Cb = pu_b.Z_C(f)

            Vn = string_position_weight(f, neck_position) if neck_position is not None else 1.0
            Vb = string_position_weight(f, bridge_position) if bridge_position is not None else 1.0

            Y_N = 1.0 / Z_src_n + 1.0 / Z_Cn + 1.0 / R_bn
            Y_B = 1.0 / Z_src_b + 1.0 / Z_Cb + 1.0 / R_bb
            Y_S = 1.0 / R_bn + 1.0 / R_bb + 1.0 / Z_vol_input

            Y_eff = Y_S - 1.0 / (R_bn**2 * Y_N) - 1.0 / (R_bb**2 * Y_B)
            I_eff = (Vn / (Z_src_n * R_bn * Y_N) +
                     Vb / (Z_src_b * R_bb * Y_B))

            H_pickup = I_eff / Y_eff

        else:
            # Single pickup active
            responses = []
            for pu, blend_level, pos in [
                (pu_n, blend_n, neck_position),
                (pu_b, blend_b, bridge_position),
            ]:
                if blend_level < 0.01:
                    continue

                Z_series_pu = pu.Z_series(f)
                Z_C_pu = pu.Z_C(f)

                R_blend_series = max(circuit.R_blend * (1 - blend_level), 0.1)
                Z_after_blend = R_blend_series + Z_vol_input
                Z_load_at_pu = parallel(Z_C_pu, Z_after_blend)

                H_to_pu_out = Z_load_at_pu / (Z_series_pu + Z_load_at_pu)
                H_blend = Z_vol_input / Z_after_blend

                H_pu = H_to_pu_out * H_blend

                if pos is not None:
                    H_pu = H_pu * string_position_weight(f, pos)

                responses.append(H_pu)

            if responses:
                H_pickup = sum(responses)
            else:
                H_pickup = np.zeros_like(f, dtype=complex)

        H_total = H_pickup * H_vol

    # Normalize to 200Hz
    idx_200 = np.argmin(np.abs(f - 200))
    ref = np.abs(H_total[idx_200])
    if ref > 0:
        H_normalized = H_total / ref
    else:
        H_normalized = H_total

    magnitude_dB = 20 * np.log10(np.abs(H_normalized) + 1e-30)

    return {
        "f": f,
        "H": H_total,
        "dB": magnitude_dB,
        "phase": np.degrees(np.angle(H_total)),
    }


# =============================================================================
# ANALYSIS SCENARIOS
# =============================================================================

def find_resonant_peak(f, dB):
    """Find the frequency and amplitude of the resonant peak."""
    mask = f > 1000
    if not np.any(mask):
        return 0, 0
    idx = np.argmax(dB[mask])
    peak_f = f[mask][idx]
    peak_dB = dB[mask][idx]
    return peak_f, peak_dB


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def analyze_tone_caps(pickup, circuit_base, f):
    """Compare tone cap values at various tone positions."""
    print_header("TONE CAP COMPARISON")
    print(f"  Pickup: {pickup.name}")
    print(f"  Pickup resonance (unloaded): {pickup.f_res:.0f} Hz (Q={pickup.Q_factor:.1f})")

    caps = {
        "47nF (standard bass)": 47e-9,
        "33nF": 33e-9,
        "22nF (P-Bass style)": 22e-9,
        "20nF (Fralin recommended)": 20e-9,
    }

    tone_positions = [1.0, 0.7, 0.5, 0.3, 0.0]

    for cap_name, cap_val in caps.items():
        print(f"\n  --- {cap_name} ---")
        circuit = ToneCircuit(
            C_tone=cap_val,
            R_volume=circuit_base.R_volume,
            R_tone=circuit_base.R_tone,
            C_cable=circuit_base.C_cable,
            R_load=circuit_base.R_load,
        )
        for tp in tone_positions:
            resp = system_response(pickup, circuit, f, tone_position=tp)
            peak_f, peak_dB = find_resonant_peak(f, resp["dB"])
            passband = np.mean(resp["dB"][(f > 100) & (f < 500)])
            cutoff_mask = (resp["dB"] < passband - 3) & (f > 500)
            if np.any(cutoff_mask):
                f_cutoff = f[cutoff_mask][0]
            else:
                f_cutoff = 20000
            tone_label = f"Tone {tp*10:.0f}"
            print(f"    {tone_label:>8}: -3dB @ {f_cutoff:>6.0f} Hz | peak {peak_dB:>+5.1f}dB @ {peak_f:>5.0f} Hz")


def analyze_treble_bleed(pickup, circuit_base, f):
    """Compare treble bleed networks at different volume settings."""
    print_header("TREBLE BLEED COMPARISON")
    print(f"  Pickup: {pickup.name}")

    vol_positions = [1.0, 0.8, 0.6, 0.4]
    networks = {
        "No treble bleed": (False, 0, 0),
        "1nF + 150k (guitar spec)": (True, 1e-9, 150e3),
        "470pF + 39k (bass-scaled)": (True, 470e-12, 39e3),
        "1nF + 47k (alt bass)": (True, 1e-9, 47e3),
    }

    for net_name, (use_bleed, c_val, r_val) in networks.items():
        print(f"\n  --- {net_name} ---")
        circuit = ToneCircuit(
            R_volume=circuit_base.R_volume,
            R_tone=circuit_base.R_tone,
            C_tone=circuit_base.C_tone,
            C_cable=circuit_base.C_cable,
            R_load=circuit_base.R_load,
            treble_bleed=use_bleed,
            C_bleed=c_val,
            R_bleed=r_val,
        )
        for vp in vol_positions:
            resp = system_response(pickup, circuit, f, vol_position=vp)
            peak_f, peak_dB = find_resonant_peak(f, resp["dB"])
            idx_200 = np.argmin(np.abs(f - 200))
            idx_3k = np.argmin(np.abs(f - 3000))
            treble_balance = resp["dB"][idx_3k] - resp["dB"][idx_200]
            vol_label = f"Vol {vp*10:.0f}"
            print(f"    {vol_label:>7}: 3kHz vs 200Hz = {treble_balance:>+5.1f}dB | peak {peak_dB:>+5.1f}dB @ {peak_f:>5.0f} Hz")


def analyze_pot_values(pickup, f):
    """Compare 250K vs 500K pots."""
    print_header("POT VALUE COMPARISON: 250K vs 500K")
    print(f"  Pickup: {pickup.name}")

    for pot_val, label in [(250e3, "250K (J-Bass standard)"), (500e3, "500K (humbucker standard)")]:
        circuit = ToneCircuit(R_volume=pot_val, R_tone=pot_val, C_tone=33e-9)
        resp = system_response(pickup, circuit, f)
        peak_f, peak_dB = find_resonant_peak(f, resp["dB"])
        print(f"\n  {label}:")
        print(f"    Resonant peak: {peak_dB:>+5.1f}dB @ {peak_f:.0f} Hz")
        idx_200 = np.argmin(np.abs(f - 200))
        idx_5k = np.argmin(np.abs(f - 5000))
        brightness = resp["dB"][idx_5k] - resp["dB"][idx_200]
        print(f"    5kHz vs 200Hz: {brightness:>+5.1f}dB")
        idx_10k = np.argmin(np.abs(f - 10000))
        air = resp["dB"][idx_10k] - resp["dB"][idx_200]
        print(f"    10kHz vs 200Hz: {air:>+5.1f}dB")


def analyze_noload(pickup, circuit_base, f):
    """Compare NoLoad vs standard pot at position 10."""
    print_header("NoLoad vs STANDARD POT @ POSITION 10")
    print(f"  Pickup: {pickup.name}")

    for noload, label in [(False, "Standard CTS 250K @ 10"), (True, "Fender NoLoad @ 10")]:
        circuit = ToneCircuit(
            R_volume=circuit_base.R_volume,
            R_tone=circuit_base.R_tone,
            C_tone=circuit_base.C_tone,
            C_cable=circuit_base.C_cable,
            R_load=circuit_base.R_load,
            noload=noload,
        )
        resp = system_response(pickup, circuit, f, tone_position=1.0)
        peak_f, peak_dB = find_resonant_peak(f, resp["dB"])
        idx_200 = np.argmin(np.abs(f - 200))
        idx_5k = np.argmin(np.abs(f - 5000))
        idx_8k = np.argmin(np.abs(f - 8000))
        brightness = resp["dB"][idx_5k] - resp["dB"][idx_200]
        shimmer = resp["dB"][idx_8k] - resp["dB"][idx_200]
        print(f"\n  {label}:")
        print(f"    Peak: {peak_dB:>+5.1f}dB @ {peak_f:.0f} Hz")
        print(f"    5kHz vs 200Hz: {brightness:>+5.1f}dB")
        print(f"    8kHz vs 200Hz: {shimmer:>+5.1f}dB")


def analyze_series_parallel(f):
    """Compare parallel vs series mode across pickup sets."""
    print_header("PARALLEL vs SERIES MODE")

    for pu_key in ["stock_fender", "aguilar_4jhc", "fralin_split"]:
        pu = PICKUPS[pu_key]
        print(f"\n  --- {pu.name} ---")

        circuit = ToneCircuit(C_tone=33e-9)

        for mode, series in [("Parallel", False), ("Series", True)]:
            resp = system_response(pu, circuit, f, series_mode=series,
                                   pickup_neck=pu, pickup_bridge=pu)
            peak_f, peak_dB = find_resonant_peak(f, resp["dB"])
            idx_200 = np.argmin(np.abs(f - 200))
            idx_800 = np.argmin(np.abs(f - 800))
            idx_3k = np.argmin(np.abs(f - 3000))
            mids = resp["dB"][idx_800] - resp["dB"][idx_200]
            presence = resp["dB"][idx_3k] - resp["dB"][idx_200]
            print(f"    {mode:>8}: peak {peak_dB:>+5.1f}dB @ {peak_f:>5.0f} Hz | "
                  f"800Hz: {mids:>+5.1f}dB | 3kHz: {presence:>+5.1f}dB")


def analyze_pickup_comparison(f):
    """Compare all three pickup sets under identical conditions."""
    print_header("PICKUP COMPARISON (identical circuit)")

    circuit = ToneCircuit(C_tone=33e-9)

    for pu_key in ["stock_fender", "aguilar_4jhc", "fralin_split"]:
        pu = PICKUPS[pu_key]
        print(f"\n  --- {pu.name} ---")
        print(f"    R={pu.R:.0f} ohm  L={pu.L:.1f}H  C={pu.C*1e12:.0f}pF")
        print(f"    Unloaded resonance: {pu.f_res:.0f} Hz (Q={pu.Q_factor:.1f})")

        resp = system_response(pu, circuit, f)
        peak_f, peak_dB = find_resonant_peak(f, resp["dB"])
        idx_1k = np.argmin(np.abs(f - 1000))
        idx_3k = np.argmin(np.abs(f - 3000))
        idx_5k = np.argmin(np.abs(f - 5000))

        print(f"    Loaded resonant peak: {peak_dB:>+5.1f}dB @ {peak_f:.0f} Hz")
        print(f"    1kHz:  {resp['dB'][idx_1k]:>+5.1f}dB")
        print(f"    3kHz:  {resp['dB'][idx_3k]:>+5.1f}dB")
        print(f"    5kHz:  {resp['dB'][idx_5k]:>+5.1f}dB")


# =============================================================================
# PLOTTING
# =============================================================================

def plot_all(f, output_dir="output"):
    """Generate comparison plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib not available -- skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)

    colors = {
        "stock": "#888888",
        "aguilar": "#2196F3",
        "fralin": "#FF5722",
    }
    cap_colors = {
        "47nF": "#1976D2",
        "33nF": "#388E3C",
        "22nF": "#F57C00",
        "20nF": "#D32F2F",
    }

    # PLOT 1: Tone cap comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle("Tone Cap Comparison -- Tone @ 5 (half rotation)", fontsize=14, fontweight='bold')

    caps = {"47nF": 47e-9, "33nF": 33e-9, "22nF": 22e-9, "20nF": 20e-9}
    pickups_list = [
        ("stock_fender", "Stock Fender"),
        ("aguilar_4jhc", "Aguilar 4J-HC"),
        ("fralin_split", "Fralin Split Jazz"),
    ]

    for ax, (pu_key, pu_label) in zip(axes, pickups_list):
        pu = PICKUPS[pu_key]
        for cap_name, cap_val in caps.items():
            circuit = ToneCircuit(C_tone=cap_val)
            resp = system_response(pu, circuit, f, tone_position=0.5)
            ax.semilogx(f, resp["dB"], linewidth=2, label=cap_name,
                        color=cap_colors[cap_name])
        circuit_ref = ToneCircuit(C_tone=33e-9)
        resp_ref = system_response(pu, circuit_ref, f, tone_position=1.0)
        ax.semilogx(f, resp_ref["dB"], '--', linewidth=1, label="Tone @ 10",
                    color="#999999", alpha=0.7)

        ax.set_title(pu_label, fontsize=11)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xlim(20, 20000)
        ax.set_ylim(-25, 10)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=8)
        ax.axvline(41.2, color='#ccc', linestyle=':', alpha=0.5)
        ax.axvline(4000, color='#ccc', linestyle=':', alpha=0.5)

    axes[0].set_ylabel("Magnitude (dB, normalized)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tone_cap_comparison.png"), dpi=150)
    print(f"  Saved: {output_dir}/tone_cap_comparison.png")

    # PLOT 2: Treble bleed comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Treble Bleed Networks -- Fralin Split Jazz", fontsize=14, fontweight='bold')

    pu = PICKUPS["fralin_split"]
    vol_positions = [1.0, 0.7, 0.5, 0.3]
    bleed_configs = {
        "No bleed": (False, 0, 0, "#1976D2"),
        "1nF+150k (guitar)": (True, 1e-9, 150e3, "#D32F2F"),
        "470pF+39k (bass)": (True, 470e-12, 39e3, "#388E3C"),
        "1nF+47k (alt)": (True, 1e-9, 47e3, "#F57C00"),
    }

    for ax, vp in zip(axes.flat, vol_positions):
        for bl_name, (use_bl, c_bl, r_bl, color) in bleed_configs.items():
            circuit = ToneCircuit(
                C_tone=33e-9, treble_bleed=use_bl,
                C_bleed=c_bl, R_bleed=r_bl,
            )
            resp = system_response(pu, circuit, f, vol_position=vp)
            ax.semilogx(f, resp["dB"], linewidth=2, label=bl_name, color=color)

        ax.set_title(f"Volume @ {vp*10:.0f}", fontsize=11)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("dB (normalized)")
        ax.set_xlim(20, 20000)
        ax.set_ylim(-20, 10)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "treble_bleed_comparison.png"), dpi=150)
    print(f"  Saved: {output_dir}/treble_bleed_comparison.png")

    # PLOT 3: Pot values + NoLoad
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle("Pot Value & NoLoad Comparison -- Tone @ 10", fontsize=14, fontweight='bold')

    configs = {
        "250K standard": (250e3, False, "#1976D2"),
        "250K NoLoad": (250e3, True, "#388E3C"),
        "500K standard": (500e3, False, "#D32F2F"),
        "500K NoLoad": (500e3, True, "#F57C00"),
    }

    for ax, (pu_key, pu_label) in zip(axes, pickups_list):
        pu = PICKUPS[pu_key]
        for cfg_name, (r_pot, nl, color) in configs.items():
            circuit = ToneCircuit(R_volume=r_pot, R_tone=r_pot, C_tone=33e-9, noload=nl)
            resp = system_response(pu, circuit, f, tone_position=1.0)
            ax.semilogx(f, resp["dB"], linewidth=2, label=cfg_name, color=color)

        ax.set_title(pu_label, fontsize=11)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xlim(20, 20000)
        ax.set_ylim(-15, 10)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Magnitude (dB, normalized)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pot_value_comparison.png"), dpi=150)
    print(f"  Saved: {output_dir}/pot_value_comparison.png")

    # PLOT 4: Series vs Parallel
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle("Parallel vs Series Mode", fontsize=14, fontweight='bold')

    for ax, (pu_key, pu_label) in zip(axes, pickups_list):
        pu = PICKUPS[pu_key]
        for mode, series, color, ls in [
            ("Parallel", False, "#1976D2", "-"),
            ("Series", True, "#D32F2F", "-"),
            ("Series (no pad)", True, "#F57C00", "--"),
        ]:
            c = ToneCircuit(C_tone=33e-9, R_pad=220e3 if "no pad" not in mode else 1e12)
            resp = system_response(pu, c, f, series_mode=series,
                                   pickup_neck=pu, pickup_bridge=pu)
            ax.semilogx(f, resp["dB"], linewidth=2, label=mode,
                        color=color, linestyle=ls)

        ax.set_title(pu_label, fontsize=11)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xlim(20, 20000)
        ax.set_ylim(-15, 15)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Magnitude (dB, normalized)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "series_parallel_comparison.png"), dpi=150)
    print(f"  Saved: {output_dir}/series_parallel_comparison.png")

    # PLOT 5: All pickups overlaid
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_title("Pickup Comparison -- 250K / 33nF / Tone @ 10", fontsize=14, fontweight='bold')

    pu_colors = {"stock_fender": colors["stock"], "aguilar_4jhc": colors["aguilar"],
                 "fralin_split": colors["fralin"]}
    circuit = ToneCircuit(C_tone=33e-9)

    for pu_key, pu_label in pickups_list:
        pu = PICKUPS[pu_key]
        resp = system_response(pu, circuit, f)
        ax.semilogx(f, resp["dB"], linewidth=2.5, label=pu_label,
                    color=pu_colors[pu_key])

    for note, freq in [("E1", 41.2), ("A1", 55), ("D2", 73.4), ("G2", 98)]:
        ax.axvline(freq, color='#ddd', linestyle=':', alpha=0.5)
        ax.text(freq, -14, note, ha='center', fontsize=7, color='#999')

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB, normalized)")
    ax.set_xlim(20, 20000)
    ax.set_ylim(-15, 10)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pickup_comparison.png"), dpi=150)
    print(f"  Saved: {output_dir}/pickup_comparison.png")

    plt.close('all')
    print(f"\n  All plots saved to: {output_dir}/")


# =============================================================================
# MAIN
# =============================================================================

def main():
    summary_only = "--summary" in sys.argv

    f = np.logspace(np.log10(20), np.log10(20000), 1000)

    circuit_base = ToneCircuit(
        R_volume=250e3,
        R_tone=250e3,
        C_tone=33e-9,
        C_cable=500e-12,
        R_load=1e6,
    )

    print("=" * 70)
    print("  JAZZ BASS PASSIVE ELECTRONICS -- Frequency Response Analysis")
    print("=" * 70)
    print(f"\n  Base circuit: {circuit_base.R_volume/1e3:.0f}K pots, "
          f"{circuit_base.C_tone*1e9:.0f}nF tone cap, "
          f"{circuit_base.C_cable*1e12:.0f}pF cable, "
          f"{circuit_base.R_load/1e6:.0f}M load")

    pickup_for_analysis = PICKUPS["fralin_split"]

    analyze_pickup_comparison(f)
    analyze_tone_caps(pickup_for_analysis, circuit_base, f)
    analyze_treble_bleed(pickup_for_analysis, circuit_base, f)
    analyze_pot_values(pickup_for_analysis, f)
    analyze_noload(pickup_for_analysis, circuit_base, f)
    analyze_series_parallel(f)

    if not summary_only:
        print_header("GENERATING PLOTS")
        plot_all(f)

    print("\nDone.")


if __name__ == "__main__":
    main()
