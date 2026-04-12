#!/usr/bin/env python3
"""
Jazz Bass Pickup Landscape — Complete Sonic Spectrum

Maps every notable J-Bass pickup on a single frequency response chart.
Groups by architecture to show WHY pickups sound different, not just brand differences.

Architecture is the #1 predictor of brightness:
  - Single-coil: brightest (but hums when soloed)
  - Split-coil side-by-side: nearly as bright, hum-cancelling
  - Stacked humbucker: darkest hum-cancelling (high mutual coupling + capacitance)

Usage:
    python3 pickup_landscape.py           # Full analysis + plots
    python3 pickup_landscape.py --table   # Summary table only

Companion article: https://toyrobot.studio/series/jazz-bass-special/
"""

import os
import sys
import numpy as np
from dataclasses import dataclass, field

# =============================================================================
# PICKUP MODELS
# =============================================================================

@dataclass
class Pickup:
    name: str
    R: float           # DC resistance (ohms), averaged neck/bridge
    L: float           # Inductance (henries)
    C: float           # Distributed capacitance (farads)
    architecture: str  # "single-coil", "split-coil", "stacked", "active"
    magnet: str
    hum_cancel: bool   # True = hum-cancelling when soloed
    notes: str = ""
    confidence: str = "estimated"  # "measured", "derived", "estimated"

    @property
    def f_res_unloaded(self):
        return 1.0 / (2 * np.pi * np.sqrt(self.L * self.C))

    @property
    def Q_factor(self):
        return (1.0 / self.R) * np.sqrt(self.L / self.C)

    @property
    def lr_ratio(self):
        """L (henries) per kohm of DC resistance."""
        return self.L / (self.R / 1000)

    def impedance(self, f):
        w = 2 * np.pi * f
        Z_L = self.R + 1j * w * self.L
        Z_C = 1.0 / (1j * w * self.C + 1e-30)
        return (Z_L * Z_C) / (Z_L + Z_C + 1e-30)


# Helper: derive L from manufacturer's loaded resonant peak spec
def L_from_loaded_peak(f_peak, C_pu=100e-12, C_cable=500e-12):
    C_total = C_pu + C_cable
    return 1.0 / ((2 * np.pi * f_peak) ** 2 * C_total)


# -----------------------------------------------------------------------------
# TRUE SINGLE-COILS
# Hum when soloed. RWRP pairs cancel hum when both pickups are on.
# These define the "ceiling" of brightness for a J-Bass pickup.
# -----------------------------------------------------------------------------

pickups = {}

pickups["stock_fender"] = Pickup(
    name="Stock Fender Am. Special",
    R=7400, L=2.65, C=170e-12,
    architecture="single-coil", magnet="Alnico 5", hum_cancel=False,
    notes="R from spec. L/C interpolated from CS60 measured data (2.48-2.84H).",
    confidence="derived",
)

pickups["fender_cs60"] = Pickup(
    name="Fender Custom Shop '60s",
    R=7200, L=2.66, C=178e-12,
    architecture="single-coil", magnet="Alnico 5", hum_cancel=False,
    notes="MEASURED by antigua (GuitarNutz2/Echoes of Mars): N=2.48H/172pF, B=2.84H/183pF. "
          "Fender spec: N=2.73H, B=2.84H. Using avg.",
    confidence="measured",
)

pickups["fender_pv62"] = Pickup(
    name="Fender Pure Vintage '62",
    R=7375, L=3.275, C=178e-12,
    architecture="single-coil", magnet="Alnico 2", hum_cancel=False,
    notes="R/L from Fender published (N=7.25k/3.2H, B=7.5k/3.35H). "
          "C=178pF from measured CS60 (same Formvar bobbin). Alnico 2 = warmer.",
    confidence="measured",
)

pickups["fender_pv74"] = Pickup(
    name="Fender Pure Vintage '74",
    R=7690, L=3.647, C=140e-12,
    architecture="single-coil", magnet="Alnico 5", hum_cancel=False,
    notes="MEASURED by GuitarNutz2/echoesofmars: N=7.45k/3.457H/141pF, B=7.93k/3.836H/139pF. "
          "Loaded peaks: B=3.13kHz/5.0dB, N=3.28kHz/5.4dB. "
          "Fender published lower (N=3.2H, B=3.7H). Enamel wire. "
          "Perceived '70s brightness' from bridge placement, not electrical response.",
    confidence="measured",
)

pickups["fender_stock_ceramic"] = Pickup(
    name="Fender Stock Ceramic",
    R=5740, L=4.076, C=175e-12,
    architecture="single-coil", magnet="Ceramic", hum_cancel=False,
    notes="MEASURED by GuitarNutz2/echoesofmars: R=5.74k, L=4.076H, C=175pF. "
          "Steel slug pole pieces -> high permeability -> L=4.08H despite only 5.74k DCR. "
          "400G. Loaded peak: 2.96kHz/3.5dB. MIM/Squier stock pickup.",
    confidence="measured",
)

pickups["fender_vmod2"] = Pickup(
    name="Fender V-Mod II",
    R=7300, L=3.0, C=140e-12,
    architecture="single-coil", magnet="Alnico (mixed)", hum_cancel=False,
    notes="Am. Professional II. Mixed Alnico II/V. V-Mod I spec: N=2.9H, B=3.1H. "
          "V-Mod II likely similar. Using 3.0H avg.",
    confidence="derived",
)

pickups["fender_cobalt"] = Pickup(
    name="Fender Cobalt Chrome",
    R=8225, L=3.875, C=120e-12,
    architecture="single-coil", magnet="Cobalt Chrome", hum_cancel=False,
    notes="R and L MEASURED (Thomann specs: 8.0k/8.45k, 3.8H/3.95H). "
          "Cobalt magnets increase core permeability -> higher L than typical single-coil. "
          "True single-coil, not hum-cancelling.",
    confidence="measured",
)

pickups["lollar"] = Pickup(
    name="Lollar Jazz Bass",
    R=8645, L=4.19, C=131e-12,
    architecture="single-coil", magnet="Alnico 5", hum_cancel=False,
    notes="MEASURED bridge (GN2/echoesofmars): R=9.69k, L=4.881H, C=131pF, 6.07kHz/15.6dB. "
          "Neck from Lollar tech page: R=7.6k, L=3.5H. Using avg. "
          "GN2 bridge higher than published (8.5k/4.0H) -- sample variance.",
    confidence="measured",
)

pickups["byo_lightning"] = Pickup(
    name="BYO Lightning",
    R=8000, L=3.725, C=131e-12,
    architecture="single-coil", magnet="Alnico 5", hum_cancel=False,
    notes="Vendor L from BYO (N=3.45H, B=4.0H). C estimated from Lollar (131pF). "
          "Budget $60/set. Loaded peaks: B=3.24kHz, N=3.50kHz. "
          "Electrically near-identical to Lollar published specs.",
    confidence="derived",
)

pickups["sd_sjb1"] = Pickup(
    name="SD SJB-1 Vintage",
    R=9350, L=3.75, C=145e-12,
    architecture="single-coil", magnet="Alnico 5", hum_cancel=False,
    notes="R from SD spec. L=3.75H estimated from L/R ratio (~0.41 H/kohm, "
          "typical for 42AWG Alnico 5 J-Bass). True single-coil, RWRP.",
    confidence="estimated",
)

pickups["sd_sjb2"] = Pickup(
    name="SD SJB-2 Hot",
    R=15810, L=6.75, C=160e-12,
    architecture="single-coil", magnet="Alnico 5", hum_cancel=False,
    notes="R from SD spec. L=6.75H estimated from L/R ratio. Very hot overwound single-coil.",
    confidence="estimated",
)

pickups["sd_sjb3"] = Pickup(
    name="SD SJB-3 Quarter Pound",
    R=13600, L=6.716, C=244e-12,
    architecture="single-coil", magnet="Alnico 5", hum_cancel=False,
    notes="MEASURED by GN2/echoesofmars: R=13.6k, L=6.716H, C=244pF. "
          "775G. Self res: 3.85kHz/14.6dB. Loaded: 2.13kHz/4.1dB. "
          "Quarter-inch Alnico 5 pole pieces.",
    confidence="measured",
)

pickups["nordstrand_nj4"] = Pickup(
    name="Nordstrand NJ4",
    R=7500, L=3.1, C=150e-12,
    architecture="single-coil", magnet="Alnico 5", hum_cancel=False,
    notes="Faithful vintage reproduction. True single-coil. L estimated from L/R ratio.",
    confidence="estimated",
)

pickups["bartolini_9cbjs"] = Pickup(
    name="Bartolini 9CBJS",
    R=7900, L=4.0, C=180e-12,
    architecture="single-coil", magnet="Ceramic", hum_cancel=False,
    notes="R from Bartolini spec. Classic Bass series. True single-coil. "
          "Ceramic magnets + carbon steel core -> high L for its R. "
          "Resonant peak 4.3-4.6kHz (Bartolini spec). Known for warmth.",
    confidence="estimated",
)


# -----------------------------------------------------------------------------
# STACKED HUMBUCKERS
# Two coils stacked vertically inside a single-coil form factor.
# Hum-cancelling always. But mutual coupling between vertically stacked coils
# increases effective L and C, making them darker than equivalent single-coils.
# -----------------------------------------------------------------------------

pickups["dimarzio_areaj"] = Pickup(
    name="DiMarzio Area J",
    R=7800, L=3.2, C=120e-12,
    architecture="split-coil", magnet="Alnico 2", hum_cancel=True,
    notes="SPLIT-COIL side-by-side (confirmed by DiMarzio tech support: "
          "'All of our Jazz Bass pickups are split humbuckers'). "
          "Previously misidentified as stacked -- the Area *guitar* pickups are stacked, "
          "but the J-bass version uses split-coil architecture like Model J and Ultra Jazz. "
          "Alnico 2 = low gauss, low magnetic pull. 4-conductor wiring. "
          "C=120pF typical for split-coil (series combination of two half-coils). "
          "DiMarzio tone chart (4-band): Bass 2.5, Low Mid 4.0, High Mid 4.0, Treble 4.5.",
    confidence="estimated",
)

pickups["sd_sjb5"] = Pickup(
    name="SD SJB-5 Stack",
    R=16450, L=7.0, C=200e-12,
    architecture="stacked", magnet="Alnico 5", hum_cancel=True,
    notes="R from SD spec (N=11.9k, B=21k). Using avg 16.45k. Stacked humbucker. "
          "L estimated high (7H) for stacked topology with heavy winding.",
    confidence="estimated",
)

pickups["fender_ultra_noiseless"] = Pickup(
    name="Fender Ultra Noiseless",
    R=13900, L=5.2, C=180e-12,
    architecture="stacked", magnet="Alnico 5", hum_cancel=True,
    notes="STACKED humbucker. Fender spec: N=12.6-13.2k/4.6H, B=14.6-15.2k/5.8H. "
          "Using avg R=13.9k, L=5.2H.",
    confidence="measured",
)

pickups["fender_gen4"] = Pickup(
    name="Fender Gen 4 Noiseless",
    R=11750, L=4.4, C=160e-12,
    architecture="stacked", magnet="Alnico 5", hum_cancel=True,
    notes="R=11.0k/12.5k, L=4.1-4.7H from Fender product page. "
          "Fender calls it 'single-coil with shielded wire' but specs are 2x "
          "a typical single-coil. Architecture likely stacked internally.",
    confidence="measured",
)


# -----------------------------------------------------------------------------
# SPLIT-COIL SIDE-BY-SIDE
# Two coils next to each other, each sensing different strings.
# Hum-cancelling always. Lower mutual coupling than stacked -> can be as
# bright as single-coils while eliminating hum.
# This is the sweet spot for studio use.
# -----------------------------------------------------------------------------

pickups["dimarzio_ultrajazz"] = Pickup(
    name="DiMarzio Ultra Jazz",
    R=12300, L=3.8, C=120e-12,
    architecture="split-coil", magnet="Alnico 5 + Ceramic", hum_cancel=True,
    notes="Split-coil SIDE-BY-SIDE. Hybrid magnets (Alnico 5 rods + ceramic bar). "
          "250mV output. 4-conductor. "
          "CALIBRATION: DiMarzio tone guide rates Treble 7.0 (vs Model J 5.5, Area J 4.5). "
          "Highest treble despite highest DCR. DiMarzio: 'even with relatively high DCR, "
          "very strong percussive highs.' Standard L/R ratio gives ~5H but tone ratings "
          "suggest hybrid magnet reduces L per ohm. L=3.8H estimated (range 3.5-4.0H).",
    confidence="estimated",
)

pickups["dimarzio_modelj"] = Pickup(
    name="DiMarzio Model J",
    R=6820, L=4.7, C=130e-12,
    architecture="split-coil", magnet="Ceramic", hum_cancel=True,
    notes="Original DiMarzio J-Bass (1979). Split-coil side-by-side. "
          "Ceramic magnet + steel blades. 150mV output. 4-conductor wiring. "
          "L revised to 4.7H based on measured ceramic/steel L/R ratios: "
          "DiMarzio Chopper (GN2): 8.86k/5.915H = 0.668 H/kohm, "
          "Fender Stock Ceramic (GN2): 5.74k/4.076H = 0.710 H/kohm. "
          "Using 0.69 H/kohm avg for ceramic+steel: 6.82k x 0.69 = 4.7H. "
          "PARALLEL WIRING: 4-conductor allows coil-parallel mode (L/4, R/2, Cx2). "
          "In parallel: L=1.175H, f_res=9.1kHz -- brighter than any pickup in the database.",
    confidence="estimated",
)

pickups["aguilar_4jhc"] = Pickup(
    name="Aguilar 4J-HC",
    R=9300, L=3.8, C=130e-12,
    architecture="split-coil", magnet="Alnico 5", hum_cancel=True,
    notes="R from spec (9k/9.6k). Split-coil. L estimated from L/R ratio "
          "(~0.41 H/kohm). 3-conductor (hot, ground, shield drain).",
    confidence="estimated",
)

pickups["fralin_split"] = Pickup(
    name="Fralin Split Jazz",
    R=11750, L=4.8, C=120e-12,
    architecture="split-coil", magnet="Alnico 5", hum_cancel=True,
    notes="R from spec (11.5k/12k). L estimated ~3.5-3.7H per coil "
          "but series winding pushes effective L higher. Using 4.8H. Wax potted.",
    confidence="estimated",
)

pickups["sd_apollo"] = Pickup(
    name="SD Apollo Jazz",
    R=9020, L=3.3, C=120e-12,
    architecture="split-coil", magnet="Alnico 5", hum_cancel=True,
    notes="R from SD spec (8.79k/9.25k). Split-coil side-by-side confirmed. "
          "'Linear humbucker' design. Added punch and articulation.",
    confidence="estimated",
)

pickups["nordstrand_nj4sv"] = Pickup(
    name="Nordstrand NJ4SV",
    R=8500, L=3.2, C=130e-12,
    architecture="split-coil", magnet="Alnico 5", hum_cancel=True,
    notes="Split-coil version of NJ4. Designed to match NJ4 tone.",
    confidence="estimated",
)

pickups["bartolini_9j"] = Pickup(
    name="Bartolini 9J",
    R=6400, L=4.4, C=200e-12,
    architecture="split-coil", magnet="Ceramic", hum_cancel=True,
    notes="R from Bartolini spec (6.1k/6.7k). Dual-coil inline BLADE design. "
          "Ceramic magnets + carbon steel. Resonant peak 4.9-5.7kHz. "
          "L revised to 4.4H using ceramic+steel L/R ratio 0.69 H/kohm. "
          "Low R but high L and C from blade core -> darker than R suggests.",
    confidence="estimated",
)

pickups["delano_jmvc4"] = Pickup(
    name="Delano JMVC 4 FE",
    R=8500, L=3.0, C=130e-12,
    architecture="split-coil", magnet="Ferrite", hum_cancel=True,
    notes="Split-coil humbucker, 9.5mm ferrite pole pieces. R estimated. "
          "Described as ultra-fast transient, detailed mids, brilliant highs. German-made.",
    confidence="estimated",
)

pickups["emg_jv"] = Pickup(
    name="EMG JV (passive)",
    R=8000, L=3.0, C=120e-12,
    architecture="split-coil", magnet="Alnico 5", hum_cancel=True,
    notes="Passive split-coil side-by-side. Alnico 5. Solderless system. "
          "Vintage-voiced. R and L estimated. Limited published data.",
    confidence="estimated",
)

pickups["reverend_jazzbomb"] = Pickup(
    name="Reverend Jazz Bomb",
    R=11000, L=7.15, C=155e-12,
    architecture="split-coil", magnet="Ceramic", hum_cancel=True,
    notes="Split-coil with rail/blade pole pieces (confirmed by Tone Merchants: "
          "'hum cancelling, split coils with rails'). NOT a dual-coil humbucker. "
          "Asymmetric set: neck 8k/42AWG, bridge 14k/43AWG. Using averaged values. "
          "L estimated from ceramic+steel L/R ratio ~0.65 H/kohm.",
    confidence="estimated",
)

pickups["wilde_j45"] = Pickup(
    name="Wilde J-45N",
    R=20000, L=4.0, C=150e-12,
    architecture="stacked", magnet="Alnico 2", hum_cancel=True,
    notes="Bill Lawrence 'Noisefree' coaxial coil design: two concentric coils "
          "(inner inside outer), separated by air gap. R=20k and L~4H confirmed "
          "by Bill (Wilde Pickups), April 2026. 9 evenly spaced pole pieces. "
          "C=150pF estimated (coaxial air gap: lower than tight stacks "
          "like SJB-5/Ultra Noiseless at 180-200pF, higher than split-coils). "
          "J-45L variant exists: L~5H, thinner wire, higher R, same coaxial design.",
    confidence="measured",
)

pickups["nordstrand_bigj_blade"] = Pickup(
    name="Nordstrand Big J-Blade",
    R=9000, L=5.5, C=160e-12,
    architecture="single-coil", magnet="Ceramic", hum_cancel=False,
    notes="True single-coil with continuous blade pole piece. Ceramic bar magnets. "
          "Available in Warm & Wooly (darker) and Clean & Clear (brighter) winds. "
          "Using Warm & Wooly specs. R, L, C all estimated. "
          "Hums when soloed like all true single-coils.",
    confidence="estimated",
)


# =============================================================================
# CIRCUIT MODEL
# =============================================================================

def parallel(*impedances):
    Y = sum(1.0 / (Z + 1e-30) for Z in impedances)
    return 1.0 / Y


# =============================================================================
# STRING POSITION WEIGHTING
# =============================================================================

POSITION_60S_NECK = 0.28
POSITION_60S_BRIDGE = 0.13
POSITION_70S_NECK = 0.26
POSITION_70S_BRIDGE = 0.11

def string_position_weight(f, x_rel):
    """Harmonic amplitude envelope from pickup position on the string.
    Smoothed in log-frequency space to extract the tonal envelope."""
    fundamentals = np.array([41.2, 55.0, 73.4, 98.0])
    weights = np.zeros_like(f, dtype=float)
    for f0 in fundamentals:
        n = f / f0
        weights += np.abs(np.sin(n * np.pi * x_rel))
    weights /= len(fundamentals)
    window = max(3, len(f) // 10)
    kernel = np.ones(window) / window
    weights = np.convolve(weights, kernel, mode='same')
    idx_ref = np.argmin(np.abs(f - 200))
    ref = weights[idx_ref]
    if ref > 1e-10:
        weights /= ref
    return weights


def frequency_response(pickup, f, R_vol=250e3, R_tone=250e3,
                        C_cable=500e-12, R_load=1e6, tone_pos=1.0,
                        position=None):
    """
    Standard passive J-Bass circuit at tone@10:
    pickup -> 250K volume -> 250K tone (fully open) -> cable -> 1M Hi-Z
    Returns dB normalized to 200Hz.
    """
    w = 2 * np.pi * f
    Z_series_pu = pickup.R + 1j * w * pickup.L
    Z_C_pu = 1.0 / (1j * w * pickup.C + 1e-30)

    Z_cable = 1.0 / (1j * w * C_cable + 1e-30)
    Z_ext = parallel(Z_cable, R_load + 0j)

    R_upper = max(R_tone * tone_pos, 0.1)
    R_lower = max(R_tone * (1 - tone_pos), 0.1)
    Z_tone = R_upper + R_lower

    Z_total_load = parallel(Z_C_pu, parallel(R_vol + 0j, parallel(R_tone + 0j, Z_ext)))

    H = Z_total_load / (Z_series_pu + Z_total_load)

    if position is not None:
        H = H * string_position_weight(f, position)

    idx_200 = np.argmin(np.abs(f - 200))
    ref = np.abs(H[idx_200])
    if ref > 0:
        H = H / ref

    return 20 * np.log10(np.abs(H) + 1e-30)


# =============================================================================
# SPECTRAL BALANCE METRICS
# =============================================================================

def spectral_balance(resp_db, f):
    """
    Compute perceived 'thickness' as the energy balance between the low-mid
    region (80-400Hz) and the resonant peak region (1.5-6kHz).

    Returns a dict with:
      - low_mid_energy: mean dB in 80-400Hz band
      - peak_energy: mean dB in 1.5-6kHz band
      - balance: low_mid - peak (higher = thicker, lower = thinner)
      - peak_prominence: max dB in 1-8kHz minus mean dB in 80-400Hz
    """
    power = 10 ** (resp_db / 10.0)

    mask_low = (f >= 80) & (f <= 400)
    mask_peak = (f >= 1500) & (f <= 6000)
    mask_peak_wide = (f >= 1000) & (f <= 8000)

    low_mid_power = np.mean(power[mask_low])
    peak_power = np.mean(power[mask_peak])

    low_mid_db = 10 * np.log10(low_mid_power + 1e-30)
    peak_db = 10 * np.log10(peak_power + 1e-30)

    peak_max_db = np.max(resp_db[mask_peak_wide])
    low_mid_mean_db = np.mean(resp_db[mask_low])

    return {
        'low_mid_energy': low_mid_db,
        'peak_energy': peak_db,
        'balance': low_mid_db - peak_db,
        'peak_prominence': peak_max_db - low_mid_mean_db,
    }


def cap_shift_response(pickup, f, C_added, R_vol=250e3, R_tone=250e3,
                        C_cable=500e-12, R_load=1e6):
    """
    Frequency response with added parallel capacitance across the pickup.
    Simulates a switchable cap mod that shifts the resonant peak down.
    """
    w = 2 * np.pi * f
    Z_series_pu = pickup.R + 1j * w * pickup.L
    C_total = pickup.C + C_added
    Z_C_pu = 1.0 / (1j * w * C_total + 1e-30)
    Z_cable = 1.0 / (1j * w * C_cable + 1e-30)
    Z_ext = parallel(Z_cable, R_load + 0j)
    Z_total_load = parallel(Z_C_pu, parallel(R_vol + 0j, parallel(R_tone + 0j, Z_ext)))
    H = Z_total_load / (Z_series_pu + Z_total_load)
    idx_200 = np.argmin(np.abs(f - 200))
    ref = np.abs(H[idx_200])
    if ref > 0:
        H = H / ref
    return 20 * np.log10(np.abs(H) + 1e-30)


# =============================================================================
# ANALYSIS
# =============================================================================

def run_analysis():
    f = np.logspace(np.log10(20), np.log10(20000), 2000)

    results = {}
    for key, pu in pickups.items():
        resp = frequency_response(pu, f)
        idx_1k = np.argmin(np.abs(f - 1000))
        idx_3k = np.argmin(np.abs(f - 3000))
        idx_5k = np.argmin(np.abs(f - 5000))
        idx_8k = np.argmin(np.abs(f - 8000))

        sb = spectral_balance(resp, f)

        results[key] = {
            "pickup": pu,
            "response": resp,
            "db_1k": resp[idx_1k],
            "db_3k": resp[idx_3k],
            "db_5k": resp[idx_5k],
            "db_8k": resp[idx_8k],
            "f_res_unloaded": pu.f_res_unloaded,
            "spectral_balance": sb['balance'],
            "peak_prominence": sb['peak_prominence'],
        }

    return f, results


def print_table(results):
    """Print summary table sorted by brightness (5kHz level)."""
    print("\n" + "=" * 120)
    print("JAZZ BASS PICKUP LANDSCAPE -- Sorted by brightness (5kHz level, tone@10, 250K pots, 500pF cable, 1M load)")
    print("=" * 120)

    sorted_keys = sorted(results.keys(), key=lambda k: results[k]["db_5k"], reverse=True)

    arch_sym = {
        "single-coil": "SC",
        "split-coil": "SP",
        "stacked": "ST",
    }

    print(f"\n{'Pickup':<30} {'Arch':>4} {'Hum':>4} {'R':>7} {'L(H)':>6} {'f_res':>7} {'Q':>5} "
          f"{'1kHz':>6} {'3kHz':>6} {'5kHz':>6} {'8kHz':>6} {'Bal':>6} {'Prom':>5} {'Conf':<8}")
    print("-" * 140)

    for key in sorted_keys:
        r = results[key]
        pu = r["pickup"]
        arch = arch_sym.get(pu.architecture, "??")
        hum = "Y" if pu.hum_cancel else "N"

        f_res_khz = pu.f_res_unloaded / 1000
        print(f"{pu.name:<30} {arch:>4} {hum:>4} {pu.R:>7.0f} {pu.L:>6.2f} {f_res_khz:>6.1f}k {pu.Q_factor:>5.1f} "
              f"{r['db_1k']:>+6.1f} {r['db_3k']:>+6.1f} {r['db_5k']:>+6.1f} {r['db_8k']:>+6.1f} "
              f"{r['spectral_balance']:>+5.1f} {r['peak_prominence']:>5.1f} "
              f"{pu.confidence:<8}")

    print("\n" + "-" * 140)
    print("Arch: SC=single-coil  SP=split-coil  ST=stacked")
    print("Hum:  Y=hum-cancelling always  N=hums when soloed (RWRP pairs cancel in both-on)")
    print("Bal:  Spectral balance (80-400Hz energy vs 1.5-6kHz energy). Closer to 0 = thicker/fuller.")
    print("Prom: Peak prominence (max resonant peak dB above low-mid floor). Higher = sharper peak.")
    print("All dB values relative to 200Hz. Circuit: 250K vol + 250K tone @ 10 + 500pF cable + 1M load.")
    print("Active pickups (EMG J/JX) not included -- different electrical model.\n")

    # Architecture summary
    print("=" * 80)
    print("ARCHITECTURE GROUPS -- Why this matters more than brand")
    print("=" * 80)

    for arch_name, arch_code in [("single-coil", "SC"), ("split-coil", "SP"), ("stacked", "ST")]:
        group = [k for k in sorted_keys if results[k]["pickup"].architecture == arch_name]
        if group:
            levels = [results[k]["db_5k"] for k in group]
            print(f"\n  {arch_code} {arch_name.upper()} ({len(group)} pickups)")
            print(f"     5kHz range: {min(levels):+.1f} to {max(levels):+.1f} dB")
            print(f"     Hum-cancel: {'Yes' if arch_name != 'single-coil' else 'Only when both pickups on (RWRP)'}")
            names = [results[k]["pickup"].name for k in group]
            print(f"     Pickups: {', '.join(names)}")


def plot_landscape(f, results, output_dir="output"):
    """Generate the complete landscape visualization."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    os.makedirs(output_dir, exist_ok=True)

    colors = {
        "single-coil": "#4A90D9",
        "split-coil": "#2ECC71",
        "stacked": "#E74C3C",
    }
    arch_labels = {
        "single-coil": "Single-Coil (hums when soloed)",
        "split-coil": "Split-Coil Side-by-Side (hum-cancelling)",
        "stacked": "Stacked Humbucker (hum-cancelling, darker)",
    }

    fig, axes = plt.subplots(3, 1, figsize=(16, 20), gridspec_kw={'height_ratios': [3, 3, 2]})

    # Top panel: ALL pickups overlaid, colored by architecture
    ax1 = axes[0]
    ax1.set_title("Jazz Bass Pickup Landscape -- All Pickups by Architecture",
                   fontsize=14, fontweight='bold', pad=15)

    sorted_keys = sorted(results.keys(), key=lambda k: results[k]["db_5k"], reverse=True)

    for key in sorted_keys:
        r = results[key]
        pu = r["pickup"]
        color = colors[pu.architecture]
        ax1.semilogx(f, r["response"], color=color, alpha=0.7, linewidth=1.2)

    legend_elements = [
        Line2D([0], [0], color=colors["single-coil"], lw=2, label=arch_labels["single-coil"]),
        Line2D([0], [0], color=colors["split-coil"], lw=2, label=arch_labels["split-coil"]),
        Line2D([0], [0], color=colors["stacked"], lw=2, label=arch_labels["stacked"]),
    ]
    ax1.legend(handles=legend_elements, fontsize=9, loc='lower left')
    ax1.set_xlim(200, 20000)
    ax1.set_ylim(-30, 10)
    ax1.set_ylabel("Level (dB re: 200Hz)")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.grid(True, alpha=0.3, which='both')
    ax1.axvline(5000, color='gray', ls=':', alpha=0.5)
    ax1.text(5200, 8, "5kHz reference", fontsize=8, color='gray')

    # Middle panel: Hum-cancelling only
    ax2 = axes[1]
    ax2.set_title("Hum-Cancelling Pickups Only -- The Decision Space",
                   fontsize=14, fontweight='bold', pad=15)

    hc_keys = [k for k in sorted_keys if results[k]["pickup"].hum_cancel]

    for key in hc_keys:
        r = results[key]
        pu = r["pickup"]
        color = colors[pu.architecture]
        ax2.semilogx(f, r["response"], color=color, alpha=0.6, linewidth=1.2,
                      label=pu.name)

    ax2.legend(fontsize=7.5, loc='lower left', ncol=2)
    ax2.set_xlim(200, 20000)
    ax2.set_ylim(-30, 10)
    ax2.set_ylabel("Level (dB re: 200Hz)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.grid(True, alpha=0.3, which='both')
    ax2.axvline(5000, color='gray', ls=':', alpha=0.5)

    # Bottom panel: Character map scatter plot
    ax3 = axes[2]
    ax3.set_title("Character Map -- Brightness (5kHz) vs DC Resistance",
                   fontsize=14, fontweight='bold', pad=15)

    for key in sorted_keys:
        r = results[key]
        pu = r["pickup"]
        color = colors[pu.architecture]

        marker = 'o' if pu.hum_cancel else 's'
        ax3.scatter(pu.R / 1000, r["db_5k"], c=color, s=60,
                   marker=marker, edgecolors='none', zorder=3)

        ax3.annotate(pu.name, (pu.R / 1000, r["db_5k"]),
                     xytext=(0.15, 0.3), textcoords='offset fontsize',
                     fontsize=7, color=color, alpha=0.9,
                     ha='left', va='bottom')

    scatter_legend = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=colors["single-coil"],
               markersize=8, label="Single-coil (squares)"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors["split-coil"],
               markersize=8, label="Split-coil (circles)"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors["stacked"],
               markersize=8, label="Stacked (circles)"),
    ]
    ax3.legend(handles=scatter_legend, fontsize=8, loc='lower left')
    ax3.set_xlabel("DC Resistance (kohm)")
    ax3.set_ylabel("5kHz Level (dB) -- higher = brighter")
    ax3.grid(True, alpha=0.3)

    ax3.axhspan(-6, 0, alpha=0.05, color='green', zorder=0)
    ax3.axhspan(-12, -6, alpha=0.05, color='yellow', zorder=0)
    ax3.axhspan(-25, -12, alpha=0.05, color='red', zorder=0)
    ax3.text(17.5, -3, "BRIGHT", fontsize=9, color='green', alpha=0.5, ha='right')
    ax3.text(17.5, -9, "WARM", fontsize=9, color='orange', alpha=0.5, ha='right')
    ax3.text(17.5, -16, "DARK", fontsize=9, color='red', alpha=0.5, ha='right')

    plt.tight_layout()
    outpath = os.path.join(output_dir, "pickup_landscape.png")
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {outpath}")
    plt.close()


def validate_lr_ratios():
    """Check L/R ratios of measured pickups against the 0.41 H/kohm assumption."""
    print("\n" + "=" * 80)
    print("L/R RATIO VALIDATION -- Measured pickups vs 0.41 H/kohm assumption")
    print("=" * 80)
    print(f"\n{'Pickup':<30} {'R(kohm)':>7} {'L(H)':>6} {'L/R':>6} {'vs 0.41':>8} {'Magnet':<20}")
    print("-" * 80)

    measured = [(k, pu) for k, pu in pickups.items() if pu.confidence == "measured"]
    ratios = []
    for key, pu in sorted(measured, key=lambda x: x[1].lr_ratio):
        ratio = pu.lr_ratio
        delta = ((ratio - 0.41) / 0.41) * 100
        ratios.append(ratio)
        print(f"{pu.name:<30} {pu.R/1000:>7.2f} {pu.L:>6.2f} {ratio:>6.3f} {delta:>+7.1f}% {pu.magnet:<20}")

    mean_r = np.mean(ratios)
    std_r = np.std(ratios)
    print("-" * 80)
    print(f"{'Mean':<30} {'':>7} {'':>6} {mean_r:>6.3f} {((mean_r - 0.41) / 0.41) * 100:>+7.1f}%")
    print(f"{'Std dev':<30} {'':>7} {'':>6} {std_r:>6.3f}")
    print(f"{'Range':<30} {'':>7} {'':>6} {min(ratios):.3f}-{max(ratios):.3f}")
    print(f"\nAssumed ratio for estimates: 0.41 H/kohm")
    print(f"Measured mean: {mean_r:.3f} H/kohm ({((mean_r - 0.41) / 0.41) * 100:+.0f}% vs assumption)")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    f, results = run_analysis()

    print_table(results)

    if "--validate" in sys.argv:
        validate_lr_ratios()

    if "--table" not in sys.argv and "--validate" not in sys.argv:
        plot_landscape(f, results)

    print("\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print("""
Architecture predicts brightness more than brand, magnet type, or DC resistance.

  SINGLE-COILS are the brightness ceiling -- but they hum when soloed.

  SPLIT-COIL SIDE-BY-SIDE pickups approach single-coil brightness while
  cancelling hum. The best split-coils sit at the bright end of this
  group thanks to optimized magnets and moderate internal capacitance.

  STACKED HUMBUCKERS cancel hum but mutual coupling between vertically
  stacked coils pushes effective inductance and capacitance up, making
  them inherently darker.

  ACTIVE PICKUPS (EMG J/JX) have built-in preamps that fundamentally
  change the electrical model. They can't be compared in a passive
  circuit sim. Their low output impedance (~10k) means cable capacitance
  has almost no effect.
""")
