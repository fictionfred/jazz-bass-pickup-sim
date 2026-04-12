# Jazz Bass Pickup Sim

Frequency response simulator for passive Jazz Bass electronics. Pure Python math — no SPICE needed.

Models the complete signal chain from pickup coil to audio interface input, answering questions like:

- How does pickup architecture (single-coil vs split-coil vs stacked) affect tone?
- What does changing the tone cap from 47nF to 22nF actually do?
- 250K vs 500K pots — how much difference?
- Treble bleed: worth it on bass?
- Series vs parallel wiring
- How much does cable capacitance matter?

## What's here

**`pickup_landscape.py`** — Database of 30 J-Bass pickups with RLC parameters (measured where available, estimated from published specs otherwise). Generates a landscape chart showing every pickup's frequency response, grouped by architecture. Includes spectral balance metrics and L/R ratio validation against measured data.

**`jazz_bass_tone.py`** — Interactive tone circuit simulator. Models volume/tone/blend pots, treble bleed networks, NoLoad pots, series/parallel switching, cable capacitance, and string position weighting. Runs every combination and generates comparison plots.

## Quick start

```bash
pip install numpy matplotlib

# Full landscape analysis + plots
python3 pickup_landscape.py

# Table only (no plots)
python3 pickup_landscape.py --table

# Validate L/R ratio assumptions against measured data
python3 pickup_landscape.py --validate

# Full tone circuit analysis + plots
python3 jazz_bass_tone.py

# Numbers only
python3 jazz_bass_tone.py --summary
```

Plots save to `./output/`.

## The model

Each pickup is a lumped RLC circuit:

```
V_emf ---[R + jωL]---+--- output
                      |
                     [C]
                      |
                     GND
```

- **R**: DC resistance (measured with a multimeter)
- **L**: Inductance (sets the resonant frequency together with C)
- **C**: Distributed capacitance between coil windings (hard to measure, often estimated)

The transfer function is a voltage divider between the pickup's source impedance (R + jωL) and the total load (pickup C in parallel with pots, cable capacitance, and interface impedance). This produces a resonant peak — the pickup's "voice" — followed by a 2-pole rolloff.

### Data quality

Each pickup has a confidence rating:

- **measured** — Full RLC from independent measurements (GuitarNutz2/Echoes of Mars bode plots, manufacturer-published L values confirmed by third party)
- **derived** — R and L from manufacturer, C estimated from a measured pickup of the same architecture
- **estimated** — R from manufacturer, L estimated from L/R ratio (~0.41 H/kohm for Alnico 5), C estimated from architecture

The notes field on each pickup documents exactly where every value came from.

## String position weighting

The simulator includes an optional model for how pickup position on the string affects the harmonic spectrum. A pickup at position *x* from the bridge senses the *n*th harmonic with amplitude sin(*n* * pi * *x* / *L*). Bridge pickups emphasise upper harmonics; neck pickups get a stronger fundamental.

Standard J-Bass positions are included for both 60s and 70s spacing.

## Adding your own pickups

In `pickup_landscape.py`, add a new entry to the `pickups` dict:

```python
pickups["my_pickup"] = Pickup(
    name="My Custom Pickup",
    R=8000,           # DC resistance in ohms
    L=3.5,            # Inductance in henries
    C=130e-12,        # Distributed capacitance in farads
    architecture="split-coil",
    magnet="Alnico 5",
    hum_cancel=True,
    notes="Where you got the values from",
    confidence="estimated",
)
```

If you only have DC resistance and a loaded resonant peak frequency from the manufacturer:

```python
from pickup_landscape import L_from_loaded_peak

# Derive inductance from a loaded peak spec
L = L_from_loaded_peak(f_peak=3500, C_pu=130e-12, C_cable=500e-12)
```

In `jazz_bass_tone.py`, add to the `PICKUPS` dict:

```python
PICKUPS["my_pickup"] = Pickup(
    name="My Custom Pickup",
    R=8000,
    L=3.5,
    C=130e-12,
)
```

## Companion articles

This code was written alongside a series of articles about upgrading a Fender American Special Jazz Bass for studio recording:

**[Jazz Bass Special series on toyrobot.studio](https://toyrobot.studio/series/jazz-bass-special/)**

## Requirements

- Python 3.8+
- numpy
- matplotlib (optional — needed for plots, not for analysis)

## License

MIT
