# stroller-vibe

Measure how much vibration a baby gets in a stroller, with a phone now and a
$16 Seeed XIAO nRF52840 Sense later. Recordings go in; ISO 2631-1 weighted
metrics, charts and a shareable HTML report come out.

There is no vibration standard for infants. Published work so far uses a few
strollers, a few surfaces and short lab-style walks. The goal of this project
is real-world data at scale: many walks, surfaces, strollers and speeds,
published openly.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate   # needs Python 3.10+
pip install -e .
python -m strollervibe demo            # synthetic example -> demo/results/report.html
python -m strollervibe check rec.zip   # sanity-check one recording right after a walk
python -m strollervibe analyze data/ --meta walks.csv --out results/
```

On macOS the built-in `python3` is 3.9, which is too old; install a newer one
(for example `brew install python@3.12`) and use `python3.12 -m venv .venv`.

`analyze` writes `results/report.html` (one self-contained file you can send
to anyone), `walks_summary.csv`, `surface_summary.csv`,
`stroller_summary.csv` (with two or more strollers) and the charts in
`results/plots/`.

To see what a report looks like without installing anything, download
[`docs/demo-report.html`](docs/demo-report.html) and open it in a browser.
The demo data is invented by a script. Its report carries a red banner; never
cite those numbers.

## Recording protocol

### Once

1. Install **phyphox** (free, RWTH Aachen University; iOS and Android).
2. Use the experiment **"Acceleration with g"**, not "Acceleration (without g)".
   The raw values include gravity, which lets the tool find which way is up
   and cut the moments you were handling the phone.
3. Do a 2-minute test walk, export it, and run
   `python -m strollervibe check <file>`. It prints the sample rate, whether the
   phone reads gravity correctly (should be 9.81 m/s²), the tilt and any gaps.
   If the recording has gaps, turn off auto-lock while recording.

### Two kinds of runs

**A. Controlled runs (no baby).** These compare surfaces, speeds and strollers
fairly.

- Put a **5 kg bag of rice** (in a pillowcase) in the bassinet. That's about
  a two-month-old's weight and matches the dummy used in a 2025 stroller study.
- Phone **flat on the mattress, screen up, top of the phone pointing the way
  you walk**, with the rice bag on top of it. The phone then measures what
  reaches the body, like the seat pads ISO measurements use.
- One surface per recording, 2–5 minutes each. Short single-surface runs are
  far more useful than one long mixed walk.
- Repeat each surface at two speeds (normal and brisk), 3 times each.

**B. Everyday walks with your baby.** These show real exposure.

- Phone **flat at the foot end of the mattress**, screen up, top pointing the
  way you walk. Wedge it so it can't slide (under the mattress cover or in a
  thin sock).
- **Never under the baby and never near the face.**
- This measures the mattress next to the baby, not the baby's body; the
  report says so.

### Each recording

1. Press play in phyphox, place the phone, start walking. The tool detects
   when the phone was put down and picked up and cuts those parts.
2. At the end, stop, pick up the phone, press stop.
3. Export: phyphox menu (⋮) → **Export Data** → **CSV (Comma, decimal point)**.
   Save the zip into your `data/` folder (labels may differ slightly between
   versions; any of the CSV formats works).
4. Add a row to `walks.csv` straight away, while you remember.

Stops at crossings are detected and left out of the metrics automatically.
If the start or end of a recording has a different surface, set
`trim_start_s` / `trim_end_s` for that walk.

### Speed

Measure a stretch on a map (say 200 m) and put it in `distance_m`; the tool
divides by the rolling time. Or enter `speed_kmh` directly.

### A first dataset in 3–4 weeks

- Controlled: 4–5 surfaces × 2 speeds × 3 repeats ≈ 24–30 runs of 2–5 min.
- Everyday walks: as many as happen.
- A second stroller (borrow one from a friend) makes it far more interesting,
  because stroller comparison is what manufacturers and parents care about.

## walks.csv

Copy `walks_template.csv`. One row per recording; only `file` is required.

| column | meaning | suggested values |
|---|---|---|
| walk_id | short name | W01, W02… (defaults to file name) |
| file | export file name in `data/` | `Acceleration with g 2026-10-22 10-15-03.zip` |
| date, city | when / where | |
| surface | what you rolled on | indoor_floor, smooth_asphalt, rough_asphalt, paving_slabs, paving_stones, cobblestones, gravel, grass, dirt_path |
| stroller | model and tyre type | "Brand Model (air tyres)" |
| mode | how the baby lies or sits | bassinet, seat, car_seat |
| load | what's in it | rice_5kg, baby, empty |
| placement | where the phone is | under_load, mattress_foot, frame |
| speed_kmh / distance_m | speed, or a measured distance | |
| trim_start_s / trim_end_s | extra seconds to cut from the start / end | |
| notes | anything odd | "kerb drops at 1:20" |
| synthetic | only for generated data | yes |

## What the numbers mean

| metric | what it is |
|---|---|
| **a_wz** | Frequency-weighted RMS vertical acceleration (W<sub>k</sub>), m/s². The main number. |
| a_wx, a_wy | Horizontal, weighted W<sub>d</sub>. x = across the stroller, y = direction of travel. |
| **a_v** | Vibration total value √(a_wx² + a_wy² + a_wz²); what the ISO comfort table uses. |
| MTVV_z | Maximum transient vibration value: the worst 1-second running RMS (kerbs, potholes). |
| VDV_z | Vibration dose value, m/s<sup>1.75</sup>. Grows with duration and punishes shocks. |
| crest_z | Peak ÷ RMS. Past crest 9, MTVV/a_w 1.5 or VDV/(a_w·T<sup>¼</sup>) 1.75, ISO 2631-1 says RMS alone may understate peaks; the walk gets `rms_may_understate` and you should quote MTVV and VDV too. |
| peak_hz | Frequency that contributes most to a_wz. |
| comfort_ref | ISO 2631-1 Annex C band for a_v (adult passengers): < 0.315 not uncomfortable, 0.315–0.63 a little, 0.5–1 fairly, 0.8–1.6 uncomfortable, 1.25–2.5 very, > 2 extremely. |

The weighting filters are the exact analogue responses from ISO 2631-1:1997
Annex A, applied in the frequency domain; the tests check them against the
standard's table to 1%.

## Caveats to state in anything you publish

- ISO 2631-1 describes adults. It's used here as a shared reference scale,
  as published stroller studies do, not as a safety limit for babies.
- Phones sample at roughly 100–500 Hz. At 100 Hz nothing above 50 Hz is
  captured (the ISO band goes to 80 Hz). The XIAO's IMU can sample much faster.
- Phone accelerometers aren't calibrated instruments. `check` compares each
  phone's gravity reading against 9.81 m/s².
- Mattress-surface placement is not the baby's body.

## Later: the XIAO nRF52840 Sense

The loader reads any CSV with a time column and x/y/z acceleration columns, in
m/s² or g. Firmware that writes
`Time (s),Acceleration x (m/s^2),Acceleration y (m/s^2),Acceleration z (m/s^2)`
works with no changes here, so phone and device data can sit in one dataset.

## Publishing the data

- Raw exports + `walks.csv` + `results/` in a public GitHub repository.
  `results/` is in `.gitignore` so test runs don't get committed by accident;
  when a dataset is ready, remove that line or `git add -f results/`.
- Code under MIT ([LICENSE](LICENSE)), data under CC BY 4.0
  ([LICENSE-DATA](LICENSE-DATA)); archive a release on Zenodo for a DOI that
  researchers can cite.
- Recordings hold acceleration only (no GPS). Keep names and photos of your
  child out of the repository.
- phyphox zips also hold `meta/time.csv` (the real clock time of each
  recording) and `meta/device.csv` (the phone model). Together with `date` and
  `city` that shows when and where your family walks. If that bothers you,
  delete `meta/time.csv` from the zips before publishing; the analysis
  doesn't use it.

## Tests

```bash
pip install -e ".[test]" && python -m pytest
```
