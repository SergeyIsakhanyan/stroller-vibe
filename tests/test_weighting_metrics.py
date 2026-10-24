import numpy as np
import pytest

from strollervibe import metrics as M
from strollervibe.weighting import apply_weighting, magnitude

# ISO 2631-1:1997 Table 3 weighting factors (x1000).
WK = {0.5: 418, 1.0: 482, 2.0: 531, 4.0: 967, 5.0: 1039, 6.3: 1054, 8.0: 1036, 10: 988, 16: 768, 31.5: 405, 80: 132}
WD = {0.5: 853, 1.0: 1011, 2.0: 890, 4.0: 512, 5.0: 409, 6.3: 323, 8.0: 253, 16: 125, 31.5: 63.2, 80: 21.1}


@pytest.mark.parametrize("name,table", [("Wk", WK), ("Wd", WD)])
def test_weighting_matches_iso_table(name, table):
    for f, expected in table.items():
        assert magnitude(f, name) * 1000 == pytest.approx(expected, rel=0.01), f"{name} at {f} Hz"


@pytest.mark.parametrize("f0", [2.0, 6.3, 16.0])
def test_weighted_rms_of_a_sine(f0):
    fs, dur = 200.0, 60.0
    t = np.arange(int(fs * dur)) / fs
    x = 1.0 * np.sin(2 * np.pi * f0 * t)
    y = apply_weighting(x, fs, "Wk")[int(2 * fs):-int(2 * fs)]
    assert M.rms(y) == pytest.approx(magnitude(f0, "Wk") / np.sqrt(2), rel=0.01)


def test_weighting_removes_gravity_offset():
    fs = 100.0
    x = 9.81 + 0.0 * np.arange(6000)
    y = apply_weighting(x, fs, "Wk")
    assert np.max(np.abs(y)) < 1e-9


def test_vdv_of_a_sine():
    fs, dur, a = 1000.0, 10.0, 2.0
    t = np.arange(int(fs * dur)) / fs
    x = a * np.sin(2 * np.pi * 5 * t)
    assert M.vdv(x, fs) == pytest.approx(a * (3 * dur / 8) ** 0.25, rel=1e-3)


def test_mtvv_finds_the_burst():
    fs = 100.0
    x = np.zeros(6000)
    x[3000:3100] = 2.0 * np.sin(2 * np.pi * 8 * np.arange(100) / fs)  # 1 s burst
    assert M.mtvv(x, fs) == pytest.approx(2.0 / np.sqrt(2), rel=0.02)
    assert len(M.running_rms(x, fs)) == len(x)


def test_comfort_labels_follow_annex_c():
    assert M.comfort_label(0.2) == "not uncomfortable"
    assert M.comfort_label(0.4) == "a little uncomfortable"
    assert M.comfort_label(0.55) == "a little uncomfortable / fairly uncomfortable"
    assert M.comfort_label(3.0) == "extremely uncomfortable"


def test_crest_factor_of_sine():
    x = np.sin(np.linspace(0, 200 * np.pi, 100000))
    assert M.crest_factor(x) == pytest.approx(np.sqrt(2), rel=1e-3)
