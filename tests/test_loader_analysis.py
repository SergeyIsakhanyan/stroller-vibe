import zipfile

import numpy as np
import pandas as pd
import pytest

from strollervibe.analysis import Settings, analyze_folder, analyze_walk, rotation_to_z, stable_region
from strollervibe.loader import G, load_recording
from strollervibe.report import write_outputs
from strollervibe.synth import make_demo_dataset, synth_walk, write_phyphox_zip


@pytest.fixture(scope="module")
def walk():
    return synth_walk("paving_slabs", 4.0, 90.0, seed=11)


def test_phyphox_zip_roundtrip(tmp_path, walk):
    t, acc = walk
    p = tmp_path / "Acceleration with g 2026-10-21.zip"
    write_phyphox_zip(p, t, acc)
    rec = load_recording(p)
    assert rec.includes_gravity
    assert rec.device.get("deviceModel") == "Synthetic phone"
    assert len(rec.t) == len(t)
    assert np.allclose(rec.acc, acc, atol=1e-6)


@pytest.mark.parametrize("sep,decimal", [(",", "."), ("\t", "."), (";", ","), ("\t", ",")])
def test_csv_variants(tmp_path, walk, sep, decimal):
    t, acc = walk
    df = pd.DataFrame({"Time (s)": t, "Acceleration x (m/s^2)": acc[:, 0],
                       "Acceleration y (m/s^2)": acc[:, 1], "Acceleration z (m/s^2)": acc[:, 2]})
    p = tmp_path / "rec.csv"
    df.to_csv(p, sep=sep, decimal=decimal, index=False)
    rec = load_recording(p)
    assert np.allclose(rec.acc, acc, atol=1e-6)


def test_values_in_g_are_converted(tmp_path, walk):
    t, acc = walk
    df = pd.DataFrame({"time": t, "gFx": acc[:, 0] / G, "gFy": acc[:, 1] / G, "gFz": acc[:, 2] / G})
    p = tmp_path / "toolbox.csv"
    df.to_csv(p, index=False)
    rec = load_recording(p)
    assert np.allclose(rec.acc, acc, atol=1e-6)


def test_unitless_g_values_are_converted(tmp_path, walk):
    t, acc = walk
    df = pd.DataFrame({"t": t, "x": acc[:, 0] / G, "y": acc[:, 1] / G, "z": acc[:, 2] / G})
    p = tmp_path / "device.csv"
    df.to_csv(p, index=False)
    assert np.allclose(load_recording(p).acc, acc, atol=1e-6)


@pytest.mark.parametrize("units", [True, False])
def test_rough_linear_acceleration_is_not_mistaken_for_g(tmp_path, units):
    # Without gravity, cobblestones at 5 km/h give a median |a| of about 1 m/s².
    t, acc = synth_walk("cobblestones", 5.0, 60.0, seed=3, include_gravity=False)
    suffix = " (m/s^2)" if units else ""
    df = pd.DataFrame({"time": t, **{f"Linear Acceleration {a}{suffix}": acc[:, i] for i, a in enumerate("xyz")}})
    p = tmp_path / "lin.csv"
    df.to_csv(p, index=False)
    rec = load_recording(p)
    assert np.allclose(rec.acc, acc, atol=1e-6)
    assert not rec.includes_gravity


def test_epoch_microseconds(tmp_path, walk):
    t, acc = walk
    df = pd.DataFrame({"timestamp": (1.76e9 + t) * 1e6, "ax": acc[:, 0], "ay": acc[:, 1], "az": acc[:, 2]})
    p = tmp_path / "us.csv"
    df.to_csv(p, index=False)
    assert np.allclose(load_recording(p).t, t - t[0], atol=1e-3)


def test_linear_acceleration_export_is_flagged(tmp_path):
    t, acc = synth_walk("smooth_asphalt", 4.0, 60.0, seed=2, include_gravity=False)
    p = tmp_path / "lin.zip"
    write_phyphox_zip(p, t, acc, include_gravity=False)
    res = analyze_walk(p)
    assert not load_recording(p).includes_gravity
    assert any("without g" in w for w in res.warnings)


def test_bad_zip_explains(tmp_path):
    p = tmp_path / "empty.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("notes.txt", "hello")
    with pytest.raises(ValueError, match="no accelerometer table"):
        load_recording(p)


def test_rotation_to_z():
    rng = np.random.default_rng(0)
    for _ in range(20):
        g = rng.standard_normal(3)
        R = rotation_to_z(g)
        assert np.allclose(R @ (g / np.linalg.norm(g)), [0, 0, 1], atol=1e-9)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.allclose(rotation_to_z([0, 0, -1]) @ [0, 0, -1], [0, 0, 1])


def test_tilt_does_not_change_the_result(tmp_path):
    out = {}
    for tilt in (0.0, 25.0):
        t, acc = synth_walk("cobblestones", 4.0, 90.0, seed=5, tilt_deg=tilt)
        p = tmp_path / f"tilt{tilt}.zip"
        write_phyphox_zip(p, t, acc)
        out[tilt] = analyze_walk(p)
    assert out[25.0].metrics["tilt_deg"] == pytest.approx(25.0, abs=0.5)
    assert out[25.0].metrics["aw_z"] == pytest.approx(out[0.0].metrics["aw_z"], rel=0.03)
    assert any("tilted" in w for w in out[25.0].warnings)


def test_handling_is_trimmed_and_stops_excluded(tmp_path):
    t, acc = synth_walk("smooth_asphalt", 4.0, 120.0, seed=9, stop=(50.0, 30.0))
    p = tmp_path / "stop.zip"
    write_phyphox_zip(p, t, acc)
    m = analyze_walk(p).metrics
    # 4 s handling + 3 s trim + 2 s settle at the start; 3 + 3 + 2 at the end.
    assert m["analysed_s"] == pytest.approx(120 - 17, abs=1.5)
    assert m["stopped_s"] == pytest.approx(28, abs=3)
    assert m["aw_z"] > m["aw_z_incl_stops"]
    assert m["g_measured"] == pytest.approx(G, abs=0.02)


def test_user_trim_overrides(tmp_path, walk):
    t, acc = walk
    p = tmp_path / "w.zip"
    write_phyphox_zip(p, t, acc)
    m = analyze_walk(p, {"trim_start_s": "30"}).metrics
    assert m["analysed_s"] == pytest.approx(90 - 30 - 8 - 2, abs=1.5)


def test_speed_from_distance(tmp_path, walk):
    t, acc = walk
    p = tmp_path / "w.zip"
    write_phyphox_zip(p, t, acc)
    m = analyze_walk(p, {"distance_m": "100"}).metrics
    assert m["speed_kmh"] == pytest.approx(100 / m["moving_s"] * 3.6)


def test_demo_end_to_end(tmp_path):
    data, meta = make_demo_dataset(tmp_path / "demo")
    (data / "stray.zip").write_bytes((data / "walk01.zip").read_bytes())
    with open(meta, "a") as fh:
        fh.write("W99,missing.zip,,,,,,,,,,,,,yes\n")
    results, problems = analyze_folder(data, meta)
    assert len(results) == 20
    assert any("stray.zip" in p and "not listed" in p for p in problems)
    assert any("missing.zip" in p for p in problems)
    report, walks = write_outputs(results, problems, tmp_path / "out")
    html = report.read_text()
    assert "SYNTHETIC DEMO DATA" in html
    rice = walks[walks["load"] == "rice_5kg"].groupby("surface")["aw_z"].median()
    assert rice.idxmax() == "cobblestones" and rice.idxmin() == "indoor_floor"
    for f in ("walks_summary.csv", "surface_summary.csv", "stroller_summary.csv", "plots/comparison_by_surface.png"):
        assert (tmp_path / "out" / f).exists()


def test_settle_zero_keeps_the_signal(tmp_path, walk):
    t, acc = walk
    p = tmp_path / "w.zip"
    write_phyphox_zip(p, t, acc)
    m = analyze_walk(p, settings=Settings(settle_s=0.0)).metrics
    assert m["analysed_s"] == pytest.approx(90 - 13, abs=1.5)
    assert np.isfinite(m["aw_z"])


def test_phone_never_still_is_warned(tmp_path):
    t, acc = synth_walk("smooth_asphalt", 4.0, 90.0, seed=4)
    # Spin the phone two full turns about its x axis: no 3-s stretch keeps one orientation.
    th = 2 * 2 * np.pi * t / t[-1]
    c, s = np.cos(th), np.sin(th)
    acc = np.column_stack([acc[:, 0], c * acc[:, 1] - s * acc[:, 2], s * acc[:, 1] + c * acc[:, 2]])
    assert stable_region(acc, 100.0) is None
    p = tmp_path / "spin.zip"
    write_phyphox_zip(p, t, acc)
    assert any("lay still" in w for w in analyze_walk(p).warnings)


def test_duplicates_are_reported(tmp_path, walk):
    t, acc = walk
    data = tmp_path / "data"
    write_phyphox_zip(data / "a.zip", t, acc)
    write_phyphox_zip(data / "b.zip", t, acc)
    write_phyphox_zip(data / "sub" / "a.zip", t, acc)
    meta = tmp_path / "walks.csv"
    meta.write_text("walk_id,file,surface\nW1,a.zip,gravel\nW1,b.zip,grass\nW3,b.zip,dirt_path\n")
    results, problems = analyze_folder(data, meta)
    assert sorted(r.walk_id for r in results) == ["W1", "W1-2"]
    assert any("b.zip" in p and "more than once" in p for p in problems)
    assert any("also called a.zip" in p for p in problems)
    assert any("used twice" in p for p in problems)
    write_outputs(results, problems, tmp_path / "out")
    assert (tmp_path / "out" / "plots" / "W1-2.png").exists()


def test_outputs_from_an_earlier_run_are_cleaned(tmp_path, walk):
    t, acc = walk
    out = tmp_path / "out"
    (out / "plots").mkdir(parents=True)
    (out / "stroller_summary.csv").write_text("old")
    (out / "plots" / "speed.png").write_bytes(b"old")
    p = tmp_path / "w.zip"
    write_phyphox_zip(p, t, acc)
    write_outputs([analyze_walk(p)], [], out)
    assert not (out / "stroller_summary.csv").exists()
    assert not (out / "plots" / "speed.png").exists()


def test_no_walks_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="No walks"):
        write_outputs([], ["x.zip: broken"], tmp_path / "out")
