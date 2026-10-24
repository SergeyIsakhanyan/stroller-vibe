"""Command line: python -m strollervibe {analyze,check,demo} ..."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from .analysis import Settings, analyze_folder, analyze_walk
from .loader import G


def _settings(args) -> Settings:
    return Settings(trim_s=args.trim, detect_stops=not args.keep_stops, stop_threshold=args.stop_threshold)


def cmd_analyze(args):
    data = Path(args.data)
    meta = Path(args.meta) if args.meta else next((p for p in (Path("walks.csv"), data / "walks.csv", data.parent / "walks.csv") if p.exists()), None)
    if meta is None:
        print("warning: no walks.csv found; every file is analysed as surface 'unknown'", file=sys.stderr)
    from .report import write_outputs
    results, problems = analyze_folder(data, meta, _settings(args))
    try:
        report, walks = write_outputs(results, problems, args.out)
    except ValueError as e:
        sys.exit(str(e))
    print(f"{len(results)} walk(s) analysed -> {report}")
    for p in problems:
        print("  !", p)
    cols = ["walk_id", "surface", "stroller", "aw_z", "av", "mtvv_z", "moving_s"]
    print(walks[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))


def cmd_check(args):
    for f in args.files:
        print(f"\n{f}")
        try:
            res = analyze_walk(f, {}, _settings(args))
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        m = res.metrics
        print(f"  length {m['recording_s']:.0f} s, analysed {m['analysed_s']:.0f} s, rolling {m['moving_s']:.0f} s")
        print(f"  sample rate {m['fs_hz']:.0f} Hz (vibration captured up to {m['fs_hz'] / 2:.0f} Hz)")
        if math.isfinite(m["g_measured"]):  # NaN when the export has no gravity
            print(f"  gravity read as {m['g_measured']:.2f} m/s² (true {G:.2f}); phone tilt {m['tilt_deg']:.0f}°")
        print(f"  a_wz {m['aw_z']:.2f} m/s², a_v {m['av']:.2f} m/s², MTVV_z {m['mtvv_z']:.2f}, peak {m['peak_hz']:.1f} Hz")
        print("  OK" if not res.warnings else "  warnings:\n    - " + "\n    - ".join(res.warnings))


def cmd_demo(args):
    from .report import write_outputs
    from .synth import make_demo_dataset
    out = Path(args.out)
    data, meta = make_demo_dataset(out)
    results, problems = analyze_folder(data, meta, Settings())
    report, _ = write_outputs(results, problems, out / "results", title="Stroller vibration report (synthetic demo)")
    print(f"Synthetic demo data in {data}\nReport: {report}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="strollervibe", description="Stroller vibration analysis (ISO 2631-1).")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--trim", type=float, default=3.0, help="seconds cut after placing / before picking up the phone (default 3)")
        sp.add_argument("--stop-threshold", type=float, default=0.05, help="m/s²; below this the stroller counts as stopped (default 0.05)")
        sp.add_argument("--keep-stops", action="store_true", help="don't exclude stops from the metrics")

    a = sub.add_parser("analyze", help="analyse a folder of recordings and write a report")
    a.add_argument("data", help="folder with phyphox exports (.zip or .csv)")
    a.add_argument("--meta", help="walks.csv (default: ./walks.csv or next to the data folder)")
    a.add_argument("--out", default="results", help="output folder (default: results)")
    common(a)
    a.set_defaults(func=cmd_analyze)

    c = sub.add_parser("check", help="quick check of one or more recordings")
    c.add_argument("files", nargs="+")
    common(c)
    c.set_defaults(func=cmd_check)

    d = sub.add_parser("demo", help="generate a synthetic dataset and its report")
    d.add_argument("--out", default="demo")
    d.set_defaults(func=cmd_demo)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
