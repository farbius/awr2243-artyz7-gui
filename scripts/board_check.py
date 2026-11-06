"""Hardware check of the board link: deploy + init + server, then board vs mock on known covariances.

    .venv\\Scripts\\python.exe scripts\\board_check.py [--force-fpga] [--n 200]

Covariances: music.covariance (analytic and sampled, 1-2 sources) plus the trace-8 output of the
laptop processing chain on simulated frames. Reports ESPRIT angle difference, normalized spectrum
difference and peak bins per core, and the round-trip statistics over --n requests.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import music  # noqa: E402
from radar_doa_gui.board_link import BoardLink  # noqa: E402
from radar_doa_gui.mock_board import compute  # noqa: E402
from radar_doa_gui.params import RadarRequest, derive  # noqa: E402
from radar_doa_gui.processing import ProcOptions, Processor, parse_adc  # noqa: E402
from radar_doa_gui.protocol import THETA_DEG  # noqa: E402
from radar_doa_gui.settings import Settings  # noqa: E402
from radar_doa_gui.sim import RadarSimulator, Target  # noqa: E402


def cases():
    yield "analytic 1 src +23", music.covariance([23.0]), 1
    yield "analytic 2 src -20/+30", music.covariance([-20.0, 30.0]), 2
    yield "sampled 2 src -20/+30", music.covariance([-20.0, 30.0], seed=0, sampled=True, snapshots=256), 2
    yield "sampled 2 src -45/+10", music.covariance([-45.0, 10.0], seed=1, sampled=True, snapshots=64), 2
    for mode in ("TDM", "DDM"):
        cfg = derive(RadarRequest(mode=mode))
        sim = RadarSimulator(cfg, [Target(6.0, -20.0, 0.0, 0.0), Target(6.0, 30.0, 0.0, -3.0)], seed=3, clutter=False)
        proc = Processor(cfg, ProcOptions(k_frames=16))
        for _ in range(16):
            res = proc.process(parse_adc(sim.frame_bytes(), cfg.request.chirps_per_frame, cfg.num_samples))
        yield f"sim chain {mode} -20/+30 (K=16)", res.covariance, 2


def peaks(p, k):
    idx = [i for i in range(1, len(p) - 1) if p[i] > p[i - 1] and p[i] >= p[i + 1]]
    return sorted(sorted(idx, key=lambda i: p[i])[-k:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-fpga", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    link = BoardLink(Settings(), progress=lambda m: print(m))
    t = time.perf_counter()
    link.configure(force_fpga=args.force_fpga)
    print(f"configure: {time.perf_counter() - t:.1f} s; server info {link.info}\n")
    worst = 0.0
    try:
        for name, r, k in cases():
            hw = link.run(np.asarray(r, np.complex64), k, frame_id=1)
            ref_spec, ref_esp, _, errs = compute(np.asarray(r, np.complex64), k)
            print(f"{name}  (K={k})")
            print(f"  ESPRIT board {np.round(hw['esprit_deg'][:k], 3)}  mock {np.round(ref_esp[:k], 3)}  "
                  f"diff {np.max(np.abs(np.subtract(hw['esprit_deg'][:k], ref_esp[:k]))):.4f} deg")
            for alg in ("bartlett", "music", "capon"):
                a, b = hw["spectra"][alg].astype(float), ref_spec[alg].astype(float)
                err = float(np.max(np.abs(a / a.max() - b / b.max())))
                worst = max(worst, err) if alg != "capon" else worst
                pa, pb = peaks(a, k), peaks(b, k)
                print(f"  {alg:8s} max normalized diff {err:.2e}  peaks board {np.round(THETA_DEG[pa], 2)} "
                      f"mock {np.round(THETA_DEG[pb], 2)}  bins {'match' if pa == pb else f'{pa} vs {pb}'}")
            if hw.get("unwritten_cores"):
                print("  UNWRITTEN:", hw["unwritten_cores"])
            lat = {c: v / 1e3 for c, v in hw["latency_ns"].items()}
            print(f"  latency us {lat}  rtt {hw['rtt_s'] * 1e3:.2f} ms  (board {hw['board_s'] * 1e3:.2f} ms)")
            if errs:
                print("  mock notes:", errs)
        r = music.covariance([-20.0, 30.0], seed=0, sampled=True, snapshots=256).astype(np.complex64)
        rtts, boards = [], []
        for i in range(args.n):
            out = link.run(r, 2, frame_id=i)
            rtts.append(out["rtt_s"])
            boards.append(out["board_s"])
        rtts, boards = np.array(rtts) * 1e3, np.array(boards) * 1e3
        print(f"\n{args.n} requests: rtt mean {rtts.mean():.2f} ms, p50 {np.median(rtts):.2f}, "
              f"p95 {np.percentile(rtts, 95):.2f}, max {rtts.max():.2f}; board {boards.mean():.2f} ms; "
              f"transport {np.mean(rtts - boards):.2f} ms; {1e3 / rtts.mean():.0f} req/s")
        print(f"worst normalized Bartlett/MUSIC diff: {worst:.2e}")
    finally:
        link.close()
        print("closed")


if __name__ == "__main__":
    main()
