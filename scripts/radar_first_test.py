"""First hardware capture: bring-up + TDM profile, a finite run of frames, StopFrame + DCA1000 stop.

    .venv\\Scripts\\python.exe scripts\\radar_first_test.py [--frames 8] [--backoff 6] [--mode TDM]

Checks (numbers, no plots): frames received / dropped / lost packets, ADC peak (saturation),
per-RX noise floor balance (+/-3 dB), strongest range bin, and the laptop chain on the last frame.
Raw ADC frames are kept in captures/ (radar data only; no camera).
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from radar_doa_gui import params as P  # noqa: E402
from radar_doa_gui.processing import ProcOptions, Processor, parse_adc  # noqa: E402
from radar_doa_gui.radar import Awr2243Radar  # noqa: E402
from radar_doa_gui.settings import Settings  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--backoff", type=int, default=6)
    ap.add_argument("--mode", default="TDM")
    ap.add_argument("--period-ms", type=float, default=50.0)
    ap.add_argument("--bringup", choices=("auto", "skip", "force"), default="auto")
    args = ap.parse_args()
    s = Settings()
    req = P.RadarRequest(mode=args.mode, tx_backoff_db=args.backoff)
    cfg = P.derive(req)
    assert cfg.ok, cfg.issues
    period = max(args.period_ms * 1e-3, cfg.min_frame_period_s)
    print(f"{cfg.mode}: N={cfg.num_samples}, Fs={cfg.fs_hz / 1e6:.2f} Msps, S={cfg.slope_hz_per_s / 1e12:.3f} MHz/us, "
          f"Tc={cfg.chirp_period_s * 1e6:.2f} us, {cfg.request.chirps_per_frame} chirps, {cfg.bytes_per_frame} B/frame, "
          f"period {period * 1e3:.1f} ms, backoff {args.backoff} dB")
    radar = Awr2243Radar(s, progress=print)
    frames = []
    try:
        t0 = time.perf_counter()
        radar.configure(cfg, period, bringup=args.bringup)
        print(f"configure: {time.perf_counter() - t0:.1f} s")
        radar.start(args.frames)
        deadline = time.monotonic() + args.frames * period + 5
        while not radar.finished and time.monotonic() < deadline:
            item = radar.read_frame(timeout=0.5)
            if item:
                frames.append(item)
    finally:
        radar.stop()
        radar.close()
    print(f"frames received {len(frames)} / {args.frames}; assembler {radar.stats}")
    if not frames:
        return 1
    out = ROOT / "captures"
    out.mkdir(exist_ok=True)
    stem = out / f"{cfg.mode.lower()}_{datetime.now():%Y%m%d_%H%M%S}"
    np.save(stem.with_suffix(".npy"), np.stack([np.frombuffer(f, "<i2") for f, _ in frames]))
    print(f"saved {stem.with_suffix('.npy')} (frame indices {[i for _, i in frames]})")
    adc = parse_adc(frames[-1][0], cfg.request.chirps_per_frame, cfg.num_samples)
    peak = max(np.abs(adc.real).max(), np.abs(adc.imag).max())
    print(f"ADC peak |I/Q| {peak:.0f} of 32767 ({20 * np.log10(peak / 32767):.1f} dBFS)")
    rng = np.fft.fft((adc - adc.mean(-1, keepdims=True)) * np.hanning(cfg.num_samples), axis=-1)
    pw = (np.abs(rng) ** 2).mean(axis=0)                                  # [rx, bin]
    hi = slice(int(0.5 * cfg.num_samples), int(0.85 * cfg.num_samples))
    noise = 10 * np.log10(pw[:, hi].mean(axis=1))
    print("noise floor per RX [dB]:", np.round(noise, 1), f"spread {noise.max() - noise.min():.1f} dB "
          f"({'PASS' if noise.max() - noise.min() <= 3 else 'FAIL'} +/-3 dB)")
    tot = pw.sum(0)
    for b in np.argsort(tot[3:int(0.9 * cfg.num_samples)])[::-1][:3] + 3:
        print(f"  strong bin {b}: {b * cfg.range_bin_m:.2f} m, {10 * np.log10(tot[b] / np.median(tot)):.1f} dB above median")
    image = 10 * np.log10(tot[int(0.9 * cfg.num_samples):].mean() / tot[3:int(0.9 * cfg.num_samples)].mean())
    print(f"image band (0.9 Fs..Fs) vs valid band: {image:.1f} dB")
    proc = Processor(cfg, ProcOptions(k_frames=len(frames)))
    for f, _ in frames:
        res = proc.process(parse_adc(f, cfg.request.chirps_per_frame, cfg.num_samples))
    print(f"chain: target {res.target_range_m:.2f} m (bin {res.target_bin}), Doppler bin {res.doppler_bin}, "
          f"v {res.velocity_mps:+.2f} m/s, processing {res.proc_s * 1e3:.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
