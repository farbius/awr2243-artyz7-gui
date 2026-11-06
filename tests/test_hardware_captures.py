"""Regression tests on real AWR2243 captures in captures/ (skipped if none are present)."""
import glob

import numpy as np
import pytest

from radar_doa_gui import ROOT
from radar_doa_gui.params import RadarRequest, derive
from radar_doa_gui.processing import ProcOptions, Processor, parse_adc


def latest(prefix):
    files = sorted(glob.glob(str(ROOT / "captures" / f"{prefix}_*.npy")))
    if not files:
        pytest.skip(f"no {prefix} capture")
    return np.load(files[-1])


def test_tdm_capture_iq_orientation():
    """Real echoes at bin k, nothing at the mirror N-k (I first, then Q)."""
    cfg = derive(RadarRequest(mode="TDM", tx_backoff_db=6))
    frames = latest("tdm")
    n = cfg.num_samples
    p = 0
    for f in frames:
        a = parse_adc(f, cfg.request.chirps_per_frame, n)
        p = p + (np.abs(np.fft.fft((a - a.mean(-1, keepdims=True)) * np.hanning(n), axis=-1)) ** 2).mean((0, 1))
    assert all(p[k] > 100 * p[n - k] for k in (3, 4, 5))


def test_ddm_capture_tx2_offset_is_minus_quarter():
    """Static clutter: TX0 at Doppler bin 0, TX2 at -N/4; with the default sign v = 0."""
    cfg = derive(RadarRequest(mode="DDM", tx_backoff_db=6))
    frames = latest("ddm")
    proc = Processor(cfg, ProcOptions(k_frames=len(frames)))
    for f in frames:
        res = proc.process(parse_adc(f, cfg.request.chirps_per_frame, cfg.num_samples))
    assert res.doppler_bin == 0 and res.velocity_mps == 0.0
