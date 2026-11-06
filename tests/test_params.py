import itertools
import math

import pytest

from radar_doa_gui import params as P
from radar_doa_gui.params import RadarRequest, derive, fps_budget, suggest


GRID = [RadarRequest(range_res_m=dr, max_range_m=rmax, display_range_m=rmax, mode=mode)
        for dr, rmax, mode in itertools.product((0.2, 0.25, 0.5, 0.78, 1.0), (5.0, 20.0, 50.0, 90.0),
                                                ("TDM", "DDM"))]


@pytest.mark.parametrize("req", GRID, ids=lambda r: f"{r.mode}-{r.range_res_m}-{r.max_range_m}")
def test_valid_configs_meet_all_constraints(req):
    cfg = derive(req)
    if not cfg.ok:
        pytest.skip("infeasible: " + "; ".join(cfg.issues))
    # S N / Fs = B and dR = c / 2B
    assert cfg.slope_hz_per_s * cfg.num_samples / cfg.fs_hz == pytest.approx(cfg.bandwidth_hz)
    assert cfg.range_res_m == pytest.approx(P.C0 / (2 * cfg.bandwidth_hz))
    # slope quantized to the mmWaveLink LSB, within one LSB of the request
    assert cfg.slope_hz_per_s / P.SLOPE_LSB_HZ_PER_S == pytest.approx(round(cfg.slope_hz_per_s / P.SLOPE_LSB_HZ_PER_S))
    assert abs(cfg.bandwidth_hz - P.C0 / (2 * req.range_res_m)) <= P.SLOPE_LSB_HZ_PER_S * cfg.num_samples / cfg.fs_hz
    # IF and sample-rate limits
    assert cfg.fs_hz <= P.FS_MAX_HZ
    assert cfg.max_if_hz <= min(0.9 * cfg.fs_hz, P.IF_MAX_HZ) * (1 + 1e-6)
    assert cfg.max_range_m >= req.max_range_m * (1 - 1e-6)
    assert cfg.num_samples in P.SAMPLE_CHOICES
    # timing
    assert cfg.adc_start_s + cfg.num_samples / cfg.fs_hz <= cfg.ramp_end_s + 1e-12
    assert cfg.chirp_period_s >= P.MIN_CHIRP_S - 1e-12
    assert cfg.lvds_chirp_s <= cfg.chirp_period_s
    # 76-77 GHz by default
    assert P.STANDARD_BAND_HZ[0] <= cfg.start_freq_hz and cfg.ramp_top_hz <= P.STANDARD_BAND_HZ[1] + 1e3
    # Doppler
    lam = P.C0 / cfg.center_freq_hz
    factor = 8 if req.mode == "TDM" else 4
    assert cfg.v_max_mps == pytest.approx(lam / (factor * cfg.chirp_period_s))
    assert cfg.v_res_mps == pytest.approx(lam / (2 * req.chirps_per_frame * cfg.chirp_period_s))
    assert cfg.min_frame_period_s >= cfg.active_frame_s + P.FRAME_BLANK_S - 1e-12


def test_ddm_doubles_vmax_of_tdm():
    tdm = derive(RadarRequest(mode="TDM"))
    ddm = derive(RadarRequest(mode="DDM"))
    assert tdm.ok and ddm.ok
    assert ddm.v_max_mps == pytest.approx(2 * tdm.v_max_mps)
    assert ddm.v_res_mps == pytest.approx(tdm.v_res_mps)
    assert (tdm.loops, tdm.chirps_in_ram) == (64, 2)
    assert (ddm.loops, ddm.chirps_in_ram) == (32, 4)


def test_fine_resolution_needs_extended_band():
    req = RadarRequest(range_res_m=0.1, max_range_m=10.0, display_range_m=10.0)
    cfg = derive(req)
    assert not cfg.ok and any("band edge" in i for i in cfg.issues)
    ext = derive(RadarRequest(range_res_m=0.1, max_range_m=10.0, display_range_m=10.0, extended_band=True))
    assert ext.ok and ext.ramp_top_hz <= P.EXTENDED_BAND_HZ[1]


def test_too_many_samples_is_reported_and_suggestion_is_valid():
    req = RadarRequest(range_res_m=0.17, max_range_m=300.0, display_range_m=50.0)
    cfg = derive(req)
    assert not cfg.ok
    alt = suggest(req)
    assert alt is not None and alt.ok


def test_bad_inputs():
    assert not derive(RadarRequest(mode="FOO")).ok
    assert not derive(RadarRequest(range_res_m=0)).ok
    assert not derive(RadarRequest(chirps_per_frame=126, mode="DDM")).ok      # not a multiple of 4
    assert not derive(RadarRequest(chirps_per_frame=1024, mode="TDM")).ok     # 512 loops > 255


def test_display_range_is_clipped():
    cfg = derive(RadarRequest(max_range_m=10.0, display_range_m=500.0))
    assert cfg.ok and cfg.display_range_m == pytest.approx(cfg.max_range_m)
    assert cfg.warnings


def test_fps_budget_picks_slowest_stage_and_period():
    cfg = derive(RadarRequest())
    b = fps_budget(cfg, processing_s=0.050, link_s=0.002, ip_latency_s=500e-6)
    assert b.bottleneck == "processing"
    assert b.fps_max == pytest.approx(20.0)
    assert b.frame_period_s >= 0.050 and b.frame_period_s >= cfg.min_frame_period_s
    assert b.stages["board"] == pytest.approx(0.002 + 4 * 500e-6)
    assert b.measured == dict(radar=False, transfer=False, processing=True, board=True)
    fast = fps_budget(cfg, processing_s=1e-4, link_s=1e-4, ip_latency_s=1e-6)
    assert fast.frame_period_s >= cfg.min_frame_period_s
    piped = fps_budget(cfg, processing_s=0.003, link_s=0.005, ip_latency_s=500e-6, pipeline_s=0.011)
    assert piped.bottleneck == "pipeline" and piped.measured["pipeline"]
    assert piped.frame_period_s >= 0.011 * P.FRAME_PERIOD_MARGIN - 1e-4


def test_dca_throughput_estimate():
    # 1456 B every 25 us + wire time: ~39 MB/s
    assert 35e6 < P.dca_bytes_per_s() < 45e6
    assert math.isclose(P.dca_bytes_per_s(0.0), 1456 / (1508 / 125e6))


def test_report_rows_mention_infeasible():
    cfg = derive(RadarRequest(range_res_m=0.05))
    rows = dict(P.report(cfg))
    assert "INFEASIBLE" in rows
