import numpy as np
import pytest

from radar_doa_gui.mock_board import compute
from radar_doa_gui.params import RadarRequest, derive
from radar_doa_gui.processing import ProcOptions, Processor, finalize_covariance, parse_adc
from radar_doa_gui.protocol import THETA_DEG
from radar_doa_gui.sim import RadarSimulator, Target


def cfg_for(mode, **kw):
    cfg = derive(RadarRequest(mode=mode, range_res_m=0.5, max_range_m=20.0, display_range_m=15.0, **kw))
    assert cfg.ok, cfg.issues
    return cfg


def steering(theta_deg):
    return np.exp(1j * np.pi * np.arange(8) * np.sin(np.deg2rad(theta_deg)))


def bartlett_peak(x):
    grid = np.linspace(-90, 90, 3601)
    a = np.exp(1j * np.pi * np.arange(8)[:, None] * np.sin(np.deg2rad(grid)))
    return grid[np.argmax(np.abs(a.conj().T @ x))]


def test_parse_adc_layout_roundtrip():
    cfg = cfg_for("TDM")
    sim = RadarSimulator(cfg, [Target(5.0, 10.0, 0.0, 20.0)], seed=1, clutter=False)
    raw = sim.frame_bytes()
    assert len(raw) == cfg.bytes_per_frame
    adc = parse_adc(raw, cfg.request.chirps_per_frame, cfg.num_samples)
    # sample (chirp 3, sample 7): bytes = [I rx0..3, Q rx0..3]
    words = np.frombuffer(raw, "<i2").reshape(cfg.request.chirps_per_frame, cfg.num_samples, 8)
    np.testing.assert_array_equal(adc[3, :, 7].real, words[3, 7, :4])
    np.testing.assert_array_equal(adc[3, :, 7].imag, words[3, 7, 4:])


@pytest.mark.parametrize("mode", ["TDM", "DDM"])
@pytest.mark.parametrize("theta", [-40.0, 0.0, 17.0])
def test_virtual_array_order_tx0_then_tx2(mode, theta):
    """A single target must give the snapshot x[p] ~ exp(j pi p sin(theta)), p = slot*4 + rx."""
    cfg = cfg_for(mode)
    sim = RadarSimulator(cfg, [Target(7.0, theta, 0.0, 10.0, fluctuating=False)], seed=2, clutter=False)
    proc = Processor(cfg, ProcOptions(k_frames=1))
    res = proc.process(parse_adc(sim.frame_bytes(), cfg.request.chirps_per_frame, cfg.num_samples))
    assert res.target_range_m == pytest.approx(7.0, abs=cfg.range_bin_m)
    x = res.snapshot / res.snapshot[0]
    np.testing.assert_allclose(np.angle(x * steering(theta).conj()), 0, atol=0.05)
    flipped = Processor(cfg, ProcOptions(k_frames=1, flip_array=True))
    sim2 = RadarSimulator(cfg, [Target(7.0, theta, 0.0, 10.0, fluctuating=False)], seed=2, clutter=False)
    res2 = flipped.process(parse_adc(sim2.frame_bytes(), cfg.request.chirps_per_frame, cfg.num_samples))
    assert bartlett_peak(res2.snapshot) == pytest.approx(-theta, abs=0.2)


def test_covariance_definition_and_normalization():
    x = steering(20.0) * 2 + 0.1
    r = finalize_covariance(np.outer(x, x.conj()), fb=False, loading=0.0)
    expected = np.outer(x, x.conj())
    np.testing.assert_allclose(r, expected / np.trace(expected).real * 8, rtol=1e-5, atol=1e-6)
    assert r.dtype == np.complex64 and np.trace(r).real == pytest.approx(8, rel=1e-6)
    r2 = finalize_covariance(np.outer(x, x.conj()), fb=True, loading=0.01)
    np.testing.assert_allclose(r2, r2.conj().T, atol=1e-6)                       # Hermitian
    np.testing.assert_allclose(r2, r2[::-1, ::-1].conj(), atol=1e-6)              # persymmetric
    assert np.linalg.eigvalsh(r2.astype(complex)).min() > 0                       # loaded -> PD


@pytest.mark.parametrize("mode", ["TDM", "DDM"])
def test_two_targets_end_to_end_with_mock_board(mode):
    cfg = cfg_for(mode)
    truth = [-20.0, 30.0]
    sim = RadarSimulator(cfg, [Target(6.0, truth[0], 0.0, 0.0), Target(6.0, truth[1], 0.0, -3.0)], seed=3,
                         clutter=False)
    proc = Processor(cfg, ProcOptions(k_frames=32, clutter_removal=False))
    for _ in range(32):
        res = proc.process(parse_adc(sim.frame_bytes(), cfg.request.chirps_per_frame, cfg.num_samples))
    assert res.target_range_m == pytest.approx(6.0, abs=cfg.range_bin_m)
    spectra, esp, _, errors = compute(res.covariance, 2)
    assert not errors
    np.testing.assert_allclose(np.sort(esp), truth, atol=2.0)
    p = spectra["music"]
    peaks = [i for i in range(1, 511) if p[i] > p[i - 1] and p[i] >= p[i + 1]]
    top = sorted(sorted(peaks, key=lambda i: p[i])[-2:])
    np.testing.assert_allclose(THETA_DEG[top], truth, atol=2.0)


def test_tdm_motion_phase_compensation():
    cfg = cfg_for("TDM")
    v = 0.6 * cfg.v_max_mps
    errs = {}
    for comp in (True, False):
        sim = RadarSimulator(cfg, [Target(8.0, 25.0, v, 10.0, fluctuating=False)], seed=4, clutter=False)
        proc = Processor(cfg, ProcOptions(k_frames=1, tdm_phase_comp=comp))
        res = proc.process(parse_adc(sim.frame_bytes(), cfg.request.chirps_per_frame, cfg.num_samples))
        assert res.velocity_mps == pytest.approx(v, abs=cfg.v_res_mps)
        errs[comp] = abs(bartlett_peak(res.snapshot) - 25.0)
    assert errs[True] < 0.3 and errs[False] > 3.0


def test_ddm_resolves_velocity_beyond_tdm_limit():
    cfg = cfg_for("DDM")
    v = -0.8 * cfg.v_max_mps            # beyond the TDM limit v_max/2 at the same chirp period
    sim = RadarSimulator(cfg, [Target(9.0, -15.0, v, 10.0, fluctuating=False)], seed=5, clutter=False)
    res = Processor(cfg, ProcOptions(k_frames=1)).process(
        parse_adc(sim.frame_bytes(), cfg.request.chirps_per_frame, cfg.num_samples))
    assert res.velocity_mps == pytest.approx(v, abs=cfg.v_res_mps)
    assert bartlett_peak(res.snapshot) == pytest.approx(-15.0, abs=0.3)


@pytest.mark.parametrize("mode", ["TDM", "DDM"])
def test_clutter_removal_suppresses_static_reflector(mode):
    cfg = cfg_for(mode)
    # strong static reflector at 1.2 m (sim clutter) + weaker moving target at 10 m
    sim = RadarSimulator(cfg, [Target(10.0, 5.0, 3.0, -12.0, fluctuating=False)], seed=6, clutter=True)
    raw = sim.frame_bytes()
    adc = parse_adc(raw, cfg.request.chirps_per_frame, cfg.num_samples)
    with_c = Processor(cfg, ProcOptions(k_frames=1, clutter_removal=False, dc_guard_bins=1)).process(adc)
    without = Processor(cfg, ProcOptions(k_frames=1, clutter_removal=True, dc_guard_bins=1)).process(adc)
    assert with_c.target_range_m == pytest.approx(1.2, abs=cfg.range_bin_m)
    assert without.target_range_m == pytest.approx(10.0, abs=cfg.range_bin_m)


def test_chirp_snapshots_and_window():
    cfg = cfg_for("TDM")
    sim = RadarSimulator(cfg, [Target(6.0, 10.0, 0.5, 0.0)], seed=7, clutter=False)
    proc = Processor(cfg, ProcOptions(k_frames=4, snapshot_mode="chirp"))
    for i in range(6):
        res = proc.process(parse_adc(sim.frame_bytes(), cfg.request.chirps_per_frame, cfg.num_samples))
    assert res.frames_in_window == 4
    assert res.snapshots_in_window == 4 * cfg.loops
