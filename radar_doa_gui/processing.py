"""Laptop signal processing: ADC frame -> range profile, range-azimuth map, target cell, covariance.

Conventions (shared with the simulator and the DoA IP cores):
  * ADC frame bytes (DCA1000, complex 1x, 4 LVDS lanes), as written by mmWave Studio
    (repo capture/awr2243.py sidecar):  chirp > sample > [I rx0..3, Q rx0..3], int16 LE.
  * adc cube: [chirp, rx, sample] complex64.
  * Virtual array: virt = tx_slot * 4 + rx, TX0 first (repo data-contract.md section 5);
    8-element lambda/2 ULA with steering a[n] = exp(+j pi n sin(theta)) (IP-core convention).
    `flip_array` reverses the element order (theta -> -theta) if the board sign turns out mirrored.
  * Covariance R = (1/K) sum x x^H over snapshots x (column vectors), normalized to trace 8.

TDM: chirp m uses TX0 for even m and TX2 for odd m; one loop = (TX0, TX2).
     TX2 is transmitted Tc after TX0, so a target with Doppler phase phi per loop (2 Tc)
     adds phi/2 to the TX2 channels; `tdm_phase_comp` removes it using the Doppler peak.
DDM-4: TX0 and TX2 in every chirp, TX2 phase code exp(j * s * pi/2 * m) as seen in the ADC data,
     s = `ddm_tx2_sign`. After the Doppler FFT the TX2 copy of a target is offset by s * N/4 bins.
     The chirp RAM programs +90 deg per chirp; on the AWR2243BOOST that appears as s = -1
     (measured 2026-11-06). For each
     candidate TX0 bin k the score P[k] + P[k+q] - P[k+2q] - P[k+3q] (q = s N/4) is
     maximal when k is the TX0 copy (two occupied and two empty quarter bands).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time

import numpy as np

from .params import NUM_RX, NUM_VIRT, RadarConfig

ANGLE_FFT = 64
RA_ANGLES_DEG = np.linspace(-90.0, 90.0, 181)
_SIN_GRID = 2.0 * np.fft.fftshift(np.fft.fftfreq(ANGLE_FFT))     # -1 .. +1 (exclusive)
DB_FLOOR = -60.0


@dataclass
class ProcOptions:
    clutter_removal: bool = False
    snapshot_mode: str = "frame"        # "frame": one snapshot per frame; "chirp": every TDM loop
    k_frames: int = 16                  # sliding window of frames for the covariance
    fb_averaging: bool = True
    diag_loading: float = 1e-3          # fraction of tr(R)/8 added to the diagonal
    dc_guard_bins: int = 3              # bins near DC excluded from the target search
    tdm_phase_comp: bool = True
    ddm_tx2_sign: int = -1              # observed TX2 Doppler offset (AWR2243BOOST, measured)
    flip_array: bool = False
    reset_on_target_jump: bool = True   # restart the window when the target bin moves > 1 bin
    calibration: np.ndarray | None = None   # 8 complex coefficients, TX0 RX0..3, TX2 RX0..3


@dataclass
class ProcResult:
    frame_id: int
    range_m: np.ndarray                 # display range axis
    range_profile_db: np.ndarray        # [4 rx, range]
    ra_map_db: np.ndarray               # [range, RA_ANGLES_DEG]
    target_bin: int
    target_range_m: float
    doppler_bin: int
    velocity_mps: float                 # sign follows the FFT convention (not verified on hardware)
    snapshot: np.ndarray                # 8 complex
    covariance: np.ndarray              # 8x8 complex64, trace 8
    snapshots_in_window: int
    frames_in_window: int
    proc_s: float = 0.0
    t_acq: float = 0.0
    extra: dict = field(default_factory=dict)


def parse_adc(raw, num_chirps: int, num_samples: int) -> np.ndarray:
    """DCA1000 frame bytes (or int16 array) -> complex64 [chirp, rx, sample]."""
    a = np.frombuffer(raw, dtype="<i2") if isinstance(raw, (bytes, bytearray, memoryview)) else np.asarray(raw)
    expected = num_chirps * num_samples * 2 * NUM_RX
    if a.size != expected:
        raise ValueError(f"frame has {a.size} int16 values, expected {expected}")
    a = a.reshape(num_chirps, num_samples, 2 * NUM_RX).astype(np.float32)
    x = a[..., :NUM_RX] + 1j * a[..., NUM_RX:]
    return np.ascontiguousarray(x.transpose(0, 2, 1)).astype(np.complex64)


def to_db(power, ref=None):
    ref = np.max(power) if ref is None else ref
    return np.maximum(10 * np.log10(np.maximum(power, 1e-30) / max(ref, 1e-30)), DB_FLOOR).astype(np.float32)


def fb_average(r):
    return (r + r[::-1, ::-1].conj()) / 2


def finalize_covariance(r, fb=True, loading=0.0):
    """FB averaging, diagonal loading (fraction of tr/8), trace normalized to 8 (IP-core fixtures)."""
    r = np.asarray(r, dtype=np.complex128)
    r = (r + r.conj().T) / 2
    if fb:
        r = fb_average(r)
    tr = np.trace(r).real
    if tr <= 0 or not np.isfinite(tr):
        return np.eye(NUM_VIRT, dtype=np.complex64)
    r = r + loading * tr / NUM_VIRT * np.eye(NUM_VIRT)
    return (r / np.trace(r).real * NUM_VIRT).astype(np.complex64)


def ra_map(virt_bins):
    """Bartlett/FFT beamforming. virt_bins [..., 8, range] -> power [range, RA_ANGLES_DEG]."""
    spec = np.fft.fftshift(np.fft.fft(virt_bins, n=ANGLE_FFT, axis=-2), axes=-2)
    p = np.abs(spec) ** 2
    if p.ndim == 3:
        p = p.mean(axis=0)
    p = p.T                                   # [range, sin grid]
    s = np.sin(np.deg2rad(RA_ANGLES_DEG))
    return np.stack([np.interp(s, _SIN_GRID, row) for row in p]) if p.size else p


class Processor:
    def __init__(self, cfg: RadarConfig, opts: ProcOptions):
        self.cfg, self.opts = cfg, opts
        n = cfg.num_samples
        self.n_range = n
        self.range_axis = np.arange(n) * cfg.range_bin_m
        self.n_display = int(max(2, min(np.searchsorted(self.range_axis, cfg.display_range_m, "right"),
                                        int(0.9 * n))))
        self.win_range = np.hanning(n).astype(np.float32)
        nd = cfg.doppler_bins
        self.win_doppler = np.hanning(nd).astype(np.float32)
        self.window = deque(maxlen=max(1, opts.k_frames))
        self.last_target = None

    def reset(self):
        self.window.clear()
        self.last_target = None

    # -- helpers ------------------------------------------------------------------------------
    def _range_fft(self, adc):
        x = adc - adc.mean(axis=-1, keepdims=True)
        return np.fft.fft(x * self.win_range, axis=-1)

    def _channel_fix(self, v, axis):
        """Apply calibration and optional flip along the virtual-channel axis."""
        if self.opts.calibration is not None:
            shape = [1] * v.ndim
            shape[axis] = NUM_VIRT
            v = v * np.asarray(self.opts.calibration, dtype=np.complex64).reshape(shape)
        if self.opts.flip_array:
            v = np.flip(v, axis=axis)
        return v

    def _target_bin(self, power):
        g = min(self.opts.dc_guard_bins, self.n_display - 1)
        return g + int(np.argmax(power[g:self.n_display]))

    # -- per mode -------------------------------------------------------------------------------
    def _tdm(self, adc):
        cfg, o = self.cfg, self.opts
        rng = self._range_fft(adc)                                    # [chirp, rx, bin]
        virt = rng.reshape(cfg.loops, 2, NUM_RX, -1).reshape(cfg.loops, NUM_VIRT, -1)
        if o.clutter_removal:
            virt = virt - virt.mean(axis=0, keepdims=True)
        profile = (np.abs(virt.reshape(cfg.loops, 2, NUM_RX, -1)) ** 2).mean(axis=(0, 1))
        virt = self._channel_fix(virt, axis=1)
        power = (np.abs(virt) ** 2).mean(axis=(0, 1))
        tb = self._target_bin(power)
        x = virt[:, :, tb]                                             # [loop, 8]
        dop = np.fft.fft(x * self.win_doppler[:, None], axis=0)
        kd = int(np.argmax((np.abs(dop) ** 2).sum(axis=1)))
        phi = 2 * np.pi * np.fft.fftfreq(cfg.loops)[kd]               # phase per loop (2 Tc)
        comp = np.ones(NUM_VIRT, np.complex64)
        if o.tdm_phase_comp:
            # flip_array reverses the channel order, so the TX2 half moves to the front
            tx2 = slice(0, 4) if o.flip_array else slice(4, 8)
            comp[tx2] = np.exp(-1j * phi / 2)
        snap = dop[kd] * comp
        if o.snapshot_mode == "chirp":
            xs = x * comp[None, :]
            r = xs.T @ xs.conj() / xs.shape[0]
            n_snap = xs.shape[0]
        else:
            r = np.outer(snap, snap.conj())
            n_snap = 1
        vel = phi / (2 * np.pi) / (2 * cfg.chirp_period_s) * cfg.wavelength_m / 2
        ra = ra_map(virt[:, :, :self.n_display])
        return profile, ra, tb, kd, vel, snap, r, n_snap

    def _ddm(self, adc):
        cfg, o = self.cfg, self.opts
        nc = cfg.request.chirps_per_frame
        rng = self._range_fft(adc)                                    # [chirp, rx, bin]
        m = np.arange(nc)
        code = np.exp(1j * o.ddm_tx2_sign * np.pi / 2 * m).astype(np.complex64)[:, None, None]
        if o.clutter_removal:
            rng = rng - rng.mean(axis=0, keepdims=True)                              # static TX0
            rng = rng - (rng * code.conj()).mean(axis=0, keepdims=True) * code       # static TX2
        profile = (np.abs(rng) ** 2).mean(axis=0)
        y = np.fft.fft(rng * self.win_doppler[:, None, None], axis=0)             # [doppler, rx, bin]
        p = (np.abs(y) ** 2).sum(axis=1)                                          # [doppler, bin]
        q = o.ddm_tx2_sign * nc // 4
        score = p + np.roll(p, -q, axis=0) - np.roll(p, -2 * q, axis=0) - np.roll(p, -3 * q, axis=0)
        k0 = np.argmax(score, axis=0)                                             # TX0 bin per range bin
        bins = np.arange(y.shape[2])
        v_tx0 = y[k0, :, bins]                                                    # [bin, rx]
        v_tx2 = y[(k0 + q) % nc, :, bins]
        virt = np.concatenate([v_tx0, v_tx2], axis=1).T                           # [8, bin]
        virt = self._channel_fix(virt, axis=0)
        power = p.mean(axis=0)
        tb = self._target_bin(power)
        kd = int(k0[tb])
        snap = virt[:, tb]
        r = np.outer(snap, snap.conj())
        vel = np.fft.fftfreq(nc)[kd] / cfg.chirp_period_s * cfg.wavelength_m / 2
        ra = ra_map(virt[:, :self.n_display])
        return profile, ra, tb, kd, vel, snap, r, 1

    # -- public -----------------------------------------------------------------------------------
    def process(self, adc, frame_id=0, t_acq=0.0) -> ProcResult:
        t0 = time.perf_counter()
        fn = self._tdm if self.cfg.mode == "TDM" else self._ddm
        profile, ra, tb, kd, vel, snap, r, n_snap = fn(adc)
        if (self.opts.reset_on_target_jump and self.last_target is not None
                and abs(tb - self.last_target) > 1):
            self.window.clear()
        self.last_target = tb
        self.window.append((r, n_snap))
        total = sum(n for _, n in self.window)
        r_avg = sum(rr * n for rr, n in self.window) / total
        cov = finalize_covariance(r_avg, self.opts.fb_averaging, self.opts.diag_loading)
        nd = self.n_display
        return ProcResult(
            frame_id=frame_id, range_m=self.range_axis[:nd],
            range_profile_db=to_db(profile[:, :nd]), ra_map_db=to_db(ra),
            target_bin=tb, target_range_m=float(self.range_axis[tb]), doppler_bin=kd,
            velocity_mps=float(vel), snapshot=snap.astype(np.complex64), covariance=cov,
            snapshots_in_window=total, frames_in_window=len(self.window),
            proc_s=time.perf_counter() - t0, t_acq=t_acq)
