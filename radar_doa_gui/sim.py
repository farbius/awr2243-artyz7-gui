"""Synthetic AWR2243 frames (simulation mode) in the DCA1000 byte layout.

Each target: range [m], azimuth [deg] (steering exp(+j pi p sin(theta)), p = virtual index),
radial velocity [m/s] (positive -> positive Doppler frequency), SNR per ADC sample [dB],
fluctuating (random phase per frame, so two targets in one cell are uncorrelated over frames).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import C0, NUM_RX, RadarConfig

NOISE_COUNTS = 8.0


@dataclass
class Target:
    range_m: float = 6.0
    azimuth_deg: float = -20.0
    velocity_mps: float = 0.0
    snr_db: float = 0.0
    fluctuating: bool = True


DEFAULT_TARGETS = (Target(6.0, -20.0, 0.0, 0.0), Target(6.0, 30.0, 0.0, -3.0))


class RadarSimulator:
    def __init__(self, cfg: RadarConfig, targets=DEFAULT_TARGETS, ddm_tx2_sign=-1, seed=0,
                 clutter=True):
        self.cfg, self.targets, self.sign = cfg, list(targets), ddm_tx2_sign
        self.rng = np.random.default_rng(seed)
        self.clutter = clutter
        self.frame = 0

    def adc(self) -> np.ndarray:
        """complex [chirp, rx, sample] in ADC counts."""
        cfg = self.cfg
        nc, n = cfg.request.chirps_per_frame, cfg.num_samples
        m = np.arange(nc)[:, None, None]
        rx = np.arange(NUM_RX)[None, :, None]
        t = (np.arange(n) / cfg.fs_hz)[None, None, :]
        tm = m * cfg.chirp_period_s + self.frame * 0.0
        x = np.zeros((nc, NUM_RX, n), np.complex128)
        targets = list(self.targets)
        if self.clutter:   # static, non-fluctuating reflector near the radar (bridge/desk)
            targets.append(Target(1.2, 0.0, 0.0, -6.0, fluctuating=False))
        for tg in targets:
            amp = NOISE_COUNTS * 10 ** (tg.snr_db / 20)
            ph0 = self.rng.uniform(0, 2 * np.pi) if tg.fluctuating else 0.0
            fb = 2 * cfg.slope_hz_per_s * tg.range_m / C0
            fd = 2 * tg.velocity_mps / cfg.wavelength_m
            u = np.pi * np.sin(np.deg2rad(tg.azimuth_deg))
            base = np.exp(1j * (2 * np.pi * fb * t + 2 * np.pi * fd * tm + ph0))
            if cfg.mode == "TDM":
                slot = (m % 2)
                x += amp * base * np.exp(1j * u * (slot * NUM_RX + rx))
            else:
                code = np.exp(1j * self.sign * np.pi / 2 * m)
                x += amp * base * (np.exp(1j * u * rx) + code * np.exp(1j * u * (NUM_RX + rx)))
        x += NOISE_COUNTS / np.sqrt(2) * (self.rng.standard_normal(x.shape) + 1j * self.rng.standard_normal(x.shape))
        self.frame += 1
        return x

    def frame_bytes(self) -> bytes:
        """One frame in the DCA1000 layout: chirp > sample > [I rx0..3, Q rx0..3], int16 LE."""
        x = self.adc().transpose(0, 2, 1)                               # [chirp, sample, rx]
        out = np.empty(x.shape[:2] + (2 * NUM_RX,), dtype="<i2")
        out[..., :NUM_RX] = np.clip(np.round(x.real), -32768, 32767)
        out[..., NUM_RX:] = np.clip(np.round(x.imag), -32768, 32767)
        return out.tobytes()
