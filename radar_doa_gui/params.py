"""Radar parameter derivation for the AWR2243 (TX0 + TX2, RX0..3, complex 1x, 16 bit).

Pure module: no hardware, no GUI. All values in SI units (m, s, Hz, Hz/s, m/s).

Derivation from range resolution dR and maximum range R_max:
    B   = c / (2 dR)                          sampled bandwidth
    N   = smallest of SAMPLE_CHOICES with 0.9 * N * dR >= R_max
          (complex 1x sampling covers beat frequencies 0..Fs, i.e. ranges
          0..N*dR; the top 10 % is the receiver's image band)
    Fs  = largest rate (100 ksps steps) with Fs <= 22.5 Msps,
          max IF = Fs * R_max / (N dR) <= 20 MHz and slope B*Fs/N <= S_max
    S   = B Fs / N, quantized to the mmWaveLink slope LSB; B and dR are then
          recomputed from the quantized slope.
Chirp timing: ADC start, ramp end = ADC start + N/Fs + margin, idle time; the
chirp cycle is at least 13 us (legacy chirp API, DFP user guide).

TDM: chirps alternate TX0, TX2 (two chirp-RAM entries); one TX repeats every
2 Tc, so v_max = lambda / (8 Tc).
DDM-4: TX0 and TX2 in every chirp, TX2 phase k * 90 deg (k = chirp mod 4). The
TX2 copy sits a quarter of the Doppler spectrum away from TX0; the two empty
quarter bands identify TX0, so v_max = lambda / (4 Tc) with chirp period Tc.

Device and link constants below that are not taken from the repository configs
are marked "assumption"; they are checked on hardware in a later step.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

C0 = 299_792_458.0

# --- AWR2243 limits ---------------------------------------------------------------
FS_MAX_HZ = 22.5e6              # complex 1x ADC rate limit (task spec)
FS_MIN_HZ = 2.0e6               # assumption: lowest rate used
FS_STEP_HZ = 100e3              # Fs chosen in 100 ksps steps (Studio takes ksps)
IF_MAX_HZ = 20e6                # IF bandwidth limit
IF_FS_FRACTION = 0.9            # usable fraction of Fs (image band above)
SLOPE_MAX_HZ_PER_S = 100e12     # assumption: conservative ramp-rate limit (100 MHz/us)
SLOPE_LSB_HZ_PER_S = 3.6e9 * 900 / 2**26 * 1e6   # 48.279 kHz/us (awr2243/recording.py)
FREQ_LSB_HZ = 3.6e9 / 2**26
TIME_LSB_S = 10e-9              # idle, ADC start, ramp end
FRAME_TICK_S = 5e-9             # frame periodicity unit
MIN_CHIRP_S = 13e-6             # legacy chirp API minimum cycle (DFP user guide 11.2)
MIN_IDLE_S = 2e-6               # assumption: synthesizer reset time
MAX_RAMP_END_S = 500e-6         # assumption
FRAME_BLANK_S = 0.2e-3          # inter-frame blank time (DFP user guide 11.1, repo)
FRAME_PERIOD_MAX_S = (2**30 - 1) * FRAME_TICK_S   # 5.37 s
MAX_LOOPS = 255
MAX_CHIRP_RAM = 512
STANDARD_BAND_HZ = (76.0e9, 77.0e9)     # fixed infrastructure band (repo requirements.md)
EXTENDED_BAND_HZ = (76.0e9, 81.0e9)
START_FREQ_HZ = 76.1e9                  # as configs/awr2243_common.lua
SAMPLE_CHOICES = (64, 128, 256, 512, 1024)   # ADC ping-pong buffer: N * 4 RX * 4 B <= 16 KiB
NUM_RX, NUM_TX, NUM_VIRT = 4, 2, 8
BYTES_PER_SAMPLE = 4            # complex 16 bit: I + Q, per RX
RX_GAIN_CODE = 94               # as configs/awr2243_common.lua

# --- LVDS / DCA1000 -------------------------------------------------------------------
LVDS_BYTES_PER_S = 4 * 600e6 / 8        # assumption: 4 lanes x 600 Mbps (LvdsClkConfig(1, 1))
DCA_PAYLOAD_BYTES = 1456                # DCA1000 UDP payload per packet
DCA_PACKET_DELAY_S = 25e-6              # CaptureCardConfig_PacketDelay(25) in the repo configs
DCA_WIRE_BYTES = 1456 + 10 + 42         # payload + DCA1000 header + UDP/IP/Ethernet overhead
GIGE_BYTES_PER_S = 125e6

# --- Defaults for the fps model ---------------------------------------------------------
DEFAULT_PROCESSING_S = 5e-3             # until measured
DEFAULT_LINK_S = 5e-3                   # board transport round trip until measured
NUM_CORES = 4                           # Bartlett, MUSIC, ESPRIT, Capon run in sequence
FRAME_PERIOD_MARGIN = 1.25              # programmed period = 1.25 x slowest stage (SSH jitter, GIL sharing)


def dca_bytes_per_s(packet_delay_s: float = DCA_PACKET_DELAY_S) -> float:
    """Estimated sustained DCA1000 Ethernet throughput in B/s (one packet per delay + wire time)."""
    return DCA_PAYLOAD_BYTES / (packet_delay_s + DCA_WIRE_BYTES / GIGE_BYTES_PER_S)


@dataclass(frozen=True)
class RadarRequest:
    """GUI inputs. Lengths in m, times in s."""
    range_res_m: float = 0.5            # 300 MHz, repo requirements.md; ~0.28 m is the 76-77 GHz limit
    max_range_m: float = 20.0
    display_range_m: float = 15.0
    mode: str = "TDM"                   # "TDM" or "DDM"
    chirps_per_frame: int = 128         # physical chirps (TDM: 2 per loop, DDM: 4 per loop)
    extended_band: bool = False
    idle_s: float = 5e-6
    adc_start_s: float = 3e-6
    ramp_margin_s: float = 1e-6
    tx_backoff_db: int = 0


@dataclass(frozen=True)
class RadarConfig:
    """A derived, quantized configuration. `issues` is empty when it is valid."""
    request: RadarRequest
    bandwidth_hz: float = 0.0
    range_res_m: float = 0.0
    max_range_m: float = 0.0            # usable: 0.9 * Fs * c / (2 S), capped by 20 MHz IF
    num_samples: int = 0
    fs_hz: float = 0.0
    slope_hz_per_s: float = 0.0
    start_freq_hz: float = START_FREQ_HZ
    idle_s: float = 0.0
    adc_start_s: float = 0.0
    ramp_end_s: float = 0.0
    chirp_period_s: float = 0.0
    center_freq_hz: float = 0.0
    wavelength_m: float = 0.0
    ramp_top_hz: float = 0.0
    max_if_hz: float = 0.0
    chirps_in_ram: int = 0
    loops: int = 0
    v_max_mps: float = 0.0
    v_res_mps: float = 0.0
    active_frame_s: float = 0.0
    min_frame_period_s: float = 0.0
    bytes_per_chirp: int = 0
    bytes_per_frame: int = 0
    lvds_chirp_s: float = 0.0
    display_range_m: float = 0.0
    angle_res_deg: float = 0.0
    fov_deg: float = 90.0
    issues: tuple = ()
    warnings: tuple = ()

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def mode(self) -> str:
        return self.request.mode

    @property
    def range_bin_m(self) -> float:
        """Range per FFT bin (no zero padding): c Fs / (2 S N)."""
        return C0 * self.fs_hz / (2 * self.slope_hz_per_s * self.num_samples)

    @property
    def doppler_bins(self) -> int:
        """Slow-time FFT length: loops for TDM (per TX), all chirps for DDM."""
        return self.loops if self.mode == "TDM" else self.request.chirps_per_frame


def _ceil_to(value: float, step: float) -> float:
    return math.ceil(value / step - 1e-9) * step


def derive(req: RadarRequest) -> RadarConfig:
    """Derive a valid, quantized configuration; infeasibility is reported in `issues`."""
    issues: list[str] = []
    warns: list[str] = []
    mode = req.mode.upper()
    if mode not in ("TDM", "DDM"):
        return RadarConfig(req, issues=(f"Unknown mode {req.mode!r} (TDM or DDM)",))
    if req.range_res_m <= 0 or req.max_range_m <= 0:
        return RadarConfig(req, issues=("Range resolution and maximum range must be positive",))
    band_lo, band_hi = EXTENDED_BAND_HZ if req.extended_band else STANDARD_BAND_HZ

    b_req = C0 / (2 * req.range_res_m)
    n_needed = req.max_range_m / (IF_FS_FRACTION * req.range_res_m)
    n = next((c for c in SAMPLE_CHOICES if c >= n_needed), None)
    if n is None:
        issues.append(f"R_max/dR needs {math.ceil(n_needed)} ADC samples; at most {SAMPLE_CHOICES[-1]}")
        n = SAMPLE_CHOICES[-1]

    fs_cap = min(FS_MAX_HZ,
                 IF_MAX_HZ * n * req.range_res_m / req.max_range_m,
                 SLOPE_MAX_HZ_PER_S * n / b_req)
    fs = math.floor(fs_cap / FS_STEP_HZ + 1e-9) * FS_STEP_HZ
    if fs < FS_MIN_HZ:
        issues.append(f"Sample rate {fs / 1e6:.2f} Msps below {FS_MIN_HZ / 1e6:.1f} Msps "
                      "(20 MHz IF or slope limit); increase dR or reduce R_max")
        fs = FS_MIN_HZ

    slope = max(1, round(b_req * fs / n / SLOPE_LSB_HZ_PER_S)) * SLOPE_LSB_HZ_PER_S
    if slope > SLOPE_MAX_HZ_PER_S * (1 + 1e-9):
        issues.append(f"Slope {slope / 1e12:.1f} MHz/us above {SLOPE_MAX_HZ_PER_S / 1e12:.0f} MHz/us")
    bandwidth = slope * n / fs
    range_res = C0 / (2 * bandwidth)
    max_if = 2 * slope * req.max_range_m / C0
    if max_if > min(IF_FS_FRACTION * fs, IF_MAX_HZ) * (1 + 1e-6):
        issues.append(f"Max IF {max_if / 1e6:.2f} MHz above min(0.9 Fs, 20 MHz)")
    max_range = min(IF_FS_FRACTION * fs, IF_MAX_HZ) * C0 / (2 * slope)

    sampling = n / fs
    adc_start = _ceil_to(req.adc_start_s, TIME_LSB_S)
    ramp_end = _ceil_to(adc_start + sampling + req.ramp_margin_s, 0.1e-6)
    idle = _ceil_to(max(req.idle_s, MIN_IDLE_S, MIN_CHIRP_S - ramp_end), TIME_LSB_S)
    if idle > req.idle_s + 1e-12:
        warns.append(f"Idle time raised to {idle * 1e6:.2f} us (13 us minimum chirp cycle)")
    if ramp_end > MAX_RAMP_END_S:
        issues.append(f"Ramp end {ramp_end * 1e6:.1f} us above {MAX_RAMP_END_S * 1e6:.0f} us")
    tc = idle + ramp_end
    f0 = START_FREQ_HZ
    ramp_top = f0 + slope * ramp_end
    if f0 < band_lo or ramp_top > band_hi + 1e3:
        b_avail = (band_hi - f0) * sampling / ramp_end
        issues.append(f"Ramp ends at {ramp_top / 1e9:.3f} GHz, above the {band_hi / 1e9:.0f} GHz band edge; "
                      f"dR must be >= {C0 / (2 * b_avail):.3f} m"
                      + ("" if req.extended_band else " (or enable the extended band)"))
    fc = f0 + slope * (adc_start + sampling / 2)
    lam = C0 / fc

    nc = int(req.chirps_per_frame)
    per_loop = 2 if mode == "TDM" else 4
    if nc % per_loop:
        issues.append(f"{mode}: chirps per frame must be a multiple of {per_loop}")
    loops = nc // per_loop
    if not 1 <= loops <= MAX_LOOPS:
        issues.append(f"{mode}: {loops} loops; must be 1..{MAX_LOOPS} "
                      f"(chirps per frame {per_loop}..{per_loop * MAX_LOOPS})")
    if nc & (nc - 1):
        warns.append("Chirps per frame is not a power of two (Doppler FFT is zero-padded)")
    if mode == "TDM":
        v_max = lam / (8 * tc)
    else:
        v_max = lam / (4 * tc)
    v_res = lam / (2 * nc * tc)

    bytes_per_chirp = n * NUM_RX * BYTES_PER_SAMPLE
    lvds = bytes_per_chirp / LVDS_BYTES_PER_S
    if lvds > tc:
        issues.append(f"LVDS needs {lvds * 1e6:.1f} us per chirp but the chirp cycle is {tc * 1e6:.1f} us; "
                      "increase idle time or reduce samples")
    active = nc * tc
    min_period = _ceil_to(active + FRAME_BLANK_S, FRAME_TICK_S)
    if min_period > FRAME_PERIOD_MAX_S:
        issues.append("Frame too long for the frame periodicity register")

    display = req.display_range_m
    if display <= 0:
        issues.append("Max display range must be positive")
    elif display > max_range:
        warns.append(f"Display range clipped to the usable maximum range {max_range:.2f} m")
        display = max_range

    return RadarConfig(
        request=replace(req, mode=mode), bandwidth_hz=bandwidth, range_res_m=range_res,
        max_range_m=max_range, num_samples=n, fs_hz=fs, slope_hz_per_s=slope, start_freq_hz=f0,
        idle_s=idle, adc_start_s=adc_start, ramp_end_s=ramp_end, chirp_period_s=tc,
        center_freq_hz=fc, wavelength_m=lam, ramp_top_hz=ramp_top, max_if_hz=max_if,
        chirps_in_ram=per_loop, loops=loops, v_max_mps=v_max, v_res_mps=v_res,
        active_frame_s=active, min_frame_period_s=min_period, bytes_per_chirp=bytes_per_chirp,
        bytes_per_frame=nc * bytes_per_chirp, lvds_chirp_s=lvds, display_range_m=display,
        angle_res_deg=math.degrees(2 / NUM_VIRT), fov_deg=90.0,
        issues=tuple(issues), warnings=tuple(warns))


def suggest(req: RadarRequest, step: float = 1.03, span: int = 40) -> RadarConfig | None:
    """Nearest valid configuration, varying dR and R_max by factors of `step` (log distance)."""
    best, best_cost = None, math.inf
    for i in range(-span, span + 1):
        for j in range(-span, span + 1):
            cost = abs(i) + abs(j)
            if cost >= best_cost:
                continue
            cand = derive(replace(req, range_res_m=req.range_res_m * step**i,
                                  max_range_m=req.max_range_m * step**j))
            if cand.ok:
                best, best_cost = cand, cost
    return best


# --- Frame rate budget ----------------------------------------------------------------------

@dataclass(frozen=True)
class FpsBudget:
    """Per-stage time per frame (s); the pipeline runs the stages concurrently."""
    stages: dict = field(default_factory=dict)
    measured: dict = field(default_factory=dict)    # stage -> True if measured
    fps_max: float = 0.0
    bottleneck: str = ""
    frame_period_s: float = 0.0                      # value to program
    latency_s: float = 0.0                           # sum of the stages (one frame end to end)


def fps_budget(cfg: RadarConfig, processing_s: float | None = None,
               link_s: float | None = None, ip_latency_s: float = 500e-6,
               packet_delay_s: float = DCA_PACKET_DELAY_S,
               pipeline_s: float | None = None) -> FpsBudget:
    """Maximum frame rate = 1 / slowest stage.

    radar:      active frame time + inter-frame blank (device minimum period)
    transfer:   bytes per frame / estimated DCA1000 Ethernet throughput
    processing: measured laptop FFT + covariance time (default until measured)
    board:      measured transport round trip + NUM_CORES x IP-core latency
                (the GUI field is the latency of one core; the board runs four in sequence)
    pipeline:   optional, measured interval between board results while the pipeline was
                backlogged (frames dropped). It covers what the stage times miss: all stages
                share one Python interpreter (GIL), so together they run slower than alone.
    """
    stages = dict(
        radar=cfg.min_frame_period_s,
        transfer=cfg.bytes_per_frame / dca_bytes_per_s(packet_delay_s),
        processing=DEFAULT_PROCESSING_S if processing_s is None else processing_s,
        board=(DEFAULT_LINK_S if link_s is None else link_s) + NUM_CORES * ip_latency_s,
    )
    measured = dict(radar=False, transfer=False, processing=processing_s is not None,
                    board=link_s is not None)
    if pipeline_s is not None:
        stages["pipeline"] = pipeline_s
        measured["pipeline"] = True
    bottleneck = max(stages, key=stages.get)
    slowest = stages[bottleneck]
    period = max(cfg.min_frame_period_s, _ceil_to(slowest * FRAME_PERIOD_MARGIN, 1e-4))
    return FpsBudget(stages=stages, measured=measured, fps_max=1 / slowest, bottleneck=bottleneck,
                     frame_period_s=period, latency_s=sum(stages.values()))


def report(cfg: RadarConfig, budget: FpsBudget | None = None) -> list[tuple[str, str]]:
    """Human-readable (name, value) rows for the GUI report."""
    if not cfg.num_samples:
        return [("Status", "; ".join(cfg.issues))]
    rows = [
        ("Range resolution dR", f"{cfg.range_res_m:.3f} m (requested {cfg.request.range_res_m:.3f})"),
        ("Max range (usable)", f"{cfg.max_range_m:.2f} m (requested {cfg.request.max_range_m:.2f})"),
        ("Bandwidth B (sampled)", f"{cfg.bandwidth_hz / 1e6:.1f} MHz"),
        ("Slope S", f"{cfg.slope_hz_per_s / 1e12:.3f} MHz/us"),
        ("Sample rate Fs", f"{cfg.fs_hz / 1e6:.2f} Msps complex 1x"),
        ("ADC samples N", f"{cfg.num_samples}"),
        ("Max IF", f"{cfg.max_if_hz / 1e6:.2f} MHz"),
        ("Ramp", f"{cfg.start_freq_hz / 1e9:.3f} .. {cfg.ramp_top_hz / 1e9:.3f} GHz, fc {cfg.center_freq_hz / 1e9:.3f} GHz"),
        ("Idle / ADC start / ramp end", f"{cfg.idle_s * 1e6:.2f} / {cfg.adc_start_s * 1e6:.2f} / {cfg.ramp_end_s * 1e6:.2f} us"),
        ("Chirp period Tc", f"{cfg.chirp_period_s * 1e6:.2f} us"),
        ("Chirps per frame", f"{cfg.request.chirps_per_frame} ({cfg.loops} loops x {cfg.chirps_in_ram}, {cfg.mode})"),
        ("v_max", f"+/-{cfg.v_max_mps:.2f} m/s ({cfg.v_max_mps * 3.6:.1f} km/h)"),
        ("Velocity resolution dv", f"{cfg.v_res_mps:.3f} m/s"),
        ("Angular resolution", f"~{cfg.angle_res_deg:.1f} deg at boresight (8-element lambda/2 ULA)"),
        ("Field of view", f"+/-{cfg.fov_deg:.0f} deg unambiguous"),
        ("Active frame time", f"{cfg.active_frame_s * 1e3:.3f} ms"),
        ("Data per frame", f"{cfg.bytes_per_frame / 1024:.1f} KiB"),
    ]
    if budget:
        period = budget.frame_period_s
        rows += [
            ("Frame period (programmed)", f"{period * 1e3:.2f} ms ({1 / period:.1f} fps)"),
            ("Duty cycle", f"{100 * cfg.active_frame_s / period:.1f} %"),
            ("Data rate", f"{cfg.bytes_per_frame / period / 1e6:.2f} MB/s"),
            ("Max fps (theory)", f"{budget.fps_max:.1f} fps, limited by {budget.bottleneck}"),
        ]
        for name, t in budget.stages.items():
            rows.append((f"  stage {name}", f"{t * 1e3:.2f} ms" + (" (measured)" if budget.measured[name] else " (estimate)")))
    for w in cfg.warnings:
        rows.append(("Warning", w))
    for i in cfg.issues:
        rows.append(("INFEASIBLE", i))
    return rows
