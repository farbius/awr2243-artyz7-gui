# Radar + FPGA DoA GUI

Live direction-of-arrival estimation with a TI **AWR2243BOOST + DCA1000EVM** radar, a Windows
laptop and the four DoA IP cores (BartlettEval, MUSICEval, ESPRITEval, CaponEval) on the
**Arty Z7-20**, with live USB camera video.

The laptop configures the radar through mmWave Studio and receives the ADC data over UDP. It
then runs the range FFT and builds the 8x8 covariance at the strongest range cell, which the
board turns into three spatial spectra and two ESPRIT angles over one persistent SSH channel.
The GUI plots the range-azimuth map, the spectra, the range profile and the camera view.

Built and tested on the hardware on 2026-11-06. See [Verified on hardware](#verified-on-hardware)
for what has and has not been verified.

## Contents

```
radar_doa_gui/            host package (python -m radar_doa_gui)
  params.py               pure: dR / R_max -> chirp, frame, limits, report, fps model
  processing.py           range FFT, clutter removal, range-azimuth map, target cell,
                          TDM / DDM virtual array, covariance (FB averaging, loading)
  radar.py                mmWave Studio RSTD client, Lua generation, DCA1000 UDP receiver
  board_link.py           paramiko: deploy, configfs + u-dma-buf init, persistent channel
  protocol.py             binary request/response protocol (also deployed to the board)
  sim.py / mock_board.py  simulation mode: synthetic frames, numpy DoA reference models
  workers.py              acquisition / processing / board / camera threads
  gui.py, settings.py     PySide6 + pyqtgraph window, JSON settings
board_server/doa_server.py  long-running board server (deployed to /root/doa_app)
lua/doa_common.lua        bring-up + waveform helpers, copied from the autobahn-labs repo
scripts/                  hardware checks: board_check.py, link_check.py, radar_first_test.py
tests/                    pytest: params, processing, protocol, radar framing, real captures
captures/                 raw ADC frames of the hardware tests (.npy, radar data only)
```

The earlier batch and JSON tools in this folder (`doa4_eval.py`, `doa4_gui.py`,
`doa4_gui_server.py`, see `CLAUDE.md`) are unchanged. The new board server reuses
`doa4_eval.DoAEvalSystem` and the SWIG driver modules as they are.

## Architecture

```
 Windows laptop                                                           Arty Z7-20
 ┌──────────────────────────────────────────────────────────────────┐   ┌────────────────────────┐
 │ GUI thread (draw only, 30 Hz)                                     │   │ doa_server.py          │
 │   ▲ Latest holders                                                │   │  (python3, numpy)      │
 │   │                                                               │   │                        │
 │ acquisition ──DropQueue(2)──► processing ──DropQueue(1)──► board ─┼───┼► stdin  (24 B + 512 B) │
 │  thread                        thread                     thread  │SSH│  udmabuf0 ← R (8x8)    │
 │   ▲ UDP 4098 (ADC)              range FFT, RA map,         ◄──────┼───┼─ stdout (68 B + 6 KiB) │
 │   │ UDP 4096 (DCA1000 cmds)     target cell, R             paramiko   │  Bartlett → MUSIC →    │
 │   │                                                               │   │  ESPRIT → Capon (PL)   │
 │ camera thread (OpenCV, live only)                                 │   │  udmabuf1 → spectra    │
 │                                                                   │   └────────────────────────┘
 │ control: pythonnet → RtttNetClientAPI.dll → mmWave Studio (RSTD, TCP 2777)
 └───────┬──────────────────────────────────────────┬─────────────────┘
         │ Ethernet 192.168.33.30 ↔ .180             │ USB (XDS110 UART COM3, FTDI SPI)
     DCA1000EVM ◄──── LVDS (4 lanes) ──── AWR2243BOOST (TX0 + TX2, RX0..3)
```

The queues drop their oldest entry when full, so a slow stage never builds up lag. The drop
count is shown in the report.

## Setup

**Laptop** (Windows, Python 3.13, venv in this folder):

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt     # versions pinned
```

**mmWave Studio 3.0.0.14** with DFP 02.02.05.00 (firmware paths in `lua/doa_common.lua`):

- Start Studio and run `RSTD.NetStart()` in its Lua shell.
- The DCA1000 Ethernet adapter must be 192.168.33.30, with the DCA1000 at 192.168.33.180.
- The AWR2243BOOST control UART is COM3 (*XDS110 Class Application/User UART*).

**Board:** `ssh root@192.168.1.123` with key login. The GUI does the rest (see
[Board configuration](#board-configuration)). It needs `/root/u-dma-buf.ko` on the board,
which is not shipped from this folder.

## Run sequence

```powershell
.venv\Scripts\python.exe -m radar_doa_gui            # settings from radar_doa_gui_settings.json
.venv\Scripts\python.exe -m radar_doa_gui --sim      # no hardware: simulated radar + mock board
```

1. Pick **Radar** `sim` / `awr2243` and **DoA engine** `mock` / `board`.
2. Set the radar fields (range resolution, maximum range, display range, TDM/DDM, chirps per
   frame, TX backoff, extended band). The report previews the derived configuration as you type.
3. **Configure (radar + board)** runs these steps:
   1. Derive and check the configuration. If it is infeasible, the reasons and the nearest
      valid dR / R_max are shown.
   2. Deploy the board files and initialize the board, then start the server.
   3. Measure the laptop processing time and the board link time (20 requests).
   4. Set the frame period from the fps model.
   5. Program the radar: bring-up when needed, then profile, chirps and frame.
4. **Start streaming** or **Single frame**. *Radar frames per run* (0 = continuous) limits a
   run to a finite number of frames. The first radar transmit of a session asks for confirmation.
5. Processing fields (K, number of sources, snapshots, FB averaging, diagonal loading, clutter
   removal, flip array, IP-core latency) apply immediately.
6. Closing the window stops everything: StopFrame and the DCA1000 stop, the board server
   (which releases the UIO devices and buffers), and the camera. The settings are saved to JSON.

Logs go to `logs/radar_doa_gui.log`. `--snapshot PNG` configures, streams for about 6 s, saves
the window and exits. The camera stays off in snapshot mode, so no camera frame is ever saved.

## Capture path (mmWave Studio + direct UDP)

- **Control** goes through mmWave Studio. Python sends Lua code over RSTD; each command
  writes its result to a file in `.studio_tmp/`, because a successful SendCommand only means
  the command was delivered.
- **Bring-up** (`C.bringup`, unchanged from the repo) covers SOP mode, COM3 connect,
  BSS/MSS firmware, PowerOn, RfEnable, DCA1000 Ethernet/mode/packet delay 25,
  `ChanNAdcConfig` (TX0 + TX2, RX0..3, 16 bit, complex 1x), `SetMiscConfig(1)` (per-chirp
  phase shifter), RfInit, DataPath and 4-lane LVDS. The *Radar bring-up* setting chooses
  between:
  - `auto`: once per Studio session;
  - `skip`: reuse a bring-up done by hand in Studio (TDM only, because the phase shifter
    state is unknown);
  - `force`: run it again.
- **Waveform:** one profile and a one-subframe advanced frame, generated from the GUI values.
  The frame count is 0 for continuous runs, otherwise N.
- **Data:** Studio's recorder is **not** used, so UDP 4098 stays free. `radar.py` binds
  4096/4098 on 192.168.33.30 itself and sends DCA1000 `RECORD_START` before StartFrame and
  `RECORD_STOP` after StopFrame. It rebuilds the frames from the 48-bit byte counter in each
  packet. A frame with a missing packet never completes and is dropped, never zero-filled.
- **ADC layout:** chirp > sample > [I rx0..3, Q rx0..3], int16 LE, I first. This was confirmed
  on hardware: the echoes appear at bin k and nothing at the mirror N-k.
- **Calibration:** no RF calibration data is uploaded; RfInit runs the boot calibrations. The
  only calibration is 8 complex coefficients per virtual channel (TX0 RX0..3, TX2 RX0..3) from
  an `.npz` file (key `correction`), applied after the FFT. The default path is
  `awr2243-autobahn-labs/datasets/awr2243_calibration.npz`. **That file does not exist yet**,
  so unity coefficients are used; the status bar says so.

## TDM and DDM

Virtual array: `virt = tx_slot * 4 + rx`, TX0 first. On the AWR2243BOOST, TX0 and TX2 are 2λ
apart and the RX elements λ/2 apart, which gives a gap-free 8-element λ/2 ULA. The steering
model is a[n] = exp(+jπ n sin θ), the same as the IP cores.

| | TDM | DDM-4 |
|---|---|---|
| Chirp RAM | chirp 0 TX0, chirp 1 TX2 | 4 chirps, TX0 + TX2 in each, TX2 phase k·90° (k = chirp mod 4) |
| Same-TX repetition | 2 Tc | Tc |
| v_max | λ / (8 Tc) | λ / (4 Tc) (2x TDM) |
| Δv | λ / (2 N_c Tc) | λ / (2 N_c Tc) |
| Channel separation | by chirp slot | after the Doppler FFT, by sub-band |

- **TDM motion compensation:** TX2 transmits Tc after TX0, so a target with Doppler phase φ
  per loop adds φ/2 on the TX2 channels. The phase is removed using the Doppler peak
  (`tdm_phase_comp`, on by default; checked in the simulator).
- **DDM-4 separation:**
  - The TX2 copy of a target sits a quarter of the Doppler spectrum away from TX0, and the
    two remaining quarter bands stay empty.
  - For each range bin, the TX0 bin k maximizes P[k] + P[k+q] − P[k+2q] − P[k+3q] with
    q = s·N/4, and the 8-element vector is [Y(k, rx0..3), Y(k+q, rx0..3)].
  - Because the empty pair identifies TX0, velocities up to λ/(4 Tc) are unambiguous.
  - The cost: one target per range-Doppler cell, and the empty-band decision needs SNR.
- **Measured TX2 offset sign:** the chirp RAM steps TX2 by +90° per chirp, but on this board
  the copy appears at **−N/4**, so `ddm_tx2_sign = -1`. This was measured on a static scene
  on 2026-11-06: bins 0 and 96 occupied, 32 and 64 empty 30–40 dB lower. A regression test
  runs on that capture.
- **DDM clutter removal** removes the static TX0 part (mean over chirps) and the static TX2
  part (mean of the code-demodulated signal).

## Processing and covariance

- **Range:** Hann window, FFT over N samples. Bins below *dc_guard_bins* (3) are excluded from
  the target search, and the display is limited to *Max display range*.
- **Range-azimuth map:** FFT beamforming over the 8 virtual channels (64-point, mapped to
  −90..90°), non-coherently averaged over loops.
- **Target cell:** the range bin with the most energy, then the strongest Doppler bin there.
- **Snapshots:**
  - Default: one snapshot per frame, the 8-vector at the target range and Doppler cell, with
    R averaged over a sliding window of K frames (default 16; a warning appears below 16).
  - The chirp-snapshot option (TDM only) uses every loop, with TX2 motion compensation.
  - The window restarts when the target bin moves by more than one bin.
- **Covariance:** R = (1/K) Σ x xᴴ, Hermitian-symmetrized, forward-backward averaged (default
  on), loaded with δ·tr(R)/8 (default δ = 1e-3) and normalized to trace 8, as in the IP-core
  test fixtures.

## Board configuration

`board_link.py`, triggered by Configure:

1. **Deploy** to `/root/doa_app`, uploading only files whose sha256 differs. The board has no
   sftp-server, so files go over an exec channel (`cat > tmp && mv`). `/root/deploy4` is not
   touched.
2. **u-dma-buf:**
   - If `u_dma_buf` is loaded and `/dev/udmabuf0/1` both have 0x1000000 bytes, nothing happens.
   - If it isn't loaded, `insmod /root/u-dma-buf.ko udmabuf0=0x1000000 udmabuf1=0x1000000`.
   - A missing module or wrong buffers are reported as an error, never fixed silently.
3. **FPGA (configfs, as in `CLAUDE.md`):**
   - Skipped when overlay `doa_app` is applied, the four UIO devices exist and a marker matches
     the bitstream/overlay hashes and the boot id.
   - Otherwise: stop the server, remove every overlay, copy the files to `/lib/firmware` if
     they differ, apply the overlay, check `fpga0` and the UIO names.
4. **Server:**
   - `python3 -u doa_server.py` runs on one paramiko channel; old servers are stopped first
     (pid file or `pgrep '[d]oa_server.py'`).
   - Requests time out after 3 s, and the link reconnects and re-initializes on failure.
   - *Number of sources* (1 or 2) is sent with every request and stays editable, because the
     server reports `k_fixed: null`.

**Protocol** (`protocol.py`):

- *Request:* 24 B header (magic, kind, frame id, K, host timestamp) + the 512 B covariance
  (row-major complex64).
- *Response:* 68 B header with frame id, board timestamps, the per-core PS-measured latency,
  an unwritten-output bitmask and the 2 ESPRIT angles, followed by 3 × 512 float32 spectra
  (Bartlett, MUSIC as 1/null spectrum, Capon) on `linspace(-90, 90, 512)`.

## Frame rate model

The pipeline stages run concurrently, so `fps_max = 1 / slowest stage`:

| Stage | Value used |
|---|---|
| radar | active frame time + 0.2 ms inter-frame blank |
| transfer | bytes per frame / DCA1000 throughput, estimated as 1456 B per (25 µs packet delay + wire time) ≈ 39 MB/s |
| processing | measured at Configure (90th percentile) |
| board | measured link time (round trip minus core time) + 4 × the *IP-core latency* field (default 500 µs; the four cores run in sequence) |
| pipeline | measured result interval of the previous streaming run, if it dropped frames (all stages share one Python interpreter) |

The programmed frame period is 1.25 × the slowest stage, but never below the radar minimum.
The report shows the theoretical and measured values for each stage, the end-to-end latency,
the fps and the dropped frames.

## Verified on hardware

Everything below was measured on 2026-11-06. The latencies are PS-measured start→done times:
upper bounds that include UIO and Python overhead, not the cores' cycle latency. Do not use
them in datasheets.

| Item | Result |
|---|---|
| Unit tests | 70 passed (8 grid points correctly reported as infeasible) |
| Board init | deploy + u-dma-buf + configfs overlay + server in 6.8 s; repeat run detects "already loaded" (3.3 s) |
| Board vs numpy reference (6 covariances) | ESPRIT difference 0.0000°; Bartlett/MUSIC/Capon peaks on the same bins; Bartlett ≤ 2.4e-7, MUSIC ≤ 3.7e-3 normalized difference; Capon up to 1.4e-2 on an ill-conditioned 64-snapshot case (known Capon issue, `CLAUDE.md`) |
| Board round trip (binary protocol) | median 6.9 ms (cores ~1.55 ms, server ~2.7 ms); the old JSON server took ~30 ms |
| Radar bring-up | 37 s, all boot calibrations passed |
| 8-frame TDM / DDM captures | 8/8 frames, 0 lost packets, ADC peak −31 dBFS, RX noise floors within 0.9 dB |
| Continuous TDM, 6 dB backoff, 9.6 ms period | 574 frames, 0 lost packets, 106 fps acquisition, 91.6 fps results, end-to-end 16.6 ms, 15 % pipeline drops |
| Continuous DDM-4, 6 dB backoff, 10.4 ms period | 530 frames, 0 lost packets, 96 fps acquisition, 89.8 fps results, end-to-end 12.3 ms, 11 % pipeline drops; static scene at v = 0 |

**Not yet verified on hardware:**

- **Angle sign and accuracy.** There was no target at a known angle; the scene was the static
  room at 1.5–2 m, which is coherent and gives a nearly rank-1 covariance. Check with a person
  or a metal object at a measured angle (3–6 m), with clutter removal on. Use *Flip array* if
  the sign is mirrored.
- **Channel calibration.** No `awr2243_calibration.npz` exists. In the DDM capture the TX2 half
  was 1.32x stronger than the TX0 half, which is either a TX imbalance or near field.
- **DDM velocity disambiguation beyond the TDM limit** (tested only in the simulator).
- **The frame-period correction from the measured pipeline rate** (unit-tested only; needs a
  second Configure in the same session).
- **The camera view** (all hardware runs used snapshot mode, which keeps the camera off).

## Known limitations

- **mmWave Studio RSTD:** the server serves one client at a time and twice missed a client's
  disconnect, after which it serves no one until Studio is restarted. Every RSTD call now has
  a time limit, and the GUI reports *"Studio's RSTD server does not respond… restart Studio,
  run RSTD.NetStart()"* instead of hanging. Use a single GUI session rather than many
  short-lived scripts. After a restart, power-cycle the board and use bring-up `auto`; a
  repeat bring-up without `FullReset` is untested.
- **Pipeline throughput:** at ~100 fps, 11–15 % of the frames are dropped. All stages share one
  Python interpreter (UDP receive at ~9,400 packets/s, paramiko, numpy), so processing takes
  ~6 ms under load against 2.5 ms alone. Reconfiguring after a run lowers the frame period to
  the measured rate. Two board requests in flight, or a C/NumPy UDP receiver, would raise it.
- **One target cell:** DoA uses the single strongest range-Doppler cell. Two targets are
  resolved only if they share that cell and are not coherent; static reflectors are coherent,
  and FB averaging helps only partly.
- **DDM-4:** one target per range-Doppler cell; the empty-band decision fails at low SNR.
- **Device constants** marked *assumption* in `params.py` (100 MHz/µs slope limit, 2 Msps
  minimum, LVDS 4 × 600 Mbps, DCA1000 throughput) are conservative estimates, not measured
  limits. dR below ~0.28 m needs the extended band (76–81 GHz, off by default).
- **The camera** is never recorded and never shown in snapshots (GDPR).
