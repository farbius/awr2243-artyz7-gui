# Task: Radar + FPGA DoA GUI (AWR2243BOOST + DCA1000EVM + Arty Z7-20 + USB camera)

## Ground rules
- Write files ONLY in the current working directory (CWD). Never create, modify or delete anything outside it.
- C:\workspace\github\awr2243-autobahn-labs is a READ-ONLY reference. Read it, import from it via sys.path if useful, but never write there. If you need to change something from it, copy it into CWD first.
- Before writing any code, explore and send me a short plan (see "Phase 0"). Ask me about anything ambiguous instead of guessing.
- Hardware is connected and live. Ask before any action that is hard to undo on the board (overwriting files outside a dedicated deploy folder, rebooting, changing network config).

## Hardware setup
- AWR2243BOOST + DCA1000EVM connected to this Windows laptop. mmWave Studio is running.
- Arty Z7-20 reachable at: ssh root@192.168.1.123
- USB web camera connected to the laptop.

## Goal
A Python GUI that:
1. Configures the AWR2243BOOST (and DCA1000) and the Arty Z7-20.
2. Triggers and captures radar frames (continuous streaming or single frame).
3. Computes the 8x8 spatial covariance matrix on the laptop.
4. Sends it to the Arty Z7-20, where the DoA IP-core application computes 3 spatial spectra (Bartlett, MUSIC, Capon) and 2 angles.
5. Receives the results back and plots them together with live camera video.

## Phase 0: Explore and plan (no code yet)
Read and summarize:
- C:\workspace\github\awr2243-autobahn-labs\capture\awr2243.py: how it configures the radar (mmWave Studio RSTD/Lua? direct?), how capture works (file-based via Studio vs. direct UDP from DCA1000), and how calibration data is uploaded.
- C:\workspace\github\awr2243-autobahn-labs\configs: available configs and the calibration file format/flow.
- CWD: the board-side files (Python app, device-tree overlay, bitstream, kernel module, etc.) and the DoA application's interface: its input format (covariance layout, dtype, element ordering), its output format (spectra length, angle grid, the 2 angles), and how it is invoked (one-shot CLI vs. long-running service).
- The antenna geometry of AWR2243BOOST, so that TX0 + TX2 with RX0..RX3 form an 8-element ULA. Confirm the virtual element ordering from the repo or documentation, and state it.

Then report back:
- Your understanding of the capture path and whether real-time streaming is possible while mmWave Studio holds the DCA1000. Studio's capture usually writes to files, and only one process can bind UDP port 4098. Propose the streaming approach.
- The proposed architecture, file layout, dependencies, and a list of open questions.
Wait for my OK before implementing.

## Arty Z7-20 configuration ("Configure" button)
- Copy all required files from CWD to a dedicated folder on the board, e.g. /root/doa_app (use paramiko/SFTP, and only upload changed files).
- Initialize:
    fpgautil -b sys_wrapper.bit.bin -o doa4_eval.dtbo
    insmod u-dma-buf.ko udmabuf0=0x1000000 udmabuf1=0x1000000
  Handle "already loaded" cases gracefully (check lsmod and /dev/udmabuf*; skip or reload as appropriate).
- Start the DoA application as a long-running process with ONE persistent connection (a paramiko channel with a binary protocol over stdin/stdout, or a TCP socket). Do NOT open a new SSH session or scp per frame. If the existing board app is one-shot only, write a thin board-side server wrapper (kept in CWD, deployed to the board) that loads the IP core once and loops over requests.
- Protocol: compact binary (e.g. 8x8 complex64 = 512 B in; spectra + angles + IP-core latency out), with a header containing a frame id and timestamps so round-trip latency can be measured.

## AWR2243BOOST configuration
- Reuse the logic in capture\awr2243.py and the configs folder, including calibration file upload.
- Generate the profile, chirp and frame configuration from the GUI parameters (below) instead of fixed files. Keep a "load preset" option for the existing configs.
- Use TX0 and TX2 only, and RX0..RX3, giving 8 virtual channels:
  - TDM: alternating chirps TX0, TX2. Account for the doubled effective chirp period in v_max, and compensate the Doppler-induced phase between TX slots if you implement it (mark as optional).
  - DDM: both TX in every chirp, with a per-chirp phase code on TX2 via the per-chirp phase shifter, so the TX signals separate in Doppler. Explain the scheme you chose, its v_max penalty, and how the channels are separated after the Doppler FFT.

## Processing (laptop)
- Range FFT (windowed) per virtual channel. Optionally remove static clutter (mean subtraction), with a GUI toggle.
- Range-azimuth map: computed on the laptop (e.g. Bartlett/FFT beamforming over the 8 virtual channels, per range bin, up to "max display range").
- Target range cell: the range bin with maximum energy within the display range (exclude the first few bins near DC).
- Covariance: R = (1/K) * sum(x_k x_k^H) over K snapshots, where x_k is the 8-element vector at the target range cell. K = "number of frames used as snapshots" (sliding window over frames; also use chirps within a frame if that matches the board app's expectation, and state the choice). Optional diagonal loading.
- Send R to the board, receive 3 spectra + 2 angles.

## GUI layout
- Framework: PySide6 (or PyQt5) + pyqtgraph for fast plotting. OpenCV for the camera.
- Top: ONE canvas with a 2x2 grid of equally sized plots:
  1. Range-azimuth map (heatmap; range axis limited by "max display range")
  2. Azimuth spectra for the target range cell: Bartlett, MUSIC, Capon overlaid and normalized in dB, with the 2 estimated angles marked as vertical lines
  3. Range profile: 4 RX curves in dB, limited by "max display range", with the target bin marked
  4. Live USB camera view
- Bottom: a Radar Configuration panel and controls:
  - Inputs: range resolution [m], maximum range [m], max display range [m], TDM/DDM mode, number of frames used as snapshots for covariance (K), IP-core latency [µs] (default 500, used in the fps calculation; also show the measured value).
  - Buttons: Configure (radar + board), Start/Stop streaming, Single frame capture.
  - Read-only report: derived parameters and status (see below).
  - Status bar: connection states (Studio, DCA1000, board, camera), errors.
- Threads: separate workers for radar acquisition, processing, board communication and camera. The GUI thread only draws. Use bounded queues and drop stale frames instead of building up lag.

## Parameter derivation (implement as a pure, unit-tested module)
From range resolution ΔR and maximum range R_max, derive a valid configuration:
- B = c / (2·ΔR) (sampled bandwidth); N_samples ≈ R_max / ΔR, rounded up to a power of two or a value supported by the device.
- Choose F_s (complex 1x, ≤ 22.5 Msps) and slope S so that S·N/F_s = B and the max IF = 2·S·R_max/c ≤ min(0.9·F_s, 20 MHz).
- ADC start time, ramp end time (sampling window plus margin) and idle time: use sane defaults consistent with the repo configs.
- Respect device limits (ramp rate, start frequency 76–81 GHz, chirp RAM, LVDS/DCA1000 throughput). If the request is infeasible, show why and suggest the nearest valid values.
- Report: ΔR, R_max, B, S, F_s, N_samples, chirp period, chirps per frame, v_max and Δv (per TDM/DDM), angular resolution and FoV of the 8-element array, active frame time, duty cycle, data per frame and data rate.

## Maximum achievable fps (show in the report)
Compute and display the maximum frame rate as the inverse of the slowest stage:
- Radar: active frame time + required inter-frame time (calibration/monitoring margin), within device limits on frame periodicity.
- Data transfer: bytes per frame / realistic DCA1000 Ethernet throughput.
- Laptop processing: measured FFT/covariance time.
- Board: measured SSH round trip + IP-core latency (500 µs default from the GUI field).
Show both the theoretical value and the measured end-to-end fps and latency, broken down per stage. Set the frame periodicity in the radar configuration so it never exceeds the achievable rate.

## Camera
Use OpenCV with sensible defaults (e.g. 640x480 @ 30 fps, MJPG if supported), in a separate thread. Auto-detect the camera index, and show "no camera" gracefully if none is found. Optional: camera index/resolution in a settings dialog.

## Robustness and testing
- A simulation mode (synthetic radar frames with 1–2 targets at known angles, and a mock board that computes Bartlett/MUSIC/Capon in numpy) so the whole GUI runs without hardware. Use it to verify that angles and spectra match between the mock and the real board.
- Unit tests for the parameter derivation, covariance computation, virtual array ordering, and the protocol encoding/decoding.
- Clear logging to a file; timeouts and reconnect logic for SSH; clean shutdown (stop the sensor, stop the board app, release the camera).
- Save and load GUI settings to JSON.

## Deliverables (in CWD)
- The Python package with an entry point (e.g. python -m radar_doa_gui) and requirements.txt.
- The board-side wrapper/server, if needed.
- README: setup steps, the run sequence, the architecture diagram (text), the capture path explanation, the TDM/DDM explanation, and known limitations.

Work incrementally: Phase 0 plan → parameter module + tests → simulation-mode GUI → board link → radar capture → full integration. After each step, tell me what works and what you verified on the real hardware.