# CLAUDE.md: four-core DoA Eval system on the Arty Z7-20

Board package for the combined BartlettEval + MUSICEval + ESPRITEval + CaponEval design
(XC7Z020-1CLG400C, 100 MHz), plus an interactive host GUI. Written after the session of
2026-11-04 that deployed and tested this folder on the board. Local work, not part of the
`doa-ip-cores` repository (`local/` is git-ignored); see `local/README.md` there for how
the bitstream, overlay and SWIG modules were built.

## Contents

| File | Runs on | Purpose |
| --- | --- | --- |
| `sys_wrapper.bit.bin` | board | Bitstream with the four cores |
| `doa4_eval.dtbo` | board | Device-tree overlay: FPGA load + four UIO devices |
| `_<alg>_eval.so`, `<alg>_eval.py` | board | SWIG driver modules (Bartlett, MUSIC, ESPRIT, Capon) |
| `bartlett.py`, `music.py`, `esprit.py`, `capon.py` | both | Family applications / reference models (unchanged) |
| `doa4_eval.py` | board | Batch validation: 21 MUSIC testbench scenarios on all four cores |
| `doa4_gui_server.py` | board | JSON-lines server used by the GUI (uploaded by the GUI on start) |
| `doa4_gui.py` | host PC | Tk + matplotlib GUI |

Address map: BartlettEval 0x40000000, MUSICEval 0x40010000, ESPRITEval 0x40020000,
CaponEval 0x40030000. All read the 8x8 complex64 covariance from `udmabuf0`; Bartlett and
Capon write 512 float64, MUSIC 512 float32 into the shared `udmabuf1`; ESPRIT returns its
angles in registers.

## Board setup (after every board reboot)

Board: `ssh root@192.168.1.123` (key login, hostname `artyz720`, Linux 6.6.40-xilinx).
The board has **no sftp-server**, so `scp` fails; copy with tar over ssh:

```sh
tar -cf - --exclude=__pycache__ . | ssh root@192.168.1.123 'mkdir -p /root/deploy4 && tar -xf - -C /root/deploy4'
```

Load the bitstream + overlay through configfs (it is not mounted by default) and u-dma-buf:

```sh
cd /root/deploy4
cp sys_wrapper.bit.bin doa4_eval.dtbo /lib/firmware/
mount -t configfs configfs /sys/kernel/config 2>/dev/null
mkdir -p /sys/kernel/config/device-tree/overlays/doa4
echo -n doa4_eval.dtbo > /sys/kernel/config/device-tree/overlays/doa4/path
cat /sys/kernel/config/device-tree/overlays/doa4/status   # applied
cat /sys/class/fpga_manager/fpga0/state                   # operating
insmod /root/u-dma-buf.ko udmabuf0=0x1000000 udmabuf1=0x1000000
```

Expect four UIO devices (`/sys/class/uio/uio*/name`: BartlettEval, CaponEval, ESPRITEval,
MUSICEval) and `udmabuf0`/`udmabuf1` of 16 MiB below the 512 MiB HP DDR window (in the
session: 0x15900000 and 0x16900000). `mountpoint` is not available on the board.

## Batch validation: `doa4_eval.py`

```sh
python3 doa4_eval.py            # on the board; results.json in doa4_eval_results/
```

Result of the 2026-11-04 hardware run: **14/21 scenarios passed on all scored cores**
(about 10 s). Matplotlib is not installed on the board, so no PNGs are written.

- BartlettEval 21/21 PASS, MUSICEval 21/21 PASS, ESPRITEval 11/11 scored PASS
  (0.0000 deg angle error).
- CaponEval: 12 PASS, 7 FAIL, 2 not scored (zero matrix). It fails all five *sampled*
  signal scenarios (max normalized error 7.5e-4 ... 2.4e-3) and both non-centro-Hermitian
  PSD stress cases (5.2e-2) against `capon.py` `MAX_NORMALIZED_ERROR = 1e-4`; its peak
  angles still match the reference. Analytic, identity and invalid-K cases pass (<= 1e-6).
  Root cause not investigated yet (precision on ill-conditioned inputs vs. a too tight
  tolerance). Next step: run CaponEval alone (`capon_eval.py` or `--cores capon`) to rule
  out the shared-buffer system. Relevant to Capon PR #1 (`capon-ip-core`).

## Interactive GUI: `doa4_gui.py`

Runs on the host (WSL2 with WSLg display). Needs numpy, matplotlib and tkinter; the
system `python3` has no numpy, use the repository venv:

```sh
cd local/deploy4 && ../../.venv/bin/python doa4_gui.py \
    [--host root@192.168.1.123] [--remote-dir /root/deploy4] [--snapshot plot.png]
```

- Input windows (entry fields, not knobs): Target 1 and Target 2 angle (-90..90 deg) and
  SNR (-30..60 dB), Target 2 enable checkbox (K = number of enabled targets, used by MUSIC
  and ESPRIT), covariance mode sampled/analytic, snapshots, seed, "new seed every run".
  "Run on FPGA" or Enter in any field runs; out-of-range values are rejected.
- The host builds the covariance with `music.covariance` (half-wavelength 8-element ULA,
  uncorrelated sources in unit-power noise, trace normalized to 8) and sends it as 128
  float32 words over one persistent `ssh ... python3 -u doa4_gui_server.py` session.
- The server writes it to `udmabuf0`, runs Bartlett -> MUSIC -> ESPRIT -> Capon via
  `doa4_eval.DoAEvalSystem`, and returns spectra, ESPRIT angles, latencies and
  unwritten-output flags as one JSON line. stdout carries only the protocol; diagnostics
  go to stderr.
- Plot: normalized dB spectra (Bartlett, MUSIC, Capon), ESPRIT angles as green vertical
  lines, true DoA as dotted lines with markers. Status report on the canvas: per-core
  latency and DoA estimate, total time for all four cores, host round trip.
- Latency is **PS-measured start->done** (`time.perf_counter_ns` around Start/IsDone
  polling), so it includes UIO and Python overhead and is an upper bound, not the core's
  cycle latency. Do not publish these numbers in datasheets (repository rule: only
  verified synthesis/cosim/hardware figures).

Session example (-20 deg @ 15 dB, +30 deg @ 10 dB, sampled, 256 snapshots, seed 0):

| Core | Latency (PS-measured) | DoA |
| --- | --- | --- |
| BartlettEval | ~450 us | -20.25, +30.12 deg |
| MUSICEval | ~350 us | -19.90, +30.12 deg |
| ESPRITEval | ~325 us | -20.06, +29.94 deg |
| CaponEval | ~460 us | -19.90, +30.12 deg |
| All four | ~2.9 ms | host round trip ~30 ms |

## Gotchas

- Closing the GUI closes the SSH session; the server releases the UIO devices and buffers
  on stdin EOF. If a server is left over, stop it on the board with
  `kill $(pgrep -f 'python3 -u doa4_gui_server')`. Do not run `pkill -f doa4_gui_server.py`
  through ssh: the pattern also matches the remote shell running the command.
- The overlay and u-dma-buf stay loaded until reboot; removing the overlay logs
  "memory leak will occur if overlay removed" warnings (harmless, expected).
- `__pycache__/` is a build artifact; do not ship or commit it.
