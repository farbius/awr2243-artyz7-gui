"""Interactive DoA GUI for the four-core Eval system (runs on the host PC).

The host generates the 8x8 float covariance (music.covariance: half-wavelength ULA,
uncorrelated sources in unit-power noise, trace normalized to 8), sends it over one
persistent SSH session to doa4_gui_server.py on the Arty Z7-20, which runs
BartlettEval -> MUSICEval -> ESPRITEval -> CaponEval on it, and plots the returned
spectra (ESPRIT angles as vertical lines) with the measured core latencies.

    python doa4_gui.py [--host root@192.168.1.123] [--remote-dir /root/deploy4]

Requires numpy, matplotlib and tkinter on the host; on the board the bitstream,
doa4_eval.dtbo overlay and u-dma-buf (udmabuf0/udmabuf1) must already be loaded.
doa4_gui_server.py is uploaded to --remote-dir on start.
"""
import argparse
import json
import queue
import subprocess
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import music  # noqa: E402  (covariance model, scan grid, peak search)

HERE = Path(__file__).resolve().parent
THETA = music.THETA_SCAN_DEG
INK, SURFACE, GRID, MUTED = "#0b0b0b", "#fcfcfb", "#e4e3df", "#6b6a66"
COLORS = dict(bartlett="#2a78d6", music="#eb6834", esprit="#1f9d55", capon="#8a3ffc")
NAMES = dict(bartlett="BartlettEval", music="MUSICEval", esprit="ESPRITEval", capon="CaponEval")
DB_FLOOR = -50.0


# -----------------------------------------------------------------------------
# Board link: one SSH session, JSON lines
# -----------------------------------------------------------------------------

class BoardLink:
    def __init__(self, host, remote_dir):
        self.host, self.remote_dir = host, remote_dir
        self.proc = None
        self.info = None

    def _ssh(self, *args, **kw):
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", self.host, *args]

    def connect(self):
        server = (HERE / "doa4_gui_server.py").read_bytes()
        subprocess.run(self._ssh(f"cat > {self.remote_dir}/doa4_gui_server.py"),
                       input=server, check=True, timeout=30)
        self.proc = subprocess.Popen(
            self._ssh(f"cd {self.remote_dir} && exec python3 -u doa4_gui_server.py"),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1)
        self.info = self._read()
        if not self.info.get("ready"):
            raise RuntimeError(self.info.get("error", "board server did not start"))
        return self.info

    def _read(self):
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read() if self.proc.poll() is not None else ""
            raise RuntimeError(f"board connection closed. {err.strip()[-400:]}")
        return json.loads(line)

    def run(self, r, k):
        words = np.ascontiguousarray(r, dtype=np.complex64).view(np.float32).ravel()
        t0 = time.perf_counter()
        self.proc.stdin.write(json.dumps(dict(rxx=words.tolist(), k=int(k))) + "\n")
        self.proc.stdin.flush()
        reply = self._read()
        reply["roundtrip_ms"] = (time.perf_counter() - t0) * 1e3
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "board error"))
        return reply

    def close(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                self.proc.kill()


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

class Field:
    """Labelled entry window with range checking."""

    def __init__(self, parent, row, label, value, lo, hi, unit, integer=False):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(8, 6), pady=3)
        self.var = tk.StringVar(value=str(value))
        self.entry = ttk.Entry(parent, textvariable=self.var, width=9, justify="right")
        self.entry.grid(row=row, column=1, sticky="e", pady=3)
        ttk.Label(parent, text=unit, foreground=MUTED).grid(row=row, column=2, sticky="w", padx=(4, 8))
        self.label, self.lo, self.hi, self.integer = label, lo, hi, integer

    def get(self):
        text = self.var.get().strip()
        try:
            v = int(text) if self.integer else float(text)
        except ValueError:
            raise ValueError(f"{self.label}: '{text}' is not a number") from None
        if not self.lo <= v <= self.hi:
            raise ValueError(f"{self.label}: {v} outside [{self.lo}, {self.hi}]")
        return v

    def state(self, enabled):
        self.entry.configure(state="normal" if enabled else "disabled")


class App:
    def __init__(self, root, link):
        self.root, self.link = root, link
        self.results = queue.Queue()
        self.busy = False
        root.title("DoA Eval: Bartlett / MUSIC / ESPRIT / Capon on Arty Z7-20")
        root.configure(background=SURFACE)
        style = ttk.Style(root)
        style.configure("Run.TButton", font=("TkDefaultFont", 11, "bold"), padding=6)

        side = ttk.Frame(root, padding=10)
        side.pack(side="left", fill="y")
        t1 = ttk.LabelFrame(side, text="Target 1", padding=6)
        t1.pack(fill="x", pady=(0, 8))
        self.ang1 = Field(t1, 0, "Angle", -20.0, -90, 90, "deg")
        self.snr1 = Field(t1, 1, "SNR", 15.0, -30, 60, "dB")

        t2 = ttk.LabelFrame(side, text="Target 2", padding=6)
        t2.pack(fill="x", pady=(0, 8))
        self.use2 = tk.BooleanVar(value=True)
        ttk.Checkbutton(t2, text="Enabled", variable=self.use2, command=self._toggle2).grid(
            row=0, column=0, columnspan=3, sticky="w", padx=8)
        self.ang2 = Field(t2, 1, "Angle", 30.0, -90, 90, "deg")
        self.snr2 = Field(t2, 2, "SNR", 10.0, -30, 60, "dB")

        cv = ttk.LabelFrame(side, text="Covariance", padding=6)
        cv.pack(fill="x", pady=(0, 8))
        self.sampled = tk.BooleanVar(value=True)
        ttk.Radiobutton(cv, text="Sampled (snapshots)", variable=self.sampled, value=True,
                        command=self._toggle_cov).grid(row=0, column=0, columnspan=3, sticky="w", padx=8)
        ttk.Radiobutton(cv, text="Analytic (exact)", variable=self.sampled, value=False,
                        command=self._toggle_cov).grid(row=1, column=0, columnspan=3, sticky="w", padx=8)
        self.snaps = Field(cv, 2, "Snapshots", 256, 16, 100000, "", integer=True)
        self.seed = Field(cv, 3, "Seed", 0, 0, 2**31 - 1, "", integer=True)
        self.newseed = tk.BooleanVar(value=False)
        ttk.Checkbutton(cv, text="New seed every run", variable=self.newseed).grid(
            row=4, column=0, columnspan=3, sticky="w", padx=8)

        self.run_btn = ttk.Button(side, text="Run on FPGA", style="Run.TButton", command=self.run)
        self.run_btn.pack(fill="x", pady=(4, 6))
        ttk.Label(side, text="Enter in any field also runs.", foreground=MUTED).pack(anchor="w")
        self.msg = tk.StringVar(value="Connecting...")
        ttk.Label(side, textvariable=self.msg, wraplength=220, foreground=MUTED).pack(anchor="w", pady=(10, 0))

        self.fig = Figure(figsize=(10, 6.4), dpi=100, facecolor=SURFACE)
        self.ax = self.fig.add_axes([0.07, 0.36, 0.9, 0.57])
        self.status_ax = self.fig.add_axes([0.07, 0.02, 0.9, 0.24])
        self.status_ax.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side="right", fill="both", expand=True)
        root.bind("<Return>", lambda _e: self.run())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._draw_empty()
        self.root.after(50, self._poll)

    # -- inputs ---------------------------------------------------------------
    def _toggle2(self):
        for f in (self.ang2, self.snr2):
            f.state(self.use2.get())

    def _toggle_cov(self):
        for f in (self.snaps, self.seed):
            f.state(self.sampled.get())

    def scenario(self):
        angles, snr = [self.ang1.get()], [self.snr1.get()]
        if self.use2.get():
            angles.append(self.ang2.get())
            snr.append(self.snr2.get())
        sampled = self.sampled.get()
        if sampled and self.newseed.get():
            self.seed.var.set(str(int(self.seed.get()) + 1))
        seed = self.seed.get() if sampled else 0
        snaps = self.snaps.get() if sampled else music.SNAPSHOTS
        r = music.covariance(angles, seed=seed, sampled=sampled, snr_db=np.array(snr), snapshots=snaps)
        return dict(angles=angles, snr=snr, sampled=sampled, seed=seed, snapshots=snaps,
                    k=len(angles)), np.ascontiguousarray(r, dtype=np.complex64)

    # -- worker ---------------------------------------------------------------
    def connect(self):
        def work():
            try:
                self.results.put(("connected", self.link.connect()))
            except Exception as e:  # noqa: BLE001
                self.results.put(("error", f"Connection failed: {e}"))
        self.busy = True
        threading.Thread(target=work, daemon=True).start()

    def run(self):
        if self.busy:
            return
        try:
            sc, r = self.scenario()
        except ValueError as e:
            self.msg.set(str(e))
            return
        self.busy = True
        self.run_btn.state(["disabled"])
        self.msg.set("Running on the FPGA...")

        def work():
            try:
                self.results.put(("result", (sc, self.link.run(r, sc["k"]))))
            except Exception as e:  # noqa: BLE001
                self.results.put(("error", str(e)))
        threading.Thread(target=work, daemon=True).start()

    def _poll(self):
        try:
            while True:
                kind, payload = self.results.get_nowait()
                self.busy = False
                self.run_btn.state(["!disabled"])
                if kind == "connected":
                    self.msg.set(f"Connected to {self.link.host}: {', '.join(payload['cores'])}. "
                                 f"udmabuf0 {payload['input']}, udmabuf1 {payload['output']}.")
                    self.run()
                elif kind == "result":
                    self.show(*payload)
                else:
                    self.msg.set(payload)
        except queue.Empty:
            pass
        self.root.after(50, self._poll)

    # -- plotting -------------------------------------------------------------
    def _style_axes(self):
        ax = self.ax
        ax.set_facecolor(SURFACE)
        ax.set(xlim=(-90, 90), ylim=(DB_FLOOR, 3), xlabel="Angle (deg)", ylabel="Normalized spectrum (dB)")
        ax.set_xticks(np.arange(-90, 91, 15))
        ax.grid(True, color=GRID, lw=.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    def _draw_empty(self):
        self.ax.clear()
        self._style_axes()
        self.canvas.draw_idle()

    def show(self, sc, reply):
        out, lat = reply["outputs"], reply["latency_us"]
        ax = self.ax
        ax.clear()
        self._style_axes()
        peaks = {}
        for alg in ("bartlett", "capon", "music"):
            p = np.asarray(out[alg], dtype=float)
            if not (np.all(np.isfinite(p)) and p.max() > 0):
                peaks[alg] = None
                continue
            ax.plot(THETA, 10 * np.log10(np.maximum(p / p.max(), 10 ** (DB_FLOOR / 10))),
                    color=COLORS[alg], lw=1.7, label=NAMES[alg])
            peaks[alg] = THETA[music.peak_indices(p, sc["k"])].tolist()
        esprit = np.asarray(out["esprit"][:sc["k"]], dtype=float)
        for i, a in enumerate(esprit):
            ax.axvline(a, color=COLORS["esprit"], lw=3.2, alpha=.85, zorder=3,
                       label="ESPRITEval" if i == 0 else None)
        for i, t in enumerate(sc["angles"]):
            ax.axvline(t, color=INK, ls=(0, (2, 3)), lw=.9, zorder=4, label="True DoA" if i == 0 else None)
            ax.plot([t], [3], marker="v", color=INK, ms=7, clip_on=False, zorder=5)
        ax.legend(loc="lower left", fontsize=8, ncol=5, frameon=False)
        cov = (f"sampled, {sc['snapshots']} snapshots, seed {sc['seed']}" if sc["sampled"] else "analytic")
        targets = ", ".join(f"{a:+.1f} deg @ {s:g} dB" for a, s in zip(sc["angles"], sc["snr"]))
        ax.set_title(f"Targets: {targets}   |   covariance: {cov}", fontsize=10, color=INK, loc="left")

        st = self.status_ax
        st.clear()
        st.axis("off")
        rows = [f"{'Core':<14}{'Latency (us)':>13}   DoA estimate"]
        for alg in ("bartlett", "music", "esprit", "capon"):
            est = esprit.tolist() if alg == "esprit" else peaks.get(alg)
            est_s = "unresolved" if not est else ", ".join(f"{v:+7.2f}" for v in est) + " deg"
            flag = "  (output words not written!)" if reply.get("unwritten", {}).get(alg) else ""
            rows.append(f"{NAMES[alg]:<14}{lat.get(alg, float('nan')):>13.1f}   {est_s}{flag}")
        rows.append(f"{'All 4 cores':<14}{reply['total_us']:>13.1f}   "
                    f"(incl. buffer writes/reads)   host round trip {reply['roundtrip_ms']:.1f} ms")
        st.text(0, 1, "Status report: Arty Z7-20 @ 100 MHz, "
                "latency = PS-measured start->done incl. UIO/Python polling overhead",
                fontsize=8.5, color=MUTED, va="top", transform=st.transAxes)
        st.text(0, 0.82, "\n".join(rows), family="monospace", fontsize=9.5, color=INK,
                va="top", transform=st.transAxes, linespacing=1.45)
        self.canvas.draw_idle()
        self.msg.set(f"Done at {time.strftime('%H:%M:%S')}.")
        if self.snapshot:
            self.fig.savefig(self.snapshot, dpi=110, facecolor=SURFACE)
            self.snapshot = None

    def close(self):
        self.link.close()
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="root@192.168.1.123")
    ap.add_argument("--remote-dir", default="/root/deploy4")
    ap.add_argument("--snapshot", type=Path, help="Save the plot canvas as PNG after the first run")
    args = ap.parse_args()
    root = tk.Tk()
    app = App(root, BoardLink(args.host, args.remote_dir))
    app.snapshot = args.snapshot
    app.connect()
    root.mainloop()


if __name__ == "__main__":
    main()
