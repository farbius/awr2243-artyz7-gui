"""PySide6 + pyqtgraph main window. The GUI thread only draws; work runs in workers.py threads."""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from . import params as P
from .mock_board import MockBoard
from .processing import RA_ANGLES_DEG, ProcOptions, Processor, parse_adc
from .protocol import THETA_DEG
from .settings import Settings
from .sim import RadarSimulator, Target
from .workers import CameraWorker, Pipeline, SimSource

log = logging.getLogger(__name__)
pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

SPEC_COLORS = dict(bartlett="#2a78d6", music="#eb6834", capon="#8a3ffc")
RX_COLORS = ("#2a78d6", "#eb6834", "#1f9d55", "#8a3ffc")
ESPRIT_COLOR = "#1f9d55"
DB_FLOOR = -50.0


def load_calibration(path):
    """8 complex coefficients (TX0 RX0..3, TX2 RX0..3) from an .npz 'correction', or None."""
    p = Path(path) if path else None
    if not p or not p.is_file():
        return None, "calibration: none (unity)"
    try:
        with np.load(p) as f:
            c = np.asarray(f["correction"], dtype=np.complex64).reshape(-1)
        if c.size != 8 or not np.all(np.isfinite(c)):
            return None, f"calibration: {p.name} invalid (need 8 finite values), unity used"
        return c, f"calibration: {p.name}"
    except Exception as e:  # noqa: BLE001
        return None, f"calibration: cannot load {p.name}: {e}"


class Bridge(QtCore.QObject):
    status = QtCore.Signal(str)
    configured = QtCore.Signal(object)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, settings: Settings, snapshot: str | None = None, autostart: bool = False,
                 transmit_confirmed: bool = False):
        super().__init__()
        self.s = settings
        self.setWindowTitle("Radar DoA: AWR2243BOOST + DCA1000 -> Arty Z7-20 (Bartlett / MUSIC / Capon / ESPRIT)")
        self.bridge = Bridge()
        self.bridge.status.connect(self._on_status)
        self.bridge.configured.connect(self._on_configured)
        self.pipeline = Pipeline(on_status=self.bridge.status.emit)
        self.pipeline.start_threads()
        self.cfg = None
        self.budget = None
        self.source = None
        self.board = None
        self.radar = None
        self.calibration, self.cal_text = load_calibration(self.s.calibration_path)
        self.transmit_confirmed = transmit_confirmed     # --confirm-transmit: approved in advance
        self._configuring = False
        self._seen = dict(proc=0, board=0, cam=0)
        self._build()
        self._load_fields()
        self.camera = CameraWorker(self.s.camera_index, self.s.camera_width, self.s.camera_height,
                                   self.s.camera_fps)
        if snapshot:
            # a saved snapshot must never contain camera frames (live video only)
            self.camera.status = "camera: off (snapshot mode)"
        else:
            self.camera.start()
        self.timer = QtCore.QTimer(self, interval=33, timeout=self._draw)
        self.timer.start()
        self.report_timer = QtCore.QTimer(self, interval=1000, timeout=self._update_report)
        self.report_timer.start()
        self._snapshot_path = snapshot
        if snapshot:
            QtCore.QTimer.singleShot(200, self.configure)      # _on_configured starts + snapshots
        elif autostart:
            QtCore.QTimer.singleShot(200, self.configure)

    # ------------------------------------------------------------------ layout
    def _build(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        v = QtWidgets.QVBoxLayout(central)
        self.canvas = pg.GraphicsLayoutWidget()
        v.addWidget(self.canvas, stretch=3)

        self.p_ra = self.canvas.addPlot(row=0, col=0, title="Range-azimuth (Bartlett FFT, laptop)")
        self.p_ra.setLabel("bottom", "Azimuth", "deg")
        self.p_ra.setLabel("left", "Range", "m")
        self.img_ra = pg.ImageItem()
        self.img_ra.setColorMap(pg.colormap.get("viridis"))
        self.p_ra.addItem(self.img_ra)
        self.ra_target = pg.InfiniteLine(angle=0, pen=pg.mkPen("w", style=QtCore.Qt.DashLine))
        self.p_ra.addItem(self.ra_target)

        self.p_spec = self.canvas.addPlot(row=0, col=1, title="Azimuth spectra at the target cell (board)")
        self.p_spec.setLabel("bottom", "Azimuth", "deg")
        self.p_spec.setLabel("left", "Normalized power", "dB")
        self.p_spec.setXRange(-90, 90)
        self.p_spec.setYRange(DB_FLOOR, 2)
        self.p_spec.showGrid(x=True, y=True, alpha=0.3)
        self.p_spec.addLegend(offset=(-10, 10))
        self.curves = {n: self.p_spec.plot([], [], pen=pg.mkPen(c, width=2), name=n.capitalize())
                       for n, c in SPEC_COLORS.items()}
        self.esprit_lines = []
        for i in range(2):
            line = pg.InfiniteLine(angle=90, pen=pg.mkPen(ESPRIT_COLOR, width=2, style=QtCore.Qt.DashLine),
                                   label="ESPRIT" if i == 0 else None,
                                   labelOpts=dict(position=0.95, color=ESPRIT_COLOR))
            line.setVisible(False)
            self.p_spec.addItem(line)
            self.esprit_lines.append(line)
        self.p_spec.plot([], [], pen=pg.mkPen(ESPRIT_COLOR, width=2, style=QtCore.Qt.DashLine), name="ESPRIT")

        self.p_rng = self.canvas.addPlot(row=1, col=0, title="Range profile")
        self.p_rng.setLabel("bottom", "Range", "m")
        self.p_rng.setLabel("left", "Power", "dB")
        self.p_rng.showGrid(x=True, y=True, alpha=0.3)
        self.p_rng.addLegend(offset=(-10, 10))
        self.rx_curves = [self.p_rng.plot([], [], pen=pg.mkPen(c, width=1.5), name=f"RX{i}")
                          for i, c in enumerate(RX_COLORS)]
        self.rng_target = pg.InfiniteLine(angle=90, pen=pg.mkPen("r", style=QtCore.Qt.DashLine),
                                          label="target", labelOpts=dict(position=0.9, color="r"))
        self.p_rng.addItem(self.rng_target)

        self.p_cam = self.canvas.addPlot(row=1, col=1, title="Camera (live only, not recorded)")
        for ax in ("left", "bottom"):
            self.p_cam.hideAxis(ax)
        self.p_cam.invertY(True)
        self.p_cam.setAspectLocked(True)
        self.img_cam = pg.ImageItem()
        self.p_cam.addItem(self.img_cam)
        self.cam_text = pg.TextItem("no camera", color="w", anchor=(0.5, 0.5))
        self.p_cam.addItem(self.cam_text)
        lay = self.canvas.ci.layout
        for i in range(2):
            lay.setColumnStretchFactor(i, 1)
            lay.setRowStretchFactor(i, 1)

        # ---- bottom panel
        bottom = QtWidgets.QHBoxLayout()
        v.addLayout(bottom, stretch=1)
        self.f = {}
        radar = QtWidgets.QGroupBox("Radar configuration (applied with Configure)")
        form = QtWidgets.QFormLayout(radar)
        self.f["range_res_m"] = self._dspin(0.05, 20, 0.05, 3, form, "Range resolution [m]")
        self.f["max_range_m"] = self._dspin(1, 500, 1, 2, form, "Maximum range [m]")
        self.f["display_range_m"] = self._dspin(0.5, 500, 1, 2, form, "Max display range [m]")
        self.f["mode"] = self._combo(["TDM", "DDM"], form, "MIMO mode")
        self.f["chirps_per_frame"] = self._ispin(8, 1020, form, "Chirps per frame")
        self.f["tx_backoff_db"] = self._ispin(0, 26, form, "TX backoff [dB]")
        self.f["extended_band"] = self._check(form, "Extended band 76-81 GHz")
        bottom.addWidget(radar)

        proc = QtWidgets.QGroupBox("Processing / DoA (applied immediately)")
        form2 = QtWidgets.QFormLayout(proc)
        self.f["k_frames"] = self._ispin(1, 4096, form2, "Frames as snapshots K")
        self.f["num_sources"] = self._ispin(1, 2, form2, "Number of sources")
        self.f["snapshot_mode"] = self._combo(["frame", "chirp"], form2, "Snapshots")
        self.f["fb_averaging"] = self._check(form2, "Forward-backward averaging")
        self.f["diag_loading"] = self._dspin(0, 1, 1e-3, 4, form2, "Diagonal loading (x tr/8)")
        self.f["clutter_removal"] = self._check(form2, "Static clutter removal")
        self.f["flip_array"] = self._check(form2, "Flip array (mirror angles)")
        self.f["ip_latency_us"] = self._dspin(1, 100000, 50, 0, form2, "IP-core latency [us]")
        for key in ("k_frames", "num_sources", "snapshot_mode", "fb_averaging", "diag_loading",
                    "clutter_removal", "flip_array", "ip_latency_us"):
            self._on_change(self.f[key], self._apply_processing)
        for key in ("range_res_m", "max_range_m", "display_range_m", "mode", "chirps_per_frame",
                    "tx_backoff_db", "extended_band"):
            self._on_change(self.f[key], self._radar_changed)
        bottom.addWidget(proc)

        ctl = QtWidgets.QGroupBox("Control")
        cv = QtWidgets.QVBoxLayout(ctl)
        form3 = QtWidgets.QFormLayout()
        self.f["radar_source"] = self._combo(["sim", "awr2243"], form3, "Radar")
        self.f["board_target"] = self._combo(["mock", "board"], form3, "DoA engine")
        self.f["test_frame_count"] = self._ispin(0, 65535, form3, "Radar frames per run (0 = continuous)")
        self.f["bringup"] = self._combo(["auto", "skip", "force"], form3, "Radar bring-up")
        cv.addLayout(form3)
        self.btn_cfg = QtWidgets.QPushButton("Configure (radar + board)")
        self.btn_run = QtWidgets.QPushButton("Start streaming")
        self.btn_single = QtWidgets.QPushButton("Single frame")
        self.btn_cfg.clicked.connect(self.configure)
        self.btn_run.clicked.connect(self.start_stop)
        self.btn_single.clicked.connect(self.single)
        for b in (self.btn_cfg, self.btn_run, self.btn_single):
            cv.addWidget(b)
        row = QtWidgets.QHBoxLayout()
        b_save, b_load = QtWidgets.QPushButton("Save settings"), QtWidgets.QPushButton("Load settings")
        b_save.clicked.connect(self._save_settings)
        b_load.clicked.connect(self._load_settings)
        row.addWidget(b_save)
        row.addWidget(b_load)
        cv.addLayout(row)
        self.k_warn = QtWidgets.QLabel("")
        self.k_warn.setStyleSheet("color: #c0392b")
        self.k_warn.setWordWrap(True)
        cv.addWidget(self.k_warn)
        cv.addStretch(1)
        bottom.addWidget(ctl)

        self.report = QtWidgets.QPlainTextEdit(readOnly=True)
        self.report.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.report.setMinimumWidth(520)
        bottom.addWidget(self.report, stretch=2)

        sb = self.statusBar()
        self.st = {k: QtWidgets.QLabel(f"{k}: -") for k in ("radar", "dca", "board", "camera", "cal")}
        for w in self.st.values():
            w.setFrameShape(QtWidgets.QFrame.StyledPanel)
            sb.addPermanentWidget(w)
        self.st["cal"].setText(self.cal_text)
        self._update_buttons()

    def _dspin(self, lo, hi, step, dec, form, label):
        w = QtWidgets.QDoubleSpinBox(minimum=lo, maximum=hi, singleStep=step, decimals=dec)
        form.addRow(label, w)
        return w

    def _ispin(self, lo, hi, form, label):
        w = QtWidgets.QSpinBox(minimum=lo, maximum=hi)
        form.addRow(label, w)
        return w

    def _combo(self, items, form, label):
        w = QtWidgets.QComboBox()
        w.addItems(items)
        form.addRow(label, w)
        return w

    def _check(self, form, label):
        w = QtWidgets.QCheckBox()
        form.addRow(label, w)
        return w

    @staticmethod
    def _on_change(w, fn):
        for sig in ("valueChanged", "currentTextChanged", "toggled"):
            if hasattr(w, sig):
                getattr(w, sig).connect(lambda *_: fn())
                return

    # ------------------------------------------------------------------ settings <-> fields
    def _load_fields(self):
        for key, w in self.f.items():
            val = getattr(self.s, key)
            w.blockSignals(True)
            if isinstance(w, QtWidgets.QCheckBox):
                w.setChecked(bool(val))
            elif isinstance(w, QtWidgets.QComboBox):
                w.setCurrentText(str(val))
            else:
                w.setValue(val)
            w.blockSignals(False)
        self._check_k()

    def _read_fields(self):
        for key, w in self.f.items():
            if isinstance(w, QtWidgets.QCheckBox):
                setattr(self.s, key, w.isChecked())
            elif isinstance(w, QtWidgets.QComboBox):
                setattr(self.s, key, w.currentText())
            elif isinstance(w, QtWidgets.QSpinBox):
                setattr(self.s, key, int(w.value()))
            else:
                setattr(self.s, key, float(w.value()))

    def _save_settings(self):
        self._read_fields()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save settings", str(Path.cwd() / "radar_doa_gui_settings.json"), "JSON (*.json)")
        if path:
            self.s.save(Path(path))
            self._on_status(f"settings saved to {path}")

    def _load_settings(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load settings", str(Path.cwd()), "JSON (*.json)")
        if path:
            self.s = Settings.load(Path(path))
            self._load_fields()
            self._on_status(f"settings loaded from {path}; press Configure")

    def _check_k(self):
        k = self.f["k_frames"].value()
        self.k_warn.setText(f"K = {k} < 16: covariance estimate is noisy" if k < 16 else "")

    def _radar_changed(self):
        if self.cfg is not None:
            self._on_status("radar parameters changed: press Configure to apply")
        self._read_fields()
        self._update_report()

    def proc_options(self):
        return ProcOptions(clutter_removal=self.s.clutter_removal, snapshot_mode=self.s.snapshot_mode,
                           k_frames=self.s.k_frames, fb_averaging=self.s.fb_averaging,
                           diag_loading=self.s.diag_loading, dc_guard_bins=self.s.dc_guard_bins,
                           tdm_phase_comp=self.s.tdm_phase_comp, ddm_tx2_sign=self.s.ddm_tx2_sign,
                           flip_array=self.s.flip_array, calibration=self.calibration)

    def _apply_processing(self):
        self._read_fields()
        self._check_k()
        self.pipeline.k_sources = self.s.num_sources
        if self.cfg is not None:
            if self.s.snapshot_mode == "chirp" and self.cfg.mode != "TDM":
                self._on_status("chirp snapshots need TDM; using one snapshot per frame")
            self.pipeline.set_processor(self.cfg, Processor(self.cfg, self.proc_options()))

    # ------------------------------------------------------------------ actions
    def configure(self):
        if self._configuring:
            return
        self._read_fields()
        if self.pipeline.acquiring:
            self.pipeline.stop_acquisition()
        req = P.RadarRequest(range_res_m=self.s.range_res_m, max_range_m=self.s.max_range_m,
                             display_range_m=self.s.display_range_m, mode=self.s.mode,
                             chirps_per_frame=self.s.chirps_per_frame, extended_band=self.s.extended_band,
                             idle_s=self.s.idle_us * 1e-6, tx_backoff_db=self.s.tx_backoff_db)
        cfg = P.derive(req)
        if not cfg.ok:
            alt = P.suggest(req)
            msg = "Infeasible configuration:\n- " + "\n- ".join(cfg.issues)
            if alt:
                msg += (f"\n\nNearest valid: dR = {alt.request.range_res_m:.3f} m, "
                        f"R_max = {alt.request.max_range_m:.2f} m")
            QtWidgets.QMessageBox.warning(self, "Configuration", msg)
            return
        self._configuring = True
        self._update_buttons()
        threading.Thread(target=self._configure_worker, args=(cfg,), daemon=True, name="configure").start()

    def _configure_worker(self, cfg):
        emit = self.bridge.status.emit
        try:
            # board first: it is needed by the processing chain either way
            if self.s.board_target == "mock":
                if self.board is not None and not isinstance(self.board, MockBoard):
                    self.board.close()
                self.board = MockBoard()
                emit("board: mock (numpy reference models)")
            else:
                from .board_link import BoardLink
                if not isinstance(self.board, BoardLink) or not self.board.connected:
                    if self.board is not None:
                        self.board.close()
                    self.board = BoardLink(self.s, progress=emit)
                self.board.configure(force_fpga=self.s.force_fpga_reload)
            proc_s, link_s = self._measure_stages(cfg)
            budget = P.fps_budget(cfg, proc_s, link_s, self.s.ip_latency_us * 1e-6,
                                  pipeline_s=self._pipeline_interval())
            emit(f"measured: processing {proc_s * 1e3:.2f} ms, board link {link_s * 1e3:.2f} ms -> "
                 f"frame period {budget.frame_period_s * 1e3:.2f} ms ({budget.bottleneck} bound)")
            if self.s.radar_source == "sim":
                if self.radar is not None:
                    self.radar.close()
                    self.radar = None
                targets = [Target(*t[:4]) for t in self.s.sim_targets]
                self.source = SimSource(RadarSimulator(cfg, targets, self.s.ddm_tx2_sign), budget.frame_period_s)
                emit("radar: simulation")
            else:
                from .radar import Awr2243Radar
                if self.radar is None:
                    self.radar = Awr2243Radar(self.s, progress=emit)
                self.radar.configure(cfg, budget.frame_period_s, bringup=self.s.bringup)
                self.source = self.radar
            self.bridge.configured.emit((cfg, budget))
        except Exception as e:  # noqa: BLE001
            log.exception("configure failed")
            emit(f"configure failed: {e}")
            self.bridge.configured.emit(None)

    def _pipeline_interval(self):
        """Measured result interval of the last streaming run, if it dropped frames (backlogged)."""
        pl = self.pipeline
        if pl.raw_q.dropped + pl.cov_q.dropped == 0 or pl.stats["fps"].value is None:
            return None
        return pl.stats["fps"].value

    def _measure_stages(self, cfg):
        """90th percentile of laptop processing time (synthetic frames) and board link time (20 requests)."""
        proc = Processor(cfg, self.proc_options())
        sim = RadarSimulator(cfg, clutter=False)
        raws = [sim.frame_bytes() for _ in range(3)]
        times = []
        for raw in raws * 3:
            t0 = time.perf_counter()
            proc.process(parse_adc(raw, cfg.request.chirps_per_frame, cfg.num_samples))
            times.append(time.perf_counter() - t0)
        eye = np.eye(8, dtype=np.complex64)
        links = [self.board.run(eye, 1, 0xFFFF0000 + i)["link_s"] for i in range(20)]
        return float(np.percentile(times, 90)), float(np.percentile(links, 90))

    def _on_configured(self, result):
        self._configuring = False
        if result is not None:
            self.cfg, self.budget = result
            self.pipeline.set_board(self.board)
            self.pipeline.k_sources = self.s.num_sources
            fixed = (self.board.info or {}).get("k_fixed") if self.board else None
            self.f["num_sources"].setEnabled(fixed is None)
            if fixed is not None:
                self.f["num_sources"].setValue(int(fixed))
            self._apply_processing()
            self._setup_axes()
            self._on_status(f"configured: {self.cfg.mode}, frame period {self.budget.frame_period_s * 1e3:.2f} ms")
        self._update_buttons()
        self._update_report()
        if self._snapshot_path:
            path, self._snapshot_path = self._snapshot_path, None
            if result is None:
                QtCore.QTimer.singleShot(500, lambda: self._snapshot(path))
            else:
                QtCore.QTimer.singleShot(300, self.start_stop)
                QtCore.QTimer.singleShot(6000, lambda: self._snapshot(path))

    def _confirm_transmit(self):
        if self.s.radar_source == "sim" or self.transmit_confirmed:
            return True
        ans = QtWidgets.QMessageBox.question(
            self, "Radar transmit",
            "The AWR2243 will start transmitting (TX0 + TX2, backoff "
            f"{self.s.tx_backoff_db} dB). Continue?")
        self.transmit_confirmed = ans == QtWidgets.QMessageBox.Yes
        return self.transmit_confirmed

    def start_stop(self):
        if self.pipeline.acquiring:
            self.pipeline.stop_acquisition()
        elif self.source is not None and self._confirm_transmit():
            self.pipeline.processor.reset()
            frames = None
            if self.s.radar_source != "sim" and self.s.test_frame_count > 0:
                frames = self.s.test_frame_count
            self.pipeline.start_acquisition(self.source, frames)
        QtCore.QTimer.singleShot(100, self._update_buttons)

    def single(self):
        if self.source is not None and not self.pipeline.acquiring and self._confirm_transmit():
            self.pipeline.start_acquisition(self.source, max(1, self.s.single_frame_count))
            QtCore.QTimer.singleShot(100, self._update_buttons)

    def _update_buttons(self):
        ready = self.source is not None and not self._configuring
        self.btn_cfg.setEnabled(not self._configuring)
        self.btn_run.setEnabled(ready)
        self.btn_single.setEnabled(ready and not self.pipeline.acquiring)
        self.btn_run.setText("Stop streaming" if self.pipeline.acquiring else "Start streaming")

    # ------------------------------------------------------------------ drawing
    def _setup_axes(self):
        d = self.cfg.display_range_m
        self.p_ra.setXRange(-90, 90, padding=0)
        self.p_ra.setYRange(0, d, padding=0)
        self.p_rng.setXRange(0, d, padding=0.02)
        self.p_rng.setYRange(-60, 2)

    def _draw(self):
        if not self.pipeline.acquiring and "Stop" in self.btn_run.text():
            self._update_buttons()
        res, seq = self.pipeline.latest_proc.get()
        if res is not None and seq != self._seen["proc"]:
            self._seen["proc"] = seq
            r = res.range_m
            dr = r[1] - r[0] if len(r) > 1 else 1.0
            self.img_ra.setImage(res.ra_map_db, autoLevels=False, levels=(-35, 0))
            self.img_ra.setRect(QtCore.QRectF(RA_ANGLES_DEG[0], r[0] - dr / 2, RA_ANGLES_DEG[-1] - RA_ANGLES_DEG[0], dr * len(r)))
            self.ra_target.setValue(res.target_range_m)
            for i, c in enumerate(self.rx_curves):
                c.setData(r, res.range_profile_db[i])
            self.rng_target.setValue(res.target_range_m)
        out, seq = self.pipeline.latest_board.get()
        if out is not None and seq != self._seen["board"]:
            self._seen["board"] = seq
            for name, curve in self.curves.items():
                p = np.asarray(out["spectra"][name], dtype=float)
                if np.all(np.isfinite(p)) and p.max() > 0:
                    curve.setData(THETA_DEG, np.maximum(10 * np.log10(np.maximum(p / p.max(), 1e-12)), DB_FLOOR))
                else:
                    curve.setData([], [])
            k = out.get("k", 2)
            for i, line in enumerate(self.esprit_lines):
                a = out["esprit_deg"][i] if i < len(out["esprit_deg"]) else np.nan
                ok = i < k and np.isfinite(a)
                line.setVisible(bool(ok))
                if ok:
                    line.setValue(a)
            pr = out["proc"]
            angles = ", ".join(f"{a:+.1f}" for a in out["esprit_deg"][:k] if np.isfinite(a))
            self.p_spec.setTitle(f"Spectra at {pr.target_range_m:.2f} m, v {pr.velocity_mps:+.2f} m/s | "
                                 f"ESPRIT {angles} deg | frame {out['frame_id']}")
        frame, seq = self.camera.latest.get()
        if frame is not None and seq != self._seen["cam"]:
            self._seen["cam"] = seq
            self.img_cam.setImage(frame, autoLevels=False, levels=(0, 255))
            self.cam_text.setVisible(False)
        elif frame is None:
            self.cam_text.setText(self.camera.status)
            self.cam_text.setPos(320, 240)

    def _update_report(self):
        self._read_fields()
        cfg = self.cfg if self.cfg is not None else P.derive(P.RadarRequest(
            range_res_m=self.s.range_res_m, max_range_m=self.s.max_range_m, display_range_m=self.s.display_range_m,
            mode=self.s.mode, chirps_per_frame=self.s.chirps_per_frame, extended_band=self.s.extended_band,
            idle_s=self.s.idle_us * 1e-6))
        st = self.pipeline.stats
        budget = P.fps_budget(cfg, st["proc"].value, st["link"].value, self.s.ip_latency_us * 1e-6,
                              pipeline_s=self._pipeline_interval()) if cfg.ok else None
        rows = [(k.replace("Frame period (programmed)", "Frame period (achievable)"), v) for k, v in P.report(cfg, budget)]
        if self.budget is not None:
            rows.insert(0, ("Frame period (programmed)", f"{self.budget.frame_period_s * 1e3:.2f} ms "
                                                         f"({1 / self.budget.frame_period_s:.1f} fps); reconfigure to apply measured values"))
        lines = [("" if self.cfg else "(not configured, preview)")]
        lines += [f"{k:30s} {v}" for k, v in rows]

        def ms(x):
            return "-" if x is None else f"{x * 1e3:.2f} ms"
        out, _ = self.pipeline.latest_board.get()
        lines += ["", "Measured"]
        lines.append(f"{'  processing (FFT + covariance)':30s} {ms(st['proc'].value)}")
        lines.append(f"{'  board round trip':30s} {ms(st['rtt'].value)}  (server on board {ms(st['board'].value)}, "
                     f"link w/o cores {ms(st['link'].value)})")
        if out is not None:
            lat = ", ".join(f"{c} {v / 1e3:.0f}" for c, v in out["latency_ns"].items())
            lines.append(f"{'  IP-core latency [us]':30s} {lat} (sum {sum(out['latency_ns'].values()) / 1e3:.0f}; "
                         "PS-measured upper bound)")
            if out.get("unwritten_cores"):
                lines.append(f"{'  WARNING unwritten output':30s} {', '.join(out['unwritten_cores'])}")
            for e in out.get("errors", []):
                lines.append(f"{'  board note':30s} {e}")
            pr = out["proc"]
            lines.append(f"{'  covariance window':30s} {pr.frames_in_window} frames / {pr.snapshots_in_window} snapshots")
        lines.append(f"{'  end-to-end latency':30s} {ms(st['e2e'].value)}")
        fps, acq = st["fps"].rate, st["acq_fps"].rate
        lines.append(f"{'  end-to-end fps':30s} {'-' if fps is None else f'{fps:.1f}'} "
                     f"(acquisition {'-' if acq is None else f'{acq:.1f}'})")
        lines.append(f"{'  frames in / dropped':30s} {self.pipeline.frames_in} / "
                     f"{self.pipeline.raw_q.dropped + self.pipeline.cov_q.dropped} (bad {self.pipeline.frames_bad})")
        if self.camera.measured_fps.rate:
            lines.append(f"{'  camera':30s} {self.camera.measured_fps.rate:.1f} fps")
        self.report.setPlainText("\n".join(lines))
        self.st["camera"].setText(self.camera.status)
        self.st["board"].setText("board: " + (getattr(self.board, "state", None) or
                                              (self.board.name if self.board else "-")))
        if self.radar is not None:
            self.st["radar"].setText("Studio: " + self.radar.studio_state)
            self.st["dca"].setText("DCA1000: " + self.radar.dca_state)
        else:
            self.st["radar"].setText("radar: " + self.s.radar_source)
            self.st["dca"].setText("DCA1000: -")

    def _on_status(self, msg):
        log.info(msg)
        self.statusBar().showMessage(msg, 15000)

    def _snapshot(self, path):
        self._update_report()
        log.info("report at snapshot:\n%s", self.report.toPlainText())
        self.grab().save(path)
        log.info("snapshot saved to %s", path)
        self.close()

    # ------------------------------------------------------------------ shutdown
    def closeEvent(self, ev):
        self._read_fields()
        try:
            self.s.save()
        except OSError:
            log.exception("cannot save settings")
        self.timer.stop()
        self.pipeline.shutdown()               # stops the source (radar: StopFrame + DCA1000 stop)
        for obj in (self.radar, self.board):
            if obj is not None:
                try:
                    obj.close()
                except Exception:  # noqa: BLE001
                    log.exception("close failed")
        self.camera.stop()
        super().closeEvent(ev)
