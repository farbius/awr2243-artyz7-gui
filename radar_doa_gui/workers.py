"""Worker threads: acquisition -> processing -> board, plus the camera.

Bounded queues drop the oldest item instead of blocking, so a slow stage never builds
up lag. The GUI thread only reads the `Latest` holders and draws.
"""
from __future__ import annotations

import logging
import queue
import threading
import time

import numpy as np

from .processing import Processor, parse_adc

log = logging.getLogger(__name__)


class DropQueue(queue.Queue):
    """Bounded queue; put() discards the oldest entry when full. `dropped` counts them."""

    def __init__(self, maxsize=2):
        super().__init__(maxsize)
        self.dropped = 0

    def put(self, item, block=False, timeout=None):
        while True:
            try:
                return super().put(item, block=False)
            except queue.Full:
                try:
                    self.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass


class Latest:
    """Thread-safe 'most recent value' with a sequence number."""

    def __init__(self):
        self._lock = threading.Lock()
        self.value, self.seq = None, 0

    def set(self, value):
        with self._lock:
            self.value, self.seq = value, self.seq + 1

    def get(self):
        with self._lock:
            return self.value, self.seq


class Ema:
    def __init__(self, alpha=0.1):
        self.alpha, self.value = alpha, None

    def add(self, x):
        self.value = x if self.value is None else (1 - self.alpha) * self.value + self.alpha * x
        return self.value


class RateEma(Ema):
    """Event rate: EMA of the interval, reported as 1 / interval (unbiased, unlike EMA of 1/dt)."""

    def __init__(self, alpha=0.1):
        super().__init__(alpha)
        self._last = None

    def tick(self, now):
        if self._last is not None:
            self.add(now - self._last)
        self._last = now

    @property
    def rate(self):
        return None if not self.value else 1.0 / self.value


class Pipeline:
    """Owns the acquisition, processing and board threads."""

    def __init__(self, on_status=lambda msg: None):
        self.on_status = on_status
        self.raw_q, self.cov_q = DropQueue(2), DropQueue(1)
        self.latest_proc, self.latest_board = Latest(), Latest()
        self.source = self.board = self.processor = None
        self.cfg = None
        self.k_sources = 2
        self._stop = threading.Event()
        self._acq_stop = threading.Event()
        self._threads, self._acq = [], None
        self.stats = dict(proc=Ema(), link=Ema(), rtt=Ema(), board=Ema(), ip=Ema(), e2e=Ema(), fps=RateEma(0.1),
                          acq_fps=RateEma(0.1))
        self.frames_in = self.frames_bad = 0
        self.errors = 0

    # -- lifecycle -------------------------------------------------------------------------
    def start_threads(self):
        if self._threads:
            return
        self._stop.clear()
        for fn, name in ((self._proc_loop, "processing"), (self._board_loop, "board")):
            t = threading.Thread(target=fn, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def set_processor(self, cfg, processor: Processor):
        self.cfg, self.processor = cfg, processor

    def set_board(self, board):
        self.board = board

    def start_acquisition(self, source, frames=None):
        """frames=None: continuous; otherwise stop after that many frames."""
        self.stop_acquisition()
        self.source = source
        self._acq_stop.clear()
        self._acq = threading.Thread(target=self._acq_loop, args=(source, frames), name="acquisition",
                                     daemon=True)
        self._acq.start()

    @property
    def acquiring(self):
        return self._acq is not None and self._acq.is_alive()

    def stop_acquisition(self):
        self._acq_stop.set()
        if self._acq is not None:
            self._acq.join(timeout=5)
            self._acq = None

    def shutdown(self):
        self.stop_acquisition()
        self._stop.set()
        for t in self._threads:
            t.join(timeout=3)
        self._threads = []

    # -- loops ---------------------------------------------------------------------------------
    def _acq_loop(self, source, frames):
        try:
            source.start(frames)
            self.on_status(f"{source.name}: acquiring" + ("" if frames is None else f" {frames} frame(s)"))
            got = 0
            while not self._acq_stop.is_set():
                item = source.read_frame(timeout=0.5)
                if item is None:
                    if source.finished:
                        break
                    continue
                raw, frame_id = item
                now = time.perf_counter()
                self.stats["acq_fps"].tick(now)
                self.frames_in += 1
                got += 1
                self.raw_q.put((raw, frame_id, now))
        except Exception as e:  # noqa: BLE001 - surfaced in the status bar
            log.exception("acquisition failed")
            self.on_status(f"{getattr(source, 'name', 'source')} error: {e}")
        finally:
            try:
                source.stop()
            except Exception as e:  # noqa: BLE001
                log.exception("source stop failed")
                self.on_status(f"stop failed: {e}")
            self.on_status(f"{getattr(source, 'name', 'source')}: stopped")

    def _proc_loop(self):
        while not self._stop.is_set():
            try:
                raw, frame_id, t_acq = self.raw_q.get(timeout=0.2)
            except queue.Empty:
                continue
            proc, cfg = self.processor, self.cfg
            if proc is None:
                continue
            try:
                adc = parse_adc(raw, cfg.request.chirps_per_frame, cfg.num_samples)
                res = proc.process(adc, frame_id, t_acq)
            except Exception as e:  # noqa: BLE001
                self.frames_bad += 1
                log.exception("processing failed")
                self.on_status(f"processing error: {e}")
                continue
            self.stats["proc"].add(res.proc_s)
            self.latest_proc.set(res)
            self.cov_q.put(res)

    def _board_loop(self):
        while not self._stop.is_set():
            try:
                res = self.cov_q.get(timeout=0.2)
            except queue.Empty:
                continue
            board = self.board
            if board is None:
                continue
            try:
                out = board.run(res.covariance, self.k_sources, res.frame_id)
            except Exception as e:  # noqa: BLE001
                self.errors += 1
                log.warning("board run failed: %s", e)
                self.on_status(f"board error: {e}")
                time.sleep(0.2)
                continue
            now = time.perf_counter()
            out["proc"] = res
            out["e2e_s"] = now - res.t_acq
            self.stats["rtt"].add(out["rtt_s"])
            self.stats["link"].add(out["link_s"])
            self.stats["board"].add(out["board_s"])
            self.stats["ip"].add(sum(out["latency_ns"].values()) / 1e9)
            self.stats["e2e"].add(out["e2e_s"])
            self.stats["fps"].tick(now)
            self.latest_board.set(out)


class SimSource:
    """Simulated radar frames, paced at the programmed frame period."""
    name = "simulation"

    def __init__(self, simulator, frame_period_s):
        self.sim, self.period = simulator, frame_period_s
        self.finished, self.remaining, self.frame_id, self.next_t = False, None, 0, 0.0

    def start(self, frames=None):
        self.remaining, self.finished = frames, False
        self.next_t = time.perf_counter()

    def read_frame(self, timeout=0.5):
        if self.remaining == 0:
            self.finished = True
            return None
        delay = self.next_t - time.perf_counter()
        if delay > timeout:
            time.sleep(timeout)
            return None
        if delay > 0:
            time.sleep(delay)
        self.next_t = max(self.next_t + self.period, time.perf_counter() - self.period)
        data = self.sim.frame_bytes()
        self.frame_id += 1
        if self.remaining is not None:
            self.remaining -= 1
        return data, self.frame_id

    def stop(self):
        self.remaining = 0


class CameraWorker:
    """OpenCV capture thread; keeps only the newest RGB frame. Live view only, never saved."""

    def __init__(self, index=-1, width=640, height=480, fps=30):
        self.index, self.size, self.fps = index, (width, height), fps
        self.latest = Latest()
        self.status = "camera: starting"
        self.measured_fps = RateEma(0.1)
        self._stop = threading.Event()
        self._t = None

    def start(self):
        self._stop.clear()
        self._t = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=3)

    def _open(self):
        import cv2
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        candidates = [self.index] if self.index >= 0 else range(5)
        for i in candidates:
            cap = cv2.VideoCapture(i, backend)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            ok, _ = cap.read()
            if ok:
                return i, cap
            cap.release()
        return None, None

    def _loop(self):
        try:
            import cv2
        except ImportError:
            self.status = "camera: OpenCV missing"
            return
        idx, cap = self._open()
        if cap is None:
            self.status = "camera: no camera"
            return
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.status = f"camera: #{idx} {w}x{h}"
        try:
            fails = 0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    fails += 1
                    if fails > 30:
                        self.status = "camera: read failed"
                        break
                    time.sleep(0.02)
                    continue
                fails = 0
                self.measured_fps.tick(time.perf_counter())
                self.latest.set(np.ascontiguousarray(frame[:, :, ::-1]))
        finally:
            cap.release()
