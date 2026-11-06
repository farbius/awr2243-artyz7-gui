"""Arty Z7-20 link: deploy, FPGA/u-dma-buf init, one persistent SSH channel to doa_server.py.

The board has no sftp-server, so files are uploaded over an exec channel
(`cat > tmp && mv tmp file`), only when their sha256 differs from the board copy.

Init (configfs method, deploy4 CLAUDE.md):
  * skip when our overlay `doa_app` is applied, the four UIO devices exist and the marker
    file matches the bitstream/overlay hashes and the current boot id (unless forced);
  * otherwise stop our server, remove every existing overlay, copy bitstream + overlay to
    /lib/firmware (only if different), apply the overlay, check fpga0 and the UIO names.
u-dma-buf: if u_dma_buf is loaded and /dev/udmabuf0/1 exist with 0x1000000 bytes, skip;
if not loaded, insmod /root/u-dma-buf.ko udmabuf0=0x1000000 udmabuf1=0x1000000;
anything else is reported as an error (never fixed silently).
"""
from __future__ import annotations

import hashlib
import json
import logging
import shlex
import threading
import time
from pathlib import Path

from . import ROOT
from . import protocol as P

log = logging.getLogger(__name__)

PKG = Path(__file__).resolve().parent
UDMABUF_SIZE = 0x1000000
UIO_NAMES = {"BartlettEval", "MUSICEval", "ESPRITEval", "CaponEval"}
OVERLAY = "doa_app"
CFG = "/sys/kernel/config"
OVDIR = f"{CFG}/device-tree/overlays"
BITSTREAM, DTBO = "sys_wrapper.bit.bin", "doa4_eval.dtbo"
REQUEST_TIMEOUT_S = 3.0


def deploy_files():
    """local path -> remote file name."""
    files = {ROOT / "board_server" / "doa_server.py": "doa_server.py",
             PKG / "protocol.py": "doa_protocol.py",
             ROOT / "doa4_eval.py": "doa4_eval.py",
             ROOT / BITSTREAM: BITSTREAM,
             ROOT / DTBO: DTBO}
    for alg in ("bartlett", "music", "esprit", "capon"):
        files[ROOT / f"{alg}_eval.py"] = f"{alg}_eval.py"
        files[ROOT / f"_{alg}_eval.so"] = f"_{alg}_eval.so"
    return files


class BoardError(RuntimeError):
    pass


class BoardLink:
    name = "Arty Z7-20"

    def __init__(self, settings, progress=lambda msg: None):
        self.s, self.progress = settings, progress
        self.dir = settings.board_dir.rstrip("/")
        if not self.dir.startswith("/root/") or " " in self.dir:
            raise BoardError(f"refusing deploy folder {self.dir!r} (must be below /root/)")
        self.client = self.chan = None
        self.info = {}
        self.state = "disconnected"
        self._lock = threading.Lock()
        self._next_reconnect = 0.0

    @property
    def connected(self):
        return self.chan is not None and not self.chan.closed and not self.chan.exit_status_ready()

    # ------------------------------------------------------------------ ssh
    def _connect(self):
        import paramiko
        if self.client is not None and self.client.get_transport() and self.client.get_transport().is_active():
            return
        c = paramiko.SSHClient()
        c.load_system_host_keys()
        c.set_missing_host_key_policy(paramiko.RejectPolicy())
        try:
            c.connect(self.s.board_host, username=self.s.board_user, timeout=8, banner_timeout=8,
                      auth_timeout=8, look_for_keys=True, allow_agent=True)
        except paramiko.SSHException as e:
            raise BoardError(f"SSH to {self.s.board_user}@{self.s.board_host} failed: {e}") from e
        except OSError as e:
            raise BoardError(f"board {self.s.board_host} unreachable: {e}") from e
        c.get_transport().set_keepalive(5)
        self.client = c
        self.state = "ssh connected"

    def exec(self, cmd, timeout=30, data=None, check=True):
        """Run one command; returns (rc, stdout, stderr)."""
        self._connect()
        chan = self.client.get_transport().open_session()
        chan.settimeout(timeout)
        chan.exec_command(cmd)
        if data is not None:
            chan.sendall(data)
            chan.shutdown_write()
        out, err = bytearray(), bytearray()
        deadline = time.monotonic() + timeout
        while True:
            if chan.recv_ready():
                out += chan.recv(65536)
            if chan.recv_stderr_ready():
                err += chan.recv_stderr(65536)
            if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
                break
            if time.monotonic() > deadline:
                chan.close()
                raise BoardError(f"timeout after {timeout} s: {cmd}")
            time.sleep(0.01)
        rc = chan.recv_exit_status()
        chan.close()
        out, err = out.decode(errors="replace"), err.decode(errors="replace")
        if check and rc != 0:
            raise BoardError(f"`{cmd}` failed ({rc}): {err.strip() or out.strip()}")
        return rc, out, err

    # ------------------------------------------------------------------ deploy
    def deploy(self):
        files = deploy_files()
        missing = [str(p) for p in files if not p.is_file()]
        if missing:
            raise BoardError(f"missing local files: {', '.join(missing)}")
        q = shlex.quote(self.dir)
        names = " ".join(shlex.quote(n) for n in files.values())
        self.exec(f"mkdir -p {q}")
        _, out, _ = self.exec(f"cd {q} && sha256sum {names} 2>/dev/null", check=False)
        remote = {line.split()[1]: line.split()[0] for line in out.splitlines() if len(line.split()) == 2}
        uploaded = []
        for path, name in files.items():
            data = path.read_bytes()
            if remote.get(name) == hashlib.sha256(data).hexdigest():
                continue
            dst = shlex.quote(f"{self.dir}/{name}")
            self.exec(f"cat > {dst}.tmp && mv {dst}.tmp {dst}", data=data, timeout=60)
            uploaded.append(name)
        self.progress(f"board: deployed to {self.dir} ({len(uploaded)} file(s) updated)")
        log.info("uploaded: %s", uploaded)
        return uploaded

    # ------------------------------------------------------------------ init
    def _uio_names(self):
        _, out, _ = self.exec("cat /sys/class/uio/uio*/name 2>/dev/null", check=False)
        return set(out.split())

    def init_fpga(self, force=False):
        q = shlex.quote(self.dir)
        self.exec(f"grep -q ' {CFG} configfs' /proc/mounts || mount -t configfs configfs {CFG}")
        _, h, _ = self.exec(f"cd {q} && sha256sum {BITSTREAM} {DTBO} | cut -d' ' -f1 | tr '\\n' ' '; "
                            "cat /proc/sys/kernel/random/boot_id")
        marker_want = h.strip()
        _, marker, _ = self.exec(f"cat {q}/.fpga_loaded 2>/dev/null", check=False)
        _, status, _ = self.exec(f"cat {OVDIR}/{OVERLAY}/status 2>/dev/null", check=False)
        if (not force and status.strip() == "applied" and UIO_NAMES <= self._uio_names()
                and marker.strip() == marker_want):
            self.progress("board: bitstream + overlay already loaded")
            return "already loaded"
        self.stop_server()
        _, others, _ = self.exec("ps -ef 2>/dev/null | grep -E '[d]oa4_gui_server|[d]oa4_eval[.]py' || ps | grep -E '[d]oa4_gui_server|[d]oa4_eval[.]py'",
                                 check=False)
        if others.strip():
            raise BoardError(f"another DoA process holds the cores: {others.strip()}")
        _, ovs, _ = self.exec(f"ls {OVDIR} 2>/dev/null", check=False)
        for ov in ovs.split():
            self.exec(f"rmdir {shlex.quote(OVDIR + '/' + ov)}")
            self.progress(f"board: removed overlay {ov}")
        self.exec(f"cd {q} && for f in {BITSTREAM} {DTBO}; do cmp -s $f /lib/firmware/$f || cp $f /lib/firmware/$f; done")
        self.exec(f"mkdir {OVDIR}/{OVERLAY} && echo -n {DTBO} > {OVDIR}/{OVERLAY}/path")
        _, status, _ = self.exec(f"cat {OVDIR}/{OVERLAY}/status")
        _, fpga, _ = self.exec("cat /sys/class/fpga_manager/fpga0/state")
        uio = self._uio_names()
        if status.strip() != "applied" or fpga.strip() != "operating" or not UIO_NAMES <= uio:
            raise BoardError(f"overlay {status.strip()}, fpga0 {fpga.strip()}, UIO {sorted(uio)}")
        self.exec(f"echo {shlex.quote(marker_want)} > {q}/.fpga_loaded")
        self.progress("board: bitstream + overlay loaded (configfs), 4 UIO cores present")
        return "loaded"

    def init_udmabuf(self):
        _, lsmod, _ = self.exec("lsmod", check=False)
        loaded = any(line.split()[:1] == ["u_dma_buf"] for line in lsmod.splitlines())

        def sizes():
            res = {}
            for b in ("udmabuf0", "udmabuf1"):
                _, out, _ = self.exec(f"[ -e /dev/{b} ] && cat /sys/class/u-dma-buf/{b}/size", check=False)
                res[b] = int(out.strip()) if out.strip().isdigit() else None
            return res

        if loaded:
            sz = sizes()
            if all(v == UDMABUF_SIZE for v in sz.values()):
                self.progress("board: u-dma-buf already loaded (2 x 16 MiB)")
                return "already loaded"
            raise BoardError(f"u_dma_buf is loaded but the buffers are wrong: {sz} (expected {UDMABUF_SIZE} B each); "
                             "reload it manually or reboot the board")
        ko = shlex.quote(self.s.udmabuf_ko)
        rc, _, _ = self.exec(f"[ -f {ko} ]", check=False)
        if rc != 0:
            raise BoardError(f"kernel module {self.s.udmabuf_ko} not found on the board")
        self.exec(f"insmod {ko} udmabuf0={UDMABUF_SIZE:#x} udmabuf1={UDMABUF_SIZE:#x}")
        sz = sizes()
        if not all(v == UDMABUF_SIZE for v in sz.values()):
            raise BoardError(f"insmod done but buffers are {sz}")
        self.progress("board: u-dma-buf loaded (2 x 16 MiB)")
        return "loaded"

    # ------------------------------------------------------------------ server
    def stop_server(self):
        if self.chan is not None:
            try:
                self.chan.sendall(P.encode_request(P.QUIT))
                self.chan.shutdown_write()
                t = time.monotonic() + 2
                while not self.chan.exit_status_ready() and time.monotonic() < t:
                    time.sleep(0.05)
            except Exception:  # noqa: BLE001
                pass
            self.chan.close()
            self.chan = None
        # leftovers from an earlier GUI session; [d] keeps the pattern from matching this shell
        q = shlex.quote(self.dir)
        self.exec(f"p=$(cat {q}/doa_server.pid 2>/dev/null); "
                  "[ -n \"$p\" ] && grep -q doa_server /proc/$p/cmdline 2>/dev/null && kill $p; "
                  "for p in $(pgrep -f '[d]oa_server.py'); do kill $p; done; true", check=False)

    def start_server(self):
        self.stop_server()
        chan = self.client.get_transport().open_session()
        chan.exec_command(f"cd {shlex.quote(self.dir)} && exec python3 -u doa_server.py")
        chan.settimeout(15)
        threading.Thread(target=self._drain_stderr, args=(chan,), daemon=True, name="board-stderr").start()
        try:
            hello = P.read_response(chan.recv)
        except Exception as e:
            chan.close()
            raise BoardError(f"server did not start: {e}") from e
        if hello["kind"] != P.HELLO or hello["status"] != 0:
            chan.close()
            raise BoardError(f"server start failed: {hello.get('text', '')}")
        self.info = json.loads(hello["text"])
        chan.settimeout(REQUEST_TIMEOUT_S)
        self.chan = chan
        self.state = f"server ready (in {self.info['input']}, out {self.info['output']})"
        self.progress(f"board: {self.state}")

    @staticmethod
    def _drain_stderr(chan):
        buf = b""
        while not chan.closed:
            try:
                if chan.recv_stderr_ready():
                    buf += chan.recv_stderr(4096)
                    *lines, buf = buf.split(b"\n")
                    for line in lines:
                        log.info("board: %s", line.decode(errors="replace"))
                elif chan.exit_status_ready():
                    break
                else:
                    time.sleep(0.05)
            except Exception:  # noqa: BLE001
                break

    # ------------------------------------------------------------------ public
    def configure(self, force_fpga=False):
        with self._lock:
            self.progress(f"board: connecting to {self.s.board_host}")
            self._connect()
            self.deploy()
            self.init_udmabuf()
            self.init_fpga(force=force_fpga)
            self.start_server()
            self.ping()

    def ping(self):
        t0 = time.perf_counter_ns()
        self.chan.sendall(P.encode_request(P.PING, t_host_ns=t0))
        r = P.read_response(self.chan.recv)
        return (time.perf_counter_ns() - t0) / 1e9, r

    def _reconnect(self):
        if time.monotonic() < self._next_reconnect:
            raise BoardError("board disconnected (retrying)")
        self._next_reconnect = time.monotonic() + 3
        self.progress("board: reconnecting")
        try:
            if self.client is not None:
                self.client.close()
        except Exception:  # noqa: BLE001
            pass
        self.client = self.chan = None
        self._connect()
        self.init_udmabuf()
        self.init_fpga()
        self.start_server()

    def run(self, cov, k, frame_id=0):
        with self._lock:
            if not self.connected:
                self.state = "disconnected"
                self._reconnect()
            t0 = time.perf_counter_ns()
            try:
                self.chan.sendall(P.encode_request(P.RUN, frame_id, k, t0, cov))
                r = P.read_response(self.chan.recv)
            except Exception as e:
                self.chan.close()
                self.state = f"link error: {e}"
                raise BoardError(f"board link: {e}") from e
            t1 = time.perf_counter_ns()
        if r["kind"] == P.ERROR:
            raise BoardError(f"board: {r.get('text')}")
        if r["kind"] != P.RESULT or r["frame_id"] != frame_id & 0xFFFFFFFF:
            raise BoardError(f"unexpected response kind {r['kind']} frame {r['frame_id']}")
        r["rtt_s"] = (t1 - t0) / 1e9
        r["board_s"] = (r["t_done_ns"] - r["t_recv_ns"]) / 1e9          # server time on the board
        # link = everything except the IP cores (SSH transport + server overhead); the fps model
        # adds NUM_CORES x the GUI IP-latency field to it
        r["link_s"] = r["rtt_s"] - sum(r["latency_ns"].values()) / 1e9
        r["errors"] = []
        return r

    def close(self):
        with self._lock:
            try:
                if self.client is not None:
                    self.stop_server()
            except Exception:  # noqa: BLE001
                log.exception("stop server")
            if self.client is not None:
                self.client.close()
            self.client = self.chan = None
            self.state = "closed"
