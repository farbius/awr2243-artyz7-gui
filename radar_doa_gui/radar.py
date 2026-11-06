"""AWR2243BOOST + DCA1000EVM: configuration through mmWave Studio (RSTD), streaming over UDP.

Capture path (approach A):
  * mmWave Studio stays the control path: Python -> pythonnet -> RtttNetClientAPI.dll ->
    Studio Lua (RSTD.NetStart(), port 2777). Bring-up (firmware, DCA1000 Ethernet/mode, ADC,
    LVDS, RfInit), profile, chirps and the advanced frame are programmed from Lua
    (lua/doa_common.lua, copied from the repo).
  * Studio's recorder (CaptureCardConfig_StartRecord) is NOT used, so UDP port 4098 stays
    free. This module binds 4096 (DCA1000 commands) and 4098 (data) on the host IP itself,
    sends RECORD_START / RECORD_STOP to the DCA1000 and starts/stops the frames via Studio.
  * Frames are rebuilt from the DCA1000 byte counter; a frame with a missing packet never
    completes and is dropped (data contract: no silent zero filling).
Every stop (normal, error, exit) sends StopFrame and the DCA1000 stop command.
"""
from __future__ import annotations

import json
import logging
import socket
import struct
import threading
import time
import uuid
from pathlib import Path

from . import ROOT
from .params import RadarConfig

log = logging.getLogger(__name__)

LUA_COMMON = ROOT / "lua" / "doa_common.lua"
TMP = ROOT / ".studio_tmp"
CFG_PORT, DATA_PORT = 4096, 4098
DCA_HEADER, DCA_FOOTER = 0xA55A, 0xEEAA
CMD_RECORD_START, CMD_RECORD_STOP, CMD_SYSTEM_CONNECT, CMD_READ_FPGA_VERSION = 0x05, 0x06, 0x09, 0x0E
DCA_PACKET_HEADER = 10            # u32 sequence number + u48 byte count


def lua_string(value):
    """Lua string literal; JSON escaping doubles backslashes, which Lua decodes back."""
    return json.dumps(str(value), ensure_ascii=False)


# ---------------------------------------------------------------------------------- Studio
STALE_CLIENT_HELP = (
    "mmWave Studio's RSTD server does not respond. It serves one client at a time and sometimes "
    "misses a client's disconnect (seen 2026-11-06), so it may still be serving an earlier one. "
    "Restart mmWave Studio, run RSTD.NetStart() in its Lua shell and configure again (bring-up 'auto'). "
    "The radar was not programmed by this call.")


def _with_timeout(fn, timeout, what):
    """Run a blocking .NET call in a daemon thread; raise if it does not return in time."""
    box = {}

    def target():
        try:
            box["result"] = fn()
        except BaseException as e:  # noqa: BLE001 - re-raised in the caller
            box["error"] = e
    t = threading.Thread(target=target, name=f"studio-{what}", daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"{what} did not return within {timeout:.0f} s. {STALE_CLIENT_HELP}")
    if "error" in box:
        raise box["error"]
    return box["result"]


class Studio:
    """RSTD client (as repo capture/awr2243.py, result files in this folder's .studio_tmp).

    Every call into the .NET client has a watchdog: a Studio that does not serve us (stale
    client) gives a clear error instead of hanging the GUI."""

    def __init__(self, dll, port, timeout=10):
        import clr  # pythonnet
        clr.AddReference(str(Path(dll).absolute()))
        from RtttNetClientAPI import RtttNetClient
        self.client = RtttNetClient
        self.lock = threading.Lock()
        TMP.mkdir(exist_ok=True)
        if _with_timeout(self.client.Init, timeout, "RSTD Init") != 0:
            raise RuntimeError("RSTD client initialization failed")
        ok = _with_timeout(lambda: self.client.Connect("127.0.0.1", port), timeout, "RSTD Connect")
        if ok != 0 or not self.client.IsConnected():
            self.close()
            raise RuntimeError("Cannot connect to mmWave Studio. Run RSTD.NetStart() in its Lua shell.")

    def run(self, code, timeout=120):
        result = TMP / (".studio_" + uuid.uuid4().hex + ".txt")
        command = ("local ok, result = pcall(function() " + code + "\nend); "
                   + f"local f = assert(io.open({lua_string(result)}, 'w')); "
                   + "f:write(ok and (result == nil and 'OK' or tostring(result)) or ('ERROR: ' .. tostring(result))); f:close()")
        with self.lock:
            deadline = time.monotonic() + timeout
            try:
                # full Lua timeout: SendCommand may block until the Lua code ends (bring-up ~40 s)
                reply = _with_timeout(lambda: self.client.SendCommand(command), timeout, "RSTD SendCommand")
                status = reply[0] if isinstance(reply, tuple) else reply
                if status not in (0, 30000):
                    raise RuntimeError(f"Studio SendCommand failed: {status}; {self.client.GetLastError()}")
                while time.monotonic() < deadline:
                    if result.exists():
                        answer = result.read_text(encoding="utf-8")
                        if answer:
                            if answer.startswith("ERROR:"):
                                raise RuntimeError(answer)
                            return answer
                    time.sleep(0.02)
                raise RuntimeError("Studio Lua completion timed out; check Studio before retrying.")
            finally:
                result.unlink(missing_ok=True)

    def close(self):
        """Disconnect, give Studio time to register it, then close the client."""
        for name, pause in (("Disconnect", 0.5), ("Close", 0.0)):
            try:
                _with_timeout(getattr(self.client, name), 5, f"RSTD {name}")
            except Exception as exc:  # noqa: BLE001
                log.warning("Studio %s cleanup failed: %s", name, exc)
            time.sleep(pause)


# ---------------------------------------------------------------------------------- Lua
def subframes(cfg: RadarConfig):
    if cfg.mode == "TDM":
        return [dict(id="DOA-TDM", profile=0, chirp_start=0, num_chirps=2, loops=cfg.loops)]
    return [dict(id="DOA-DDM4", profile=0, chirp_start=0, num_chirps=4, loops=cfg.loops)]


def _lua_table(d):
    return "{" + ", ".join(f"{k} = {lua_string(v) if isinstance(v, str) else repr(v)}" for k, v in d.items()) + "}"


BRINGUP_MODES = ("auto", "skip", "force")


def lua_configure(cfg: RadarConfig, bringup="auto", ddm_tx2_sign=+1):
    """Lua for bring-up, profile and chirp RAM.

    bringup: "auto"  run this GUI's bring-up once per Studio session (DOA_GUI_BRINGUP flag);
             "skip"  never run it: use a bring-up already done in Studio (e.g. by hand). The
                     per-chirp phase shifter state is then unknown, so only TDM is allowed and
                     its chirps are programmed without phase commands;
             "force" run it again (e.g. after a board power cycle).
    Our bring-up always enables the per-chirp phase shifter (needed by DDM, set before RfInit).
    """
    if bringup not in BRINGUP_MODES:
        raise ValueError(f"bring-up mode must be one of {BRINGUP_MODES}")
    r = cfg.request
    prof = dict(id=0, idle_us=round(cfg.idle_s * 1e6, 2), adc_start_us=round(cfg.adc_start_s * 1e6, 2),
                ramp_end_us=round(cfg.ramp_end_s * 1e6, 2), slope_mhz_us=round(cfg.slope_hz_per_s / 1e12, 4),
                samples=cfg.num_samples, fs_ksps=int(round(cfg.fs_hz / 1e3)), backoff_db=int(r.tx_backoff_db))
    lines = [f"local C = dofile({lua_string(LUA_COMMON)})"]
    if bringup != "skip":
        lines += [
            f"if DOA_GUI_BRINGUP ~= 'v1' or {str(bringup == 'force').lower()} then",
            "  DOA_GUI_BRINGUP = nil",
            "  C.bringup(true)",
            "  DOA_GUI_BRINGUP = 'v1'",
            "end",
        ]
    lines += [
        "local PCP = (DOA_GUI_BRINGUP == 'v1')   -- per-chirp phase shifter enabled by our bring-up",
        f"C.START_FREQ_GHZ = {cfg.start_freq_hz / 1e9!r}",
        f"C.MAX_FREQ_GHZ = {81.0 if r.extended_band else 77.0!r}",
        f"C.profile({_lua_table(prof)})",
    ]
    if cfg.mode == "TDM":
        lines += ["if PCP then",
                  "  C.chirp(0, 0, 1, 0, 0, 0, 0)   -- TX0 (phase 0 programmed explicitly)",
                  "  C.chirp(1, 0, 0, 1, 0, 0, 0)   -- TX2",
                  "else",
                  "  C.chirp(0, 0, 1, 0, 0)          -- TX0",
                  "  C.chirp(1, 0, 0, 1, 0)          -- TX2",
                  "end"]
    else:
        lines.append("assert(PCP, 'DDM needs the per-chirp phase shifter: use bring-up auto/force "
                     "(restart Studio and power-cycle the board first if it was brought up by hand)')")
        for k in range(4):     # TX2 phase k * 90 deg = 16 codes of 5.625 deg
            lines.append(f"C.chirp({k}, 0, 1, 1, 0, 0, {(ddm_tx2_sign * k * 16) % 64})")
    lines += [
        "ar1.DisableTestSource(0)   -- returns -1 'already disabled' on a fresh setup",
        "DOA_C = C",
        "DOA_SUBFRAMES = {" + ", ".join(_lua_table(sf) for sf in subframes(cfg)) + "}",
    ]
    return "\n".join(lines)


def lua_start(frame_count: int, frame_period_s: float):
    return (f"DOA_C.advanced_frame(DOA_SUBFRAMES, {int(frame_count)}, {frame_period_s * 1e3!r})\n"
            "DOA_C.check(ar1.StartFrame(), 'StartFrame')")


LUA_STOP = "ar1.StopFrame(0)"


# ---------------------------------------------------------------------------------- DCA1000
def dca_command(code, data=b""):
    return struct.pack("<HHH", DCA_HEADER, code, len(data)) + data + struct.pack("<H", DCA_FOOTER)


def parse_dca_response(msg):
    """-> (command code, status); status 0 = success."""
    if len(msg) < 8:
        raise ValueError(f"short DCA1000 response {msg.hex()}")
    header, code, status = struct.unpack("<HHH", msg[:6])
    if header != DCA_HEADER:
        raise ValueError(f"bad DCA1000 response {msg.hex()}")
    return code, status


def parse_data_packet(pkt):
    seq = struct.unpack_from("<I", pkt, 0)[0]
    byte_count = int.from_bytes(pkt[4:10], "little")
    return seq, byte_count, memoryview(pkt)[DCA_PACKET_HEADER:]


class FrameAssembler:
    """Rebuild fixed-size frames from (byte_count, payload); incomplete frames are dropped."""

    def __init__(self, frame_bytes):
        self.fb = frame_bytes
        self.buf = bytearray(frame_bytes)
        self.cur, self.filled = 0, 0
        self.complete = self.dropped = self.late = 0
        self.last_seq = None
        self.lost_packets = 0

    def feed(self, seq, byte_count, payload):
        """Returns a list of (frame_index, bytes) completed by this packet."""
        if self.last_seq is not None and seq > self.last_seq + 1:
            self.lost_packets += seq - self.last_seq - 1
        self.last_seq = seq if self.last_seq is None else max(seq, self.last_seq)
        out, off, data = [], byte_count, payload
        while len(data):
            fi = off // self.fb
            if fi < self.cur:
                self.late += 1
                skip = min(len(data), (self.cur * self.fb) - off)
                off, data = off + skip, data[skip:]
                continue
            if fi > self.cur:
                if self.filled:
                    self.dropped += 1
                self.dropped += fi - self.cur - 1 if fi - self.cur > 1 else 0
                self.cur, self.filled = fi, 0
            pos = off - fi * self.fb
            n = min(len(data), self.fb - pos)
            self.buf[pos:pos + n] = data[:n]
            self.filled += n
            off, data = off + n, data[n:]
            if self.filled >= self.fb:
                out.append((fi, bytes(self.buf)))
                self.complete += 1
                self.cur, self.filled = fi + 1, 0
        return out


class Dca1000:
    def __init__(self, host_ip, dca_ip):
        self.dca = (dca_ip, CFG_PORT)
        self.cfg = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.data = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.cfg.bind((host_ip, CFG_PORT))
            self.data.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 << 20)
            self.data.bind((host_ip, DATA_PORT))
        except OSError as e:
            self.close()
            raise RuntimeError(f"cannot bind UDP {host_ip}:{CFG_PORT}/{DATA_PORT} "
                               f"(is Studio's recorder or another tool holding them?): {e}") from e

    def command(self, code, data=b"", timeout=1.0, retries=3):
        """Send a command and return its status. Retries: the first packet after an idle period
        can be lost while Windows re-resolves the DCA1000's ARP entry."""
        for attempt in range(retries):
            self.cfg.sendto(dca_command(code, data), self.dca)
            deadline = time.monotonic() + timeout
            while (left := deadline - time.monotonic()) > 0:
                self.cfg.settimeout(left)
                try:
                    msg, _ = self.cfg.recvfrom(2048)
                except socket.timeout:
                    break
                rc, status = parse_dca_response(msg)
                if rc == code:
                    return status
                # 0x0A: asynchronous DCA1000 message (e.g. record stopped / error), not our answer
                log.info("DCA1000 async message 0x%02x status 0x%04x while waiting for 0x%02x", rc, status, code)
            log.info("DCA1000 command 0x%02x: no answer (attempt %d)", code, attempt + 1)
        raise RuntimeError(f"DCA1000 did not answer command 0x{code:02x} ({retries} attempts)")

    def flush(self):
        self.data.setblocking(False)
        n = 0
        try:
            while True:
                self.data.recv(65536)
                n += 1
        except (BlockingIOError, OSError):
            pass
        self.data.setblocking(True)
        return n

    def close(self):
        for s in (self.cfg, self.data):
            try:
                s.close()
            except OSError:
                pass


# ---------------------------------------------------------------------------------- source
class Awr2243Radar:
    name = "AWR2243"

    def __init__(self, settings, progress=lambda m: None):
        self.s, self.progress = settings, progress
        self.studio = self.dca = None
        self.cfg = None
        self.period = None
        self.studio_state, self.dca_state = "disconnected", "-"
        self.finished, self.remaining = True, None
        self.assembler = None
        self._running = False

    def configure(self, cfg: RadarConfig, frame_period_s: float, bringup="auto"):
        if self.studio is None:
            self.progress("radar: connecting to mmWave Studio")
            self.studio = Studio(self.s.studio_dll, self.s.studio_port)
            # one RSTD connection per process: Studio's server serves one client at a time
            self.studio.run("return 'ok'", timeout=10)
            self.studio_state = "connected"
        self.progress(f"radar: programming AWR2243 (bring-up {bringup}, then profile/chirps)")
        # chirp RAM: TX2 always +90 deg per chirp; settings.ddm_tx2_sign is the *observed* offset
        self.studio.run(lua_configure(cfg, bringup, ddm_tx2_sign=+1), timeout=180)
        self.studio_state = f"configured {cfg.mode}"
        if self.dca is None:
            self.dca = Dca1000(self.s.host_ip, self.s.dca_ip)
        status = self.dca.command(CMD_SYSTEM_CONNECT)
        self.dca_state = "connected" if status == 0 else f"connect status {status}"
        self.cfg, self.period = cfg, frame_period_s
        self.progress(f"radar: configured {cfg.mode}, {cfg.bytes_per_frame} B/frame, "
                      f"period {frame_period_s * 1e3:.2f} ms")

    def start(self, frames=None):
        """frames=None: continuous (frame count 0); otherwise that many frames."""
        self.assembler = FrameAssembler(self.cfg.bytes_per_frame)
        self.remaining, self.finished = frames, False
        self.dca.flush()
        status = self.dca.command(CMD_RECORD_START)
        if status != 0:
            raise RuntimeError(f"DCA1000 record start failed (status {status})")
        self.dca_state = "recording"
        self._running = True
        try:
            self.studio.run(lua_start(0 if frames is None else frames, self.period), timeout=30)
        except Exception:
            self.stop()
            raise
        self.studio_state = "transmitting"

    def read_frame(self, timeout=0.5):
        if not self._running:
            self.finished = True
            return None
        deadline = time.monotonic() + timeout
        sock = self.dca.data
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                if self.remaining is not None and self.assembler.complete + self.assembler.dropped >= self.remaining:
                    self.finished = True
                return None
            sock.settimeout(left)
            try:
                pkt = sock.recv(65536)
            except socket.timeout:
                continue
            frames = self.assembler.feed(*parse_data_packet(pkt))
            if frames:
                fi, data = frames[-1]
                if self.remaining is not None and self.assembler.complete >= self.remaining:
                    self.finished = True
                    self._running = False
                return data, fi

    @property
    def stats(self):
        a = self.assembler
        return {} if a is None else dict(complete=a.complete, dropped=a.dropped, lost_packets=a.lost_packets)

    def stop(self):
        """Always both: StopFrame via Studio and DCA1000 record stop. Errors are reported, not raised."""
        errors = []
        self._running = False
        self.finished = True
        if self.studio is not None:
            try:
                self.studio.run(LUA_STOP, timeout=10)
                self.studio_state = "stopped"
            except Exception as e:  # noqa: BLE001 - StopFrame fails if frames already ended
                if "-1" not in str(e):
                    errors.append(f"StopFrame: {e}")
                self.studio_state = "stopped"
        if self.dca is not None:
            try:
                self.dca.command(CMD_RECORD_STOP)
                self.dca_state = "stopped"
            except Exception as e:  # noqa: BLE001
                errors.append(f"DCA1000 stop: {e}")
        if self.assembler is not None:
            log.info("radar stop: %s", self.stats)
        if errors:
            log.warning("; ".join(errors))
            self.progress("radar stop: " + "; ".join(errors))

    def close(self):
        if self._running:
            self.stop()
        if self.dca is not None:
            self.dca.close()
            self.dca = None
        if self.studio is not None:
            self.studio.close()
            self.studio = None
        self.studio_state, self.dca_state = "disconnected", "-"
