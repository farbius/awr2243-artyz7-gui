"""Long-running DoA server on the Arty Z7-20 (deployed to /root/doa_app by the GUI).

Loads the four IP cores once (doa4_eval.DoAEvalSystem: udmabuf0 input, udmabuf1 output,
UIO devices BartlettEval / MUSICEval / ESPRITEval / CaponEval) and then serves binary
requests on stdin, answering on stdout (protocol: doa_protocol.py). Diagnostics go to
stderr. Exits and releases the devices on stdin EOF or a QUIT request.

Latencies are PS-measured start->done (time.perf_counter_ns around Start/IsDone polling),
an upper bound of the core latency including UIO and Python overhead.
"""
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import doa_protocol as P
from doa4_eval import CORES, DoAEvalSystem, unwritten

HERE = Path(__file__).resolve().parent
PID_FILE = HERE / "doa_server.pid"
MAX_SOURCES = 2


class TimedSystem(DoAEvalSystem):
    def __init__(self, algs):
        super().__init__(algs)
        self.latency_ns = {}

    def _start_and_wait(self, alg):
        t0 = time.perf_counter_ns()
        super()._start_and_wait(alg)
        self.latency_ns[alg] = time.perf_counter_ns() - t0


def log(msg):
    sys.stderr.write(f"doa_server: {msg}\n")
    sys.stderr.flush()


def main():
    inp, out = sys.stdin.buffer, sys.stdout.buffer

    def send(msg):
        out.write(msg)
        out.flush()

    PID_FILE.write_text(str(os.getpid()))
    try:
        with contextlib.redirect_stdout(sys.stderr):
            system = TimedSystem(list(P.CORES)).__enter__()
    except Exception as e:  # noqa: BLE001 - reported to the GUI
        send(P.encode_response(P.HELLO, status=1, payload=f"{type(e).__name__}: {e}".encode()))
        PID_FILE.unlink()
        return 1
    info = dict(server="doa_server", protocol=P.VERSION, pid=os.getpid(),
                cores=[CORES[a][0] for a in P.CORES], k_fixed=None, max_sources=MAX_SOURCES,
                input=f"0x{system.tx.phys_addr:08x}", output=f"0x{system.rx.phys_addr:08x}")
    send(P.encode_response(P.HELLO, payload=json.dumps(info).encode()))
    log(f"ready: {info}")
    try:
        while True:
            try:
                head = P.decode_request_header(P.read_exact(inp.read, P.REQ.size))
            except EOFError:
                break
            t_recv = time.perf_counter_ns()
            fid, k = head["frame_id"], head["k"]
            if head["kind"] == P.QUIT:
                break
            if head["kind"] == P.PING:
                send(P.encode_response(P.PONG, fid, t_host_ns=head["t_host_ns"], t_recv_ns=t_recv,
                                       t_done_ns=time.perf_counter_ns()))
                continue
            if head["kind"] != P.RUN:
                send(P.encode_response(P.ERROR, fid, status=2, payload=b"unknown request kind"))
                continue
            cov = P.decode_covariance(P.read_exact(inp.read, P.COV_BYTES))
            try:
                if not 1 <= k <= MAX_SOURCES:
                    raise ValueError(f"number of sources must be 1..{MAX_SOURCES}, got {k}")
                system.latency_ns = {}
                outputs = system.run(cov, k, MAX_SOURCES)
                t_done = time.perf_counter_ns()
                bits = sum(bit for alg, bit in P.UNWRITTEN_BITS.items() if unwritten(outputs[alg]))
                spectra = {alg: np.asarray(outputs[alg], dtype=np.float32) for alg in P.SPECTRA}
                send(P.encode_result(fid, k, spectra, outputs["esprit"][:2],
                                     [system.latency_ns.get(a, 0) for a in P.CORES], bits,
                                     head["t_host_ns"], t_recv, t_done))
            except Exception as e:  # noqa: BLE001 - keep serving
                log(f"frame {fid}: {type(e).__name__}: {e}")
                send(P.encode_response(P.ERROR, fid, status=1, t_host_ns=head["t_host_ns"],
                                       payload=f"{type(e).__name__}: {e}".encode()))
    finally:
        system.__exit__(None, None, None)
        with contextlib.suppress(OSError):
            PID_FILE.unlink()
        log("stopped, devices released")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
