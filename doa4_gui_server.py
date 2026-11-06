"""Board-side server for doa4_gui.py (runs on the Arty Z7-20).

Reads one JSON request per line on stdin:
    {"rxx": [128 float32 words: 8x8 complex64, row-major, re/im interleaved], "k": K}
writes the covariance into udmabuf0, runs BartlettEval -> MUSICEval -> ESPRITEval ->
CaponEval on it and answers with one JSON line on stdout:
    {"ok": true, "outputs": {alg: [...]}, "latency_us": {alg: t}, "total_us": t}

latency_us is the start->done time measured on the PS with time.perf_counter_ns()
around XStart/IsDone polling, so it includes the UIO register access and Python
polling overhead (an upper bound of the core latency).
Diagnostics go to stderr; stdout carries only the JSON protocol.
"""
import contextlib
import json
import sys
import time
import numpy as np

import doa4_eval
from doa4_eval import CORES, DoAEvalSystem, IN_WORDS

MAX_SOURCES = 2


class TimedSystem(DoAEvalSystem):
    def __init__(self, algs):
        super().__init__(algs)
        self.latency_ns = {}

    def _start_and_wait(self, alg):
        t0 = time.perf_counter_ns()
        super()._start_and_wait(alg)
        self.latency_ns[alg] = time.perf_counter_ns() - t0


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    algs = list(CORES)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            system = TimedSystem(algs).__enter__()
    except Exception as e:  # noqa: BLE001 - reported to the GUI
        send(dict(ok=False, ready=False, error=f"{type(e).__name__}: {e}"))
        return 1
    send(dict(ok=True, ready=True, cores=[CORES[a][0] for a in algs],
              input=f"0x{system.tx.phys_addr:08x}", output=f"0x{system.rx.phys_addr:08x}"))
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                req = json.loads(line)
                words = np.asarray(req["rxx"], dtype=np.float32)
                if words.size != IN_WORDS:
                    raise ValueError(f"expected {IN_WORDS} float32 words, got {words.size}")
                r = words.view(np.complex64).reshape(8, 8)
                k = int(req["k"])
                system.latency_ns = {}
                t0 = time.perf_counter_ns()
                with contextlib.redirect_stdout(sys.stderr):
                    outputs = system.run(r, k, MAX_SOURCES)
                total = time.perf_counter_ns() - t0
                send(dict(ok=True,
                          outputs={a: np.asarray(o, dtype=float).tolist() for a, o in outputs.items()},
                          unwritten={a: doa4_eval.unwritten(o) for a, o in outputs.items() if a != "esprit"},
                          latency_us={a: v / 1e3 for a, v in system.latency_ns.items()},
                          total_us=total / 1e3))
            except Exception as e:  # noqa: BLE001 - keep serving
                send(dict(ok=False, error=f"{type(e).__name__}: {e}"))
    finally:
        system.__exit__(None, None, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
