"""Standalone Capon Eval signal validation through the native UIO driver."""

import argparse
import importlib
from itertools import permutations
from pathlib import Path

import numpy as np


NLEN = 8
THETA_NUM = 512
THETA_SCAN_DEG = np.linspace(-90.0, 90.0, THETA_NUM)
SNR_DB = np.array([20.0, 5.0])
SNAPSHOTS = 256
IN_ELEMS, OUT_ELEMS = 2 * NLEN * NLEN, THETA_NUM
IN_BYTES, OUT_BYTES = IN_ELEMS * 4, OUT_ELEMS * 8
MAX_NORMALIZED_ERROR = 1e-4


def steering_vector(theta_deg):
    """Eight-element, half-wavelength ULA steering vector."""

    return np.exp(1j * np.pi * np.arange(NLEN) * np.sin(np.deg2rad(theta_deg)))


def steering_matrix():
    return np.stack([steering_vector(theta) for theta in THETA_SCAN_DEG])


def active_snr(snr_db, count):
    values = np.asarray(snr_db, dtype=float)
    if count not in (1, 2) or values.ndim != 1 or len(values) < count:
        raise ValueError("Supply one or two finite per-source SNR values")
    if not np.isfinite(values[:count]).all():
        raise ValueError("SNR values must be finite")
    return values[:count]


def covariance(truth, snr_db):
    manifold = np.column_stack([steering_vector(theta) for theta in truth])
    power = 10.0 ** (active_snr(snr_db, len(truth)) / 10.0)
    matrix = (manifold * power) @ manifold.conj().T + np.eye(NLEN)
    return matrix / np.mean(np.real(np.diag(matrix)))


def signal_cases(num_sources=None, snr_db=SNR_DB):
    definitions = (
        ("one positive analytic", [23.0]),
        ("one negative analytic", [-35.0]),
        ("symmetric pair analytic", [-35.0, 35.0]),
        ("asymmetric pair analytic", [-42.0, 38.0]),
    )
    for seed, (name, truth) in enumerate(definitions):
        count = len(truth)
        if num_sources is not None and count != num_sources:
            continue
        snr = active_snr(snr_db, count)
        case = dict(name=name, k=count, truth=truth, snr_db=snr.tolist(),
                    sampled=False, seed=seed, snapshots=SNAPSHOTS)
        yield case, covariance(truth, snr)


def capon(matrix, steering):
    """Independent complex-domain Capon oracle using a high-level solve."""

    matrix = np.asarray(matrix, dtype=np.complex128)
    steering = np.asarray(steering, dtype=np.complex128)
    if matrix.shape != (NLEN, NLEN) or steering.shape != (THETA_NUM, NLEN):
        raise ValueError("Expected covariance [8,8] and steering [512,8]")
    if not np.isfinite(matrix).all() or not np.isfinite(steering).all():
        raise ValueError("Reference inputs must be finite")
    if not np.allclose(matrix, matrix.conj().T, rtol=1e-6, atol=1e-7):
        raise ValueError("Covariance must be Hermitian")
    np.linalg.cholesky(matrix)
    solved = np.linalg.solve(matrix, steering.T)
    denominator = np.einsum("ti,it->t", steering.conj(), solved).real
    if not np.isfinite(denominator).all() or np.any(denominator <= 0.0):
        raise ValueError("Capon denominator must be finite and positive")
    return 1.0 / denominator


def matched_peaks(values, truth):
    values = np.asarray(values)
    maxima = np.ones(values.shape, dtype=bool)
    maxima[:-1] &= values[:-1] >= values[1:]
    maxima[1:] &= values[1:] >= values[:-1]
    indices = np.flatnonzero(maxima)
    indices = indices[np.argsort(values[indices])[::-1]][:len(truth)]
    if len(indices) != len(truth):
        raise ValueError("Insufficient distinct spectrum peaks")
    candidates = THETA_SCAN_DEG[indices]
    return min((np.asarray(order) for order in permutations(candidates)),
               key=lambda order: np.sum((order - truth) ** 2))


def compare(case, matrix, actual):
    expected = capon(matrix, steering_matrix())
    actual = np.asarray(actual, dtype=float)
    scale = max(float(np.max(np.abs(expected))), 1.0)
    normalized = actual / scale - expected / scale
    metric = dict(
        max_abs_spectrum_error=float(np.max(np.abs(actual - expected))),
        spectrum_rmse=float(np.sqrt(np.mean((actual - expected) ** 2))),
        normalized_rmse=float(np.sqrt(np.mean(normalized ** 2))),
        normalized_max_abs_error=float(np.max(np.abs(normalized))),
    )
    python_peaks = matched_peaks(expected, case["truth"])
    hls_peaks = matched_peaks(actual, case["truth"])
    peak_error = np.abs(hls_peaks - python_peaks)
    metric.update(python_peaks=python_peaks.tolist(), hls_peaks=hls_peaks.tolist(),
                  peak_doa_error=peak_error.tolist())
    grid_bin = THETA_SCAN_DEG[1] - THETA_SCAN_DEG[0]
    metric["passed"] = bool(
        np.isfinite(actual).all()
        and np.all(actual > 0.0)
        and metric["normalized_max_abs_error"] <= MAX_NORMALIZED_ERROR
        and np.all(peak_error <= grid_bin + 1e-9)
    )
    return metric


dev = None


class CaponEvalIP:
    def __enter__(self):
        global dev
        dev = importlib.import_module("capon_eval")
        self.tx, self.rx = dev.Membuf(), dev.Membuf()
        self.ip, self.initialized = dev.XCaponeval(), False
        try:
            for buffer, name, size in ((self.tx, "udmabuf0", IN_BYTES),
                                       (self.rx, "udmabuf1", OUT_BYTES)):
                if dev.udma_allocate(buffer, name):
                    raise RuntimeError(f"Cannot map {name}")
                if buffer.buf_size < size or buffer.phys_addr + size > 2**32:
                    raise ValueError(f"{name}: buffer too small or address exceeds 32 bits")
            if dev.XCaponeval_Initialize(self.ip, "CaponEval") != dev.XST_SUCCESS:
                raise RuntimeError("Cannot initialize CaponEval")
            self.initialized = True
            dev.XCaponeval_InterruptGlobalDisable(self.ip)
            dev.XCaponeval_DisableAutoRestart(self.ip)
            dev.XCaponeval_Set_Rxx_cmpx(self.ip, self.tx.phys_addr)
            dev.XCaponeval_Set_Pspec(self.ip, self.rx.phys_addr)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        if self.initialized:
            dev.XCaponeval_Release(self.ip)
            self.initialized = False
        dev.membuf_release(self.rx)
        dev.membuf_release(self.tx)

    def write_input(self, matrix):
        values = np.ascontiguousarray(matrix, dtype=np.complex64).view(np.float32).ravel()
        if values.size != IN_ELEMS or dev.wr_input(self.tx, values):
            raise ValueError(f"Expected {IN_ELEMS} interleaved float32 values")

    def read_output(self):
        output = np.empty(OUT_ELEMS, dtype=np.float64)
        if dev.rd_output(self.rx, output):
            raise RuntimeError("Cannot read spectrum buffer")
        return output

    def run(self):
        if not dev.XCaponeval_IsReady(self.ip):
            raise RuntimeError("IP is not ready")
        dev.XCaponeval_Start(self.ip)
        for _ in range(1_000_000):
            if dev.XCaponeval_IsDone(self.ip):
                return
        raise TimeoutError("Reset the accelerator before reusing DMA buffers")


def replay_fixtures():
    raw = Path("Rxx_eval_float32.bin").read_bytes()
    if not raw or len(raw) % IN_BYTES:
        raise ValueError("Empty or incomplete covariance records")
    matrices = np.frombuffer(raw, dtype="<f4").reshape(-1, IN_ELEMS)
    with CaponEvalIP() as ip, open("Pspec_eval.bin", "wb") as output:
        for matrix in matrices:
            ip.write_input(matrix.view(np.complex64).reshape(NLEN, NLEN))
            ip.run()
            ip.read_output().astype("<f8", copy=False).tofile(output)
    print(f"CaponEval: {len(matrices)} records written")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-sources", type=int, choices=(1, 2))
    parser.add_argument("--snr-db", type=float, nargs="+", default=SNR_DB.tolist())
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--fixtures", action="store_true")
    args = parser.parse_args()
    active_snr(args.snr_db, args.num_sources or 2)
    if args.fixtures:
        replay_fixtures()
        return 0
    definitions = list(signal_cases(args.num_sources, args.snr_db))
    if args.reference_only:
        for case, matrix in definitions:
            reference = capon(matrix.astype(np.complex64), steering_matrix())
            assert np.isfinite(reference).all() and np.all(reference > 0.0)
            peaks = matched_peaks(reference, case["truth"])
            print(f"{case['name']}: K={case['k']} peaks={peaks.tolist()} reference PASS")
        return 0
    passed = True
    with CaponEvalIP() as ip:
        for case, matrix in definitions:
            matrix = matrix.astype(np.complex64)
            ip.write_input(matrix)
            ip.run()
            metric = compare(case, matrix, ip.read_output())
            print(f"Scenario: {case['name']}")
            print(f"Normalized max error: {metric['normalized_max_abs_error']:.9g}")
            print(f"Python peaks: {metric['python_peaks']}; HLS peaks: {metric['hls_peaks']}")
            print("RESULT:", "PASS" if metric["passed"] else "FAIL")
            passed = metric["passed"] and passed
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
