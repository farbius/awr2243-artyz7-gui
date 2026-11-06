"""Standalone Bartlett Eval signal validation through the native UIO driver."""
import numpy as np
import importlib
import argparse


def normalize_covariance(R: np.ndarray):
    """Return covariance with unit mean diagonal and its original power scale."""
    cov_scale = np.mean(np.real(np.diag(R)))
    return R / cov_scale, cov_scale


NLEN        = 8


SNAPSHOTS   = 256


C_LIGHT     = 3e8


F_C         = 77e9


LAMBDA      = C_LIGHT / F_C


D_SPACING   = LAMBDA / 2


SNR_DB      = np.array([20.0, 5.0])


THETA_NUM   = 512


THETA_SCAN_DEG  = np.linspace(-90.0, 90.0, THETA_NUM)


THETA_SCAN_RAD  = np.deg2rad(THETA_SCAN_DEG)


def steering_vector(theta_deg, N, d, lam):
    th = np.deg2rad(theta_deg)
    n  = np.arange(N)
    return np.exp(1j * 2 * np.pi * d * n * np.sin(th) / lam)


def steering_matrix(N, d, lam):
    """
    Generate steering matrix of shape [THETA_NUM, N]

    Each row corresponds to one steering vector.
    """
    n = np.arange(N)  # antenna indices [0, 1, ..., N-1]

    # Broadcast:
    # THETA_SCAN_RAD[:, None] -> [THETA_NUM, 1]
    # n[None, :]              -> [1, N]
    phase = 2 * np.pi * d * n[None, :] * np.sin(THETA_SCAN_RAD[:, None]) / lam

    A = np.exp(1j * phase)  # shape [THETA_NUM, N]

    return A


MAX_VALIDATION_SOURCES = 2


def active_snr(snr_db, k):
    values = np.asarray(snr_db, dtype=float)
    if not 1 <= k <= MAX_VALIDATION_SOURCES:
        raise ValueError("Validation supports K=1 or K=2")
    if values.ndim != 1 or len(values) < k or not np.isfinite(values).all():
        raise ValueError("Supply at least K finite per-source SNR values")
    return values[:k]


def covariance(manifold, snr_db=SNR_DB, sampled=False, seed=0, snapshots=SNAPSHOTS):
    """One unit-power receiver noise process; source k power is 10**(SNR[k]/10)."""
    manifold = np.asarray(manifold, dtype=complex)
    n, k = manifold.shape
    power = 10.0 ** (active_snr(snr_db, k) / 10.0)
    if sampled:
        rng = np.random.default_rng(seed)
        signals = (rng.normal(size=(k, snapshots)) + 1j*rng.normal(size=(k, snapshots)))/np.sqrt(2)
        noise = (rng.normal(size=(n, snapshots)) + 1j*rng.normal(size=(n, snapshots)))/np.sqrt(2)
        samples = manifold @ (np.sqrt(power)[:, None]*signals) + noise
        matrix = samples @ samples.conj().T / snapshots
    else:
        matrix = (manifold*power) @ manifold.conj().T + np.eye(n)
    return normalize_covariance(matrix)[0]


def matched_peaks(values, coordinates, truth):
    """Local maxima ranked by response, then one-to-one matching to true sources.

    End bins participate; assignment is exhaustive because validation K <= 2.
    This prevents two sources from being matched to the same strong peak.
    """
    from itertools import permutations
    values = np.asarray(values)
    coordinates = np.asarray(coordinates)
    truth = np.asarray(truth).reshape(len(truth), -1)
    maxima = np.ones(values.shape, dtype=bool)
    for axis in range(values.ndim):
        lo = [slice(None)]*values.ndim; hi = lo.copy()
        lo[axis] = slice(None, -1); hi[axis] = slice(1, None)
        maxima[tuple(lo)] &= values[tuple(lo)] >= values[tuple(hi)]
        maxima[tuple(hi)] &= values[tuple(hi)] >= values[tuple(lo)]
    indices = np.flatnonzero(maxima.ravel())
    indices = indices[np.argsort(values.ravel()[indices])[::-1]][:len(truth)]
    if len(indices) != len(truth):
        raise ValueError("Insufficient distinct spectrum peaks for the sources")
    candidates = coordinates.reshape(-1, truth.shape[1])[indices]
    return min((np.asarray(order) for order in permutations(candidates)),
               key=lambda order: np.sum((order-truth)**2))


def spectrum_metrics(expected, actual):
    error = np.abs(np.asarray(actual)-expected)
    return dict(max_spectrum_error=float(error.max()),
                spectrum_rmse=float(np.sqrt(np.mean(error**2))))


def normalized_spectrum_metrics(references, outputs):
    """Normalized linear errors using one Python peak scale across all regions.

    A zero reference uses unit scale so nonzero HLS output remains detectable.
    Regions are concatenated by sample count, not averaged by ROI.
    """
    reference = np.concatenate([np.asarray(r, dtype=float).ravel() for r in references])
    actual = np.concatenate([np.asarray(a, dtype=float).ravel() for a in outputs])
    scale = float(np.max(np.abs(reference)))
    if scale == 0: scale = 1.0
    error = actual/scale-reference/scale
    return dict(normalization_scale=scale, error_domain="normalized linear response",
                rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(np.abs(error))),
                max_abs_error=float(np.max(np.abs(error))))


ROWS = COLS = 4


N = ROWS * COLS


def positions(rows=ROWS, cols=COLS):
    """Y/Z positions in wavelengths, row-major, half-wavelength spacing."""
    y, z = np.meshgrid((np.arange(rows) - (rows - 1) / 2) / 2,
                       (np.arange(cols) - (cols - 1) / 2) / 2, indexing="ij")
    return np.column_stack((y.ravel(), z.ravel()))


def steering(azimuth, elevation, rows=ROWS, cols=COLS):
    az, el = np.broadcast_arrays(np.deg2rad(azimuth), np.deg2rad(elevation))
    direction = np.stack((np.sin(az) * np.cos(el), np.sin(el)), axis=-1)
    return np.exp(2j * np.pi * (direction @ positions(rows, cols).T))


def bartlett(covariance, steering):
    """Full unnormalized complex spectrum a^H R a; no unitary transform."""
    covariance = np.asarray(covariance, dtype=np.complex128)
    steering = np.asarray(steering, dtype=np.complex128)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Covariance must be square")
    if steering.ndim != 2 or steering.shape[1] != covariance.shape[0]:
        raise ValueError("Steering vectors must match the covariance dimension")
    if not np.all(np.isfinite(covariance)) or not np.all(np.isfinite(steering)):
        raise ValueError("Reference inputs must be finite")
    if not np.allclose(covariance, covariance.conj().T, rtol=1e-6, atol=1e-7):
        raise ValueError("Covariance must be Hermitian")
    return np.array([np.real(a.conj() @ covariance @ a) for a in steering])


def signal_cases(num_sources=None, snr_db=SNR_DB):
    definitions = [("one positive", [23.]), ("one negative", [-35.]),
                   ("symmetric pair", [-35., 35.]), ("asymmetric pair", [-42., 38.])]
    for index, (name, truth) in enumerate(definitions):
        k = len(truth)
        if num_sources is not None and k != num_sources: continue
        snr = active_snr(snr_db, k)
        manifold = np.column_stack([steering_vector(t, NLEN, D_SPACING, LAMBDA) for t in truth])
        for sampled in (False, True):
            case = dict(name=name+(" sampled" if sampled else " analytic"), k=k,
                        truth=truth, snr_db=snr.tolist(), sampled=sampled, seed=index,
                        snapshots=256)
            yield case, covariance(manifold, snr, sampled, index)


def compare(case, matrix, actual):
    expected = bartlett(matrix, steering_matrix(NLEN, D_SPACING, LAMBDA))
    scale = max(float(np.max(np.abs(expected))), 1.0)
    # Validate raw linear power; normalization is only a display operation.
    ok = bool(np.isfinite(actual).all() and np.allclose(actual, expected, rtol=2e-5, atol=2e-6*scale))
    metric = dict(name=case['name'], k=case['k'], snr_db=case.get('snr_db', []),
                  truth=case['truth'], **spectrum_metrics(expected, actual))
    if case['truth']:
        coords = THETA_SCAN_DEG[:, None]
        python = matched_peaks(expected, coords, case['truth'])[:, 0]
        hls = matched_peaks(actual, coords, case['truth'])[:, 0]
        error = np.abs(hls-python)
        metric.update(python_peaks=python.tolist(), hls_peaks=hls.tolist(),
                      peak_doa_error=error.tolist(), max_peak_error=float(error.max()),
                      true_doa_error=np.abs(hls-case['truth']).tolist())
        ok = ok and bool(np.all(error <= THETA_SCAN_DEG[1]-THETA_SCAN_DEG[0]+1e-9))
    metric['passed'] = ok
    return metric, expected


dev = None

IN_ELEMS, OUT_ELEMS = 128, 512
IN_BYTES, OUT_BYTES = IN_ELEMS * 4, OUT_ELEMS * 8


class BartlettEvalIP:
    def __enter__(self):
        global dev
        dev = importlib.import_module("bartlett_eval")
        self.udmabuf_tx, self.udmabuf_rx = dev.Membuf(), dev.Membuf()
        self.ip, self.initialized = dev.XBartletteval(), False
        try:
            for buffer, name, size in ((self.udmabuf_tx, "udmabuf0", IN_BYTES),
                                       (self.udmabuf_rx, "udmabuf1", OUT_BYTES)):
                if dev.udma_allocate(buffer, name):
                    raise RuntimeError(f"Cannot map {name}")
                if buffer.buf_size < size or buffer.phys_addr + size > 2**32:
                    raise ValueError(f"{name}: buffer too small or address exceeds 32 bits")
            if dev.XBartletteval_Initialize(self.ip, "BartlettEval") != dev.XST_SUCCESS:
                raise RuntimeError("Cannot initialize BartlettEval")
            self.initialized = True
            dev.XBartletteval_InterruptGlobalDisable(self.ip)
            dev.XBartletteval_DisableAutoRestart(self.ip)
            dev.XBartletteval_Set_Rxx_cmpx(self.ip, self.udmabuf_tx.phys_addr)
            dev.XBartletteval_Set_Pspec(self.ip, self.udmabuf_rx.phys_addr)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        if self.initialized:
            dev.XBartletteval_Release(self.ip)
            self.initialized = False
        dev.membuf_release(self.udmabuf_rx)
        dev.membuf_release(self.udmabuf_tx)

    def write_in(self, covariance):
        values = np.asarray(covariance)
        if np.iscomplexobj(values):
            values = np.ascontiguousarray(values, dtype=np.complex64).view(np.float32)
        values = np.ascontiguousarray(values, dtype=np.float32).ravel()
        if values.size != IN_ELEMS or dev.wr_input(self.udmabuf_tx, values):
            raise ValueError(f"Expected {IN_ELEMS} interleaved float32 values")

    def read_out(self):
        output = np.empty(OUT_ELEMS, dtype=np.float64)
        if dev.rd_output(self.udmabuf_rx, output):
            raise RuntimeError("Cannot read spectrum buffer")
        return output

    def run(self):
        if not dev.XBartletteval_IsReady(self.ip):
            raise RuntimeError("IP is not ready")
        dev.XBartletteval_Start(self.ip)
        for _ in range(1_000_000):
            if dev.XBartletteval_IsDone(self.ip):
                return
        raise TimeoutError("Reset the accelerator before reusing DMA buffers")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-sources", type=int, choices=(1, 2))
    parser.add_argument("--snr-db", type=float, nargs="+", default=SNR_DB.tolist())
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--fixtures", action="store_true", help="Preserve binary record replay mode")
    args = parser.parse_args()
    active_snr(args.snr_db, args.num_sources or 2)
    if args.fixtures:
        replay_fixtures()
        return 0
    definitions = list(signal_cases(args.num_sources, args.snr_db))
    if args.reference_only:
        for case, matrix in definitions:
            reference = bartlett(matrix.astype(np.complex64), steering_matrix(NLEN,D_SPACING,LAMBDA))
            assert np.isfinite(reference).all()
            print(f"{case['name']}: K={case['k']} SNR={case['snr_db']} dB; covariance/reference check PASS")
        return 0
    passed = True
    with BartlettEvalIP() as ip:
        for case, matrix in definitions:
            matrix = matrix.astype(np.complex64)
            ip.write_in(matrix)
            ip.run()
            actual = ip.read_out()
            metric, reference = compare(case, matrix, actual)
            normalized = normalized_spectrum_metrics([reference], [actual])
            metric.update(raw_spectrum_rmse=metric['spectrum_rmse'], spectrum_rmse=normalized['rmse'],
                          spectrum_mae=normalized['mae'], max_abs_spectrum_error=normalized['max_abs_error'],
                          normalization_scale=normalized['normalization_scale'])
            print(f"Scenario: {case['name']}\nK: {case['k']}\nSNR per source: {case['snr_db']} dB")
            print(f"RMSE: {normalized['rmse']:.9g} (normalized response)")
            print(f"Max abs error: {normalized['max_abs_error']:.9g}")
            print(f"True: {case['truth']}; Python peaks: {metric['python_peaks']}; HLS peaks: {metric['hls_peaks']}")
            print("RESULT:", "PASS" if metric['passed'] else "FAIL")
            passed = metric['passed'] and passed
    return 0 if passed else 1


def replay_fixtures():
    # Read bytes first so even a trailing partial scalar is rejected.
    from pathlib import Path
    raw = Path("Rxx_eval_float32.bin").read_bytes()
    if not raw or len(raw) % IN_BYTES:
        raise ValueError("Empty or incomplete covariance records")
    matrices = np.frombuffer(raw, dtype="<f4").reshape(-1, IN_ELEMS)
    with BartlettEvalIP() as ip, open("Pspec_eval.bin", "wb") as output:
        for index, matrix in enumerate(matrices):
            ip.write_in(matrix)
            ip.run()
            ip.read_out().astype("<f8", copy=False).tofile(output)
    print(f"BartlettEval: {len(matrices)} records written")


if __name__ == "__main__":
    raise SystemExit(main())
