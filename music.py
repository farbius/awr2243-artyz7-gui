"""MUSICEval validation of the HLS testbench scenarios through the native UIO/SWIG driver.

Hardware run (target):   python3 music.py [--num-sources K] [--output-dir DIR]
Offline replay (no FPGA): python3 music.py --replay DIR   (HLS fixtures or a saved hardware run)
Reference check only:     python3 music.py --reference-only
"""
import argparse
import importlib
import json
import re
from pathlib import Path
import numpy as np

EDITION, IP_NAME, MODULE, SLUG = "MUSIC Eval", "MUSICEval", "music_eval", "music_eval"
NLEN = 8
THLEN = 512
MAX_SOURCES = 2
# Tolerances and conventions from hls_validation/music_eval.py.
D_FLOOR = 1.0e-6
D_ATOL = 5.0e-5
D_RTOL = 1.0e-3
PEAK_BINS = 1
SNAPSHOTS = 2048
SNR_DB = np.array([15.0, 10.0])
THETA_SCAN_DEG = np.linspace(-90.0, 90.0, THLEN)
GRID_STEP_DEG = float(THETA_SCAN_DEG[1]-THETA_SCAN_DEG[0])
DDR_LIMIT = 0x20000000
IN_WORDS, IN_BYTES, OUT_BYTES = 2*NLEN*NLEN, 2*NLEN*NLEN*4, THLEN*4
SENTINELS = (0xDEADBEEF, 0x7FC00001)
FILES = dict(input="Rxx_eval_float32.bin", counts="num_sources.bin", output="Pspec_eval_float32.bin")


def steering(angles_deg, n=NLEN):
    """Positive-phase half-wavelength ULA manifold a[n] = exp(j*pi*n*sin(theta))."""
    angles = np.atleast_1d(np.asarray(angles_deg, dtype=float))
    return np.exp(1j*np.pi*np.arange(n)[:, None]*np.sin(np.deg2rad(angles))[None, :])


def covariance(angles, seed=0, sampled=False, snr_db=SNR_DB, snapshots=SNAPSHOTS):
    """Uncorrelated sources in unit-power noise, normalized to trace NLEN (as the HLS fixtures)."""
    power = 10.0 ** (np.asarray(snr_db, dtype=float)[:len(angles)] / 10.0)
    a = steering(angles)
    if sampled:
        rng = np.random.default_rng(seed)
        s = (rng.normal(size=(len(angles), snapshots))+1j*rng.normal(size=(len(angles), snapshots)))/np.sqrt(2)
        w = (rng.normal(size=(NLEN, snapshots))+1j*rng.normal(size=(NLEN, snapshots)))/np.sqrt(2)
        x = a@(np.sqrt(power)[:, None]*s) + w
        r = x@x.conj().T/snapshots
    else:
        r = (a*power)@a.conj().T + np.eye(NLEN)
    return r/np.trace(r).real*NLEN


def fb_average(r):
    return (r+r[::-1, ::-1].conj())/2


def reference(r, k):
    """Independent complex MUSIC null spectrum a^H En En^H a on the FB-averaged covariance."""
    _, v = np.linalg.eigh(fb_average(np.asarray(r, dtype=complex)))
    noise = v[:, :NLEN-k]
    return np.sum(np.abs(noise.conj().T@steering(THETA_SCAN_DEG))**2, axis=0)


def peak_indices(values, count):
    """Indices of the `count` largest local maxima (end bins participate), ascending."""
    values = np.asarray(values, dtype=float)
    maxima = np.ones(values.shape, dtype=bool)
    maxima[:-1] &= values[:-1] >= values[1:]
    maxima[1:] &= values[1:] >= values[:-1]
    # Plateaus (e.g. clamped floors) keep only their first bin.
    maxima[1:] &= ~(maxima[:-1] & (values[1:] == values[:-1]))
    indices = np.flatnonzero(maxima)
    indices = indices[np.argsort(values[indices], kind='stable')[::-1]][:count]
    return np.sort(indices)


def scenarios(num_sources=None):
    """The 21 HLS testbench scenarios (hls_validation.music_eval.prepare), in fixture order."""
    cases = []
    signal = [[23.], [-35., 35.], [-42., 38.], [40., 70.], [10., 18.]]
    for index, angles in enumerate(signal):
        if num_sources is not None and len(angles) != num_sources:
            continue
        for sampled in (False, True):
            cases.append((dict(name=f'{angles} '+('sampled' if sampled else 'analytic'), kind='signal',
                               k=len(angles), truth=angles, sampled=sampled),
                          covariance(angles, seed=index, sampled=sampled)))
    rng = np.random.default_rng(2718)
    x = rng.normal(size=(NLEN, NLEN+3))+1j*rng.normal(size=(NLEN, NLEN+3))
    arbitrary = x@x.conj().T/x.shape[1]
    for k in ([num_sources] if num_sources is not None else range(1, MAX_SOURCES+1)):
        cases.append((dict(name=f'non-centro-Hermitian PSD K={k}', kind='stress', k=k, truth=[]),
                      arbitrary/np.trace(arbitrary).real*NLEN))
        for name, matrix in (('identity', np.eye(NLEN)), ('zero', np.zeros((NLEN, NLEN)))):
            cases.append((dict(name=f'{name} K={k}', kind='degenerate', k=k, truth=[]), matrix))
    for k in (0, MAX_SOURCES+1, 4, 8, 4294967295):
        cases.append((dict(name=f'invalid K={k}', kind='invalid', k=k, truth=[]), covariance([23.])))
    return [(c, np.ascontiguousarray(r, dtype=np.complex64)) for c, r in cases]


def case_id(case):
    """Deterministic file identifier, e.g. two_targets_m35_p35_analytic."""
    if case['kind'] != 'signal':
        return re.sub(r'[^a-z0-9]+', '_', case['name'].lower().replace('k=', 'k')).strip('_')
    count = ['single_target', 'two_targets', 'three_targets', 'four_targets'][len(case['truth'])-1]
    angles = '_'.join(('m' if t < 0 else 'p')+f'{abs(t):g}'.replace('.', 'p') for t in case['truth'])
    return f"{count}_{angles}_{'sampled' if case['sampled'] else 'analytic'}"


def spectrum_values(out):
    return np.asarray(out, dtype=np.float32).astype(float)


def compare(case, r, out):
    """HLS compare() criteria; returns a metric dict with 'passed' and 'reason'."""
    p = spectrum_values(out)
    m = dict(name=case['name'], kind=case['kind'], k=case['k'], passed=False, reason='')
    if case['kind'] == 'invalid':
        m['passed'] = bool(np.all(p == 0))
        m['reason'] = '' if m['passed'] else 'expected an all-zero spectrum'
        return m
    if not (np.all(np.isfinite(p)) and np.all(p > 0)):
        m['reason'] = 'nonfinite or nonpositive spectrum'
        return m
    d = 1/p
    m.update(d=d, min_d=float(d.min()), max_d=float(d.max()))
    if case['kind'] == 'degenerate':
        m['passed'] = bool(np.all(d <= NLEN*(1+D_RTOL)+D_ATOL))
        m['reason'] = '' if m['passed'] else 'null spectrum exceeds NLEN'
        return m
    ref = reference(np.asarray(r).astype(complex), case['k'])
    expected = np.maximum(ref, D_FLOOR)
    error = np.abs(d-expected)
    bound = D_ATOL+D_RTOL*expected
    m.update(ref_d=ref, error=error, bound=bound, max_d_error=float(error.max()),
             tolerance_used=float(np.max(error/bound)), worst_bin=int(np.argmax(error/bound)))
    if not np.all(error < bound+1e-15):
        m['reason'] = 'null spectrum outside the HLS tolerance'
        return m
    if case['kind'] == 'signal':
        hls, python = peak_indices(p, case['k']), peak_indices(1/expected, case['k'])
        m.update(hw_bins=hls.tolist(), ref_bins=python.tolist(), hw_deg=THETA_SCAN_DEG[hls].tolist(),
                 ref_deg=THETA_SCAN_DEG[python].tolist())
        if len(hls) != case['k'] or len(python) != case['k']:
            m['reason'] = 'fewer than K spectrum peaks'
            return m
        if np.any(np.abs(hls-python) > PEAK_BINS):
            m['reason'] = 'hardware peak more than one bin from the reference'
            return m
        if not case['sampled'] and np.any(np.abs(THETA_SCAN_DEG[hls]-case['truth']) > GRID_STEP_DEG):
            m['reason'] = 'hardware peak more than one grid step from the true DoA'
            return m
    m['passed'] = True
    return m


def angles(values):
    return ', '.join(f'{v:+.3f}' for v in values) + ' deg'


def print_banner():
    print("========================================")
    print(f"{EDITION} - 1D ULA, {NLEN} RX")
    print("========================================")
    print(f"Grid: {THLEN} bins, -90..+90 deg, step {GRID_STEP_DEG:.3f} deg; K = 1..{MAX_SOURCES}\n")


def print_result(case, m):
    kinds = dict(signal='signal', stress='PSD stress (no true DoA)', degenerate='degenerate (bounds only)',
                 invalid='invalid/zero output')
    print(f"Test: {case['name']}\nType: {kinds[case['kind']]}")
    print(f"Snapshots: {SNAPSHOTS} (seeded sampled covariance)" if case.get('sampled') else
          "Snapshots: analytic covariance")
    print(f"Sources (K): {case['k']}")
    if case['kind'] == 'signal':
        print(f"Expected DoA: {angles(case['truth'])}")
    if 'hw_deg' in m and len(m['hw_deg']) == case['k']:
        hw, ref, truth = np.array(m['hw_deg']), np.array(m['ref_deg']), np.array(case['truth'])
        print(f"Reference DoA: {angles(ref)}\nEstimated DoA: {angles(hw)}")
        limit = 'informative for sampled data' if case['sampled'] else 'limit one grid step'
        print(f"Angular errors: {', '.join(f'{e:.3f}' for e in np.abs(hw-truth))} deg vs truth "
              f"({limit}; grid step {GRID_STEP_DEG:.3f} deg)")
        print(f"Grid/model error: {', '.join(f'{e:.3f}' for e in np.abs(ref-truth))} deg (reference peak vs truth)")
        bins = np.abs(np.array(m['hw_bins'])-m['ref_bins'])
        print(f"Implementation error: {', '.join(str(b) for b in bins)} bins (hardware vs reference peak, limit 1)")
    if 'max_d_error' in m:
        print(f"Null spectrum: max |D_hw - D_ref| = {m['max_d_error']:.3g} "
              f"({100*m['tolerance_used']:.1f}% of tolerance, worst bin {m['worst_bin']})")
    elif case['kind'] == 'degenerate' and 'max_d' in m:
        print(f"Null spectrum: D in [{m['min_d']:.3g}, {m['max_d']:.3g}] (bound {NLEN})")
    if not m['passed']:
        print(f"Failure: {m['reason']}")
    print(f"Result: {'PASS' if m['passed'] else 'FAIL'}\n")


def print_summary(passed, total):
    print("========================================")
    print(f"Summary: {passed}/{total} tests passed")
    print("========================================")


INK, MUTED, SURFACE, HW, EST = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6", "#eb6834"


def plot_case(case, m, out, path, origin):
    """Pseudospectrum and null-spectrum error versus the hardware angular grid (output data unchanged)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = spectrum_values(out)
    title = f"{EDITION} - {case['name']} (K={case['k']})"
    rows = 1 if case['kind'] == 'invalid' or 'd' not in m else 2
    fig, axes = plt.subplots(rows, 1, figsize=(9, 3.4*rows+0.6), sharex=True, squeeze=False,
                             layout='constrained', facecolor=SURFACE)
    top = axes[0, 0]
    if rows == 1:
        top.plot(THETA_SCAN_DEG, p, color=HW, lw=2, label='Hardware output')
        top.set(ylabel='Output value (linear)')
        top.set_title('Expected all-zero spectrum' if case['kind'] == 'invalid' else 'Raw output', color=INK)
    else:
        ref_p = 1/np.maximum(m['ref_d'], D_FLOOR) if 'ref_d' in m else None
        scale = ref_p.max() if ref_p is not None else p.max()
        if ref_p is not None:
            top.plot(THETA_SCAN_DEG, 10*np.log10(ref_p/scale), color=MUTED, lw=3, alpha=.55,
                     label='NumPy reference')
        top.plot(THETA_SCAN_DEG, 10*np.log10(p/scale), color=HW, lw=1.5, label='Hardware')
        for i, angle in enumerate(case['truth']):
            top.axvline(angle, color=INK, ls='--', lw=1, label='Expected DoA' if i == 0 else None)
        if m.get('hw_bins'):
            top.plot(THETA_SCAN_DEG[m['hw_bins']], 10*np.log10(p[m['hw_bins']]/scale), 'o', ms=8, color=EST,
                     markeredgecolor=SURFACE, markeredgewidth=2, label='Estimated DoA', zorder=5)
            for b in m['hw_bins']:
                top.annotate(f"{THETA_SCAN_DEG[b]:+.2f} deg", (THETA_SCAN_DEG[b], 10*np.log10(p[b]/scale)),
                             xytext=(6, 4), textcoords='offset points', fontsize=8, color=INK)
        top.set(ylabel='Pseudospectrum (dB, normalized)', ylim=(-50, 5))
        top.set_title('MUSIC pseudospectrum P = 1/max(D, D_floor)', color=INK, fontsize=10)
        low = axes[1, 0]
        if 'error' in m:
            low.semilogy(THETA_SCAN_DEG, np.maximum(m['error'], 1e-12), color=HW, lw=1.2, label='|D_hw - D_ref|')
            low.semilogy(THETA_SCAN_DEG, m['bound'], color=MUTED, ls='--', lw=1.2, label='HLS tolerance')
            low.set(ylabel='Null-spectrum error')
        else:
            low.semilogy(THETA_SCAN_DEG, m['d'], color=HW, lw=1.2, label='D_hw = 1/P')
            low.axhline(NLEN, color=MUTED, ls='--', lw=1.2, label=f'Bound D <= {NLEN}')
            low.set(ylabel='Null spectrum D')
        low.legend(loc='lower right', fontsize=8)
    for row in axes[:, 0]:
        row.grid(True, color='#e4e3df', lw=.8)
        row.set_facecolor(SURFACE)
        row.set_xlim(-90, 90)
    axes[-1, 0].set_xlabel('Angle (deg)')
    top.legend(loc='lower left', fontsize=8)
    fig.suptitle(f"{title}: {'PASS' if m['passed'] else 'FAIL'}\n{origin}", fontsize=11, color=INK)
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def report(cases, matrices, outputs, origin, output_dir, plots=True, repeatable=None):
    print_banner()
    print(f"Data origin: {origin}\n")
    metrics = [compare(c, r, o) for c, r, o in zip(cases, matrices, outputs)]
    for m, same in zip(metrics, repeatable or []):
        if not same:
            m.update(passed=False, reason='outputs differ between two transactions')
    for c, m in zip(cases, metrics):
        print_result(c, m)
    passed = sum(m['passed'] for m in metrics)
    print_summary(passed, len(cases))
    if plots:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            print(f"Matplotlib is not installed: PNG plots skipped. Copy {output_dir} to a host and run "
                  f"'python3 music.py --replay {output_dir.name}'.")
        else:
            for c, m, o in zip(cases, metrics, outputs):
                plot_case(c, m, o, output_dir/f"{SLUG}_{case_id(c)}.png", origin)
            print(f"Wrote {len(cases)} PNG plots to {output_dir}")
    return passed == len(cases)


class MUSICEvalIP:
    def __enter__(self):
        self.dev = dev = importlib.import_module(MODULE)
        self.tx, self.rx, self.ip, self.initialized = dev.Membuf(), dev.Membuf(), dev.XMusiceval(), False
        try:
            for buffer, name, size in ((self.tx, "udmabuf0", IN_BYTES), (self.rx, "udmabuf1", OUT_BYTES)):
                if dev.udma_allocate(buffer, name):
                    raise RuntimeError(f"Cannot map {name}")
                if buffer.buf_size < size or buffer.phys_addr + size > DDR_LIMIT or buffer.phys_addr % 4:
                    raise ValueError(f"{name}: buffer too small, unaligned or outside the 512 MiB HP DDR window")
            if dev.XMusiceval_Initialize(self.ip, IP_NAME) != dev.XST_SUCCESS:
                raise RuntimeError(f"Cannot initialize {IP_NAME} UIO device")
            self.initialized = True
            if self.ip.Control_BaseAddress in (0, 0xffffffff):
                raise RuntimeError("UIO mmap failed")
            dev.XMusiceval_InterruptGlobalDisable(self.ip)
            dev.XMusiceval_DisableAutoRestart(self.ip)
            dev.XMusiceval_Set_Rxx_cmpx(self.ip, self.tx.phys_addr)
            dev.XMusiceval_Set_Pspec(self.ip, self.rx.phys_addr)
            print(f"Input 0x{self.tx.phys_addr:08x}, output 0x{self.rx.phys_addr:08x}")
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        if self.initialized:
            self.dev.XMusiceval_Release(self.ip)
            self.initialized = False
        self.dev.membuf_release(self.rx)
        self.dev.membuf_release(self.tx)

    def run_once(self, k, sentinel):
        dev = self.dev
        if dev.fill_output(self.rx, sentinel):
            raise RuntimeError("Cannot initialize the output buffer")
        if not dev.XMusiceval_IsReady(self.ip):
            raise RuntimeError(f"IP is not ready (ap_ctrl=0x{dev.ap_ctrl(self.ip):08x})")
        dev.XMusiceval_Set_num_sources(self.ip, k)
        dev.XMusiceval_Start(self.ip)
        for _ in range(1_000_000):
            if dev.XMusiceval_IsDone(self.ip):
                break
        else:
            raise TimeoutError(f"Timeout (ap_ctrl=0x{dev.ap_ctrl(self.ip):08x}); "
                               "reset the accelerator before reusing DMA buffers")
        out = np.empty(THLEN, dtype=np.float32)
        if dev.rd_output(self.rx, out):
            raise RuntimeError("Cannot read spectrum buffer")
        return out

    def run(self, r, k):
        """Two transactions, as the HLS testbench; returns the output and its repeatability."""
        words = np.ascontiguousarray(r, dtype=np.complex64).view(np.float32).ravel()
        if words.size != IN_WORDS or self.dev.wr_input(self.tx, words):
            raise ValueError("Expected an 8x8 complex64 covariance")
        first, second = (self.run_once(k, s) for s in SENTINELS)
        return second, first.view(np.uint32).tobytes() == second.view(np.uint32).tobytes()


def save_run(output_dir, cases, matrices, outputs, origin):
    """Store a run in the HLS fixture layout so --replay can re-check and plot it on a host."""
    np.asarray(matrices, dtype='<c8').tofile(output_dir/FILES['input'])
    np.array([c['k'] for c in cases], dtype='<u4').tofile(output_dir/FILES['counts'])
    np.asarray(outputs, dtype='<f4').tofile(output_dir/FILES['output'])
    (output_dir/'cases.json').write_text(json.dumps(cases, indent=2))
    (output_dir/'origin.txt').write_text(origin+'\n')


def load_run(directory):
    """Read HLS C-simulation fixtures (build/eval/hls/io) or a saved hardware run."""
    cases = json.loads((directory/'cases.json').read_text())
    n = len(cases)
    matrices = np.fromfile(directory/FILES['input'], dtype='<c8')
    outputs = np.fromfile(directory/FILES['output'], dtype='<f4')
    counts = np.fromfile(directory/FILES['counts'], dtype='<u4')
    if matrices.size != n*NLEN*NLEN or outputs.size != n*THLEN or counts.size != n:
        raise ValueError(f"{directory}: fixture/output size mismatch for {n} cases")
    if any(int(k) != c['k'] for k, c in zip(counts, cases)):
        raise ValueError(f"{directory}: num_sources does not match cases.json")
    origin = (directory/'origin.txt').read_text().strip() if (directory/'origin.txt').is_file() \
        else "HLS C-simulation fixtures"
    return cases, matrices.reshape(n, NLEN, NLEN), outputs.reshape(n, THLEN), origin


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-sources", type=int, choices=range(1, MAX_SOURCES+1),
                        help="Run only valid scenarios with this K")
    parser.add_argument("--reference-only", action="store_true", help="Check NumPy references without hardware")
    parser.add_argument("--replay", type=Path, metavar="DIR", help="Validate and plot saved outputs offline")
    parser.add_argument("--output-dir", type=Path, default=Path(f"{SLUG}_results"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.replay and args.num_sources is not None:
        parser.error("--num-sources selects hardware scenarios; omit it with --replay")
    if args.replay:
        cases, matrices, outputs, origin = load_run(args.replay)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        origin = f"OFFLINE REPLAY of {origin} from {args.replay} (no FPGA execution)"
        return 0 if report(cases, matrices, outputs, origin, args.output_dir, not args.no_plots) else 1
    definitions = scenarios(args.num_sources)
    if args.reference_only:
        for case, r in definitions:
            if case['kind'] == 'signal' and not case['sampled']:
                ref = reference(r.astype(complex), case['k'])
                peaks = THETA_SCAN_DEG[peak_indices(1/np.maximum(ref, D_FLOOR), case['k'])]
                assert np.all(np.abs(peaks-case['truth']) <= GRID_STEP_DEG), case['name']
                print(f"{case['name']}: reference peaks {angles(peaks)}; reference check PASS")
        print(f"{len(definitions)} scenarios generated (reference only, no hardware)")
        return 0
    cases, matrices, outputs, repeatable = [], [], [], []
    with MUSICEvalIP() as ip:
        for case, r in definitions:
            out, same = ip.run(r, case['k'])
            cases.append(case); matrices.append(r); outputs.append(out); repeatable.append(same)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    origin = f"{IP_NAME} hardware run (UIO/u-dma-buf)"
    save_run(args.output_dir, cases, matrices, outputs, origin)
    return 0 if report(cases, matrices, outputs, origin, args.output_dir, not args.no_plots, repeatable) else 1


if __name__ == "__main__":
    raise SystemExit(main())
