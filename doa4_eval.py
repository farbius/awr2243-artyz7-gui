"""Combined DoA Eval application: BartlettEval, MUSICEval, ESPRITEval and CaponEval in one PL design.

All four cores share one u-dma-buf input (udmabuf0, 8x8 complex64 covariance) and one
u-dma-buf output (udmabuf1, spectrum). The covariance is written once per scenario and
the cores run cascaded on it: Bartlett -> MUSIC -> ESPRIT -> Capon. Before each spectrum
core runs, the output buffer is pre-filled with a sentinel so unwritten words are caught.

Hardware run (target):  python3 doa4_eval.py [--cores bartlett,music,esprit,capon]
                                             [--output-dir DIR] [--no-plots]
Reference check only:   python3 doa4_eval.py --reference-only

The SWIG modules (bartlett_eval, music_eval, esprit_eval, capon_eval) and the family
applications (bartlett.py, music.py, esprit.py, capon.py) are imported unchanged; they
must be on sys.path (this script's directory, the build tree, or --module-dir).
The scenarios are the 21 MUSIC testbench cases; each core is checked with its own
application's reference model and tolerance.
"""
import argparse
import importlib
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BUILD_SW = HERE.parent / "build" / "sw"

# alg -> (UIO name = linux,uio-name in doa4_eval.dtsi, SWIG module, driver prefix, output dtype)
CORES = dict(
    bartlett=("BartlettEval", "bartlett_eval", "XBartletteval", np.float64),
    music=("MUSICEval", "music_eval", "XMusiceval", np.float32),
    esprit=("ESPRITEval", "esprit_eval", "XEspriteval", None),      # angles in registers
    capon=("CaponEval", "capon_eval", "XCaponeval", np.float64),
)
NLEN, THLEN, IN_WORDS = 8, 512, 128
IN_BYTES = IN_WORDS * 4
OUT_BYTES = THLEN * 8          # largest output: Bartlett/Capon, 512 float64
DDR_LIMIT = 0x20000000         # 512 MiB HP DDR window of every core
MAX_POLL = 1_000_000
SENTINEL = 0xDEADBEEF          # 32-bit fill pattern of the output buffer


def load_modules(extra_dirs, algs):
    """Import the family applications (reference models) as-is."""
    for d in [*extra_dirs, HERE, *sorted(BUILD_SW.glob("*/python/eval"))]:
        if Path(d).is_dir() and str(d) not in sys.path:
            sys.path.append(str(d))
    # music.py supplies the scenarios and the scan grid, so it is always loaded.
    return {alg: importlib.import_module(alg) for alg in dict.fromkeys(["music", *algs])}


# -----------------------------------------------------------------------------
# Hardware: one input and one output buffer shared by the cores
# -----------------------------------------------------------------------------

class DoAEvalSystem:
    """Map udmabuf0/udmabuf1 once and attach the selected UIO cores to them."""

    def __init__(self, algs):
        self.algs = list(algs)
        self.mods, self.ips, self.initialized = {}, {}, []
        self.tx = self.rx = None

    def call(self, alg, name, *args):
        return getattr(self.mods[alg], f"{CORES[alg][2]}_{name}")(self.ips[alg], *args)

    def __enter__(self):
        self.mods = {alg: importlib.import_module(CORES[alg][1]) for alg in self.algs}
        # music_eval provides the buffer helpers (wr_input, fill_output); the modules
        # share the SWIG runtime, so one Membuf type serves all of them.
        self.mem = importlib.import_module("music_eval")
        self.tx, self.rx = self.mem.Membuf(), self.mem.Membuf()
        self.ips = {alg: getattr(self.mods[alg], CORES[alg][2])() for alg in self.algs}
        try:
            for buffer, name, size in ((self.tx, "udmabuf0", IN_BYTES), (self.rx, "udmabuf1", OUT_BYTES)):
                if self.mem.udma_allocate(buffer, name):
                    raise RuntimeError(f"Cannot map {name}")
                if buffer.buf_size < size or buffer.phys_addr + size > DDR_LIMIT or buffer.phys_addr % 8:
                    raise ValueError(f"{name}: buffer too small, unaligned or outside the 512 MiB HP DDR window")
            for alg in self.algs:
                uio, mod = CORES[alg][0], self.mods[alg]
                if self.call(alg, "Initialize", uio) != mod.XST_SUCCESS:
                    raise RuntimeError(f"Cannot initialize {uio} UIO device")
                self.initialized.append(alg)
                if self.ips[alg].Control_BaseAddress in (0, 0xffffffff):
                    raise RuntimeError(f"{uio}: UIO mmap failed")
                self.call(alg, "InterruptGlobalDisable")
                self.call(alg, "DisableAutoRestart")
                self.call(alg, "Set_Rxx_cmpx", self.tx.phys_addr)
                if CORES[alg][3] is not None:
                    self.call(alg, "Set_Pspec", self.rx.phys_addr)
            print(f"Shared input 0x{self.tx.phys_addr:08x} ({self.tx.buf_size} B), "
                  f"shared output 0x{self.rx.phys_addr:08x} ({self.rx.buf_size} B); "
                  f"cores: {', '.join(CORES[a][0] for a in self.algs)}")
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        for alg in reversed(self.initialized):
            self.call(alg, "Release")
        self.initialized = []
        for buffer in (self.rx, self.tx):
            if buffer is not None:
                self.mem.membuf_release(buffer)
        self.tx = self.rx = None

    def _start_and_wait(self, alg):
        if not self.call(alg, "IsReady"):
            raise RuntimeError(f"{CORES[alg][0]} is not ready")
        self.call(alg, "Start")
        for _ in range(MAX_POLL):
            if self.call(alg, "IsDone"):
                return
        raise TimeoutError(f"{CORES[alg][0]} timeout; reset the accelerators before reusing DMA buffers")

    def write_covariance(self, r):
        words = np.ascontiguousarray(r, dtype=np.complex64).view(np.float32).ravel()
        if words.size != IN_WORDS or self.mem.wr_input(self.tx, words):
            raise ValueError("Expected an 8x8 complex64 covariance")

    def run_spectrum(self, alg, k=None):
        """Bartlett, MUSIC or Capon: sentinel-fill the shared output, run, read the spectrum."""
        dtype = CORES[alg][3]
        if self.mem.fill_output(self.rx, SENTINEL):
            raise RuntimeError("Cannot initialize the output buffer")
        if k is not None:
            self.call(alg, "Set_num_sources", k)
        self._start_and_wait(alg)
        out = np.empty(THLEN, dtype=dtype)
        if self.mods[alg].rd_output(self.rx, out):
            raise RuntimeError(f"Cannot read {CORES[alg][0]} spectrum")
        return out

    def run_esprit(self, k, max_sources):
        self.call("esprit", "Set_num_sources", k)
        for i in range(max_sources):        # clear-on-read validity flags
            self.call("esprit", f"Get_angle_{i}_vld")
        self._start_and_wait("esprit")
        if not all(self.call("esprit", f"Get_angle_{i}_vld") for i in range(max_sources)):
            raise RuntimeError("ESPRIT: missing angle validity")
        raw = np.array([self.call("esprit", f"Get_angle_{i}") for i in range(max_sources)], dtype=np.uint32)
        return raw.view(np.float32).astype(np.float64)

    def run(self, r, k, max_sources):
        """One scenario: write the shared input once, then cascade the selected cores."""
        self.write_covariance(r)
        outputs = {}
        for alg in self.algs:
            if alg == "esprit":
                outputs[alg] = self.run_esprit(k, max_sources)
            else:
                outputs[alg] = self.run_spectrum(alg, k if alg == "music" else None)
        return outputs


# -----------------------------------------------------------------------------
# Validation with the family reference models
# -----------------------------------------------------------------------------

def unwritten(out):
    """True if any 32-bit word of the spectrum still holds the sentinel."""
    return bool(np.any(np.ascontiguousarray(out).view(np.uint32) == SENTINEL))


def check_bartlett(app, case, r, out):
    """bartlett.py spectrum tolerance; peaks matched to the truth are informative."""
    expected = app.bartlett(np.asarray(r, dtype=np.complex64), app.steering_matrix(NLEN, app.D_SPACING, app.LAMBDA))
    scale = max(float(np.max(np.abs(expected))), 1.0)
    ok = bool(np.isfinite(out).all() and np.allclose(out, expected, rtol=2e-5, atol=2e-6 * scale))
    m = dict(passed=ok, ref=expected, max_rel_error=float(np.max(np.abs(out - expected)) / scale))
    if case['kind'] == 'signal':
        try:
            m['peaks'] = app.matched_peaks(out, app.THETA_SCAN_DEG[:, None], case['truth'])[:, 0].tolist()
        except ValueError:
            m['peaks'] = None       # beamwidth too wide to resolve the sources
    if not ok:
        m['reason'] = 'spectrum outside the bartlett.py tolerance'
    return m


def check_music(app, case, r, out):
    m = app.compare(case, r, out)
    if 'hw_deg' in m:
        m['peaks'] = m['hw_deg']
    return m


def check_esprit(app, case, r, out):
    """esprit.py tolerance on signal scenarios, all-zero on invalid K, others informative."""
    m = dict(peaks=out[:case['k']].tolist() if case['kind'] == 'signal' else out.tolist())
    if case['kind'] == 'invalid':
        m['passed'] = bool(np.all(out == 0))
        m['reason'] = '' if m['passed'] else 'expected all-zero angles'
    elif case['kind'] == 'signal':
        try:
            ref = app.reference(np.asarray(r, dtype=complex), case['k'])
        except ValueError as e:
            return dict(m, passed=False, reason=f'reference: {e}')
        error = np.abs(out - ref)
        m.update(ref=ref.tolist(), max_error=float(error.max()),
                 passed=bool(np.all(np.isfinite(out)) and np.all(error <= app.TOLERANCE)))
        m['reason'] = '' if m['passed'] else f'angle error above {app.TOLERANCE} deg'
    else:
        m.update(passed=None, reason='not scored (no ESPRIT reference for this scenario)')
    return m


def check_capon(app, case, r, out):
    """capon.py tolerance (normalized spectrum error, peaks within one bin) on every
    scenario with a positive-definite Hermitian covariance; others are informative.
    Capon has no K input, so invalid-K scenarios are scored as plain spectra."""
    r = np.asarray(r, dtype=np.complex64)
    try:
        expected = app.capon(r, app.steering_matrix())
    except (ValueError, np.linalg.LinAlgError) as e:
        return dict(passed=None, reason=f'not scored (no Capon reference: {e})')
    if case['kind'] == 'signal':
        try:
            m = app.compare(case, r, out)
            m['peaks'] = m['hls_peaks']
            if not m['passed']:
                m['reason'] = 'spectrum or peaks outside the capon.py tolerance'
            return dict(m, ref=expected)
        except ValueError:
            pass                    # sources not resolved as distinct peaks: spectrum only
    actual = np.asarray(out, dtype=float)
    scale = max(float(np.max(np.abs(expected))), 1.0)
    error = float(np.max(np.abs(actual / scale - expected / scale)))
    ok = bool(np.isfinite(actual).all() and np.all(actual > 0) and error <= app.MAX_NORMALIZED_ERROR)
    m = dict(passed=ok, ref=expected, normalized_max_abs_error=error)
    if case['kind'] == 'signal':
        m['peaks'] = None
    if not ok:
        m['reason'] = 'spectrum outside the capon.py tolerance'
    return m


CHECKS = dict(bartlett=check_bartlett, music=check_music, esprit=check_esprit, capon=check_capon)


def check(alg, app, case, r, out):
    m = CHECKS[alg](app, case, r, out)
    if CORES[alg][3] is not None and unwritten(out):
        m.update(passed=False, reason=f'output words not written by {CORES[alg][0]}')
    return m


def fmt_angles(values):
    return 'unresolved' if values is None else ', '.join(f'{v:+.2f}' for v in values) + ' deg'


def print_result(case, results):
    print(f"Scenario: {case['name']}  K={case['k']}"
          + (f"  truth: {fmt_angles(case['truth'])}" if case['truth'] else ''))
    for alg, m in results.items():
        state = 'n/a ' if m['passed'] is None else ('PASS' if m['passed'] else 'FAIL')
        detail = []
        if 'peaks' in m:
            detail.append(f"DoA {fmt_angles(m['peaks'])}")
        if 'max_rel_error' in m:
            detail.append(f"max |err|/peak {m['max_rel_error']:.2e}")
        if 'max_d_error' in m:
            detail.append(f"max |D err| {m['max_d_error']:.2e}")
        if 'max_error' in m:
            detail.append(f"max angle err {m['max_error']:.4f} deg")
        if 'normalized_max_abs_error' in m:
            detail.append(f"max normalized err {m['normalized_max_abs_error']:.2e}")
        if m['passed'] is not True and m.get('reason'):
            detail.append(m['reason'])
        print(f"  {CORES[alg][0]:<13}{state}  {'; '.join(detail)}")
    print()


INK, SURFACE = "#0b0b0b", "#fcfcfb"
COLORS = dict(bartlett="#2a78d6", music="#eb6834", esprit="#1f9d55", capon="#8a3ffc")
LABELS = dict(bartlett="Bartlett", music="MUSIC", esprit="ESPRIT", capon="Capon")


def plot_case(theta, case, results, outputs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained', facecolor=SURFACE)
    for alg, values in outputs.items():
        if alg == 'esprit':
            continue
        p = np.asarray(values, dtype=float)
        if np.all(np.isfinite(p)) and np.max(p) > 0:
            ax.plot(theta, 10 * np.log10(np.maximum(p / p.max(), 1e-6)), color=COLORS[alg], lw=1.6,
                    label=f"{LABELS[alg]} hardware")
    if case['kind'] == 'signal':
        for i, a in enumerate(outputs.get('esprit', [])[:case['k']]):
            ax.axvline(a, color=COLORS['esprit'], lw=1.6, label='ESPRIT hardware' if i == 0 else None)
        for i, t in enumerate(case['truth']):
            ax.axvline(t, color=INK, ls='--', lw=1, label='True DoA' if i == 0 else None)
    ax.set(xlim=(-90, 90), ylim=(-50, 3), xlabel='Angle (deg)', ylabel='Normalized spectrum (dB)')
    ax.grid(True, color='#e4e3df', lw=.8)
    ax.set_facecolor(SURFACE)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='lower left', fontsize=8)
    verdict = ', '.join(f"{CORES[a][0]} {'n/a' if m['passed'] is None else 'PASS' if m['passed'] else 'FAIL'}"
                        for a, m in results.items())
    ax.set_title(f"{case['name']} (K={case['k']})\n{verdict}", fontsize=10, color=INK)
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items() if not isinstance(v, np.ndarray) or v.size <= 8}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def parse_cores(text):
    algs = [a.strip().lower() for a in text.split(',') if a.strip()]
    unknown = [a for a in algs if a not in CORES]
    if unknown or not algs:
        raise argparse.ArgumentTypeError(f"cores must be a comma list of {', '.join(CORES)}")
    return [a for a in CORES if a in algs]          # always run in cascade order


def reference_only(apps, definitions):
    """Check that every family reference model runs on the scenarios; no hardware."""
    for case, r in definitions:
        if case['kind'] != 'signal' or case['sampled']:
            continue
        line = []
        if 'esprit' in apps:
            ref = apps['esprit'].reference(r.astype(complex), case['k'])
            assert np.allclose(ref[:case['k']], case['truth'], atol=0.05), case['name']
            line.append(f"ESPRIT {fmt_angles(ref[:case['k']])}")
        if 'capon' in apps:
            spectrum = apps['capon'].capon(r, apps['capon'].steering_matrix())
            assert np.isfinite(spectrum).all() and np.all(spectrum > 0), case['name']
            try:
                line.append(f"Capon {fmt_angles(apps['capon'].matched_peaks(spectrum, case['truth']))}")
            except ValueError:
                line.append("Capon peaks unresolved")
        print(f"{case['name']}: {'; '.join(line) or 'scenario generated'}; PASS")
    print(f"{len(definitions)} scenarios generated (reference only, no hardware)")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--module-dir", type=Path, action="append", default=[],
                        help="Directory containing the SWIG modules and family apps (repeatable)")
    parser.add_argument("--cores", type=parse_cores, default=list(CORES),
                        help="Comma list of cores to run (default: bartlett,music,esprit,capon)")
    parser.add_argument("--reference-only", action="store_true", help="Check the references without hardware")
    parser.add_argument("--output-dir", type=Path, default=Path("doa4_eval_results"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    apps = load_modules(args.module_dir, args.cores)
    max_sources = apps['esprit'].MAX_SOURCES if 'esprit' in args.cores else 0
    definitions = apps['music'].scenarios()     # MUSIC testbench scenarios: signal, stress, degenerate, invalid K

    if args.reference_only:
        return reference_only({a: apps[a] for a in args.cores}, definitions)

    names = ' -> '.join(CORES[a][0] for a in args.cores)
    records, failed = [], 0
    print("========================================")
    print(f"DoA Eval system: {names} (shared buffers)")
    print("========================================")
    with DoAEvalSystem(args.cores) as system:
        for case, r in definitions:
            outputs = system.run(r, case['k'], max_sources)
            results = {alg: check(alg, apps[alg], case, r, out) for alg, out in outputs.items()}
            failed += any(m['passed'] is False for m in results.values())
            print_result(case, results)
            records.append((case, outputs, results))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(
        [dict(case=c, outputs={a: o.tolist() for a, o in out.items()}, results=jsonable(res))
         for c, out, res in records], indent=1))
    if not args.no_plots:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            print("Matplotlib is not installed: PNG plots skipped.")
        else:
            for i, (c, out, res) in enumerate(records):
                plot_case(apps['music'].THETA_SCAN_DEG, c, res, out,
                          args.output_dir / f"doa4_eval_{i:02d}_{apps['music'].case_id(c)}.png")
            print(f"Wrote {len(records)} PNG plots to {args.output_dir}")
    print("========================================")
    print(f"Summary: {len(records) - failed}/{len(records)} scenarios passed on all scored cores")
    print("========================================")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
