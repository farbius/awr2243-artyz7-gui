"""Mock board: Bartlett / MUSIC / Capon spectra and ESPRIT angles in numpy.

Uses the family reference models from the deploy folder unchanged (bartlett.py, music.py,
capon.py, esprit.py), i.e. the same oracles the hardware was validated against. Output
types match the board: float32 spectra on linspace(-90, 90, 512), MUSIC as the
pseudo-spectrum 1 / (null spectrum), ESPRIT angles sorted ascending, zero-padded.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from . import ROOT
from .protocol import CORES, THLEN

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import bartlett as _bartlett  # noqa: E402
import capon as _capon        # noqa: E402
import esprit as _esprit      # noqa: E402
import music as _music        # noqa: E402

_A_BARTLETT = _bartlett.steering_matrix(_bartlett.NLEN, _bartlett.D_SPACING, _bartlett.LAMBDA)
_A_CAPON = _capon.steering_matrix()
MAX_SOURCES = 2


def compute(r, k):
    """Return (spectra dict, esprit angles [2], latency_ns dict, errors list)."""
    r = np.asarray(r, dtype=np.complex64)
    spectra, lat, errors = {}, {}, []
    t = time.perf_counter_ns()
    spectra["bartlett"] = np.real(np.einsum("ti,ij,tj->t", _A_BARTLETT.conj(), r.astype(complex), _A_BARTLETT)).astype(np.float32)
    lat["bartlett"] = time.perf_counter_ns() - t
    t = time.perf_counter_ns()
    if 1 <= k <= MAX_SOURCES:
        spectra["music"] = (1 / np.maximum(_music.reference(r.astype(complex), k), _music.D_FLOOR)).astype(np.float32)
    else:
        spectra["music"] = np.zeros(THLEN, np.float32)
    lat["music"] = time.perf_counter_ns() - t
    t = time.perf_counter_ns()
    try:
        esp = np.asarray(_esprit.reference(r.astype(complex), k), dtype=np.float32)
    except (ValueError, np.linalg.LinAlgError) as e:
        esp = np.full(MAX_SOURCES, np.nan, np.float32)
        errors.append(f"ESPRIT: {e}")
    lat["esprit"] = time.perf_counter_ns() - t
    t = time.perf_counter_ns()
    try:
        spectra["capon"] = _capon.capon(r, _A_CAPON).astype(np.float32)
    except (ValueError, np.linalg.LinAlgError) as e:
        spectra["capon"] = np.zeros(THLEN, np.float32)
        errors.append(f"Capon: {e}")
    lat["capon"] = time.perf_counter_ns() - t
    return spectra, esp, {c: lat[c] for c in CORES}, errors


class MockBoard:
    """Same interface as board_link.BoardLink."""
    name = "mock board"

    def __init__(self):
        self.connected = True
        self.info = dict(server="mock", cores=["Bartlett", "MUSIC", "ESPRIT", "Capon"], k_fixed=None)

    def run(self, cov, k, frame_id=0):
        t0 = time.perf_counter_ns()
        spectra, esp, lat, errors = compute(cov, k)
        t1 = time.perf_counter_ns()
        return dict(kind=1, status=0, frame_id=frame_id, k=k, spectra=spectra, esprit_deg=list(map(float, esp)),
                    latency_ns=lat, unwritten_cores=[], errors=errors, rtt_s=(t1 - t0) / 1e9,
                    board_s=(t1 - t0) / 1e9, link_s=max(0.0, (t1 - t0 - sum(lat.values())) / 1e9))

    def close(self):
        self.connected = False
