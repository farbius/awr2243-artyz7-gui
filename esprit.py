"""ESPRITEval hardware validation: native UIO/SWIG driver, one covariance DMA buffer."""
import argparse
import importlib
import numpy as np

MAX_SOURCES = 2
IP_BASE = 0x40000000
INPUT_BYTES = 512
TOLERANCE = 0.05


def unitary_basis(n):
    q = np.zeros((n,n), complex)
    for i in range(n//2):
        q[i,i] = q[n-1-i,i] = 1/np.sqrt(2)
        q[i,n-n//2+i] = 1j/np.sqrt(2)
        q[n-1-i,n-n//2+i] = -1j/np.sqrt(2)
    if n%2: q[n//2,n//2] = 1
    return q


def reference(r, k):
    if not 1 <= k <= MAX_SOURCES: return np.zeros(MAX_SOURCES)
    fb = (r + r[::-1,::-1].conj())/2
    q8, q7 = unitary_basis(8), unitary_basis(7)
    _, v = np.linalg.eigh((q8.conj().T @ fb @ q8).real)
    h = q7.conj().T @ q8[1:,:]
    e1, e2 = h.real @ v[:,-k:], h.imag @ v[:,-k:]
    eta = np.linalg.eigvals(np.linalg.solve(e1.T @ e1, e1.T @ e2))
    if np.max(np.abs(eta.imag)) > 1e-5: raise ValueError("Complex spatial-frequency roots")
    angles = np.rad2deg(np.arcsin(np.clip(2*np.arctan(eta.real)/np.pi,-1,1)))
    return np.pad(np.sort(angles), (0,MAX_SOURCES-k))


def covariance(angles):
    a = np.exp(1j*np.pi*np.arange(8)[:,None]*np.sin(np.deg2rad(angles)))
    r = a @ a.conj().T + .001*np.eye(8)
    return np.ascontiguousarray(r / np.trace(r).real * 8, dtype=np.complex64)


def scenarios():
    for k in range(1,MAX_SOURCES+1):
        angles = [23.] if k == 1 else np.rad2deg(np.arcsin(np.linspace(-.75,.75,k))).tolist()
        yield f"K={k} analytic", k, covariance(angles), angles, [0,0,0,0]
    for k in (0,MAX_SOURCES+1):
        yield f"invalid K={k}", k, np.eye(8,dtype=np.complex64), [], [0,0,0,0]


class ESPRITEvalIP:
    def __enter__(self):
        self.dev = importlib.import_module("esprit_eval")
        dev = self.dev
        self.buffer, self.ip = dev.Membuf(), dev.XEspriteval()
        self.initialized = False
        try:
            if dev.udma_allocate(self.buffer, "udmabuf0"): raise RuntimeError("Cannot map udmabuf0")
            if self.buffer.buf_size < INPUT_BYTES or self.buffer.phys_addr + INPUT_BYTES > 0x20000000:
                raise ValueError("Covariance buffer too small or outside mapped DDR")
            if dev.XEspriteval_Initialize(self.ip, "ESPRITEval") != dev.XST_SUCCESS:
                raise RuntimeError("Cannot initialize ESPRITEval UIO device")
            self.initialized = True
            if self.ip.Control_BaseAddress in (0,0xffffffff): raise RuntimeError("UIO mmap failed")
            dev.XEspriteval_InterruptGlobalDisable(self.ip)
            dev.XEspriteval_DisableAutoRestart(self.ip)
            dev.XEspriteval_Set_Rxx_cmpx(self.ip, self.buffer.phys_addr)
            return self
        except BaseException:
            self.__exit__(None,None,None)
            raise

    def __exit__(self, *_):
        if self.initialized:
            self.dev.XEspriteval_Release(self.ip)
            self.initialized = False
        self.dev.membuf_release(self.buffer)

    def run(self, r, k):
        dev = self.dev
        if not dev.XEspriteval_IsReady(self.ip): raise RuntimeError("IP is not ready")
        words = np.ascontiguousarray(r, dtype=np.complex64).view(np.float32).ravel()
        if words.size != 128 or dev.wr_input(self.buffer, words): raise ValueError("Expected 8x8 covariance")
        dev.XEspriteval_Set_num_sources(self.ip, k)
        for i in range(MAX_SOURCES): getattr(dev, f"XEspriteval_Get_angle_{i}_vld")(self.ip)
        dev.XEspriteval_Start(self.ip)
        for _ in range(1_000_000):
            if dev.XEspriteval_IsDone(self.ip): break
        else: raise TimeoutError("Reset accelerator before reusing covariance buffer")
        for i in range(MAX_SOURCES):
            if not getattr(dev, f"XEspriteval_Get_angle_{i}_vld")(self.ip): raise RuntimeError("Missing angle validity")
        raw = np.array([getattr(dev, f"XEspriteval_Get_angle_{i}")(self.ip) for i in range(MAX_SOURCES)],dtype=np.uint32)
        angles = raw.view(np.float32).astype(np.float64)
        status = None
        return angles, status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-only', action='store_true', help='Check NumPy references without opening hardware')
    args = parser.parse_args()
    cases = list(scenarios())
    for name,k,r,truth,_ in cases:
        if truth: np.testing.assert_allclose(reference(r,k)[:k],truth,atol=.01)
    if args.reference_only:
        for name,k,r,truth,_ in cases:
            ref = reference(r,k) if truth else np.zeros(MAX_SOURCES)
            print(f"{name}: K={k}, Python reference={ref}; reference check PASS")
        return 0
    passed = True
    print("ESPRITEval validation, UIO base", hex(IP_BASE))
    with ESPRITEvalIP() as ip:
        for name,k,r,truth,expected_status in cases:
            ref = reference(r,k) if truth else np.zeros(MAX_SOURCES)
            actual,status = ip.run(r,k)
            error = np.abs(actual-ref)
            inactive = actual[k:] if 1 <= k <= MAX_SOURCES else actual
            ok = bool(np.all(np.isfinite(actual)) and np.all(error <= TOLERANCE) and np.all(inactive == 0))
            print(f"Scenario: {name}  K: {k}")
            for i in range(MAX_SOURCES):
                print(f"  angle_{i}: reference={ref[i]: .6f} deg hardware={actual[i]: .6f} deg error={error[i]:.6f} deg")
            if status is not None:
                flags,convert,fb,unitary = status
                print(f"  Status: flags=0x{flags:08x} [saturated={bool(flags&1)} nonfinite={bool(flags&2)}]")
                print(f"    covariance_float_to_fixed saturated={convert}\n    fb_average saturated={fb}\n    unitary saturated={unitary}")
                ok = ok and status == expected_status
            print("RESULT:", "PASS" if ok else "FAIL", "\n")
            passed = ok and passed
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
