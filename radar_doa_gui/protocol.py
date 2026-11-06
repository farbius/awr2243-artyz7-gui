"""Binary protocol between the GUI and the board server (deployed as doa_protocol.py).

Standard library + numpy only (runs on the board with Python 3.11 / numpy 1.25).
All integers little-endian. One request -> one response over a byte stream
(SSH channel stdin/stdout).

Request (24 B header, + 512 B covariance for RUN):
    magic b"DOAQ", version u8, kind u8, reserved u16, frame_id u32,
    k u8 (number of sources for MUSIC/ESPRIT), flags u8, reserved u16, t_host_ns u64
    RUN payload: 8x8 complex64, row-major, re/im interleaved (128 float32),
                 R = E[x x^H] with steering a[n] = exp(+j pi n sin(theta)), n = 0..7

Response (68 B header + payload_len bytes):
    magic b"DOAR", version u8, kind u8, status u16, frame_id u32, k u8,
    unwritten u8 (bit0 Bartlett, bit1 MUSIC, bit2 Capon: sentinel words left),
    reserved u16, t_host_ns u64 (echo), t_recv_ns u64, t_done_ns u64 (board clock),
    latency_ns u32 x 4 (Bartlett, MUSIC, ESPRIT, Capon; PS-measured start->done),
    esprit_deg f32 x 2, payload_len u32
    RESULT payload: 3 x 512 float32 spectra (Bartlett, MUSIC, Capon) on linspace(-90, 90, 512)
    HELLO / ERROR payload: UTF-8 text (HELLO: JSON with server info)
"""
import struct

import numpy as np

VERSION = 1
REQ_MAGIC, RESP_MAGIC = b"DOAQ", b"DOAR"
REQ = struct.Struct("<4sBBHIBBHQ")
RESP = struct.Struct("<4sBBHIBBHQQQ4I2fI")
RUN, PING, QUIT = 1, 2, 3                       # request kinds
HELLO, RESULT, PONG, ERROR = 0, 1, 2, 255       # response kinds
NLEN, THLEN = 8, 512
COV_BYTES = NLEN * NLEN * 8
SPECTRA = ("bartlett", "music", "capon")
CORES = ("bartlett", "music", "esprit", "capon")
UNWRITTEN_BITS = dict(bartlett=1, music=2, capon=4)
THETA_DEG = np.linspace(-90.0, 90.0, THLEN)


class ProtocolError(RuntimeError):
    pass


def encode_request(kind, frame_id=0, k=0, t_host_ns=0, cov=None, flags=0):
    head = REQ.pack(REQ_MAGIC, VERSION, kind, 0, frame_id & 0xFFFFFFFF, k, flags, 0, t_host_ns)
    if kind != RUN:
        return head
    r = np.ascontiguousarray(cov, dtype=np.complex64)
    if r.shape != (NLEN, NLEN):
        raise ValueError("covariance must be 8x8")
    return head + r.tobytes()


def decode_request_header(buf):
    magic, ver, kind, _, frame_id, k, flags, _, t_host = REQ.unpack(buf)
    if magic != REQ_MAGIC or ver != VERSION:
        raise ProtocolError(f"bad request header {magic!r} v{ver}")
    return dict(kind=kind, frame_id=frame_id, k=k, flags=flags, t_host_ns=t_host)


def decode_covariance(buf):
    if len(buf) != COV_BYTES:
        raise ProtocolError(f"expected {COV_BYTES} covariance bytes, got {len(buf)}")
    return np.frombuffer(buf, dtype=np.complex64).reshape(NLEN, NLEN).copy()


def encode_response(kind, frame_id=0, status=0, k=0, unwritten=0, t_host_ns=0, t_recv_ns=0,
                    t_done_ns=0, latency_ns=(0, 0, 0, 0), esprit_deg=(0.0, 0.0), payload=b""):
    lat = [min(max(int(v), 0), 0xFFFFFFFF) for v in latency_ns]
    head = RESP.pack(RESP_MAGIC, VERSION, kind, status, frame_id & 0xFFFFFFFF, k, unwritten, 0,
                     t_host_ns, t_recv_ns, t_done_ns, *lat, *[float(a) for a in esprit_deg],
                     len(payload))
    return head + payload


def encode_result(frame_id, k, spectra, esprit_deg, latency_ns, unwritten=0, t_host_ns=0,
                  t_recv_ns=0, t_done_ns=0):
    payload = b"".join(np.asarray(spectra[name], dtype=np.float32).reshape(THLEN).tobytes()
                       for name in SPECTRA)
    return encode_response(RESULT, frame_id, 0, k, unwritten, t_host_ns, t_recv_ns, t_done_ns,
                           latency_ns, esprit_deg, payload)


def decode_response_header(buf):
    (magic, ver, kind, status, frame_id, k, unwritten, _, t_host, t_recv, t_done,
     l0, l1, l2, l3, e0, e1, plen) = RESP.unpack(buf)
    if magic != RESP_MAGIC or ver != VERSION:
        raise ProtocolError(f"bad response header {magic!r} v{ver}")
    return dict(kind=kind, status=status, frame_id=frame_id, k=k, unwritten=unwritten,
                t_host_ns=t_host, t_recv_ns=t_recv, t_done_ns=t_done,
                latency_ns=dict(zip(CORES, (l0, l1, l2, l3))), esprit_deg=[e0, e1],
                payload_len=plen)


def decode_payload(head, payload):
    """Attach the payload to a decoded header: spectra for RESULT, text otherwise."""
    if head["kind"] == RESULT:
        if len(payload) != 3 * THLEN * 4:
            raise ProtocolError(f"RESULT payload {len(payload)} B, expected {3 * THLEN * 4}")
        arr = np.frombuffer(payload, dtype=np.float32).reshape(3, THLEN)
        head["spectra"] = {name: arr[i].copy() for i, name in enumerate(SPECTRA)}
        head["unwritten_cores"] = [n for n, bit in UNWRITTEN_BITS.items() if head["unwritten"] & bit]
    else:
        head["text"] = payload.decode("utf-8", errors="replace")
    return head


def read_exact(read, n):
    """Read exactly n bytes with read(size) -> bytes; b'' at EOF raises EOFError."""
    chunks, got = [], 0
    while got < n:
        b = read(n - got)
        if not b:
            raise EOFError(f"stream closed after {got} of {n} bytes")
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


def read_response(read):
    head = decode_response_header(read_exact(read, RESP.size))
    payload = read_exact(read, head["payload_len"]) if head["payload_len"] else b""
    return decode_payload(head, payload)
