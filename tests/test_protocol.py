import io

import numpy as np
import pytest

from radar_doa_gui import protocol as P


def test_request_roundtrip():
    r = (np.arange(64) + 1j * np.arange(64)[::-1]).reshape(8, 8).astype(np.complex64)
    msg = P.encode_request(P.RUN, frame_id=0x12345678, k=2, t_host_ns=123456789012, cov=r)
    assert len(msg) == P.REQ.size + P.COV_BYTES == 24 + 512
    head = P.decode_request_header(msg[:P.REQ.size])
    assert head == dict(kind=P.RUN, frame_id=0x12345678, k=2, flags=0, t_host_ns=123456789012)
    np.testing.assert_array_equal(P.decode_covariance(msg[P.REQ.size:]), r)
    # layout: row-major, re/im interleaved float32 (what DoAEvalSystem.write_covariance expects)
    words = np.frombuffer(msg[P.REQ.size:], np.float32)
    assert words[2] == r[0, 1].real and words[3] == r[0, 1].imag and words[16] == r[1, 0].real


def test_ping_has_no_payload_and_bad_magic_is_rejected():
    assert len(P.encode_request(P.PING, 7)) == P.REQ.size
    with pytest.raises(P.ProtocolError):
        P.decode_request_header(b"XXXX" + P.encode_request(P.PING)[4:])


def test_result_roundtrip():
    spectra = {n: np.random.default_rng(i).random(512).astype(np.float32) for i, n in enumerate(P.SPECTRA)}
    msg = P.encode_result(9, 2, spectra, (-20.5, 30.25), (450_000, 350_000, 325_000, 460_000),
                          unwritten=P.UNWRITTEN_BITS["capon"], t_host_ns=5, t_recv_ns=6, t_done_ns=7)
    assert len(msg) == P.RESP.size + 3 * 512 * 4
    out = P.read_response(io.BytesIO(msg).read)
    assert out["kind"] == P.RESULT and out["frame_id"] == 9 and out["k"] == 2
    assert out["esprit_deg"] == [-20.5, 30.25]
    assert out["latency_ns"] == dict(bartlett=450_000, music=350_000, esprit=325_000, capon=460_000)
    assert out["unwritten_cores"] == ["capon"]
    assert (out["t_host_ns"], out["t_recv_ns"], out["t_done_ns"]) == (5, 6, 7)
    for n in P.SPECTRA:
        np.testing.assert_array_equal(out["spectra"][n], spectra[n])


def test_error_and_hello_carry_text():
    msg = P.encode_response(P.ERROR, 3, status=1, payload="boom: ü".encode())
    out = P.read_response(io.BytesIO(msg).read)
    assert out["kind"] == P.ERROR and out["status"] == 1 and out["text"] == "boom: ü"


def test_truncated_stream_raises_eof():
    msg = P.encode_response(P.PONG, 1)
    with pytest.raises(EOFError):
        P.read_response(io.BytesIO(msg[:-3]).read)
