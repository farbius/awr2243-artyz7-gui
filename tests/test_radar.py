import numpy as np
import pytest

from radar_doa_gui import radar as R
from radar_doa_gui.params import RadarRequest, derive


def packets(stream, size=1456, start_seq=1):
    """Split a byte stream into DCA1000 packets: u32 seq, u48 byte count, payload."""
    out = []
    for i, off in enumerate(range(0, len(stream), size)):
        chunk = stream[off:off + size]
        out.append((start_seq + i).to_bytes(4, "little") + off.to_bytes(6, "little") + chunk)
    return out


def test_dca_command_encoding_matches_openradar():
    assert R.dca_command(R.CMD_RECORD_START).hex() == "5aa50500" + "0000" + "aaee"
    assert R.parse_dca_response(bytes.fromhex("5aa505000000aaee")) == (0x05, 0)


def test_frame_assembler_in_order_frames_straddling_packets():
    fb = 5000                                     # not a multiple of the packet size
    frames = [bytes([i]) * fb for i in range(4)]
    asm = R.FrameAssembler(fb)
    got = []
    for pkt in packets(b"".join(frames)):
        got += asm.feed(*R.parse_data_packet(pkt))
    assert [i for i, _ in got] == [0, 1, 2, 3]
    assert all(data == frames[i] for i, data in got)
    assert asm.dropped == 0 and asm.lost_packets == 0


def test_frame_assembler_drops_frame_with_lost_packet():
    fb = 4 * 1456
    frames = [bytes([i]) * fb for i in range(3)]
    pk = packets(b"".join(frames))
    del pk[5]                                     # second frame loses a packet
    asm = R.FrameAssembler(fb)
    got = []
    for p in pk:
        got += asm.feed(*R.parse_data_packet(p))
    assert [i for i, _ in got] == [0, 2]
    assert asm.dropped == 1 and asm.lost_packets == 1


def test_frame_assembler_reordered_packets_within_frame():
    fb = 3 * 1456
    pk = packets(bytes(range(256)) * 17 + b"x" * (fb - 256 * 17))
    pk[0], pk[1] = pk[1], pk[0]
    asm = R.FrameAssembler(fb)
    got = []
    for p in pk:
        got += asm.feed(*R.parse_data_packet(p))
    assert len(got) == 1 and got[0][1] == bytes(range(256)) * 17 + b"x" * (fb - 256 * 17)


@pytest.mark.parametrize("mode", ["TDM", "DDM"])
def test_lua_configure(mode):
    cfg = derive(RadarRequest(mode=mode, tx_backoff_db=6))
    lua = R.lua_configure(cfg)
    assert "C.bringup(true)" in lua and "DOA_GUI_BRINGUP" in lua
    skip = R.lua_configure(cfg, bringup="skip")
    assert "C.bringup" not in skip and "local PCP" in skip
    assert "or true then" in R.lua_configure(cfg, bringup="force")
    with pytest.raises(ValueError):
        R.lua_configure(cfg, bringup="maybe")
    assert "backoff_db = 6" in lua and f"samples = {cfg.num_samples}" in lua
    assert f"fs_ksps = {int(round(cfg.fs_hz / 1e3))}" in lua
    assert "C.MAX_FREQ_GHZ = 77.0" in lua
    if mode == "TDM":
        assert "C.chirp(0, 0, 1, 0, 0, 0, 0)" in lua and "C.chirp(1, 0, 0, 1, 0, 0, 0)" in lua
        assert "C.chirp(0, 0, 1, 0, 0) " in lua            # variant without phase commands
        assert "num_chirps = 2" in lua and f"loops = {cfg.loops}" in lua
    else:
        assert "assert(PCP" in lua
        for k, code in enumerate((0, 16, 32, 48)):
            assert f"C.chirp({k}, 0, 1, 1, 0, 0, {code})" in lua
        assert "num_chirps = 4" in lua and f"loops = {cfg.loops}" in lua
    neg = R.lua_configure(cfg, ddm_tx2_sign=-1)
    if mode == "DDM":
        assert "C.chirp(1, 0, 1, 1, 0, 0, 48)" in neg


def test_lua_start_continuous_and_finite():
    assert "advanced_frame(DOA_SUBFRAMES, 0, 10.0)" in R.lua_start(0, 0.010)
    assert "advanced_frame(DOA_SUBFRAMES, 8, 25.0)" in R.lua_start(8, 0.025)
    assert "StartFrame" in R.lua_start(1, 0.01)


def test_lua_string_keeps_windows_paths():
    assert R.lua_string(r"C:\a\b.lua") == r'"C:\\a\\b.lua"'
