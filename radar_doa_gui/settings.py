"""GUI settings, saved to / loaded from JSON."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path

from . import ROOT, REPO

SETTINGS_FILE = ROOT / "radar_doa_gui_settings.json"


@dataclass
class Settings:
    # radar (applied with Configure)
    range_res_m: float = 0.5
    max_range_m: float = 20.0
    display_range_m: float = 15.0
    mode: str = "TDM"
    chirps_per_frame: int = 128
    extended_band: bool = False
    tx_backoff_db: int = 6
    idle_us: float = 5.0
    single_frame_count: int = 1
    test_frame_count: int = 8            # finite run for the first hardware tests (0 = continuous)
    # processing (applied immediately)
    k_frames: int = 16
    num_sources: int = 2
    snapshot_mode: str = "frame"
    fb_averaging: bool = True
    diag_loading: float = 1e-3
    clutter_removal: bool = False
    tdm_phase_comp: bool = True
    flip_array: bool = False
    # Observed Doppler offset of the TX2 copy in DDM, in units of N/4. The chirp RAM always
    # steps TX2 by +90 deg per chirp; on the AWR2243BOOST this shows up at -N/4
    # (measured 2026-11-06, captures/ddm_20261106_181939.npy).
    ddm_tx2_sign: int = -1
    dc_guard_bins: int = 3
    calibration_path: str = str(REPO / "datasets" / "awr2243_calibration.npz")
    ip_latency_us: float = 500.0
    # sources
    radar_source: str = "sim"            # "sim" or "awr2243"
    board_target: str = "mock"           # "mock" or "board"
    # board
    board_host: str = "192.168.1.123"
    board_user: str = "root"
    board_dir: str = "/root/doa_app"
    udmabuf_ko: str = "/root/u-dma-buf.ko"
    force_fpga_reload: bool = False
    # mmWave Studio / DCA1000
    studio_dll: str = ("C:/ti/mmwave_studio_03_00_00_14/mmWaveStudio/Clients/"
                       "RtttNetClientController/RtttNetClientAPI.dll")
    studio_port: int = 2777
    bringup: str = "auto"                # auto | skip (already done in Studio, TDM only) | force
    host_ip: str = "192.168.33.30"
    dca_ip: str = "192.168.33.180"
    # camera
    camera_index: int = -1               # -1 = auto-detect
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    # simulation targets: [range_m, azimuth_deg, velocity_mps, snr_db]
    sim_targets: list = field(default_factory=lambda: [[6.0, -20.0, 0.0, 0.0], [6.0, 30.0, 0.0, -3.0]])

    def save(self, path: Path = SETTINGS_FILE):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = SETTINGS_FILE) -> "Settings":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
