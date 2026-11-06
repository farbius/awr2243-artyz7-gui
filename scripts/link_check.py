"""Connectivity check without RF activity: mmWave Studio RSTD (Lua no-op) and DCA1000 command port."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radar_doa_gui import radar as R  # noqa: E402
from radar_doa_gui.settings import Settings  # noqa: E402

s = Settings()
try:
    studio = R.Studio(s.studio_dll, s.studio_port)
    print("Studio:", studio.run("return 'Lua ok, bring-up flag: ' .. tostring(DOA_GUI_BRINGUP) .. ', repo flag: ' .. tostring(PYTHON_RADAR_INITIALIZED)", timeout=10))
    studio.close()
except Exception as e:  # noqa: BLE001
    print("Studio: FAILED:", e)
try:
    dca = R.Dca1000(s.host_ip, s.dca_ip)
    try:
        print("DCA1000 system connect status:", dca.command(R.CMD_SYSTEM_CONNECT))
        dca.cfg.settimeout(1.0)
        dca.cfg.sendto(R.dca_command(R.CMD_READ_FPGA_VERSION), dca.dca)
        msg, _ = dca.cfg.recvfrom(2048)
        code, status = R.parse_dca_response(msg)
        print(f"DCA1000 FPGA version word: 0x{status:04x} (major {status & 0x7F}, minor {(status >> 7) & 0x7F}, "
              f"{'playback' if status & 0x4000 else 'record'} mode)")
    finally:
        dca.close()
except Exception as e:  # noqa: BLE001
    print("DCA1000: FAILED:", e)
