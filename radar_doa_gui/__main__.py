"""Entry point: python -m radar_doa_gui [--sim] [--settings FILE] [--snapshot PNG]"""
import argparse
import logging
import sys
from pathlib import Path

from . import ROOT
from .settings import SETTINGS_FILE, Settings


def setup_logging():
    logdir = ROOT / "logs"
    logdir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(logdir / "radar_doa_gui.log", encoding="utf-8"),
                  logging.StreamHandler(sys.stderr)])
    logging.getLogger("paramiko").setLevel(logging.WARNING)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", type=Path, default=SETTINGS_FILE)
    ap.add_argument("--sim", action="store_true", help="simulated radar + mock board (no hardware)")
    ap.add_argument("--snapshot", help="configure, stream ~5 s, save a PNG of the window and exit "
                                       "(camera stays off: frames are never saved)")
    ap.add_argument("--confirm-transmit", action="store_true",
                    help="skip the radar transmit dialog (only for runs approved in advance)")
    args = ap.parse_args(argv)
    setup_logging()
    settings = Settings.load(args.settings)
    if args.sim:
        settings.radar_source, settings.board_target = "sim", "mock"
    from PySide6 import QtWidgets
    from .gui import MainWindow
    app = QtWidgets.QApplication(sys.argv[:1])
    w = MainWindow(settings, snapshot=args.snapshot, transmit_confirmed=args.confirm_transmit)
    w.resize(1600, 1000)
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
