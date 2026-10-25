"""Shared setup: the app folder on the import path and no chat files written to disk."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("NASHMI_PERSIST_HISTORY", "0")
