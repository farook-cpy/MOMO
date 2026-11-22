"""Shared pytest fixtures for ZIA."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import ZiaConfig
from memory.database import ZiaDatabase


@pytest.fixture
def tmp_config(tmp_path: Path) -> ZiaConfig:
    cfg = ZiaConfig(
        root_dir=ROOT,
        db_path=tmp_path / "zia_test.db",
        backup_dir=tmp_path / "backups",
    )
    cfg.system.log_dir = tmp_path / "logs"
    cfg.system.pid_file = tmp_path / "zia.pid"
    cfg.llm.retry_base_delay_seconds = 0.01
    return cfg


@pytest.fixture
def test_db(tmp_config: ZiaConfig) -> ZiaDatabase:
    db = ZiaDatabase(tmp_config.db_path)
    db.connect()
    yield db
    db.close()
