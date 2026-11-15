"""Session state: a cache key, an explicit re-run gate and an error trap (D16)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from src.config.loader import effective_config_dict, load_config
from src.errors import PrototypeError
from src.pipeline import run


def configuration_digest(cfg) -> str:
    payload = {
        "config": effective_config_dict(cfg),
        "files": {},
    }
    for name in cfg.data.files.__dict__.values():
        path = cfg.data.raw_dir / name
        payload["files"][name] = {
            "present": path.exists(),
            "size": path.stat().st_size if path.exists() else 0,
        }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def bundle_key(cfg) -> str:
    return f"{configuration_digest(cfg)}-seed{cfg.run.seed}"


@dataclass
class BundleCache:
    key: str | None = None
    bundle: object | None = None
    error: str | None = None
    calls: int = 0

    def ensure(self, cfg, force: bool = False, runner=None):
        runner = runner or (lambda config: run(config, verbose=False))
        key = bundle_key(cfg)
        if force or self.bundle is None or key != self.key:
            self.calls += 1
            try:
                self.bundle = runner(cfg)
                self.error = None
            except PrototypeError as exc:
                self.error = exc.render()
            self.key = key
        return self.bundle

    def load(self, cfg_path: str = "config/settings.yaml", root: str = "."):
        cfg = load_config(cfg_path, root=root)
        return cfg, self.ensure(cfg)


CACHE = BundleCache()


def error_panel_text(error: str) -> str:
    return (f"{error}\n\nNext step: python -m codes.sample_data "
            f"(then python -m codes.run_pipeline)")