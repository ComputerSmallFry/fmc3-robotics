from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EpisodeBuffer:
    observations: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)


class So101DatasetAdapter:
    def __init__(self) -> None:
        self._active: bool = False
        self._session: Optional[Dict[str, Any]] = None
        self._buffer: Optional[EpisodeBuffer] = None
        self._output_dir: Optional[Path] = None

    def start(self, session: Dict[str, Any]) -> Dict[str, Any]:
        output_dir = Path(session.get("output_dir", "datasets/"))
        output_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir = output_dir
        self._session = session
        self._buffer = EpisodeBuffer()
        self._active = True
        return {
            "recording": True,
            "session_id": session.get("name", "session"),
            "output_dir": str(output_dir),
        }

    def stop(self) -> Dict[str, Any]:
        if not self._active or self._buffer is None or self._output_dir is None:
            return {"recording": False, "dataset_path": ""}

        session_name = (self._session or {}).get("name", "session")
        dataset_path = self._output_dir / f"{session_name}_episode.npz"
        metadata_path = self._output_dir / f"{session_name}_meta.json"

        obs = self._buffer.observations
        act = self._buffer.actions
        np.savez_compressed(dataset_path, observations=obs, actions=act)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._session or {}, f, indent=2)

        self._active = False
        self._session = None
        self._buffer = None
        return {"recording": False, "dataset_path": str(dataset_path)}

    def add_step(self, observation: Dict[str, Any], action: Dict[str, Any]) -> None:
        if not self._active or self._buffer is None:
            return
        self._buffer.observations.append(observation)
        self._buffer.actions.append(action)
