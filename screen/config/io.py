# screen/config/io.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .schema import HarnessCandidate, ProfileArm


def _load_profiles_from_top_json(top_json: Path) -> List[ProfileArm]:
    top = json.loads(top_json.read_text(encoding="utf-8"))
    arms: List[ProfileArm] = []
    for r in top:
        arms.append(ProfileArm(profile_id=r["profile_id"], profile=r["profile"]))
    return arms


def load_harness_candidates(
    *,
    harnesses_json: Optional[Path],
    harness: Optional[Path],
    harness_id: Optional[str],
    top_json: Optional[Path],
) -> List[HarnessCandidate]:
    if harnesses_json:
        data = json.loads(Path(harnesses_json).read_text(encoding="utf-8"))
        cands: List[HarnessCandidate] = []
        for h in data:
            hid = str(h["harness_id"])
            hpath = Path(h["harness_path"]).resolve()
            profiles = [ProfileArm(profile_id=p["profile_id"], profile=p["profile"]) for p in h["profiles"]]
            if profiles:
                cands.append(HarnessCandidate(harness_id=hid, harness_path=hpath, profiles=profiles))
        return cands

    if not harness or not top_json:
        raise SystemExit("Need --harnesses_json OR legacy (--harness and --top_json)")

    hpath = Path(harness).resolve()
    hid = harness_id or hpath.stem
    profiles = _load_profiles_from_top_json(Path(top_json).resolve())
    return [HarnessCandidate(harness_id=hid, harness_path=hpath, profiles=profiles)]
