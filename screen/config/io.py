# screen/config/io.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .schema import HarnessCandidate, ProfileArm


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_profiles_from_top_json(top_json: Path) -> List[ProfileArm]:
    top = _load_json(top_json)
    arms: List[ProfileArm] = []
    for r in top:
        arms.append(ProfileArm(profile_id=r["profile_id"], profile=r["profile"]))
    return arms


def _load_group_map(groups_map: Optional[Path]) -> Dict[str, str]:
    if not groups_map:
        return {}
    p = Path(groups_map)
    if not p.exists():
        raise SystemExit(f"groups_map not found: {p}")
    data = _load_json(p)
    out: Dict[str, str] = {}
    for k, v in data.items():
        out[str(k)] = str(v)
    return out


def load_harness_candidates(
    *,
    harnesses_json: Optional[Path],
    harness: Optional[Path],
    harness_id: Optional[str],
    top_json: Optional[Path],
    groups_map: Optional[Path] = None,
) -> List[HarnessCandidate]:
    group_by_hid = _load_group_map(groups_map)

    if harnesses_json:
        data = _load_json(Path(harnesses_json))
        cands: List[HarnessCandidate] = []
        for h in data:
            hid = str(h["harness_id"])
            hpath = Path(h["harness_path"]).resolve()
            gid = group_by_hid.get(hid, "OTHERS")

            profiles: Optional[List[ProfileArm]] = None
            if "profiles" in h and h["profiles"]:
                profiles = [ProfileArm(profile_id=p["profile_id"], profile=p["profile"]) for p in h["profiles"]]

            cands.append(HarnessCandidate(harness_id=hid, harness_path=hpath, group_id=gid, profiles=profiles))
        return cands

    # legacy mode: still allowed
    if not harness or not top_json:
        raise SystemExit("Need --harnesses_json OR legacy (--harness and --top_json)")

    hpath = Path(harness).resolve()
    hid = harness_id or hpath.stem
    gid = group_by_hid.get(hid, "OTHERS")
    profiles = _load_profiles_from_top_json(Path(top_json).resolve())
    return [HarnessCandidate(harness_id=hid, harness_path=hpath, group_id=gid, profiles=profiles)]