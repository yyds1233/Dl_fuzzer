# screen/bandit_audit_driver_hier.py
from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from screen.bandit.policy import make_bandit
from screen.bandit.rewards import compute_fast_reward, compute_proxy_reward
from screen.config.io import load_harness_candidates
from screen.config.schema import DriverConfig, HarnessCandidate, ProfileArm, StepResult
from screen.metrics.compute import compute_deltas, normalize_exec_s
from screen.metrics.parse_libfuzzer import parse_fuzzer_log
from screen.runner.audit_runner import run_cov_audit_in_cov_env
from screen.runner.fuzz_runner import run_one_epoch


# ---------------------------
# Corpus snapshot/manifest utils (先留在 driver，后续可再拆到 metrics/)
# ---------------------------
def _iter_corpus_files(corpus_dir: Path) -> List[Tuple[str, int, int]]:
    items: List[Tuple[str, int, int]] = []
    if not corpus_dir.exists():
        return items
    for p in corpus_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(corpus_dir))
        st = p.stat()
        items.append((rel, int(st.st_size), int(st.st_mtime_ns)))
    items.sort(key=lambda x: x[0])
    return items


def load_manifest(path: Path) -> Dict[str, Dict[str, int]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(path: Path, manifest: Dict[str, Dict[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def build_manifest_from_items(items: List[Tuple[str, int, int]]) -> Dict[str, Dict[str, int]]:
    m: Dict[str, Dict[str, int]] = {}
    for rel, size, mtime_ns in items:
        m[rel] = {"size": int(size), "mtime_ns": int(mtime_ns)}
    return m


def diff_manifest(old: Dict[str, Dict[str, int]], new: Dict[str, Dict[str, int]]) -> List[str]:
    delta: List[str] = []
    for rel, meta in new.items():
        prev = old.get(rel)
        if prev is None:
            delta.append(rel)
        else:
            if int(prev.get("size", -1)) != int(meta.get("size", -2)) or int(prev.get("mtime_ns", -1)) != int(meta.get("mtime_ns", -2)):
                delta.append(rel)
    return delta


def _safe_tag_name(profile_id: str, basename: str) -> str:
    if basename.startswith(profile_id + "__"):
        return basename
    return f"{profile_id}__{basename}"


def tag_delta_files_with_profile(corpus_dir: Path, delta_relpaths: List[str], profile_id: str) -> List[str]:
    new_rels: List[str] = []
    for rel in delta_relpaths:
        src = corpus_dir / rel
        if not src.exists() or not src.is_file():
            continue
        parent = src.parent
        base = src.name
        new_base = _safe_tag_name(profile_id, base)
        if new_base == base:
            new_rels.append(rel)
            continue

        dst = parent / new_base
        if dst.exists():
            stem = dst.stem
            suf = dst.suffix
            i = 0
            while dst.exists():
                i += 1
                dst = parent / f"{stem}__r{i}{suf}"
        src.rename(dst)
        new_rels.append(str(dst.relative_to(corpus_dir)))
    return new_rels


def update_current_manifest_and_tag_epoch_delta(*, corpus_dir: Path, current_manifest_path: Path, epoch_profile_id: str) -> List[str]:
    old = load_manifest(current_manifest_path)
    items_before = _iter_corpus_files(corpus_dir)
    new_before = build_manifest_from_items(items_before)
    epoch_delta = diff_manifest(old, new_before)

    tagged_delta = tag_delta_files_with_profile(corpus_dir, epoch_delta, epoch_profile_id)

    items_after = _iter_corpus_files(corpus_dir)
    new_after = build_manifest_from_items(items_after)
    save_manifest(current_manifest_path, new_after)
    return tagged_delta


def diff_audit_window_delta(*, audit_base_manifest_path: Path, current_manifest_path: Path) -> List[str]:
    base = load_manifest(audit_base_manifest_path)
    cur = load_manifest(current_manifest_path)
    return diff_manifest(base, cur)


def advance_audit_base_to_current(*, audit_base_manifest_path: Path, current_manifest_path: Path) -> None:
    cur = load_manifest(current_manifest_path)
    save_manifest(audit_base_manifest_path, cur)


def materialize_subset_corpus(corpus_dir: Path, relpaths: List[str], out_dir: Path, *, max_inputs: int = 2000) -> int:
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    rels = list(relpaths)

    if max_inputs > 0 and len(rels) > max_inputs:
        tmp: List[Tuple[int, str]] = []
        for rel in rels:
            src = corpus_dir / rel
            if not src.exists():
                continue
            try:
                tmp.append((int(src.stat().st_mtime_ns), rel))
            except FileNotFoundError:
                continue
        tmp.sort(key=lambda x: x[0], reverse=True)
        rels = [r for _, r in tmp[:max_inputs]]

    count = 0
    for rel in rels:
        src = corpus_dir / rel
        if not src.exists() or not src.is_file():
            continue
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        count += 1
    return count


# ---------------------------
# Attribution helpers
# ---------------------------
def producer_from_relpath(rel: str) -> Optional[str]:
    base = Path(rel).name
    if "__" not in base:
        return None
    pid, _ = base.split("__", 1)
    return pid or None


def group_relpaths_by_profile(relpaths: List[str]) -> Dict[str, List[str]]:
    buckets: DefaultDict[str, List[str]] = defaultdict(list)
    for rel in relpaths:
        pid = producer_from_relpath(rel) or "unknown"
        buckets[pid].append(rel)
    return dict(buckets)


def _select_topk_profiles_by_count(buckets: Dict[str, List[str]], *, topk: int) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    for pid, rels in buckets.items():
        if pid in ("unknown", "other"):
            continue
        items.append((pid, len(rels)))
    items.sort(key=lambda x: x[1], reverse=True)
    if topk > 0:
        items = items[:topk]
    return items


# ---------------------------
# Orchestrator (核心循环)
# ---------------------------
def orchestrate(cfg: DriverConfig) -> None:
    rt = cfg.runtime
    bd = cfg.bandit
    au = cfg.audit

    root = rt.root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    candidates = load_harness_candidates(
        harnesses_json=cfg.harnesses_json,
        harness=cfg.harness,
        harness_id=cfg.harness_id,
        top_json=cfg.top_json,
    )
    if not candidates:
        raise SystemExit("no harness candidates loaded")

    fuzz_flags = [x for x in rt.fuzz_flags.split(" ") if x.strip()]

    manifest_root = rt.manifest_dir
    if not manifest_root.is_absolute():
        manifest_root = (root / manifest_root).resolve()
    manifest_root.mkdir(parents=True, exist_ok=True)

    global_root = au.global_dir
    if not global_root.is_absolute():
        global_root = (root / global_root).resolve()
    global_root.mkdir(parents=True, exist_ok=True)

    # index
    harness_ids = [c.harness_id for c in candidates]
    harness_path_by_id: Dict[str, Path] = {c.harness_id: c.harness_path for c in candidates}
    profiles_by_harness: Dict[str, List[ProfileArm]] = {c.harness_id: c.profiles for c in candidates}

    # bandits
    harness_bandit = make_bandit(
        c_fast=bd.c_fast, c_slow=bd.c_slow,
        epsilon=bd.epsilon_harness,
        elim_margin=bd.elim_margin,
        elim_patience=bd.elim_patience,
        elim_min_pulls=bd.elim_min_pulls,
        alpha_min=bd.alpha_min,
        cooldown_steps=bd.cooldown_steps,
        seed=bd.seed,
    )

    profile_bandits: Dict[str, Any] = {}
    for hid in harness_ids:
        profile_bandits[hid] = make_bandit(
            c_fast=bd.c_fast, c_slow=bd.c_slow,
            epsilon=bd.epsilon_profile,
            elim_margin=bd.elim_margin,
            elim_patience=bd.elim_patience,
            elim_min_pulls=bd.elim_min_pulls,
            alpha_min=bd.alpha_min,
            cooldown_steps=bd.cooldown_steps,
            seed=bd.seed + (hash(hid) & 0xFFFF),
        )

    results: List[StepResult] = []

    t = 1
    try:
        while True:
            if rt.steps > 0 and t > rt.steps:
                break

            # 1) select harness
            hid = harness_bandit.select(harness_ids)
            harness_path = harness_path_by_id[hid]

            # 2) select profile
            prof_arms = profiles_by_harness[hid]
            prof_ids = [p.profile_id for p in prof_arms]
            pb = profile_bandits[hid]
            pid = pb.select(prof_ids)
            arm = next(p for p in prof_arms if p.profile_id == pid)

            profile_env = {k: str(v) for k, v in arm.profile.items()}
            profile_env.setdefault("OMP_NUM_THREADS", "1")
            profile_env.setdefault("MKL_NUM_THREADS", "1")
            profile_env.setdefault("TORCH_NUM_THREADS", "1")

            corpus_dir = root / "corpus" / hid
            run_dir = root / "runs" / hid / pid / f"t{t:04d}"
            crash_dir = run_dir / "crash"
            log_path = run_dir / "fuzzer.log"
            run_dir.mkdir(parents=True, exist_ok=True)

            cur_manifest = manifest_root / hid / "current.json"
            audit_base_manifest = manifest_root / hid / "audit_base.json"
            cur_manifest.parent.mkdir(parents=True, exist_ok=True)

            # 3) fuzz epoch
            run_one_epoch(
                python=rt.python,
                harness_path=harness_path,
                corpus_dir=corpus_dir,
                crash_dir=crash_dir,
                log_path=log_path,
                epoch_sec=rt.epoch,
                fuzz_flags=fuzz_flags,
                profile_env=profile_env,
            )

            # 4) update manifest + tag
            epoch_delta_relpaths = update_current_manifest_and_tag_epoch_delta(
                corpus_dir=corpus_dir,
                current_manifest_path=cur_manifest,
                epoch_profile_id=pid,
            )
            (run_dir / "epoch_delta_files.json").write_text(json.dumps(epoch_delta_relpaths, indent=2), encoding="utf-8")
            delta_files_epoch = len(epoch_delta_relpaths)

            # 5) fast reward
            p = parse_fuzzer_log(log_path)
            exec_s = normalize_exec_s(p["exec_s_last"])
            delta_ft, delta_cov = compute_deltas(p["cov_first"], p["cov_last"], p["ft_first"], p["ft_last"])

            proxy_reward = float(compute_proxy_reward(delta_ft, delta_cov, exec_s, mix=rt.mix))
            fast_reward = float(compute_fast_reward(proxy_reward, delta_files_epoch))

            pb.update_fast(pid, fast_reward)
            harness_bandit.update_fast(hid, fast_reward)

            # 6) slow audit
            audited_harnesses = 0
            slow_harness_selected: Optional[int] = None
            slow_profile_credit_selected: Optional[float] = None

            do_audit = (au.audit_every > 0 and (t % au.audit_every == 0))
            if do_audit:
                for ahid in harness_ids:
                    acorpus = root / "corpus" / ahid
                    acur = manifest_root / ahid / "current.json"
                    abase = manifest_root / ahid / "audit_base.json"
                    if not acur.exists():
                        continue

                    if not abase.exists():
                        advance_audit_base_to_current(audit_base_manifest_path=abase, current_manifest_path=acur)
                        continue

                    if au.full_corpus_audit:
                        curm = load_manifest(acur)
                        window_delta = list(curm.keys())
                    else:
                        window_delta = diff_audit_window_delta(audit_base_manifest_path=abase, current_manifest_path=acur)

                    if not window_delta:
                        continue

                    audit_root = root / "audits" / ahid / f"t{t:04d}"
                    audit_corpus_dir = audit_root / "window_corpus"
                    audited_inputs = materialize_subset_corpus(
                        acorpus, window_delta, audit_corpus_dir, max_inputs=au.audit_max_inputs
                    )
                    if audited_inputs <= 0:
                        advance_audit_base_to_current(audit_base_manifest_path=abase, current_manifest_path=acur)
                        continue

                    audit_json = run_cov_audit_in_cov_env(
                        cov_venv_activate=au.cov_venv_activate.resolve(),
                        cov_audit_script=au.cov_audit_script.resolve(),
                        harness_path=harness_path_by_id[ahid],
                        corpus_dir=audit_corpus_dir,
                        work_dir=audit_root / "work",
                        global_dir=global_root,
                        primary_object=au.primary_object,
                        extra_objects=au.extra_object,
                        ignore_filename_regex=au.ignore_filename_regex,
                        replay_extra=au.cov_replay_extra,
                    )
                    delta = audit_json.get("delta", {}) or {}
                    slow_h = int(delta.get(au.slow_metric, 0)) if isinstance(delta, dict) else 0

                    harness_bandit.update_slow(ahid, float(slow_h))

                    buckets = group_relpaths_by_profile(window_delta)
                    top_items = _select_topk_profiles_by_count(buckets, topk=int(au.audit_profile_topk))
                    denom = sum(cnt for _p, cnt in top_items)
                    pb2 = profile_bandits[ahid]

                    if slow_h > 0 and denom > 0:
                        for p2, cnt in top_items:
                            credit = float(slow_h) * (float(cnt) / float(denom))
                            pb2.update_slow(p2, credit)
                            if ahid == hid and p2 == pid:
                                slow_profile_credit_selected = credit
                    else:
                        if au.zero_slow_penalty > 0.0:
                            for p2, cnt in top_items:
                                if int(cnt) >= int(au.min_credit_inputs):
                                    pb2.update_slow(p2, -float(au.zero_slow_penalty))
                                    if ahid == hid and p2 == pid:
                                        slow_profile_credit_selected = -float(au.zero_slow_penalty)

                    audited_harnesses += 1
                    advance_audit_base_to_current(audit_base_manifest_path=abase, current_manifest_path=acur)

                    if ahid == hid:
                        slow_harness_selected = slow_h
                        if slow_profile_credit_selected is None and slow_h > 0:
                            slow_profile_credit_selected = 0.0

                harness_bandit.maybe_soft_eliminate(harness_ids)
                for ahid in harness_ids:
                    profile_bandits[ahid].maybe_soft_eliminate([p.profile_id for p in profiles_by_harness[ahid]])

            sr = StepResult(
                t=t,
                harness_id=hid,
                profile_id=pid,
                delta_ft=delta_ft,
                delta_cov=delta_cov,
                exec_s=exec_s,
                proxy_reward=proxy_reward,
                fast_reward=fast_reward,
                delta_files_epoch=delta_files_epoch,
                audited_harnesses=audited_harnesses,
                slow_harness=slow_harness_selected,
                slow_profile_credit=slow_profile_credit_selected,
            )
            results.append(sr)

            hu, hl = harness_bandit.ucb_lcb(hid)
            pu, pl = pb.ucb_lcb(pid)

            print(
                f"[t={t:04d}] harness={hid} profile={pid} "
                f"Δft={delta_ft} Δcov={delta_cov} exec/s={exec_s:.1f} "
                f"proxy={proxy_reward:.3f} fast={fast_reward:.3f} delta_files={delta_files_epoch} "
                + (f"| audit_h={audited_harnesses} slow_{au.slow_metric}(H/P)=({slow_harness_selected}/{slow_profile_credit_selected}) "
                   if do_audit else "")
                + f"| H(UCB/LCB)=({hu:.3f}/{hl:.3f}) active={harness_bandit.is_active(hid)} "
                  f"P(UCB/LCB)=({pu:.3f}/{pl:.3f}) active={pb.is_active(pid)}"
            )

            out_state = {
                "config": cfg.to_jsonable(),
                "harness_bandit": harness_bandit.to_jsonable(),
                "profile_bandits": {x: profile_bandits[x].to_jsonable() for x in profile_bandits},
                "results_tail": [asdict(x) for x in results[-50:]],
            }
            (root / "state").mkdir(parents=True, exist_ok=True)
            (root / "state" / "bandit_state.json").write_text(json.dumps(out_state, indent=2), encoding="utf-8")

            t += 1

    except KeyboardInterrupt:
        print("\n[!] interrupted by user (Ctrl+C)")

    final = root / "state" / "bandit_all_results.json"
    final.write_text(json.dumps([asdict(x) for x in results], indent=2), encoding="utf-8")
    print(f"[+] wrote {final}")
    print(f"[+] global union dir: {global_root}")
    print(f"[+] manifest root: {manifest_root}")
    print(f"[+] corpus root: {root / 'corpus'}")
