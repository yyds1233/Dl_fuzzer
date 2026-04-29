from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from screen.bandit.policy import make_bandit
from screen.bandit.rewards import compute_fast_reward, compute_proxy_reward
from screen.config.io import load_harness_candidates
from screen.config.schema import DriverConfig, StepResult
from screen.metrics.compute import compute_deltas, normalize_exec_s
from screen.metrics.parse_libfuzzer import parse_fuzzer_event_mix, parse_fuzzer_log
from screen.runner.audit_runner import run_cov_audit_in_cov_env
from screen.runner.fuzz_runner import run_one_epoch
from screen.pool.profile_pool import ProfilePoolManager
from screen.prior.group_prior import GroupPriorManager


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
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def build_manifest_from_items(items: List[Tuple[str, int, int]]) -> Dict[str, Dict[str, int]]:
    manifest: Dict[str, Dict[str, int]] = {}
    for rel, size, mtime_ns in items:
        manifest[rel] = {"size": int(size), "mtime_ns": int(mtime_ns)}
    return manifest


def diff_manifest(old: Dict[str, Dict[str, int]], new: Dict[str, Dict[str, int]]) -> List[str]:
    delta: List[str] = []
    for rel, meta in new.items():
        prev = old.get(rel)
        if prev is None:
            delta.append(rel)
            continue
        if int(prev.get("size", -1)) != int(meta.get("size", -2)):
            delta.append(rel)
            continue
        if int(prev.get("mtime_ns", -1)) != int(meta.get("mtime_ns", -2)):
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


def materialize_subset_corpus(corpus_dir: Path, relpaths: List[str], out_dir: Path, *, max_inputs: int = 0) -> int:
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


def _should_run_local_audit(*, audit_every: int, audit_min_delta_files: int, runs_since_base: int, delta_files_since_base: int) -> bool:
    by_runs = audit_every > 0 and runs_since_base >= audit_every
    by_delta_files = audit_min_delta_files > 0 and delta_files_since_base >= audit_min_delta_files
    return by_runs or by_delta_files


def _reset_local_audit_state(runs_since_audit: Dict[str, int], delta_files_since_audit: Dict[str, int], hid: str) -> None:
    runs_since_audit[hid] = 0
    delta_files_since_audit[hid] = 0


def _write_prior_from_fast_gate(*, cfg: DriverConfig, prior: GroupPriorManager, pool: ProfilePoolManager, hid: str, group_id: str, slow_h: int, t: int) -> int:
    if not cfg.prior.enabled:
        return 0
    if group_id == "OTHERS" or slow_h <= 0:
        return 0
    admitted = 0
    min_pulls = max(0, int(cfg.prior.min_pulls_for_admit))
    top_n = max(0, int(cfg.prior.top_n_fast))
    reward_clip = float(cfg.prior.reward_clip)
    for st in pool.ranked_states(hid):
        if st.pulls < min_pulls:
            continue
        reward = float(min(st.fast_ewma, reward_clip))
        prior.observe(group_id=group_id, profile_id=st.profile_id, profile=st.profile, reward=reward, t=t)
        admitted += 1
        if top_n > 0 and admitted >= top_n:
            break
    return admitted


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
        groups_map=cfg.groups_map,
    )
    if not candidates:
        raise SystemExit("no harness candidates loaded")
    import shlex
    fuzz_flags = shlex.split(rt.fuzz_flags)
    manifest_root = rt.manifest_dir
    if not manifest_root.is_absolute():
        manifest_root = (root / manifest_root).resolve()
    manifest_root.mkdir(parents=True, exist_ok=True)
    global_root = au.global_dir
    if not global_root.is_absolute():
        global_root = (root / global_root).resolve()
    global_root.mkdir(parents=True, exist_ok=True)
    harness_ids = [c.harness_id for c in candidates]
    harness_path_by_id: Dict[str, Path] = {c.harness_id: c.harness_path for c in candidates}
    group_by_hid: Dict[str, str] = {c.harness_id: c.group_id for c in candidates}
    warmup_queue: List[str] = list(harness_ids)

    prior = None
    pool = None
    if not rt.disable_profiles:
        prior = GroupPriorManager(state_dir=(root / "state" / "group_prior"), elite_size=cfg.prior.elite_size, ewma_alpha=cfg.prior.ewma_alpha)
        pool = ProfilePoolManager(
            state_dir=(root / "state" / "pools"),
            k=cfg.pool.k,
            refresh_every=cfg.pool.refresh_every,
            keep_frac=cfg.pool.keep_frac,
            replace_frac=cfg.pool.replace_frac,
            inject_each_refresh=cfg.pool.inject_each_refresh,
            min_pulls_to_kill=cfg.pool.min_pulls_to_kill,
            seed=bd.seed,
        )
    if not rt.disable_profiles:
        for c in candidates:
            hid = c.harness_id
            gid = c.group_id
            if c.profiles:
                pool.maybe_init_pool(hid=hid, group_id="OTHERS", t=1, group_prior=prior)
            pool.maybe_init_pool(hid=hid, group_id=gid, t=1, group_prior=prior)

    harness_bandit = make_bandit(
        c_fast=bd.c_fast,
        c_slow=bd.c_slow,
        epsilon=bd.epsilon_harness,
        elim_margin=bd.elim_margin,
        elim_patience=bd.elim_patience,
        elim_min_pulls=bd.elim_min_pulls,
        alpha_min=bd.alpha_min,
        cooldown_steps=bd.cooldown_steps,
        seed=bd.seed,
    )
    profile_bandits: Dict[str, Any] = {}
    if not rt.disable_profiles:
        for hid in harness_ids:
            profile_bandits[hid] = make_bandit(
                c_fast=bd.c_fast,
                c_slow=bd.c_slow,
                epsilon=bd.epsilon_profile,
                elim_margin=bd.elim_margin,
                elim_patience=bd.elim_patience,
                elim_min_pulls=bd.elim_min_pulls,
                alpha_min=bd.alpha_min,
                cooldown_steps=bd.cooldown_steps,
                seed=bd.seed + (hash(hid) & 0xFFFF),
            )

    audit_runs_since_base: Dict[str, int] = {hid: 0 for hid in harness_ids}
    audit_delta_files_since_base: Dict[str, int] = {hid: 0 for hid in harness_ids}
    zero_slow_audit_streak: Dict[str, int] = {hid: 0 for hid in harness_ids}
    results: List[StepResult] = []
    t = 1
    try:
        while True:
            if rt.steps > 0 and t > rt.steps:
                break
            if not rt.disable_profiles:
                for ahid in harness_ids:
                    pool.maybe_refresh(hid=ahid, group_id=group_by_hid[ahid], t=t, group_prior=prior)
            if warmup_queue:
                hid = warmup_queue.pop(0)
                harness_bandit.t += 1
            else:
                hid = harness_bandit.select(harness_ids)
            harness_path = harness_path_by_id[hid]
            if rt.disable_profiles:
                pid = "no_profile"
                pb = None
                profile_env: Dict[str, str] = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TORCH_NUM_THREADS": "1"}
            else:
                prof_arms = pool.get_active_profiles(hid)
                if not prof_arms:
                    pool.maybe_init_pool(hid=hid, group_id=group_by_hid[hid], t=t, group_prior=prior)
                    prof_arms = pool.get_active_profiles(hid)
                prof_ids = [p.profile_id for p in prof_arms]
                pb = profile_bandits[hid]
                for pid0 in prof_ids:
                    pb.ensure(pid0)
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
            run_one_epoch(python=rt.python, harness_path=harness_path, corpus_dir=corpus_dir, crash_dir=crash_dir, log_path=log_path, epoch_sec=rt.epoch, fuzz_flags=fuzz_flags, profile_env=profile_env)
            epoch_delta_relpaths = update_current_manifest_and_tag_epoch_delta(corpus_dir=corpus_dir, current_manifest_path=cur_manifest, epoch_profile_id=pid)
            (run_dir / "epoch_delta_files.json").write_text(json.dumps(epoch_delta_relpaths, indent=2), encoding="utf-8")
            delta_files_epoch = len(epoch_delta_relpaths)
            p = parse_fuzzer_log(log_path)
            exec_s = normalize_exec_s(p["exec_s_last"])
            delta_ft, delta_cov = compute_deltas(p["cov_first"], p["cov_last"], p["ft_first"], p["ft_last"])
            event_mix = parse_fuzzer_event_mix(log_path)
            event_total = int(event_mix["event_total"])
            new_count = int(event_mix["new_count"])
            reduce_count = int(event_mix["reduce_count"])
            pulse_count = int(event_mix["pulse_count"])
            reduce_pulse_ratio = float(event_mix["reduce_pulse_ratio"])
            proxy_reward = float(compute_proxy_reward(delta_ft, delta_cov, exec_s, mix=rt.mix))
            fast_reward = float(compute_fast_reward(proxy_reward, delta_files_epoch))
            forced_deprioritized = False
            if event_total >= 100 and reduce_pulse_ratio >= 0.90:
                hs = harness_bandit.ensure(hid)
                hs.bad_streak = 0
                hs.inactive_until = max(hs.inactive_until, harness_bandit.t + 5)
                forced_deprioritized = True
            if not rt.disable_profiles:
                pb.update_fast(pid, fast_reward)
                pool.on_fast(hid, pid, fast_reward)
            harness_bandit.update_fast(hid, fast_reward)
            audit_runs_since_base[hid] += 1
            audit_delta_files_since_base[hid] += delta_files_epoch
            audited_harnesses = 0
            slow_harness_selected: Optional[int] = None
            slow_profile_credit_selected: Optional[float] = None
            prior_admitted_selected = 0
            forced_deprioritized_by_zero_slow = False
            zero_slow_streak_now = zero_slow_audit_streak.get(hid, 0)
            do_audit = _should_run_local_audit(
                audit_every=int(au.audit_every),
                audit_min_delta_files=int(getattr(au, "audit_min_delta_files", 0)),
                runs_since_base=audit_runs_since_base[hid],
                delta_files_since_base=audit_delta_files_since_base[hid],
            )
            if do_audit:
                if not audit_base_manifest.exists():
                    advance_audit_base_to_current(audit_base_manifest_path=audit_base_manifest, current_manifest_path=cur_manifest)
                    _reset_local_audit_state(audit_runs_since_base, audit_delta_files_since_base, hid)
                else:
                    if au.full_corpus_audit:
                        curm = load_manifest(cur_manifest)
                        window_delta = list(curm.keys())
                    else:
                        window_delta = diff_audit_window_delta(audit_base_manifest_path=audit_base_manifest, current_manifest_path=cur_manifest)
                    audited_harnesses = 1
                    slow_h = 0
                    if window_delta:
                        audit_root = root / "audits" / hid / f"t{t:04d}"
                        audit_corpus_dir = audit_root / "window_corpus"
                        audited_inputs = materialize_subset_corpus(corpus_dir, window_delta, audit_corpus_dir, max_inputs=au.audit_max_inputs)
                        if audited_inputs > 0:
                            audit_json = run_cov_audit_in_cov_env(
                                cov_venv_activate=au.cov_venv_activate.resolve(),
                                cov_audit_script=au.cov_audit_script.resolve(),
                                harness_path=harness_path_by_id[hid],
                                corpus_dir=audit_corpus_dir,
                                work_dir=audit_root / "work",
                                global_dir=global_root,
                                primary_object=au.primary_object,
                                extra_objects=au.extra_object,
                                ignore_filename_regex=au.ignore_filename_regex,
                                replay_extra=au.cov_replay_extra,
                            )
                            delta = audit_json.get("delta", {}) or {}
                            if isinstance(delta, dict):
                                slow_h = int(delta.get(au.slow_metric, 0))
                    harness_bandit.update_slow(hid, float(slow_h))
                    slow_harness_selected = slow_h
                    if slow_h <= 0:
                        zero_slow_audit_streak[hid] = zero_slow_audit_streak.get(hid, 0) + 1
                    else:
                        zero_slow_audit_streak[hid] = 0
                    zero_slow_streak_now = zero_slow_audit_streak[hid]
                    if zero_slow_streak_now >= 2:
                        hs = harness_bandit.ensure(hid)
                        hs.bad_streak = 0
                        hs.inactive_until = max(hs.inactive_until, harness_bandit.t + 5)
                        forced_deprioritized_by_zero_slow = True
                    if not rt.disable_profiles:
                        prior_admitted_selected = _write_prior_from_fast_gate(cfg=cfg, prior=prior, pool=pool, hid=hid, group_id=group_by_hid[hid], slow_h=slow_h, t=t)
                    else:
                        prior_admitted_selected = 0
                    advance_audit_base_to_current(audit_base_manifest_path=audit_base_manifest, current_manifest_path=cur_manifest)
                    _reset_local_audit_state(audit_runs_since_base, audit_delta_files_since_base, hid)
                    harness_bandit.maybe_soft_eliminate(harness_ids)
                    if not rt.disable_profiles:
                        for ahid in harness_ids:
                            cur_ids = [p.profile_id for p in pool.get_active_profiles(ahid)]
                            profile_bandits[ahid].maybe_soft_eliminate(cur_ids)
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
                event_total=event_total,
                new_count=new_count,
                reduce_count=reduce_count,
                pulse_count=pulse_count,
                reduce_pulse_ratio=reduce_pulse_ratio,
                forced_deprioritized=forced_deprioritized,
                zero_slow_streak=zero_slow_streak_now,
                forced_deprioritized_by_zero_slow=forced_deprioritized_by_zero_slow,
            )
            results.append(sr)
            hu, hl = harness_bandit.ucb_lcb(hid)
            if rt.disable_profiles:
                pu, pl = 0.0, 0.0
                p_active = True
            else:
                pu, pl = pb.ucb_lcb(pid)
                p_active = pb.is_active(pid)
            print(
                f"[t={t:04d}] harness={hid} profile={pid} "
                f"Δft={delta_ft} Δcov={delta_cov} exec/s={exec_s:.1f} "
                f"proxy={proxy_reward:.3f} fast={fast_reward:.3f} delta_files={delta_files_epoch} "
                + (f"| deprioritized=1 rp_ratio={reduce_pulse_ratio:.3f} " if forced_deprioritized else f"| rp_ratio={reduce_pulse_ratio:.3f} ")
                + (
                    f"| audit_h={audited_harnesses} slow_{au.slow_metric}(H)=({slow_harness_selected}) "
                    f"zero_slow_streak={zero_slow_streak_now} "
                    + ("cut_by_zero_slow=1 " if forced_deprioritized_by_zero_slow else "")
                    + f"prior_admit={prior_admitted_selected} "
                    if do_audit else ""
                )
                + f"| H(UCB/LCB)=({hu:.3f}/{hl:.3f}) active={harness_bandit.is_active(hid)} "
                f"P(UCB/LCB)=({pu:.3f}/{pl:.3f}) active={p_active}"
            )
            if not rt.disable_profiles:
                pool.flush_all()
            out_state = {
                "config": cfg.to_jsonable(),
                "harness_bandit": harness_bandit.to_jsonable(),
                "profile_bandits": {x: profile_bandits[x].to_jsonable() for x in profile_bandits},
                "groups": group_by_hid,
                "warmup_remaining": list(warmup_queue),
                "audit_state": {
                    ahid: {
                        "runs_since_base": audit_runs_since_base[ahid],
                        "delta_files_since_base": audit_delta_files_since_base[ahid],
                        "zero_slow_audit_streak": zero_slow_audit_streak[ahid],
                    } for ahid in harness_ids
                },
                "results_tail": [asdict(x) for x in results[-50:]],
            }
            (root / "state").mkdir(parents=True, exist_ok=True)
            state_path = root / "state" / "bandit_state.json"
            state_tmp = state_path.with_suffix(".json.tmp")
            state_tmp.write_text(json.dumps(out_state, indent=2), encoding="utf-8")
            os.replace(state_tmp, state_path)
            t += 1
    except KeyboardInterrupt:
        print("\n[!] interrupted by user (Ctrl+C)")
        if not rt.disable_profiles:
            pool.flush_all()
    final = root / "state" / "bandit_all_results.json"
    final.write_text(json.dumps([asdict(x) for x in results], indent=2), encoding="utf-8")
    print(f"[+] wrote {final}")
    print(f"[+] global union dir: {global_root}")
    print(f"[+] manifest root: {manifest_root}")
    print(f"[+] corpus root: {root / 'corpus'}")
