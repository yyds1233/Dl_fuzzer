# Screen 筛选高效 Harness 执行的伪代码逻辑

## 一、总体目标

通过**双层 UCB 赌博机算法**，在多个 harness（模糊测试目标）中动态筛选最高效的来执行，同时自动选择每个 harness 的最优 profile（变异参数组合），最大化覆盖率产出。

---

## 二、核心数据结构

```
DriverConfig {
    runtime:   { epoch(每轮秒数), steps(总轮数), warmup_rounds, mix(Δft权重), ... }
    bandit:    { c_fast, c_slow, epsilon(探索率), elim_margin, elim_patience, cooldown_steps, ... }
    audit:     { audit_every(N轮一次审计), slow_metric(如"BRH"), audit_max_inputs, ... }
    pool:      { k(池大小), refresh_every, keep_frac, replace_frac, ... }
    prior:     { elite_size, ewma_alpha, min_pulls_for_admit, top_n_fast, ... }
}

HarnessCandidate {
    harness_id, harness_path, group_id,   // group_id 用于跨 harness 知识共享
    profiles: [{profile_id, profile}]      // 可选初始 profile 列表
}

// 双通道 UCB 统计 (每个 arm: harness 或 profile)
DualStats {
    n_fast, mean_fast   // 快速通道: 频繁的廉价代理奖励
    n_slow, mean_slow   // 慢速通道: 稀疏的昂贵审计奖励
    bad_streak, inactive_until, disabled   // 软淘汰/冷却状态
    t                   // 选择步数计数器
}

// Profile 池中的 arm 状态
ArmState {
    profile_id, profile, born_t, pulls, fast_ewma, slow_ewma, slow_n
}
```

---

## 三、主循环伪代码

```
function Screen(cfg):
    H = load_harnesses(cfg)                    // 所有候选 harness
    warmup = [each h in H] * warmup_rounds      // 预热队列: 每个 harness 轮流 N 轮
    B_h = DualChannelUCB(H)                     // harness 级赌博机
    B_p = {h: DualChannelUCB(P_h) for h in H}   // profile 级赌博机 (每个 harness 一个)
    pool = ProfilePool(H)                       // 每个 harness 的 profile 池
    prior = GroupPrior()                        // 跨 harness 精英知识库
    t = 1

    while t <= max_steps:
        // 1. 选择本轮执行的 harness 和 profile
        if warmup 非空:
            h = warmup.pop(0)                   // 预热: 顺序轮流
        else:
            h = B_h.select(H)                   // 赌博机: UCB 选择 (含 ε-探索)
        p = B_p[h].select(pool[h].profiles)

        // 2. 执行模糊测试 (默认 60s)
        log, new_files = fuzz(h, p, epoch=60s)

        // 3. 计算快速代理奖励
        Δcov, Δft = parse_cov_ft(log)           // 覆盖率/特征增量
        exec_s    = parse_speed(log)             // 执行速度
        r_proxy   = ((1-m)*ln(1+Δcov) + m*ln(1+Δft)) * √exec_s * (2 if Δcov>0 else 1)
        r_fast    = r_proxy * (1 + 0.1*ln(1+|new_files|))

        // 4. 快速通道更新
        B_h.update_fast(h, r_fast)
        B_p[h].update_fast(p, r_fast)
        pool[h].update(p, r_fast)

        // 5. 事件比例降权: REDUCE+pulse 占比过高 → 冷却
        if reduce_pulse_ratio(log) ≥ 0.70:
            B_h.cool_off(h, steps=5)

        // 6. 慢速审计 (每 10 轮或增量文件达标时触发)
        if t % audit_every == 0 or |new_files| ≥ audit_min_delta:
            Δglobal = cov_audit(h, new_files)   // LLVM 全局覆盖率增量
            r_slow  = Δglobal.BRH               // 分支命中增量

            B_h.update_slow(h, r_slow)

            // 持续零收益 → 冷却
            if r_slow ≤ 0:
                zero_streak[h]++
                if zero_streak[h] ≥ 2:
                    B_h.cool_off(h, steps=5)
            else:
                zero_streak[h] = 0
                prior.admit(pool[h].top_profiles, group[h])  // 精英入库

        // 7. 软淘汰: UCB < best_LCB 持续 N 轮 → 冷却 (非永久禁用)
        daily_soft_eliminate(B_h, B_p)

        // 8. Profile 池定期刷新: 保留 top 50%, 变异生成 30%, 从 prior 注入精英
        if t % 200 == 0:
            pool.refresh_all(prior)

        t += 1
```

---

## 3.1 Profile 池刷新逻辑 (`ProfilePoolManager.maybe_refresh`)

```
function maybe_refresh(hid, group_id, t, prior):
    if t % refresh_every != 0:  return false    // 默认每 200 轮
    if pool_size < k/2:         return false    // 池太小不刷新

    items = sort_by_fitness(所有 profile)
    // fitness = fast_ewma (快速代理奖励的指数移动平均)

    keep_n  = k * keep_frac    // 默认保留 50%
    replace_n = k * replace_frac  // 默认替换 30%

    keep = items[:keep_n]           // 保留最优秀的
    tail = items[keep_n:]           // 候选淘汰区

    kill_candidates = [x in tail if x.pulls >= min_pulls_to_kill]  // 默认 ≥30 次
    kill = kill_candidates[-replace_n:]   // 淘汰最差的

    // 生成新 profile 填补空缺:
    //   1. 从 GroupPrior 注入 inject_each_refresh 个精英 profile (同 group)
    //   2. 对 keep 中的 profile 做变异 (mutate) 生成后代
    //   3. 变异策略: 15% 概率修改每个参数; 离散值跳邻居; 浮点值 ±0.02 并吸附到网格
```

---

## 四、UCB 选择与更新 (`DualChannelUCBSoftElim`)

### 4.1 组合均值

```
mu(arm) = alpha * mean_fast + (1 - alpha) * mean_slow

其中 alpha = 1 / (1 + n_slow)          // 慢速样本越多，越信任慢速通道
     alpha = max(alpha_min, alpha)      // 默认 alpha_min = 0.2
```

### 4.2 探索奖励 (Bonus)

```
bonus(arm) = c_fast * sqrt(ln(t+1) / n_fast) + c_slow * sqrt(ln(t+1) / max(1, n_slow))
```

### 4.3 置信区间

```
UCB(arm) = mu + bonus
LCB(arm) = mu - bonus
```

### 4.4 选择策略 (`select`)

```
function select(arm_ids):
    // 排除 disabled arms
    valid = [aid not disabled]

    // ε-贪心探索: 概率 epsilon 随机选
    if random() < epsilon:
        return random_choice(valid)

    // 只从活跃的 (冷却结束) arms 中选
    active = [aid in valid if is_active(aid)]   // t >= inactive_until
    candidates = active (若为空则退回到 valid)

    // 优先选从未被快速通道采样过的 arm (cold-start)
    for each candidate:
        if n_fast == 0:  return candidate

    // 否则选 UCB 最大的
    return argmax(candidates, key=UCB)
```

### 4.5 软淘汰 (`maybe_soft_eliminate`)

```
function maybe_soft_eliminate(arm_ids):
    // 只在充分采样的活跃 arms 中找最佳 LCB
    best_lcb = -∞
    for each active arm with total_pulls >= elim_min_pulls:
        best_lcb = max(best_lcb, LCB(arm))

    // 对每个 arm: 若 UCB < best_lcb - elim_margin → bad_streak++
    // 若 bad_streak >= elim_patience:
    //     inactive_until = t + cooldown_steps   (冷却 N 步，之后可重新激活)
    // 注意: 在冷却期内的 arm 不累积 bad_streak (给予恢复机会)
```

### 4.6 更新

```
update_fast(arm_id, reward):
    n_fast += 1
    mean_fast += (reward - mean_fast) / n_fast   // 增量均值

update_slow(arm_id, reward):
    n_slow += 1
    mean_slow += (reward - mean_slow) / n_slow
```

---

## 五、Reward 计算逻辑

### 5.1 代理奖励 (`compute_proxy_reward`)

```
function compute_proxy_reward(delta_ft, delta_cov, exec_s, mix):
    cov_score  = ln(1 + max(0, Δcov))
    ft_score   = ln(1 + max(0, Δft))
    speed_term = sqrt(exec_s)          // 速度因子: 太快可能没深度, 太慢浪费资源
    cov_bonus  = 2.0 if Δcov > 0 else 1.0   // 有覆盖率增益翻倍

    return ((1 - mix) * cov_score + mix * ft_score) * speed_term * cov_bonus
    // mix=0.25 → 75% cov + 25% ft
```

### 5.2 快速奖励 (`compute_fast_reward`)

```
function compute_fast_reward(proxy_reward, delta_files):
    return proxy_reward * (1 + 0.1 * ln(1 + delta_files))
    // 产生新语料文件的 harness 获得小幅加权
```

### 5.3 慢速奖励 (审计)

```
slow_reward = audit_json.delta[slow_metric]  // 默认 "BRH" (Branches Hit 增量)
// 通过 LLVM source-based coverage 精确计算全局覆盖率增量
// 所有 harness 共享一个全局 profdata，计算 union 覆盖率的增量
```

---

## 六、降权/过滤机制总结

| 机制 | 触发条件 | 效果 | 可恢复? |
|------|---------|------|--------|
| **事件比例降权** | reduce_pulse_ratio ≥ 0.70 且 event_total ≥ min | inactive_until = t + cooldown | ✅ 冷却后恢复 |
| **零慢速降权** | 连续 zero_slow_streak ≥ 2 次审计收益为 0 | inactive_until = t + cooldown | ✅ 冷却后恢复 |
| **软淘汰** | UCB < best_LCB - margin 持续 patience 轮 | inactive_until = t + cooldown | ✅ 冷却后恢复 |
| **Epsilon 探索** | 概率 epsilon | 即使冷却也可被随机选中 | ✅ 始终有机会 |

> **关键设计理念**: 从不永久禁用任何 harness，只使用**冷却 (cooldown)** 机制。冷却期结束后，harness 自动重新进入候选池。

---

## 七、Profile 演进流程

```
                    ┌─────────────┐
                    │  GroupPrior │  (跨 harness 精英记忆)
                    │  per group  │
                    └──────┬──────┘
                           │ 注入精英 profile
                           ▼
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │ Harness A │     │ Harness B │     │ Harness C │
    │ Profile池 │     │ Profile池 │     │ Profile池 │
    │ (k=10)   │     │ (k=10)   │     │ (k=10)   │
    └──────────┘     └──────────┘     └──────────┘
         │                │                │
         │ 定期刷新:       │                │
         │ 保留 top 50%   │                │
         │ 变异生成 30%    │                │
         │ 注入 prior 精英 │                │
         ▼                ▼                ▼
    Profile Bandit   Profile Bandit   Profile Bandit
    (选最优 profile)  (选最优 profile)  (选最优 profile)
```

---

## 八、预热阶段

```
warmup_queue = []
for round in 1..warmup_rounds:
    for each harness_id:
        warmup_queue.append(harness_id)

// 预热期间:
//   - 严格按照队列顺序执行
//   - 不触发降权/软淘汰
//   - 确保每个 harness 都有初始快速采样数据
//   - 预热结束后才进入 UCB 选择
```

---

## 九、关键配置默认值

| 参数 | 默认值 | 含义 |
|------|-------|------|
| `epoch` | 60s | 每轮 fuzzing 时长 |
| `warmup_rounds` | 1 | 每个 harness 预热轮数 |
| `mix` | 0.25 | proxy reward 中 Δft 权重 (75% 给 Δcov) |
| `epsilon_harness` | 0.02 | harness 级 ε-贪心探索率 |
| `epsilon_profile` | 0.05 | profile 级 ε-贪心探索率 |
| `cooldown_steps` | 50 | 软淘汰冷却步数 |
| `elim_patience` | 3 | 连续 UCB<best_LCB 容忍轮数 |
| `audit_every` | 10 | 每 N 轮触发慢速审计 |
| `pool.k` | 10 | 每个 harness 的 profile 池大小 |
| `pool.refresh_every` | 200 | profile 池刷新间隔 |
| `pool.keep_frac` | 0.5 | 刷新时保留比例 |
| `pool.replace_frac` | 0.3 | 刷新时替换比例 |
| `zero_slow_deprioritize_streak` | 2 | 容忍连续零审计轮数 |
| `event_deprioritize_ratio` | 0.70 | REDUCE+pulse 比例阈值 |
