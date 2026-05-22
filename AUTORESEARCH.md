# Autoresearch Session Runbook

This file is for running end-to-end reinforcement-learning autoresearch
sessions: creating a session, mutating session-local candidates, running
fixed-budget experiments, maintaining the experiment log, inspecting artifacts,
confirming apparent winners, and deciding when to stop or pivot.

For project maintenance, task design, environment implementation, dependencies,
tests, dashboard development, or release work, read `AGENTS.md` instead.

## Goal

Improve a trainable controller for the fixed Gymnasium benchmark selected by
`--benchmark`.

Optimize the benchmark's `primary_metric` according to `primary_metric_mode`.
Use benchmark-declared secondary metrics, average eval return, simpler code, and
lower training cost as tie breakers, in that order.

Never compare candidates across different training or evaluation budgets unless
the log explicitly labels the comparison as budget-mismatched.

A single pass is screening evidence. It may choose the next code parent, but it
does not establish a confirmed winner.

## Autoresearch North Star

This runbook follows the same basic research-loop shape as
`karpathy/autoresearch`: keep the evaluator fixed, edit only the training
recipe, run a fixed-budget experiment, log the result, keep or discard based on
evidence, and repeat serially.

In `karpathy/autoresearch`, `prepare.py` owns fixed data and evaluation while
the agent edits `train.py`. In this repo, the benchmark and runner own the fixed
task/evaluation contract while the agent edits one session-local candidate file.

The RL version is stricter because RL results can have high seed, environment,
and evaluation variance. Confirmation and pivot rules exist to prevent lucky
single runs from becoming claims and to prevent the agent from getting stuck in
an unfruitful mutation family.

## Serial Hypothesis Principle

Autoresearch is serial hypothesis exploration.

At any moment there is exactly one active hypothesis under test. After the run,
inspect the generated artifacts, update the experiment log, revise the frontier,
and then choose the next hypothesis.

Do not prewrite, queue, or execute a batch of candidate files. A tentative idea
backlog is allowed, but it is a scratchpad, not an execution queue. The next
pass must be chosen from current evidence.

Each ordinary candidate should test one coherent mutation to the training code.
If a change combines multiple mechanisms, the log must justify why they form one
inseparable hypothesis. Otherwise split them into separate passes.

## Session Evidence Boundary

Hypotheses must come from the current session's evidence and the code in front
of the agent.

Allowed sources for choosing the next mutation:

- the selected benchmark JSON and fixed eval contract
- the selected seed trainable and current candidate code
- this session's `outer_loop_log.md`
- this session's `results.jsonl`
- this session's run summaries, train/eval episodes, curves, frames, and
  utilization artifacts
- first-principles reasoning about the algorithm, task, and failure funnel

Do not use previous sessions, memories, chat history, old dashboards, old
rollout summaries, or results from other tasks to propose or prioritize
candidate mutations.

Prior sessions and memories may be used only for setup and navigation: finding
the right repo, command shape, dashboard behavior, known installation quirks, or
the existence of a benchmark. They must not seed the hypothesis queue unless the
user explicitly asks to import prior findings into this session.

If prior evidence is explicitly imported, write a short "Imported Prior" note in
`outer_loop_log.md` before using it. The note must name the source, why it is
being imported, and whether it changes only setup or also the research plan.

## Evidence Levels

Use these labels consistently:

- `smoke`: verifies that code runs; not learning evidence.
- `screening`: one full-budget cold-start candidate run.
- `screening_leader`: best single-run result so far; useful parent, not a claim.
- `frontier`: useful diagnostic candidate that improves secondary/failure-mode
  signal without earning promotion.
- `confirmed_champion`: candidate confirmed against baseline variance.
- `invalidated`: result discarded because runner, logging, task, render, or
  budget evidence was compromised.

## Default Mode

Default mode is cold-start learning-process search.

Each ordinary pass trains a fresh policy from scratch under the fixed benchmark
budget. The point is to improve the training recipe: code structure, losses,
reward shaping, architecture, replay behavior, update cadence, exploration,
curriculum, vectorization, utilization, or another mechanism expressed in
training code.

The parent is candidate code, not a policy checkpoint.

Do not use checkpoint warm starts unless the user explicitly asks for a
warm-start diagnostic. In normal runs, do not pass `--init-checkpoint`,
`--init-from-run`, or `--init-from-best`.

## Session Invariants

- The benchmark, eval cases, success definition, environment, and runner are
  fixed during a session.
- Only edit files under
  `autoresearch_runs/sessions/<session-id>/candidates/`.
- Candidate files are authored by the model, one at a time.
- Scripts may run one already-authored candidate and collect artifacts.
- Scripts must not generate, mutate, queue, or prewrite `passNN_*.py` files.
- Pass 1 is always a verbatim copy of the selected seed trainable, usually
  `candidates/pass01_baseline.py`.
- Do not edit pass 1 code or `get_candidate()`.
- Later passes are bounded mutations from the chosen candidate-code parent.
- Ordinary passes should produce `lineage.mode = "from_scratch"` and
  `evolution.uses_warm_start = false`.

If a task, benchmark, runner, render, or logging bug is suspected, stop the
candidate loop and write the issue to `outer_loop_log.md`. Do not silently change
the benchmark to make a candidate look better.

## Candidate Contract

Every candidate must expose the same public contract as the seed:

- `get_candidate`
- `RewardRecipeWrapper`
- `train_agent`
- `save_agent_checkpoint`

If checkpoint rendering should work, preserve the established local pattern for
`Agent` and `load_agent_checkpoint(...)`.

`get_candidate()` is metadata for logs, summaries, and dashboards. It is not the
recipe. Put the recipe in executable code.

After pass 1, update `get_candidate()` whenever you mutate a candidate. The text
must name the hypothesis, changed mechanism, and expected effect.

Good examples:

```python
def get_candidate() -> str:
    return (
        "FetchPushDense SAC candidate testing goal-delta reward shaping during "
        "training. Keeps the official dense task reward for evaluation, but "
        "adds a training-only progress bonus to encourage earlier contact and "
        "object motion toward the desired goal."
    )
```

```python
def get_candidate() -> str:
    return (
        "Hopper SAC candidate testing a slower actor update schedule and wider "
        "critics. Hypothesis: delaying policy updates while increasing critic "
        "capacity improves early stability under the fixed budget."
    )
```

Prefer CleanRL-style self-contained trainables. The candidate file should expose
the actual recipe in readable code: networks, replay, optimizer setup, losses,
update cadence, reward transforms, logging, exploration, vectorization, and any
task-relevant mechanism.

## Before Starting

Read:

1. `README.md`
2. this file
3. the selected benchmark JSON
4. the selected seed trainable

Choose:

- budget style: episode budget, wall-clock budget, or task-specific benchmark
- seed trainable
- session label
- evidence mode: smoke, screening, confirmation, warm-start diagnostic, HPO, or
  PBT

The default evidence mode is screening. Do not choose a special mode such as HPO,
PBT, population search, or warm-start unless the user explicitly asks for it or
the session log records why the default serial loop is insufficient.

Budget choice matters. Use episode budgets for sample-count comparisons and
simple debugging. Use wall-clock budgets for utilization, vectorization, and
fixed-time learning-efficiency questions.

## Start A Session

Create the session:

```bash
autoresearch-gym init-session \
  --label <label> \
  --benchmark autoresearch_gym/tasks/<task_name>/<benchmark>.json \
  --seed-candidate autoresearch_gym/tasks/<task_name>/<seed_trainable>.py
```

`init-session` records the selected seed path and creates an empty
`candidates/` directory. It does not create pass files.

For pass 1, create `candidates/pass01_baseline.py` as a verbatim copy of the
selected seed trainable. Do not edit the copied file.

Run pass 1:

```bash
autoresearch-gym run \
  --benchmark autoresearch_gym/tasks/<task_name>/<benchmark>.json \
  --seed-candidate autoresearch_gym/tasks/<task_name>/<seed_trainable>.py \
  --session-dir autoresearch_runs/sessions/<session-id> \
  --candidate autoresearch_runs/sessions/<session-id>/candidates/pass01_baseline.py \
  --tag pass01-baseline
```

Unless the user explicitly asks for headless-only execution, share and verify the
dashboard URL:

```text
http://127.0.0.1:4174/dashboard/?session=autoresearch_runs/sessions/<session-id>
```

If the dashboard server is not running, start it with `autoresearch-gym
dashboard` and verify the URL is reachable.

For long runs, use compact status output so humans and coding agents can monitor
progress without parsing the final JSON summary or relying on live stderr
streaming:

```bash
autoresearch-gym run \
  --benchmark autoresearch_gym/tasks/<task_name>/<benchmark>.json \
  --session-dir autoresearch_runs/sessions/<session-id> \
  --candidate autoresearch_runs/sessions/<session-id>/candidates/passNN_<slug>.py \
  --tag passNN-<slug> \
  --compact-status \
  --compact-status-file autoresearch_runs/sessions/<session-id>/live/status.log
```

Tail `autoresearch_runs/sessions/<session-id>/live/status.log` for compact
progress lines. On wall-clock benchmarks, `pct` tracks elapsed time over
`train_seconds`; on episode-budget benchmarks, it tracks completed episodes over
`train_episodes`. Keep stdout reserved for the final JSON summary.

## Experiment Log

Each session must maintain:

`autoresearch_runs/sessions/<session-id>/outer_loop_log.md`

Update it before and after every pass. Most recent entries should appear at the
bottom.

For each pass, record:

- pass number / epoch
- stage: smoke, screening, confirmation, warm_start, or invalidated
- run id / tag
- lineage mode
- parent candidate file
- parent run id / checkpoint when warm-started
- search mode
- mutation family / proposal id
- mutation summary
- fixed benchmark path
- train budget and eval budget
- primary metric value
- relevant secondary metrics
- baseline variance context, if available
- decision: reject, frontier, screening_leader, confirmed_champion, invalidated
- interpretation
- next planned mutation or stop/pivot note

Write the hypothesis before editing the candidate. After the run, append the
actual result, decision, and interpretation from `summary.json`.

Do not rely on chat history, live dashboard impressions, or memory alone. The
session log is the durable research record.

Only hypotheses written in this session's `outer_loop_log.md` are part of the
active research state. If an idea came from outside the current session, import
it explicitly before treating it as eligible for execution.

## Baseline Variance Gate

For noisy, sparse-success, robotics, vectorized, or wall-clock sessions, run
baseline repeats before trusting candidate improvements.

Default:

- at least three baseline repeats for normal research sessions
- at least one baseline repeat before final summary for very short sessions
- more repeats when effects are small or results are intended as claims

Baseline repeats are measurement runs, not candidate mutations. They should not
spawn a queue of candidate ideas or delay the serial candidate loop beyond what
is needed to estimate noise.

Record raw baseline run ids, primary metric mean/median/std/min/max, eval return
mean/std, per-case success when fixed eval cases exist, and task-funnel metrics
such as contact, progress, goal eligibility, or success-after-contact.

If baseline repeats disagree sharply, mark one-run candidate wins as
unconfirmed unless they clear the observed noise band.

## Per-Pass Loop

For every later pass:

1. Inspect `results.jsonl`, `outer_loop_log.md`, and relevant `summary.json`.
2. Inspect `train_episodes.json` and `eval_episodes.json` when scalar metrics
   are ambiguous.
3. Inspect utilization for wall-clock, GPU, or vectorized work.
4. Summarize the current failure mode before choosing a mutation.
5. Write one bounded hypothesis in `outer_loop_log.md` before editing.
6. Create exactly one new `candidates/passNN_<slug>.py` from the chosen parent.
7. Make one coherent recipe change.
8. Update `get_candidate()` to describe that exact change.
9. Run the already-authored candidate under the fixed session contract.
10. Inspect the generated `summary.json`.
11. Append metrics, decision, interpretation, and next idea to
    `outer_loop_log.md`.
12. Revise the frontier, then choose exactly one next code parent.

Do not create pass `N + 1` while pass `N` is running. Do not create a queue of
candidates at session start. Do not run a full-loop script that both writes
candidates and executes them.

A tentative idea backlog is allowed in `outer_loop_log.md`, but it is a
scratchpad, not an execution queue.

When choosing the next pass, ignore memories and prior-session results unless
they were explicitly imported into this session log. Reason from the current
benchmark, current candidate code, and current session artifacts.

## Failure Funnel

When secondary metrics exist, diagnose the first failing stage before choosing
the next mutation. Examples:

- exploration reaches no useful states
- contact happens rarely
- contact happens but goal progress fails
- progress happens but success does not
- train reward improves but fixed eval regresses
- throughput improves but policy quality does not
- one eval case dominates failures

Choose mutations that target the failing stage. Do not keep sweeping unrelated
scalar knobs when the bottleneck is clear.

## Keep, Frontier, Reject

Use `screening_leader` when a candidate improves the primary metric in one
full-budget run.

Use `frontier` when a candidate does not win the primary metric but reveals a
useful diagnostic signal: better contact, lower variance, better worst-case
behavior, faster time-to-threshold, or a clearer failure mode.

Keep the frontier small and actionable. If many frontier candidates accumulate,
cluster them by failure mode and choose one parent only after the latest
completed run.

Reject a candidate if it crashes, regresses the primary metric, improves only
training reward, improves only throughput, depends on changed eval conditions, or
adds complexity without fixed-eval evidence.

Do not promote contact rate, train reward, dense return, or speed as a win when
the benchmark primary metric remains inside baseline noise.

If a candidate crashes from a trivial syntax, import, or shape mistake, repair it
at most twice. After two repair attempts, mark it failed and move on.

## Confirmation

Confirmation is serial and selective. Do not confirm every promising frontier
candidate.

During active exploration, prefer one new hypothesis test over confirmation if
recent passes are still producing new screening leaders or meaningfully different
frontier candidates.

Use confirmation when:

- iteration on the current frontier stops producing a new screening leader or a
  meaningfully different frontier candidate
- the next likely action is to stop, pivot, or promote a recipe
- a result is surprisingly strong and would change the direction of the session
- baseline variance is high enough that the current leader may be a lucky run
- the result will be used outside the session as a recipe claim

Default confirmation:

- select exactly one candidate to confirm
- usually choose the current `screening_leader`
- rerun the selected candidate and baseline on independent train seeds
- keep the same benchmark and budget
- use paired fixed eval cases when possible
- report raw runs, mean, median, std, min, max, and worst seed

Use five or more seeds, larger eval banks, or bootstrap intervals when effects
are small, variance is high, success is binary and eval count is low, many
candidates were searched, or the result will be used externally.

Do not launch confirmation reruns for multiple frontier candidates in parallel.
If several candidates look interesting, rank them in `outer_loop_log.md`,
confirm the top one first, then choose the next action from the new evidence.

Only confirmed candidates should be called champions.

## Budget Discipline

Use tiny smoke runs only to check that a candidate executes. Do not treat smoke
runs as evidence of learning.

Use the benchmark default budget for normal comparisons unless the user asks for
a different budget. Use larger confirmation runs only after a candidate becomes
a screening leader at the normal budget.

Always record train budget, eval budget, seed settings, benchmark path,
candidate path, and primary metric.

Never compare episode-budget and wall-clock-budget results as direct recipe wins
unless the log labels the comparison as budget-mismatched.

## Vectorized Training Discipline

More parallel environments are not automatically better learning.

For vectorized candidates, record:

- `num_envs`
- `env_steps`
- `completed_episodes`
- `gradient_updates`
- `updates_per_env_step`
- `updates_per_completed_episode`
- replay warmup fraction
- wall-clock throughput

If vectorization changes the update-to-data ratio materially, treat it as a
different training regime, not just a faster version of the same recipe.

## Stop Or Pivot

Stop the current mutation family and write a pivot note when:

- three related mutations fail to improve the screening leader beyond baseline
  variance or fail to improve confirmed primary performance
- five screening passes leave the primary metric flat inside variance
- confirmation fails to reproduce an apparent leader
- secondary metrics improve but the same bottleneck remains
- update-heavy variants reduce fresh data and regress eval quality
- reward shaping improves train reward but not fixed eval
- vectorization improves throughput but not learning
- evidence is invalidated by runner, logging, render, or benchmark issues
- the task reaches the metric ceiling

Stop rules are guardrails, not a replacement for judgment. If a rule triggers,
finish the current pass, inspect artifacts, then choose exactly one of:
confirmation, a mechanism-level pivot, or session stop.

If several serial passes explore the current frontier without producing a new
screening leader or a meaningfully different frontier candidate, pause mutation
and choose exactly one of: confirm the current screening leader, pivot to a new
mechanism family, or stop the session.

A pivot note must include:

- failed line of exploration
- evidence that it is exhausted
- current best confirmed candidate, if any
- current screening leader
- useful frontier candidates
- bottleneck diagnosis
- next mechanism-level direction

## Pivot Families

When scalar tuning stalls, pivot to a named mechanism-level hypothesis:

- exploration or state discovery
- replay strategy, HER, or contact-heavy replay
- curriculum over task phases
- reward design with transfer-risk checks
- observation or action-space redesign
- architecture or optimizer mechanism
- model-based rollouts or reusable world-model artifacts
- vectorization/update-ratio correction
- environment or benchmark audit

The pivot should explain why the new mechanism targets the observed bottleneck.
After a pivot, resume serial one-hypothesis passes; do not fan out across the
entire pivot menu.

## Special Modes

Explicit HPO, Hyperband, ASHA, PBT, warm-start, or population search sessions are
allowed only when labeled.

For these modes, log search space, resource allocation rule, population size or
trial count, inherited checkpoints, exploit/explore events, and total compute.

Special modes are not the default autoresearch loop. If the goal is a reusable
recipe, distill the discovered settings or schedule back into a cold-start
candidate and reconfirm it under the normal protocol.

## Validation Before Promotion

Before committing any change to a seed trainable, runner logging, dashboard
visualization, benchmark, or task environment, run the repo-level validation
gate, including artifact smoke tests:

```bash
.venv/bin/python scripts/pre_commit_checks.py
```

Session-local exploratory candidates do not need the full gate for every pass,
but package-code fixes do.

Remove generated caches, temporary checkpoints, logs, rendered media, and
personal run artifacts before publishing.

## Session Summary

End every session with:

- dashboard URL
- session path
- best confirmed candidate, if any
- current screening leader
- useful frontier candidates
- baseline variance summary
- failed mutation families
- stop or pivot decision
- artifact paths needed to inspect or reproduce the result
- whether any results were invalidated

When stopped, leave the session resumable: latest metrics written, current
candidate-code parent identified, failed ideas recorded, next plausible
hypothesis noted, and dashboard URL/session path available.
