# Prospective evaluation plan

Status: planning only. No game episodes, model calls, training, runtime tests, or analysis of existing experiment logs were performed for this revision. This companion file holds operational detail outside the research manuscript.

## Research priorities

The first study evaluates frontier models and harnesses on standard ViZDoom scenarios and the opening levels of Doom and Doom II. It separates model capability, harness design, and complete agent systems. Audio/text grounding, semantic supervision, transfer to new maps, and multiplayer are follow-up studies. The former draft's broad training matrix is superseded by this staged priority; it is preserved in the draft archive.

Primary questions:

1. How do frontier models differ under a common interaction interface?
2. Within one fixed model, do selective visual memory and planning change completion and cost?
3. Does competence on compact skills translate into complete-level behavior?
4. How much does automap access change opening-level performance?

The registered primary harness contrast will be memory versus no explicit memory. Planning and automap interventions will be secondary contrasts. Between-model comparisons describe the selected models at the evaluation freeze, not a timeless ranking. Any claim about generalization requires an additional held-out map study.

## Task coverage and outcomes

| Task | Current MCP task entry | Planned primary measure |
| --- | --- | --- |
| Basic | Present | Target elimination success; also native return |
| Basic Audio | To integrate | Target elimination success with declared audio access |
| Basic Notifications | To integrate | Correct-target success with declared notification access |
| Deadly Corridor | To integrate | Goal-reaching success; also native return |
| Deathmatch | To integrate | Episode return and kills |
| Defend the Center | To integrate | Episode return and kills |
| Defend the Line | To integrate | Episode return and kills |
| Health Gathering | To integrate | Survival duration in simulated seconds |
| Health Gathering Supreme | To integrate | Survival duration in simulated seconds |
| My Way Home | Present | Goal-reaching success |
| Predict Position | To integrate | Target elimination success; also native return |
| Take Cover | To integrate | Survival duration in simulated seconds |
| Doom E1M1: Hangar | Present, primary full level | Valid level exit |
| Doom II MAP01: Entryway | Present, primary full level | Valid level exit |
| Doom E1M2: Nuclear Plant | Present, extended full level | Valid level exit |
| Doom II MAP02: Underhalls | Present, extended full level | Valid level exit |

The standard Deathmatch scenario is a single-player combat task; it should not be confused with the separate multiplayer competition study. The primary standard finite-goal aggregate includes Basic, My Way Home, Deadly Corridor, and Predict Position, with equal task weights. Do not average native returns across unrelated objectives. Report sensory tasks separately until each compared model actually receives the required channel.

The reference full-level condition retains the existing automap assistance and easiest skill setting; freeze the exact setting before evaluation. A paired intervention removes the automap while holding all other observation privileges constant. Inspect each map's actual required interactions, dependency chain, and backtracking before assigning a difficulty interpretation. First-level success alone is insufficient evidence of long-horizon planning.

Original Doom assets must be supplied by the evaluator; publish identifiers and integrity hashes rather than bundling those game assets. Plan a separate redistributable and held-out map pool for transfer. Levels are independent episodes, not persistent campaigns.

## Models and harnesses

- C1-C3: at least three frontier vision-language model slots in a common harness. Select actual models across multiple families at evaluation freeze and record exact identifiers, access dates, inference settings, and whether the provider offers a pinned revision. No claim about which models are currently frontier is encoded in the paper's placeholders.
- N1: a supported model with the Codex harness through Harbor.
- N2: a supported model with the Claude Code harness through Harbor.
- Within-model ablation: bounded-history common harness; the same harness plus selective visual memory; the same harness plus memory and structured planning.
- Add a direct native-versus-common harness contrast only when the same model is supported in both and inference settings can be aligned. Otherwise report a complete-system comparison.
- Freeze prompts and harness choices using separate validation content. Do not tune on the named final evaluation levels or recycle test traces into prompts.

The common harness must enforce the same observations and tools across models. Native harnesses may add their own history management, which is part of the system being evaluated. Keep web access and unrelated external tools disabled in the benchmark condition. Do not use local source files, level solutions, or evaluator diagnostics as agent inputs. User-facing task instructions should state the objective and available controls without supplying a walkthrough.

Memory conditions must hold ordinary conversational history constant. Disabling the explicit store is insufficient if all prior images or a hidden persistent store remain available. Freeze history truncation, image retention, textual summaries, memory capacity, and the reset policy. Evaluate whether stored observations are retrieved and followed by behavior relevant to the task, not just whether memory tools are called.

## Pre-evaluation implementation and validation work

These items are planned tasks, not completed fixes or checks. Existing experiment files were only read for the manuscript.

1. Resolve an observed timing-contract discrepancy: the current server sets `FRAME_SKIP = 2` and `MAX_ACTION_REPEAT = 4`, while its tool docstring, task documentation, and grader still refer to frame skip 4 or repeat bounds up to 8. Freeze one supported contract and align instructions, validation, grading, and logging before any pilot.
2. Replace the current generic success heuristic with task-specific event predicates. `Episode.act` currently uses cumulative `episode_return > 0` to set success, and the grader also requires a positive return threshold. Positive reward is not completion in a dense-reward or survival task; a valid delayed goal with penalties could also have nonpositive return. Validate exit, target elimination, correct target, goal pickup, death, and time-limit outcomes independently. Survival and combat tasks need continuous performance measures.
3. Count model calls, tool calls, environment steps, and game tics separately. The current action history records each repeated environment step, so its decision count is not a count of MCP calls. Record early termination within repeated actions and include observation-only calls in interaction accounting.
4. Add task entries and objective-specific grading for the standard scenarios absent from the existing six-task MCP manifest. Validate available controls and enough action budget to achieve each task with a reference controller.
5. Audit what the model actually receives. Current MCP responses include screen images and structured task/episode information; full-level entries request the automap. Rewards and game variables can be privileged information. Whitelist observation fields, and keep evaluator-only map coverage, milestones, and future information inaccessible to the agent.
6. Add audio and notification payloads before claiming an MCP audiovisual or text-buffer experiment. Core modality support does not imply MCP support. Do not substitute a generated scene description or transcript without labeling the extra model and information transformation. Match auditory and notification windows to action advancement so events are not silently omitted.
7. Verify explicit store/retrieve behavior, bounded capacity, identifier replacement, reset clearing, and prevention of arbitrary historical-state access. A stored snapshot is a memory, not a simulation checkpoint or world-state restore operation.
8. Validate each full-level reward as exit-only and check nonstandard endings. Preserve native reward definitions for the standard scenarios; log auxiliary diagnostics separately.
9. Validate reset, termination, timeout/truncation, model-call errors, and reproducibility of fixed action traces on the selected platform. Set deterministic rerun rules for infrastructure failures; never silently remove failures that lower a score.
10. Only after the user authorizes experimental execution, run small pilots on separate validation content to confirm feasibility, estimate variance, and cost the final matrix.

## Sample sizes, resource budgets, and staging

Initial sample plan: 30 paired environment seeds per task and inference condition. Assign any stochastic inference seed in advance; use independent model sampling when inference seeds cannot be controlled. Thirty episodes are a starting design choice, not a power guarantee. Five independent training seeds are planned for each learned RL baseline, with the same episode suite evaluated for each seed.

Proposed staged matrix:

| Stage | Matrix | Initial evaluation episode count |
| --- | --- | ---: |
| Core frontier/system study | 5 conditions x 6 tasks x 30 episodes | 900 |
| Additional standard nonsensory tasks | 5 conditions x 6 remaining nonsensory tasks x 30 episodes | 900 |
| Extended opening levels | 5 conditions x 2 levels x 30 episodes | 300 |
| Memory/planning ablations | 2 added harness conditions x 6 primary tasks x 30 episodes | 360 |
| Automap intervention | 3 common-harness models x 2 primary levels x 30 episodes | 180 |
| Sensory tasks | At most 5 supported conditions x 2 tasks x 30 episodes | Up to 300 |

The six primary tasks are the four standard finite-goal tasks plus Hangar and Entryway. Six additional nonsensory tasks bring the standard nonsensory total to ten. The two sensory tasks bring the full standard suite to twelve. An existing core run may serve as the no-memory ablation baseline only if its harness and all other conditions are identical. Additional matched native-harness contrasts and repeated inference runs increase these totals and must be separately costed.

Initial frontier-model ceiling per episode: 1,000 model calls and 256,000 generated tokens, with a separate elapsed-time and simulated-time limit chosen per task using validation pilots. These are prospective ceilings, not current implementation guarantees. Count input tokens and provider-specific reasoning tokens as reported, including cached-token pricing and billed failures. Record unreported token categories as unavailable. Model, tool, game-time, and wall-time ceilings may bind at different points; record the first binding limit.

Do not simply reuse the current action-step budget as a model-call ceiling. Before freezing the protocol, check that the selected action duration menu and call allowance permit the intended horizon on every level. Short fixed controls can otherwise make completion impossible regardless of model capability. A matched RL comparison must use the same control cadence; report native high-frequency RL separately.

Budget reactive and recurrent RL baselines independently. Provisional training ceilings are 20 million game tics per policy and training seed for a standard-task policy, and 100 million for a full-level policy. Final scope and policy sharing must be decided before training: multiplying these budgets by all tasks can be substantially more expensive than joint training. Publish both task-specific resources and aggregate resources; pretraining of frontier models is not equated with this training budget.

Before execution, cost the selected matrix using measured validation inference and environment throughput. Freeze a total monetary and compute ceiling as well as per-episode limits. If the full plan exceeds the available budget, narrow it before final evaluation and describe the smaller scope. No paid call, pilot, or training run is authorized by this planning document.

## Analysis plan and reserved manuscript outputs

1. Report per-task performance first. Average success equally over the four standard finite-goal tasks and, separately, the two primary full levels. Use the same six-task equal-weight aggregation for inference cost, calls, and retrieval counts in the overview tables.
2. Give 95% intervals and paired differences. For binary outcomes include intervals that remain informative when all episodes fail or all succeed; an all-zero bootstrap is not zero uncertainty. Preserve paired seeds across conditions. For trained policies account for training-seed clustering.
3. The four fixed opening maps support uncertainty conditional on those maps. Resampling a tiny fixed set does not establish transfer to an arbitrary map distribution. A population-level claim needs a separately designed held-out map study.
4. Show completion and survival against resource budgets. Plot success-conditioned completion time only alongside the success probability and failure distribution. Report infrastructure failure, death, task failure, and budget exhaustion separately.
5. Assess memory effects through the prespecified within-model contrast, retrieval traces, and changes in subsequent actions. Planning and automap analyses are secondary; report effect sizes and use a declared multiplicity rule for any family of significance tests.
6. Define the failure taxonomy before final trace review. Double-code a subset and report agreement. Do not infer private reasoning from short agent-written decision notes.
7. Replace C1-C3 and N1-N2 placeholders with actual selected systems. Populate `sections/06_results.tex` only with validated measurements, not estimates or predictions. Pending dashes do not mean zero or not applicable; unsupported pairings need a separate label.

Reserved tables cover model/system comparisons, all twelve standard environments, four opening levels, and fixed-model harness ablations. Later figures should show success-versus-cost, time-to-completion with failures, and behavior around memory retrieval. No synthetic performance plots are included.

## Follow-up research

After the initial study, add controlled audio/text cue ablations; compare semantic categories as training supervision and privileged input; evaluate transfer to independently authored layouts; and run multiplayer opponent-transfer experiments through the PettingZoo branch. For multiplayer, validate simultaneous stepping, player-slot symmetry, recovery/cleanup, and the supported observations before training. Report cross-play and held-out opponents rather than only a self-play training score.

The paper directory contains no experiment launcher or generated empirical data. Technical evidence is retained in `sources.md` for editorial verification, outside the compiled manuscript.
