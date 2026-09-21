# Source and claim ledger

Inspected 8 September 2026. This is an editorial evidence ledger, not an experimental log. Publication claims use papers, proceedings, author-maintained repositories, or official documentation. Software behavior is based on static source inspection; no runtime behavior was measured.

## Research revision, 15 September 2026

The revised manuscript omits implementation paths, release annotations, commit references, and the technical appendix. They remain here solely as editorial evidence. Publication descriptions are consolidated in Related Work. The earlier draft is archived under `build/`; it is superseded by the research-focused draft and revised experiment plan.

New static evidence for the MCP contribution:

| Claim | Local evidence | Qualification |
| --- | --- | --- |
| Common observation/action tools and selective visual memory | `experiments/harbor_vizdoom/template/common/environment/vizdoom-server/server.py` | Four tools: `get_current_state`, `make_action`, `store_state`, `get_stored_state`; memory retrieves explicitly saved observations, not arbitrary past states or world-state checkpoints |
| Images and structured feedback, optional automap | Same server, observation-content and metadata methods | Current responses do not forward core audio or notification buffers; game variables and reward feedback mean the default is not a strict pixels-only track |
| Existing task coverage | `experiments/harbor_vizdoom/environments.json` | Basic, My Way Home, Doom E1M1/E1M2, Doom II MAP01/MAP02; full-level tasks request automap and easiest skill setting |
| Native harness integration | `experiments/harbor_vizdoom/run_experiment.py` and task templates | Codex and Claude Code via Harbor; model identity is configurable; source support is not measured compatibility or a frontier-model ranking |
| Existing task grader | `experiments/harbor_vizdoom/template/common/tests/grade.py` | Outcome should be based on game events; current generic positive-return checks need task-specific revision before expanding to dense-reward/survival scenarios |
| Standard scenario pool | `docs/environments/default.md` and `experiments/vizdoom_env_descriptions.json` | Twelve named scenarios; Basic Audio and Basic Notifications require the appropriate observation channel |

Two implementation inconsistencies were recorded for future work rather than changed: server frame skip/repeat bounds differ from documentation and grader assumptions; cumulative positive reward is used as generic success. The exact observations and remediation steps are in `experiment_plan.md`. No runtime behavior was tested, no existing logs were scored, and no experiment source files were changed by the paper revision.

Bibliographic links for MazeExplorer and LevDoom now point to the [conference paper](https://ieee-cog.org/2020/papers2019/paper_210.pdf) and [IEEE publication record](https://ieeexplore.ieee.org/document/9893707/), respectively, rather than source-code repositories.

Protocol/framework background uses official documentation: [MCP introduction](https://modelcontextprotocol.io/docs/getting-started/intro) and [Harbor documentation](https://www.harborframework.com/docs), accessed 15 September 2026. They support the roles of the protocol and evaluation framework; local static inspection supports the specific ViZDoom implementation claims.

## Pinned repository evidence

- Mainline: `origin/main` at `02587a44999a735b920b1736461b8f99e6aef6be`, dated 2026-09-04. [Immutable source tree](https://github.com/Farama-Foundation/ViZDoom/tree/02587a44999a735b920b1736461b8f99e6aef6be).
- PettingZoo: `origin/pettingzoo-wrapper` at `08281b4053ab9fe38beada7edef6dff4347a73db`, dated 2026-09-04. [Immutable wrapper tree](https://github.com/Farama-Foundation/ViZDoom/tree/08281b4053ab9fe38beada7edef6dff4347a73db/pettingzoo_wrapper).
- A second locally available ref, `origin/petting-zoo-wrapper-dev`, points to `9676c43aeae93049538c0ab98c8c654d2699ef70` (2026-03-04). The paper uses the newer `pettingzoo-wrapper` snapshot. No literal branch named `pettingzoo` was present in the local branch listing.
- The active workspace branch was `harbor-vizdoom`, HEAD `c28e4838`, with existing unrelated work. The inspected mainline source paths were checked against `origin/main` and had no differences. The branch was not changed and existing experiment files were not used as evidence of new results.
- `git describe` identified mainline as `1.3.0-50-g02587a44`. This is a repository description, not a package-release lookup.

| Manuscript claim | Source at mainline snapshot, unless marked branch | Evidence and qualification |
| --- | --- | --- |
| Audio added in documented version 1.1.9 | `docs/api/cpp/game_state.md:26`; `docs/api/cpp/doom_game.md`, audio methods | Checked-in version annotation; implementation history includes audio merge `2a6b890e` (2021-08-19) |
| Signed 16-bit stereo audio | `include/ViZDoomTypes.h`; `src/lib_python/ViZDoomGamePython.cpp:304`; `examples/python/audio_buffer.py` | Python shape is samples-per-tic times window length, by two channels; configurable rates and OpenAL requirement |
| Gymnasium audio/text windows follow frame skip by default | `gymnasium_wrapper/base_gymnasium_env.py:154` | Explicit kwargs can override both window sizes |
| Text is `notifications_buffer` | `docs/api/cpp/game_state.md:27`; `include/ViZDoomTypes.h`; `src/lib_python/ViZDoomPythonModule.cpp` | Documented since 1.3.0; a string of notifications, not a generated scene description |
| Notification storage is bounded and windowed | `src/vizdoom/src/viz_game.cpp:665`; `src/vizdoom/src/viz_defines.h:49` | Collection uses game tics; shared text array limit is 32,768 bytes including string-storage overhead/terminator |
| Label semantic categories | `docs/api/cpp/game_state.md:75`; `src/vizdoom/src/viz_game.cpp:250`; `src/vizdoom/src/viz_doom_classes.h` | `object_category` documented since 1.3.0; ancestry/mapping lookup, Self/Gore cases, Unknown fallback; custom-WAD caveat |
| Category masks need an adapter | `include/ViZDoomTypes.h`; `gymnasium_wrapper/base_gymnasium_env.py:298` | Label pixels store instance values. Categories are metadata joined using `Label.value`; Gymnasium does not automatically turn these into category pixels |
| Global objects also have categories and sector IDs | `docs/api/cpp/game_state.md:112`; `include/ViZDoomTypes.h` | Documented with a 1.4.0 annotation; described as inspected-source functionality, not a verified release |
| Full-level task configurations | `scenarios/doom.cfg`, `doom2.cfg`, `freedoom1.cfg`, `freedoom2.cfg` | Include USE/weapon controls, timeout 126000, and exit reward 1; audio and automap are enabled by default |
| Per-map Gymnasium registrations | `gymnasium_wrapper/__init__.py:263` | Four game/asset families, map names, skill settings; no claim of persistent campaign integration |
| Full-map loading predates this update | `git show 1.1.5:include/ViZDoomGame.h`, lines 129/131 | Historical `setDoomGamePath` and `setDoomMap` declarations |
| Reward is configurable; sparse setting needs audit | `src/lib/ViZDoomGame.cpp:39` and `:204` | Built-in coefficient defaults and `updateReward`; map/script reward may contribute independently |
| Gymnasium terminal semantics need validation | `gymnasium_wrapper/base_gymnasium_env.py:226` | `terminated` and `truncated` are computed independently; timeout flag depends on wrapper option |
| Original commercial graphics are not bundled | Repository `README.md`, Original Doom graphics | Prefer Freedoom for redistributable experiments; original WADs need separate provisioning |
| PettingZoo Parallel API | Branch `pettingzoo_wrapper/base_env_common.py:110`; factory `__init__.py` | Derives from `ParallelEnv`; runtime conformance not tested for this manuscript |
| PettingZoo RGB observation space | Branch `pettingzoo_wrapper/base_env_common.py:185` | RGB or stacked RGB, not a core-modality dictionary |
| PettingZoo `state()` is concatenated views | Branch `pettingzoo_wrapper/base_env_common.py:231` | Not an omniscient simulator state |
| PettingZoo scenarios and reward wrappers | Branch `pettingzoo_wrapper/__init__.py` | Duel, hide-and-seek, health gathering, pitfall, remedy rush; reward semantics remain scenario-dependent |
| Branch tests exist, not executed | Branch `tests/test_pettingzoo_wrapper.py` | Contains `parallel_api_test` and deterministic-seed checks |
| Branch portability/recovery needs review | Branch `pettingzoo_wrapper/base_pettingzoo_env.py` | `mp.get_context("fork")`, timeout/recovery paths and reset-settling logic |

## Selected implementation milestones

These dates come from local git history. They identify changes, not release dates. Short hashes resolve in this repository.

| Commit | Date | Recorded change |
| --- | --- | --- |
| `2a6b890e` | 2021-08-19 | Merge of the audio implementation |
| `f82d8c0f` | 2023-03-13 | Gymnasium integration merge |
| `b5daa486` | 2025-08-10 | Category as a label attribute |
| `757c5b98` | 2025-08-21 | Audio in Gymnasium observations |
| `334a3afd` | 2025-09-01 | Basic notification buffer functionality |
| `39bba594` | 2025-11-28 | Notification buffer in Gymnasium |
| `32c52467` | 2026-02-02 | Audio backend reset for episode determinism |
| `094fe404` | 2026-02-03 | Original Doom levels as Gymnasium environments |
| `6e655072` | 2026-02-06 | Notification window fix for frame skip greater than one |
| `2b15e0cf` | 2026-06-14 | Additional object/sector category and ID metadata |

The determinism-related commit records an implementation change; it is not new cross-platform determinism evidence from this paper.

## Historical reference

The [competition preprint](https://arxiv.org/abs/1809.03470) was submitted 10 September 2018. The [full paper](https://arxiv.org/pdf/1809.03470), section II and conclusion, establishes version 1.1.5, existing observation features, and the motivation for full-level challenges. The [journal DOI](https://doi.org/10.1109/TG.2018.2877047) and repository citation identify IEEE Transactions on Games 11(3), 248-259, 2019. The bibliography therefore uses 2019 while the narrative explains the 2018 reference point.

## Publications with verified ViZDoom use

This is a selective relevance-based list, not a citation ranking. Descriptions are intentionally brief; the manuscript explains their connections.

| BibTeX key | Publication and primary verification | Verified use |
| --- | --- | --- |
| `kempka2016vizdoom` | [ViZDoom, CIG 2016](https://arxiv.org/abs/1605.02097) | Original platform and learning examples |
| `wydmuch2019competitions` | [ViZDoom Competitions](https://arxiv.org/abs/1809.03470) | Platform and multiplayer competition analysis |
| `dosovitskiy2017future` | [Learning to Act by Predicting the Future](https://arxiv.org/abs/1611.01779); [author-hosted ICLR 2017 paper](https://vladlen.info/papers/learning-to-act.pdf) | Prediction-based control on Doom tasks |
| `lample2017fps` | [Playing FPS Games with Deep Reinforcement Learning](https://arxiv.org/abs/1609.05521); [author-hosted AAAI paper](https://devendrachaplot.github.io/papers/aaai17_fps_games.pdf) | Recurrent visual combat and game-feature prediction |
| `pathak2017curiosity` | [ICML proceedings](https://proceedings.mlr.press/v70/pathak17a.html) | Intrinsic curiosity evaluated in ViZDoom |
| `ha2018worldmodels` | [Author's interactive paper](https://worldmodels.github.io/); [arXiv metadata](https://arxiv.org/abs/1803.10122) | ViZDoom Take Cover; cites the World Models preprint, not a different conference title |
| `savinov2019episodic` | [Episodic Curiosity through Reachability](https://arxiv.org/abs/1810.02274) | Explicit ViZDoom navigation experiments; ICLR 2019 acceptance stated |
| `harries2019mazeexplorer` | [Microsoft author repository and citation](https://github.com/microsoft/MazeExplorer) | ViZDoom-based procedural maze benchmark, CoG 2019 |
| `petrenko2020samplefactory` | [ICML proceedings](https://proceedings.mlr.press/v119/petrenko20a.html); [paper PDF](https://proceedings.mlr.press/v119/petrenko20a/petrenko20a.pdf) | Doom training and multiplayer self-play; reported speed is specific to that system |
| `tomilin2022levdoom` | [Author repository](https://github.com/TTomilin/LevDoom); [author university publication record](https://research.tue.nl/en/publications/levdoom-a-benchmark-for-generalization-on-level-difficulty-in-rei/) | Doom generalization; author list, CoG venue, pages, DOI checked |
| `tomilin2023coom` | [NeurIPS 2023 proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d61d9f4fe4357296cb658795fd7999f0-Abstract-Datasets_and_Benchmarks.html); [paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/d61d9f4fe4357296cb658795fd7999f0-Paper-Datasets_and_Benchmarks.pdf) | ViZDoom-based continual learning, visual and task changes |
| `tomilin2025hasard` | [HASARD paper record](https://arxiv.org/abs/2503.08241) | Doom-based safe RL; record states accepted to ICLR 2025 |

## Context and methodology, not claimed as ViZDoom applications

- [DreamerV3, Nature 2025](https://www.nature.com/articles/s41586-025-08744-2): broad-domain world-model learning; relevant candidate method and framing.
- [BALROG](https://arxiv.org/abs/2411.13543), [conference PDF](https://arxiv.org/pdf/2411.13543): interactive LLM/VLM evaluation; PDF confirms ICLR 2025. OpenReview's forum required browser verification, so the primary arXiv conference copy was used.
- [SIMA 2](https://arxiv.org/abs/2512.04797): generalist embodied agents in virtual worlds; December 2025 technical report. Its relevance to ViZDoom is this manuscript's inference, not a reported SIMA 2/ViZDoom experiment.
- [Procgen, ICML 2020](https://proceedings.mlr.press/v119/cobbe20a.html): distributional evaluation and generalization.
- [PettingZoo, NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/803f7c4c3ff61b71be53a0c803bfb57f-Abstract.html) and [official Parallel API documentation](https://pettingzoo.farama.org/api/parallel/): interface concepts; source branch determines the actual ViZDoom factory and observation contract.
- [Statistical Precipice, 2021](https://arxiv.org/abs/2108.13264): uncertainty-aware RL evaluation. Specific proposed sample counts and hierarchical analysis choices are our protocol design, not facts copied from that paper.

UNREAL and unrelated Atari, DMLab, Minecraft, or robotics work were not automatically included as ViZDoom applications. EnvPool was considered as ecosystem context but omitted from the selected application table rather than inferring experimental use from software support.

## Claims deliberately left for future evidence

- New throughput, sample efficiency, task success, or comparisons with human performance.
- Whether audio, text, or categories improve any selected policy.
- Whether current generalist agents can complete unseen full levels.
- Cross-platform or multi-agent determinism, reliable branch operation, or release readiness.
- Native PettingZoo audio/text observations, a persistent campaign benchmark, or a scene-captioning API.
- General intelligence or real-world safety inferred from Doom performance.

## Document validation

The deliverable is validated by LaTeX/BibTeX compilation, citation/reference checks, source-path checks, and visual review of the compiled pages. Runtime ViZDoom tests would launch game instances and were left to the prospective plan. This documentation-only task did not modify library APIs or require regenerating Python bindings/docstrings.
