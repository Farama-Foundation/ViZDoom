# ViZDoom research paper

Venue-neutral LaTeX draft: **ViZDoom Beyond Pixels: A Testbed for Multimodal and Generalist Agents**.

The manuscript emphasizes research questions and contributions: audio and text observations, semantic categories, full-level evaluation, PettingZoo integration, and an MCP server for language-model agents. Descriptions of selected prior publications are consolidated in Related Work. Release versions, configuration paths, source-code citations, and the technical audit appendix have been removed from the compiled paper.

The Results section reserves four tables for frontier models and native harnesses, twelve standard ViZDoom environments, the first two levels of Doom and Doom II, and a fixed-model memory/planning ablation. Hangar and Entryway form the primary full-level comparison; Nuclear Plant and Underhalls extend it. All measurements remain pending.

No experiments, gameplay evaluations, model calls, training, runtime tests, or analysis of existing experiment logs were performed for this revision. Only the paper was edited, compiled, and visually checked.

## Files

- `main.tex`: manuscript entry point.
- `sections/`: introduction, related work, research extensions, MCP interface, experimental design, reserved results, and discussion.
- `sections/06_results.tex`: editable results templates; replace model slots and pending cells after evaluation.
- `references.bib`: research literature and protocol/framework background references.
- `paper.pdf`: compiled reading copy.
- `experiment_plan.md`: prospective implementation, evaluation, budget, and analysis plan; not included in the manuscript.
- `sources.md`: editorial claim ledger and static implementation evidence; not included in the manuscript.
- `build/draft-before-research-revision.zip`: local archive of the previous draft, excluded from version control with other build files.

## Build

From the repository root:

```sh
make -C paper
```

Requires a LaTeX distribution with `latexmk`, pdfLaTeX, BibTeX, and the packages declared in `main.tex`. Intermediate files are written to `build/`, and the reading copy is `paper.pdf`. Building the paper does not import ViZDoom or start any experiments.

## Before submission

Choose the venue, author list, and final title. Select and name the frontier models at evaluation freeze, finalize budgets using a validation-only pilot, complete the task integrations and grading checks in the experiment plan, and then conduct the authorized evaluation. Replace the prospective Results section with measured outcomes and update the abstract and conclusions to match the evidence. Keep all comparisons consistent about automap access, observation channels, control cadence, and model-versus-harness attribution.
