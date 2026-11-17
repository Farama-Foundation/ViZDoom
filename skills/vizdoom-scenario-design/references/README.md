# Reference pack (pinned to ViZDoom's ZDoom 2.8.1 engine)

ViZDoom embeds a fork of **ZDoom 2.8.1** (`src/vizdoom/src/version.h`: `ZDOOM 2.8.1+`).
The ZDoom wiki (<https://zdoom.org/wiki>) documents the *current* GZDoom, and the ACC 1.60
compiler (`acc`) accepts functions from GZDoom, Zandronum, Eternity and ZDaemon. Neither
source tells you what works in ViZDoom. The files below do, because they are derived from
the engine sources that ViZDoom actually runs.

| File | Content | How it was made |
|---|---|---|
| `acs_functions.md` | ACS built-ins, statements and extension functions that ACC declares **and** the engine implements | generated |
| `acs_unsupported.md` | Functions and specials that compile but do nothing, return 0, or stop the script | generated |
| `line_specials.md` | All line specials (number, ACS/line arguments, whether they work) | generated |
| `actor_classes.md` | DECORATE classes, parents, Doom 2 editor numbers (DoomEdNum) and spawn numbers | generated |
| `vizdoom_scenario_api.md` | How a WAD talks to ViZDoom: reward, `USER` variables, `.cfg` keys, multiplayer | hand-written from `docs/` and `src/lib/` |
| `tooling.md` | DoomGen (COMRAD), raw WAD/UDMF patching, omgifol, `acc`, SLADE / Ultimate Doom Builder | hand-written |
| `zdoom_wiki_index.md` | Curated wiki links, each tagged *works in ViZDoom* or *not in ViZDoom* | hand-written, checked against the generated files |

## Rules for agents

1. Before using an ACS function, find it in `acs_functions.md` or (for line specials)
   in `line_specials.md` with `Works = yes`. If it is in neither, do not use it.
2. Before writing `ACTOR X : Parent`, check that `Parent` is in `actor_classes.md`.
   A missing parent stops the engine while loading actors (e.g. `PowerupNone`).
3. Use the wiki only for semantics of functions that passed rule 1, and read the
   version notes on the page: features marked as added in GZDoom, or anything in
   ZScript, are not available.

## Regenerating

The generated files carry a provenance header (engine version, ACC version, source
paths and SHA-256 prefixes). Regenerate them after changing the engine or the
compiler; the output is deterministic, so an unchanged engine gives identical files:

```sh
python skills/vizdoom-scenario-design/scripts/build_reference_pack.py --vizdoom-root . --acc-dir DoomGen2/acc
```

`--acc-dir` must contain the `zspecial.acs` that ships with the `acc` binary you use
(COMRAD ships both in `DoomGen/acc/`). The p-code built-ins hard-coded in ACC are
vendored in `scripts/data/acc_builtins.tsv` from <https://github.com/rheit/acc/tree/1.60>;
re-extract them with `--acc-src /path/to/acc-checkout` if you change compilers.
