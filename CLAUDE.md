# music-cluster

Sorts new music (and samples) into groups the user already keeps. The user
defines the groups; the tool scores, the user confirms, then it files. It never
invents a category or touches a file the user has not confirmed. README.md
explains the product, INSTALL.md covers setup and troubleshooting, EVALUATION.md
the eval harness, docs/cli.md every command.

Python package and CLI in `music_cluster/` (FastAPI + SQLite), SvelteKit +
Tailwind UI in `ui/`, Tauri 1 desktop shell in `src-tauri/`. The desktop app
runs a frozen copy of the API: `music_cluster/server.py` is its entry point,
built by `scripts/build_sidecar.py`.

## Never

- Never move, copy or delete user files without an explicit confirmation step.
  Every filing mode in `organizer.py` has a dry run and an undo; a new mode
  ships with both.
- Never turn a tag or sample category into a trained model: they are readable
  rules over measured quantities (`tagging.py`, `samples.py`, `rhythm.py`), so a
  user can see why a track got its tag.
- Never change a collection's profile (`music` vs `sample`) after creation; the
  stored features depend on it.
- Never hand-edit or regenerate `package-lock.json` (root and `ui/`) or
  `src-tauri/Cargo.lock` outside a dependency change; install with `npm ci`.
- Never run the CLI or API by hand against the default library
  (`~/.music-cluster/library.db`): it is the owner's real one. Point
  `MUSIC_CLUSTER_DB`, `MUSIC_CLUSTER_CONFIG` and `MUSIC_CLUSTER_CACHE` at a temp
  directory, as `tests/e2e/conftest.py` does.

## Commands

```bash
# Setup (CI runs Python 3.10–3.12; librosa/numba may lag newer Pythons)
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements-dev.txt && uv pip install -e .   # uv venvs have no pip
npm ci && (cd ui && npm ci)       # root holds the Tauri CLI; start-dev.py would npm install (and rewrite the lock) if either is missing

# Run
python start-dev.py              # API on :8000 + UI on http://localhost:1420
python start-dev.py --tauri      # desktop shell

# Python checks
ruff format . && ruff check --fix . && mypy
pytest tests/unit                # fast
pytest tests/e2e tests/integration
music-cluster eval run --baseline eval/baselines/synthetic.json   # how well, not just whether

# UI checks (from ui/)
npm run format:check && npm run lint && npm run check
npm test
npx playwright test              # seeds its own library via scripts/seed_e2e_library.py
npm run ui:snapshot              # not a gate: screenshots + a11y text of every screen → ui-snapshots/report.md

# Desktop shell (from src-tauri/, only if you touched Rust). clippy needs the
# sidecar binary and ui/build to exist: create the stubs as lint.yml does.
cargo fmt --all -- --check && cargo clippy --all-targets -- -D warnings
```

FFmpeg must be installed for any audio decoding. CI (`.github/workflows/`) runs
the same checks (ruff without `--fix`); keep them passing before opening a PR.

## How to work here

Build what the task needs now, in a shape that is cheap to change later.

- Solve the stated requirement. No speculative features, flags, config keys or
  extension points. Ideas for later go in the PR description or an issue, not
  in code.
- Before non-trivial work, state your assumptions and the simplest approach
  that would work. If the ask can be read two ways, ask rather than guess.
- Keep diffs surgical: every changed line traces to the task. A small refactor
  or rename is fine in files the task already changes, in its own commit and
  listed in the PR description; anything bigger, or elsewhere, gets proposed
  first. Never reformat code you did not otherwise touch.
- Reuse the existing pattern instead of adding a second way to do the same
  thing (see "Where code goes"). If two patterns compete and you can't tell
  which is canonical, ask.
- Extensibility comes from small functions, clear names and dependencies
  passed in, not from base classes, registries or plugin systems built ahead of
  need. Abstract on the third repetition, not the first.
- Minimise scope, never correctness: input validation at the API and CLI
  boundary, error handling, and tests still ship.
- No new dependency without saying why in the PR and checking first. Large
  dependencies or models need Jules's approval. A runtime Python dependency
  goes in `requirements.txt` and the matching `setup.py` list
  (`install_requires`, or the `api`/`llm` extra), and in `COLLECT_ALL` or
  `HIDDEN_IMPORTS` in `scripts/build_sidecar.py` if PyInstaller misses it (only
  `build.yml` catches that). Test tools go in `requirements-dev.txt`.
- Done means verified: name the test, command or eval number that proves it,
  and run it before you finish.

## Writing

Applies to code comments, docs, commit messages, PR descriptions and messages
to Jules.

- Plain, natural language. No filler, clichés or AI-sounding phrases, and no
  em-dashes. Say it once.
- Point form over long paragraphs; short headings when a doc covers more than
  one topic.
- Avoid jargon. A needed technical term is defined the first time it appears
  and then used consistently. One name for one thing across code, comments,
  docs and the UI.
- Write for a reader with no context from past sessions: give the path, link,
  issue or doc they need to act.
- Code, comments and docs describe how things are now. No "previously", "as
  of last week" or "now uses X instead of Y"; what changed goes in the commit
  message or PR description.
- Comments say why, not what.

## Where code goes

- `api.py` and `cli.py` stay thin: parse the request, call a module, shape the
  response. A behaviour both the UI and the CLI need goes in the module, once.
- Workflow modules (`sorting.py`, `groups.py`, `discovery.py`, `organizer.py`)
  are functions taking `db: Database` and `config: Config`. They drive the core
  classes (`GroupSorter`, `EmbeddingSpace`, `FeatureExtractor`,
  `ClusterEngine`, `FeatureLayout`), which take their own config objects.
  Extend the existing class rather than adding a parallel function.
- All SQL lives in `database.py`; nothing else imports `sqlite3`.
- `DEFAULT_CONFIG` in `config.py` holds only what a user or the eval tunes per
  library (sorting, discovery, organize); add a key only when someone needs it
  to vary. Measurement and tag-rule thresholds stay named module constants or
  inline in the rule, with a comment saying where the value came from.
- The UI talks to the API only through `ui/src/lib/services/api.ts`; shared
  state lives in `ui/src/lib/stores/`, components in `ui/src/lib/components/`.
- New eval suites follow "Adding a suite" in EVALUATION.md.

## Testing

Every new feature, bug fix or change in behaviour ships with its tests in the
same PR: add tests for new behaviour, and update (never just delete) tests whose
expectations changed. Lean test-first: for a bug, write the test that fails
first; for a new behaviour, write down what it should do as a test before or
while building it.

Test each behaviour once, at the lowest layer that can prove it. Aim for
sufficient coverage, not maximal: don't re-test the same rule through several
layers, and don't add a test that only restates the code.

| Layer       | Where                            | Owns                                                                                                                                            |
| ----------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Python unit | `tests/unit`                     | Scoring, tagging, grouping, filing rules: logic over synthetic vectors, fast                                                                    |
| Closed loop | `tests/e2e`, `tests/integration` | Real CLI and HTTP routes over a temp SQLite library and synthesised audio: request shapes, persistence, files really written and undone         |
| UI unit     | `ui/src/**/*.test.ts` (vitest)   | Pure UI logic: the API client, stores, formatting and maths helpers                                                                             |
| Browser     | `ui/e2e` (Playwright)            | User journeys through the real UI and API: what the user sees and that a click reaches the backend. Not edge cases a lower layer already covers |
| Screenshots | `npm run ui:snapshot`            | Looking at screens while working on them. Not a test and not a gate                                                                             |

So a new scoring rule gets a unit test; a new route gets a closed-loop test; a
new screen or button gets a browser test of its happy path (and a `shot`); a
store or helper with branching logic gets a vitest test. Anything that writes,
moves or deletes user files needs a closed-loop test of both the dry run and
the undo.

- Unit tests use synthetic feature vectors (`tests/conftest.py`), not audio.
  Mark tests that decode or synthesise audio `slow` or `e2e`.
- A change to measuring or scoring code is judged by eval numbers, not only by
  tests. The synthetic run in Commands covers beats and keys (`rhythm.py`,
  `tagging.py`, `extractor.py`, `features.py`) only. Changes to `sorter.py` or
  `embedding.py` need `--suite sorting-fma` against `eval/baselines/datasets.json`
  with `--tolerance 0.03`, which needs downloaded data: run it through
  `eval.yml`'s `workflow_dispatch` or on a machine that has it. An unchanged
  synthetic run proves nothing about sorting. Refresh a baseline file only in
  the PR whose change earned the new numbers, and say so in the PR.
- After changing the UI, run `npm run ui:snapshot` and look at the screens you
  touched (`ui-snapshots/report.md` lists them, and any browser errors). When
  you add a screen or state, add a `shot` for it in `ui/snapshots/screens.spec.ts`.

## Gotchas

- Type hints use `typing` (`List`, `Optional`, …). Write new ones that way,
  and don't rewrite existing ones to PEP 585/604 syntax; those ruff rules are
  ignored on purpose.
- mypy is progressive: annotate new code, but don't add annotations to code
  you are not otherwise changing.

When the same correction comes up twice, propose adding it to this file.
