# music-cluster

Sorts new music (and samples) into groups the user already keeps. The user
defines the groups; the tool scores, the user confirms, then it files. It never
invents a category or touches a file the user has not confirmed. README.md
explains the product, INSTALL.md covers setup and troubleshooting, EVALUATION.md
the eval harness, docs/cli.md every command.

Python package and CLI in `music_cluster/` (FastAPI + SQLite), SvelteKit +
Tailwind UI in `ui/`, Tauri 1 desktop shell in `src-tauri/`.

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

## Commands

```bash
# Setup (CI runs Python 3.10–3.12; librosa/numba may lag newer Pythons)
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements-dev.txt && uv pip install -e .   # uv venvs have no pip
cd ui && npm ci && cd ..

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

# Desktop shell (from src-tauri/, only if you touched Rust)
cargo fmt --all -- --check && cargo clippy --all-targets -- -D warnings
```

FFmpeg must be installed for any audio decoding. CI (`.github/workflows/`) runs
exactly these commands; keep them passing before opening a PR.

## How to work here

Build what the task needs now, in a shape that is cheap to change later.

- Solve the stated requirement. No speculative features, flags, config keys or
  extension points. Ideas for later go in the PR description or an issue, not
  in code.
- Before non-trivial work, state your assumptions and the simplest approach
  that would work. If the ask can be read two ways, ask rather than guess.
- Keep diffs surgical: every changed line traces to the task. No drive-by
  refactors, renames or reformatting of code you did not otherwise touch.
- Reuse the existing pattern instead of adding a second way to do the same
  thing (see "Where code goes"). If two patterns compete and you can't tell
  which is canonical, ask.
- Extensibility comes from small functions, clear names and dependencies
  passed in, not from base classes, registries or plugin systems built ahead of
  need. Abstract on the third repetition, not the first.
- Minimise scope, never correctness: input validation at the API and CLI
  boundary, error handling, and tests still ship.
- No new dependency without saying why in the PR and checking first. Python
  dependencies are declared twice, in `requirements.txt` and `setup.py`;
  update both. Large dependencies or models need Jules's approval.
- Done means verified: name the test, command or eval number that proves it,
  and run it before you finish.

## Where code goes

- Logic lives in the domain modules as plain functions that take `db: Database`
  and `config: Config` as arguments. `api.py` and `cli.py` stay thin: parse the
  request, call the module, shape the response. A behaviour both the UI and the
  CLI need goes in the module, once.
- All SQL lives in `database.py`; nothing else imports `sqlite3`.
- Tunable numbers go in `DEFAULT_CONFIG` in `config.py` with a comment saying
  where the value came from (usually an eval run). Add a key only when a
  measurement or a user asks for it to vary.
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
- A change to how audio is measured or scored (`extractor.py`, `features.py`,
  `rhythm.py`, `tagging.py`, `samples.py`, `sorter.py`, `embedding.py`) is
  judged by the eval run against the baseline, not only by tests. Refresh a
  file in `eval/baselines/` only in the PR whose change earned the new numbers,
  and say so in the PR.
- After changing the UI, run `npm run ui:snapshot` and look at the screens you
  touched (`ui-snapshots/report.md` lists them, and any browser errors). When
  you add a screen or state, add a `shot` for it in `ui/snapshots/screens.spec.ts`.

## Gotchas

- Type hints use `typing` (`List`, `Optional`, …) throughout. Don't rewrite
  them to PEP 585/604 syntax; those ruff rules are ignored on purpose.
- mypy is progressive: annotate new code, but don't add annotations to code
  you are not otherwise changing.
- Dataset eval suites need downloads that the cloud sandbox can't reach; run
  them on a real machine (EVALUATION.md).

When the same correction comes up twice, propose adding it to this file.
