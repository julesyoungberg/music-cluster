# UI snapshot

```bash
cd ui && npm run ui:snapshot      # ~20s to seed the first time, then ~35s a run
open ui-snapshots/report.md
```

Drives the real app (the Vite dev server against a real API) through every
screen and the main states inside each flow, and saves for each one:

- `<screen>.png` — a screenshot
- `<screen>.aria.yml` — the accessibility tree: every heading, button, label and
  piece of text, as YAML. Cheaper than an image to read, and easy to diff.
- any console errors, uncaught exceptions, failed requests and HTTP 4xx/5xx
  responses that happened on the way to that screen

`report.md` indexes the screens and lists anything that looks wrong at the top:
flows that broke (with a screenshot of the page where they broke) and screens
that logged unexpected errors.

## What it is for

It is a tool for looking at the UI, mainly for agents changing it: run it,
read the report, open the screenshots of the screens you touched. It is not a
regression suite. Nothing is compared against a baseline, and a step only
waits for what its screen needs, so the run passes unless a flow cannot reach
its screen at all. Behaviour that must not regress belongs in `ui/e2e/`.

CI runs it on every pull request and uploads the folder as the `ui-snapshots`
artifact, so a reviewer can see the UI without running anything.

## How it works

- `scripts/prepare_ui_snapshot.py` seeds the same synthetic library as the e2e
  suite once into `.ui-snapshot-workspace/seed`, then gives each run a fresh
  copy of the database in `.ui-snapshot-workspace/run`. Every run starts from
  the same state, and the playlists the commit step writes stay in there.
  `--reseed` rebuilds the library.
- The API runs on :8124 and the UI on :4174, apart from the e2e suite's ports.
- `screens.spec.ts` is the walk through the app. `capture.ts` provides the
  `shot(name, { note })` fixture that saves a screen.

## Adding a screen

Add a `shot` where the new screen or state appears in `screens.spec.ts`. Wait
for what the screen needs first (`await expect(...).toBeVisible()`), give it a
`note` saying what it should show, and pass `expectProblems: true` only when
the errors are the point of the screen, as with the API-down state.

Environment: `PYTHON` (default `../.venv/bin/python`), `UI_SNAPSHOT_OUT`,
`UI_SNAPSHOT_WORKSPACE`, `PLAYWRIGHT_CHROMIUM_PATH`.
