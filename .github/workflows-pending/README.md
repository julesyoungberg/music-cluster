# Workflows waiting to be activated

These are the four CI workflows for this project. They are here rather than in
`.github/workflows/` because the automation that opened this branch does not
hold GitHub's `workflows` permission, and GitHub refuses any push that writes
into `.github/workflows/` without it.

Move them to make them run:

```bash
git mv .github/workflows-pending/*.yml .github/workflows/
git rm .github/workflows-pending/README.md
git commit -m "Activate the CI workflows"
git push
```

(`mkdir -p .github/workflows` first if the directory does not exist yet.)

| File | What it runs |
| --- | --- |
| `lint.yml` | `ruff format --check`, `ruff check`, `mypy`; prettier, eslint and `svelte-check`; `cargo fmt` and `clippy` |
| `test-unit.yml` | `pytest tests/unit` on Python 3.10–3.12 across Linux, macOS and Windows, plus vitest |
| `test-e2e.yml` | `pytest tests/e2e tests/integration` against real audio, plus the Playwright browser suite |
| `build.yml` | Installers for macOS (Intel and Apple silicon), Windows and Linux; a `v*` tag drafts a release with all of them attached |

Every command they run works locally too — see
[INSTALL.md](../../INSTALL.md#continuous-integration).
