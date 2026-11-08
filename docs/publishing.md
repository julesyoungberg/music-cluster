# Publishing

How to release the command line tool to PyPI, and build and host the landing
page. Nothing here runs until someone with the accounts does it.

## Command line tool on PyPI

The package is `music-cluster` on [PyPI](https://pypi.org). Once it is there,
anyone with Python 3.10 to 3.12 and FFmpeg can install it:

```bash
pipx install music-cluster     # its own environment, `music-cluster` on the PATH
pip install music-cluster      # into the current environment
```

[pipx](https://pipx.pypa.io) is the recommended route: it keeps the tool's
dependencies (librosa, numba, scikit-learn) out of the user's other projects.

### One-time setup

PyPI trusted publishing lets the workflow upload without a stored token.

1. Create a PyPI account and turn on two-factor authentication.
2. On PyPI, go to Your projects › Publishing › Add a new pending publisher:
   - PyPI project name: `music-cluster`
   - Owner: `julesyoungberg`, repository: `music-cluster`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
3. In the GitHub repository, Settings › Environments › New environment, named
   `pypi`. Optionally add yourself as a required reviewer, so every upload
   waits for a click.

### Each release

1. Set the new version in `music_cluster/__init__.py` (`__version__`).
   `setup.py` and `music-cluster --version` read it from there.
2. Tag it: `git tag v2.1.0 && git push origin v2.1.0`. `build.yml` builds the
   desktop installers and drafts a GitHub release with them.
3. Review the draft release and publish it. Publishing runs
   [`publish.yml`](../.github/workflows/publish.yml), which builds the wheel,
   checks it installs and runs, checks the version matches the tag, and uploads
   to PyPI.

A PyPI version can never be uploaded twice, even after deleting it. To try a
release first, point the workflow at [TestPyPI](https://test.pypi.org) with
`repository-url: https://test.pypi.org/legacy/` on the upload step.

To build and check locally:

```bash
pip install build twine
python -m build && twine check --strict dist/*
```

### Homebrew

- **Desktop app:** a Homebrew cask is the easy route. It points at the `.dmg`
  on the GitHub release and its checksum. It belongs in a tap
  (`julesyoungberg/homebrew-tap`) until the app is well known enough for
  homebrew-cask. Casks for unsigned apps show the macOS warning on first launch,
  so sign and notarise the app first.
- **Command line tool:** a formula is harder. Homebrew builds every Python
  dependency from source, and numba and llvmlite need a matching LLVM. Until
  that is worth the upkeep, `brew install pipx ffmpeg && pipx install
  music-cluster` gives Mac users the same result.

## Landing page

The page is `site/index.html` with its stylesheet `site/styles.css`. Text that
changes in one place (product name, links, form endpoints) lives in
`site/site.json`, and the page refers to each value as `${key}`.

```bash
python scripts/build_site.py              # writes site/dist/
python -m http.server -d site/dist 8080   # preview at http://localhost:8080
```

The build copies the screenshots the page uses from `docs/images/`, so
retaking the docs screenshots (see the README) updates the site too. It lists
any `site.json` value that still starts with "TODO".

### Before it goes live

Fill these in `site/site.json`:

| Key | What it needs |
| --- | --- |
| `name` | The product name shown on the page |
| `license` | The license the code is released under. Add the matching `LICENSE` file and `setup.py` classifier at the same time |
| `site_url` | The address the site is served from, for link previews |
| `contact_form_action` | A form endpoint that emails you, e.g. [Formspree](https://formspree.io) or [Basin](https://usebasin.com). The form posts `name`, `email` and `message` |
| `newsletter_form_action` | The signup endpoint from a mailing list service, e.g. [Buttondown](https://buttondown.com). The form posts `email` |
| `donate_url` | A support link, e.g. GitHub Sponsors, Ko-fi or Buy Me a Coffee |

`download_url`, `releases_url`, `docs_url` and `github_url` point at the GitHub
repository, so they work once it is public.

The page loads IBM Plex from Google Fonts, which tells Google each visitor's
IP address. To avoid that (EU sites often must), download the font files and
serve them next to `styles.css` instead.

### Hosting

`site/dist/` is plain files, so any static host works: GitHub Pages,
Cloudflare Pages or Netlify. Run the build command above as the host's build
step, with `site/dist` as the output folder.
