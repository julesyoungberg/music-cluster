# music-cluster UI

Desktop front end for [music-cluster](../README.md), built with SvelteKit,
Tailwind and Tauri. It is a single-page client that talks to the local API — no
server rendering, since everything it reads (the library, the filesystem) only
exists on the user's machine.

## Running

From the repository root, which starts the API too:

```bash
python start-dev.py            # web UI at http://localhost:1420
python start-dev.py --tauri    # native shell
```

Or on its own, against an API you started yourself:

```bash
npm install
npm run dev
```

Point it at a different API with `VITE_API_BASE=http://127.0.0.1:9000 npm run dev`.

## Layout

```
src/lib/services/api.ts   typed client for every endpoint, plus task polling
src/lib/stores/           collections/groups, notifications, the shared player
src/lib/components/       waveform, folder picker, group picker, modal, ...
src/routes/               one directory per screen
```

Long-running work (analysis, sorting, discovery) is a background task on the
API; the client kicks it off and polls `/api/tasks/{id}` rather than holding a
request open for minutes.

## Checks

```bash
npm run check      # svelte-check
npm run build      # static SPA into build/
```
