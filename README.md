# back_to_the_dashboard

An Express/Node dashboard that surfaces BOCPD and regime-detection pipeline
artifacts. The app exposes a REST API for sweeps, runs, logs, and Kanban data
and serves a static frontend from the `public/` directory.

## Quick start

```bash
npm install           # install dependencies
npm start             # serves the dashboard on http://localhost:4100
```

The server reads from the `TEST_BOCPD_DIR` workspace (defaults to
`/home/jason/ml/test_bocpd`) and can proxy to the Kanban API via
`KANBAN_BASE` (defaults to `http://localhost:4000`). Override either by
exporting the relevant environment variable before running `npm start`.

#### Example

```bash
TEST_BOCPD_DIR=/path/to/runs KANBAN_BASE=http://localhost:4000 npm start
```

Then open <http://localhost:4100/> to view the dashboards. Individual pages are
served from `public/*.html` and fetch JSON from the Express API.

## Available endpoints

- `GET /api/health` &mdash; sanity check on the configured repo paths.
- `GET /api/sweeps` &mdash; list lambda sweep manifests.
- `GET /api/runs` and `GET /api/latest-run` &mdash; pull recent run summaries.
- `GET /api/builds` &mdash; Jenkins-style summary of pipeline log files with status heuristics.
- `POST /api/builds/start` &mdash; launch curated pipeline jobs (WFO, autotune, validation, promotion).
- `POST /api/builds/kill` &mdash; request termination of an active build.
- `GET /api/logs` &mdash; browse pipeline log files (optional `?file=` to read).

Refer to `server.js` for the full set of endpoints and their payload shapes.

## Development tips

- Edit the static frontend in `public/`. Assets can reference `/vendor` for
  shared libraries (Chart.js is pre-vendored). A Jenkins-style build view lives
  at `/builds.html`, where you can trigger curated jobs (WFO sweeps, autotune,
  validation/promotion) via the Start New Build form and cancel active builds.
- `npm start` uses whatever Node version is active; use `nvm` or a pinned
  interpreter if required.
- Optional helper scripts (e.g. `scripts/emit_dummy_metrics.py`) remain for ad
  hoc data generation but are not part of the runtime path.

## Cleaning up

Remove `node_modules` if you want a clean slate:

```bash
rm -rf node_modules
```

No Python/Flask components remain; all dashboard functionality now flows
through the Node server.
