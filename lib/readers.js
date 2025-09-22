const fs = require('fs/promises');
const path = require('path');
const { spawn } = require('child_process');

const EXPERIMENTS_BASE = process.env.ALPHA_EXPERIMENTS_DIR || '/home/jason/ml/alpha-experiments';
const RUNS_ROOT = process.env.RUNS_DIR || path.join(EXPERIMENTS_BASE, 'runs');
const CPD_EXTRACTOR = process.env.CPD_EXTRACTOR || path.join(__dirname, '..', 'scripts', 'cpd_extract.py');
const PYTHON_BIN = process.env.CPD_PY || process.env.BUILD_PY || '/home/jason/venvs/torchbuild/bin/python';

async function ensureRunsRoot() {
  try {
    await fs.access(RUNS_ROOT);
  } catch (err) {
    throw new Error(`Runs directory not found: ${RUNS_ROOT}`);
  }
}

async function listRuns() {
  await ensureRunsRoot();
  const entries = await fs.readdir(RUNS_ROOT, { withFileTypes: true });
  const runs = [];
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const runId = entry.name;
    const summaryPath = path.join(RUNS_ROOT, runId, 'summary.json');
    try {
      const stat = await fs.stat(summaryPath);
      runs.push({ id: runId, mtimeMs: stat.mtimeMs });
    } catch (err) {
      continue;
    }
  }
  runs.sort((a, b) => b.mtimeMs - a.mtimeMs);
  return runs.map((run) => run.id);
}

async function readSummary(runId) {
  const summaryPath = path.join(RUNS_ROOT, runId, 'summary.json');
  const raw = await fs.readFile(summaryPath, 'utf-8');
  return JSON.parse(raw);
}

function parseCsv(content) {
  const lines = content.trim().split(/\r?\n/);
  if (lines.length === 0) return [];
  const header = lines[0].split(',');
  return lines.slice(1).map((line) => {
    const parts = line.split(',');
    const row = {};
    header.forEach((key, idx) => {
      row[key.trim()] = parts[idx] ? parts[idx].trim() : '';
    });
    return row;
  });
}

async function readEquity(runId) {
  const equityPath = path.join(RUNS_ROOT, runId, 'equity.csv');
  const raw = await fs.readFile(equityPath, 'utf-8');
  const rows = parseCsv(raw);
  return rows.map((row) => ({
    timestamp: row.timestamp,
    equity: Number(row.equity ?? row.incremental_pnl ?? 0),
    signal: row.signal !== undefined ? Number(row.signal) : undefined,
  }));
}

async function readTrades(runId, { limit = 200, offset = 0 } = {}) {
  const tradesCsv = path.join(RUNS_ROOT, runId, 'trades.csv');
  const raw = await fs.readFile(tradesCsv, 'utf-8');
  const rows = parseCsv(raw);
  const sliced = rows.slice(offset, offset + limit);
  return sliced.map((row) => ({
    entry_time: row.entry_time,
    exit_time: row.exit_time,
    side: row.side,
    pnl: Number(row.pnl || 0),
  }));
}

function deriveRoundTrips(trades) {
  return trades.map((trade) => {
    const entry = trade.entry_time ? new Date(trade.entry_time) : null;
    const exit = trade.exit_time ? new Date(trade.exit_time) : null;
    const holdSeconds = entry && exit ? (exit.getTime() - entry.getTime()) / 1000 : null;
    return {
      ...trade,
      hold_seconds: holdSeconds,
    };
  });
}

async function ensureFileExists(runId, relativePath, friendlyName) {
  const filePath = path.join(RUNS_ROOT, runId, relativePath);
  try {
    await fs.access(filePath);
  } catch (err) {
    const name = friendlyName || relativePath;
    throw new Error(`${name} not found for run ${runId}`);
  }
  return filePath;
}

async function runCpdExtractor(runId, mode, options = {}) {
  const {
    downsample = 1,
    threshold,
    minSpacing,
    limit,
    center,
    seconds,
  } = options;

  const parquetPath = await ensureFileExists(runId, 'cpd_ctx.parquet', 'cpd_ctx.parquet');
  const args = [CPD_EXTRACTOR, '--file', parquetPath, '--mode', mode];

  const safeDownsample = Number.isFinite(downsample) && downsample > 1 ? Math.floor(downsample) : 1;
  if (safeDownsample > 1) args.push('--downsample', String(safeDownsample));
  if (threshold !== undefined) args.push('--threshold', String(threshold));
  if (minSpacing !== undefined) args.push('--min-spacing', String(minSpacing));
  if (limit !== undefined) args.push('--limit', String(limit));
  if (center) args.push('--center', String(center));
  if (seconds !== undefined) args.push('--seconds', String(seconds));

  return new Promise((resolve, reject) => {
    const child = spawn(PYTHON_BIN, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (chunk) => { stdout += chunk; });
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    child.on('error', (error) => {
      reject(new Error(`cpd extractor failed: ${error.message}`));
    });
    child.on('close', (code) => {
      if (code !== 0) {
        return reject(new Error(`cpd extractor exited with code ${code}: ${stderr.trim()}`));
      }
      try {
        const payload = JSON.parse(stdout || 'null');
        resolve(payload ?? null);
      } catch (err) {
        reject(new Error(`failed to parse CPD extractor output: ${err.message}`));
      }
    });
  });
}

async function readCpdSeries(runId, opts = {}) {
  return runCpdExtractor(runId, 'series', opts);
}

async function readCpdModels(runId, opts = {}) {
  return runCpdExtractor(runId, 'models', opts);
}

async function readCpdSummary(runId) {
  return runCpdExtractor(runId, 'summary');
}

async function readCpdEvents(runId, opts = {}) {
  return runCpdExtractor(runId, 'events', opts);
}

async function readCpdWindow(runId, opts = {}) {
  const { timestamp, seconds = 60 } = opts;
  if (!timestamp) throw new Error('timestamp query parameter required');
  const payload = await runCpdExtractor(runId, 'window', { center: timestamp, seconds });

  const center = new Date(timestamp);
  if (Number.isNaN(center.getTime())) {
    throw new Error('invalid timestamp');
  }
  const windowSeconds = Number.isFinite(seconds) ? Number(seconds) : 60;
  const halfWindowMs = Math.max(1, windowSeconds) * 1000;

  const equity = await readEquity(runId).catch(() => []);
  const trades = await readTrades(runId, { limit: 1000, offset: 0 }).catch(() => []);

  const filterByWindow = (rows) => rows.filter((row) => {
    if (!row || !row.timestamp) return false;
    const ts = new Date(row.timestamp);
    if (Number.isNaN(ts.getTime())) return false;
    return Math.abs(ts.getTime() - center.getTime()) <= halfWindowMs;
  });

  return {
    ...payload,
    equity: Array.isArray(equity) ? filterByWindow(equity) : [],
    trades: Array.isArray(trades) ? filterByWindow(trades) : [],
  };
}

async function saveCpdClip(runId, payload) {
  if (!payload || typeof payload !== 'object') {
    throw new Error('clip payload must be an object');
  }
  const dir = path.join(RUNS_ROOT, runId, 'cpd_clips');
  await fs.mkdir(dir, { recursive: true });
  const ts = String(payload.timestamp || Date.now()).replace(/[:]/g, '-');
  const filename = `clip-${ts}.json`;
  const outPath = path.join(dir, filename);
  await fs.writeFile(outPath, JSON.stringify(payload, null, 2));
  return { path: outPath };
}

module.exports = {
  RUNS_ROOT,
  listRuns,
  readSummary,
  readEquity,
  readTrades,
  deriveRoundTrips,
  readCpdSeries,
  readCpdModels,
  readCpdSummary,
  readCpdEvents,
  readCpdWindow,
  saveCpdClip,
};
