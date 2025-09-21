const fs = require('fs/promises');
const path = require('path');

const EXPERIMENTS_BASE = process.env.ALPHA_EXPERIMENTS_DIR || '/home/jason/ml/alpha-experiments';
const RUNS_ROOT = process.env.RUNS_DIR || path.join(EXPERIMENTS_BASE, 'runs');

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

module.exports = {
  RUNS_ROOT,
  listRuns,
  readSummary,
  readEquity,
  readTrades,
  deriveRoundTrips,
};
