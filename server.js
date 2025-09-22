const express = require('express');
const cors = require('cors');
const fs = require('fs');
const fsp = require('fs/promises');
const path = require('path');
const { spawn } = require('child_process');

const readers = require('./lib/readers');

const app = express();
const PORT = process.env.PORT || 4100;

// Where the TEST_BOCPD repo lives so we can read artifacts
const EXPERIMENTS_DIR = process.env.ALPHA_EXPERIMENTS_DIR || '/home/jason/ml/alpha-experiments';
const REPO = process.env.TEST_BOCPD_DIR || EXPERIMENTS_DIR;
const KANBAN = process.env.KANBAN_BASE || 'http://localhost:4000';
const BUILD_PY = process.env.BUILD_PY || '/home/jason/venvs/torchbuild/bin/python';
const TRACK_B_REPO = process.env.TRACK_B_REPO || '/home/jason/ml/bocpdms-track';
const TRACK_B_PIPELINE_SCRIPT = process.env.TRACK_B_PIPELINE || path.join(TRACK_B_REPO, 'scripts', 'track_b_pipeline.py');

const activeBuilds = new Map();

const API_ENDPOINTS = [
  { method: 'GET', path: '/api', description: 'List available REST endpoints' },
  { method: 'GET', path: '/api/health', description: 'Check repository and Kanban wiring' },
  { method: 'GET', path: '/api/builds', description: 'Enumerate build logs and pipeline metadata' },
  { method: 'POST', path: '/api/builds/start', description: 'Trigger a pipeline job' },
  { method: 'POST', path: '/api/builds/kill', description: 'Request termination of an active job' },
  { method: 'GET', path: '/api/sweeps', description: 'List lambda sweep manifests' },
  { method: 'GET', path: '/api/runs', description: 'Enumerate recent backtest runs' },
  { method: 'GET', path: '/api/runs/:id/summary', description: 'Fetch summary.json for a run' },
  { method: 'GET', path: '/api/runs/:id/equity', description: 'Stream equity.csv for a run' },
  { method: 'GET', path: '/api/runs/:id/trades', description: 'Paginate trades.csv for a run' },
  { method: 'GET', path: '/api/runs/:id/roundtrips', description: 'Derive round-trip stats from trades' },
  { method: 'GET', path: '/api/runs/:id/cpd/series', description: 'Stream CPD context series (cp_prob, runlen)' },
  { method: 'GET', path: '/api/runs/:id/cpd/models', description: 'Stream CPD model posterior series' },
  { method: 'GET', path: '/api/runs/:id/cpd/summary', description: 'Aggregate CPD metrics for a run' },
  { method: 'GET', path: '/api/latest-run', description: 'Return most recent run summary' },
  { method: 'GET', path: '/api/logs', description: 'Browse raw pipeline log files' },
  { method: 'GET', path: '/api/autotune', description: 'Summarize autotune sweep logs' },
  { method: 'GET', path: '/api/kanban/tasks', description: 'Proxy Kanban tasks from the board API' },
  { method: 'PATCH', path: '/api/kanban/tasks/:id', description: 'Proxy Kanban task updates' },
  { method: 'GET', path: '/api/monitors/runtime', description: 'Return latest runtime health metrics' },
  { method: 'GET', path: '/api/cp_sweeps', description: 'List available cp-threshold sweep directories' },
  { method: 'GET', path: '/api/cp_sweep', description: 'Load metrics for a specific sweep' },
];

function makeEnv(overrides = {}) {
  const env = { ...process.env };
  env.PYTHONPATH = overrides.PYTHONPATH || env.PYTHONPATH || '/home/jason/ml/sparrow/src';
  env.PY = overrides.PY || env.PY || BUILD_PY;
  for (const [k, v] of Object.entries(overrides)) {
    env[k] = v;
  }
  return env;
}

function resolveScriptPath(script) {
  if (!script) return null;
  if (path.isAbsolute(script)) return script;
  return path.join(REPO, script);
}

function resolveMissingScripts(definition) {
  if (!definition) return [];
  const scripts = Array.isArray(definition.requiredScripts) ? definition.requiredScripts : [];
  const missing = [];
  for (const script of scripts) {
    const resolved = resolveScriptPath(script);
    if (!resolved || !fs.existsSync(resolved)) {
      missing.push(script);
    }
  }
  return missing;
}

function definitionAvailability(definition) {
  const baseEnabled = definition?.enabled !== false;
  const missingScripts = resolveMissingScripts(definition);
  if (missingScripts.length > 0) {
    const plural = missingScripts.length > 1 ? 'scripts' : 'script';
    return {
      enabled: false,
      missingScripts,
      reason: `Missing ${plural}: ${missingScripts.join(', ')}`
    };
  }
  const reason = baseEnabled ? null : definition?.disabled_reason || null;
  return { enabled: baseEnabled, missingScripts, reason };
}

const PIPELINE_RUN_MARKER = path.join(REPO, 'runs', 'pipeline_demo', 'latest_run.txt');

const TRACK_B_PIPELINE_STEPS = [
  'synth-data',
  'discovery',
  'training-set',
  'lambda-sweep',
  'tune-breakout',
  'run-strategy-demo',
  'bundle-release',
];

function loadLatestPipelineRun() {
  try {
    const raw = fs.readFileSync(PIPELINE_RUN_MARKER, 'utf8').trim();
    if (!raw) return null;
    const candidate = path.isAbsolute(raw) ? raw : path.join(REPO, 'runs', raw);
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  } catch (err) {
    if (err.code !== 'ENOENT') throw err;
  }
  return null;
}

function resolveTrackBBundleRun(input) {
  const candidate = (input || '').trim();
  if (!candidate || candidate.toLowerCase() === 'latest') {
    const latest = loadLatestPipelineRun();
    if (!latest) {
      throw new Error('No pipeline demo run recorded yet. Run the demo strategy first.');
    }
    return latest;
  }
  const resolved = path.isAbsolute(candidate) ? candidate : path.join(REPO, candidate);
  if (!fs.existsSync(resolved)) {
    throw new Error(`Run directory not found: ${candidate}`);
  }
  return resolved;
}

function normaliseTrackBStep(step) {
  if (!step) return null;
  const key = String(step).trim();
  if (!key) return null;
  return TRACK_B_PIPELINE_STEPS.includes(key) ? key : null;
}

function experimentsPythonPath() {
  const experimentsSrc = path.join(REPO, 'src');
  const base = process.env.PYTHONPATH || '/home/jason/ml/sparrow/src';
  const parts = base ? base.split(path.delimiter) : [];
  if (parts.includes(experimentsSrc)) return base;
  return parts.length ? `${experimentsSrc}${path.delimiter}${base}` : experimentsSrc;
}

const APPROACHES = {
  track_b: {
    id: 'track_b',
    label: 'BOCPDMS Track',
    description:
      'BOCPDMS workflow: synthetic data, hazard sweeps, breakout tuning, demo strategy, and packaging.',
  },
  bocpd: {
    id: 'bocpd',
    label: 'BOCPD Pipeline',
    description:
      'Bayesian online change-point detection workflow spanning discovery, tuning, validation, and promotion.',
  },
};

const BUILD_DEFINITIONS = {
  make_synth_data: {
    id: 'make_synth_data',
    label: 'Generate Synthetic MES Bars',
    description: 'Populate data/cache/mes/bars_1s with deterministic 1s bars for demo runs.',
    group: 'BOCPDMS Track',
    approach: 'track_b',
    requiredScripts: ['scripts/make_synth_data.py'],
    details: [
      'Writes two days of MES bars under data/cache/mes/bars_1s/date=*.',
      'Feeds downstream lambda sweep, breakout tuning, and pipeline demos.',
    ],
    fields: [],
    buildCommand() {
      return {
        cmd: [BUILD_PY, 'scripts/make_synth_data.py'],
        cwd: REPO,
        env: makeEnv({ PYTHONPATH: experimentsPythonPath() }),
        publicParams: {},
      };
    },
  },
  lambda_sweep: {
    id: 'lambda_sweep',
    label: 'Run Lambda Sweep',
    description: 'Score hazard lambdas by time-of-day bucket and emit manifest.',
    group: 'BOCPDMS Track',
    approach: 'track_b',
    requiredScripts: ['scripts/lambda_sweep.py', 'configs/lambda_sweep.yaml'],
    details: [
      'Evaluates cp_lambda schedules over cached MES bars.',
      'Writes artifacts/wfo/lambda_sweep/mes_1s/manifest.json for reuse.',
    ],
    fields: [
      { name: 'config', label: 'Config', type: 'text', default: 'configs/lambda_sweep.yaml' },
    ],
    buildCommand(params = {}) {
      const config = params.config || 'configs/lambda_sweep.yaml';
      return {
        cmd: [BUILD_PY, 'scripts/lambda_sweep.py', '--config', config],
        cwd: REPO,
        env: makeEnv({ PYTHONPATH: experimentsPythonPath() }),
        publicParams: { config },
      };
    },
  },
  tune_bocpd_breakout: {
    id: 'tune_bocpd_breakout',
    label: 'Tune BOCPD Breakout',
    description: 'Grid cp_lambda / threshold candidates over discovery signals.',
    group: 'BOCPDMS Track',
    approach: 'track_b',
    requiredScripts: ['scripts/tune_bocpd_breakout.py', 'configs/run_mes_bocpd_breakout.yaml'],
    details: [
      'Ranks breakout candidates and writes leaderboard CSV.',
      'Emits top_k_recheck.jsonl for downstream validation.',
    ],
    fields: [
      { name: 'config', label: 'Config', type: 'text', default: 'configs/run_mes_bocpd_breakout.yaml' },
      { name: 'profile', label: 'Profile', type: 'text', default: '' },
    ],
    buildCommand(params = {}) {
      const config = params.config || 'configs/run_mes_bocpd_breakout.yaml';
      const profile = params.profile ? String(params.profile).trim() : '';
      const cmd = [BUILD_PY, 'scripts/tune_bocpd_breakout.py', '--config', config];
      if (profile) cmd.push('--profile', profile);
      return {
        cmd,
        cwd: REPO,
        env: makeEnv({ PYTHONPATH: experimentsPythonPath() }),
        publicParams: { config, profile: profile || undefined },
      };
    },
  },
  run_strategy_demo: {
    id: 'run_strategy_demo',
    label: 'Run Demo Strategy',
    description: 'Execute demo BOCPD breakout run and record latest marker.',
    group: 'BOCPDMS Track',
    approach: 'track_b',
    requiredScripts: ['src/intraday_futures/runners/run_strategy.py', 'configs/run_demo.yaml'],
    details: [
      'Runs intraday_futures.runners.run_strategy with demo config.',
      'Updates runs/pipeline_demo/latest_run.txt for bundling.',
    ],
    fields: [
      { name: 'config', label: 'Config', type: 'text', default: 'configs/run_demo.yaml' },
    ],
    buildCommand(params = {}) {
      const config = params.config || 'configs/run_demo.yaml';
      return {
        cmd: [BUILD_PY, '-m', 'intraday_futures.runners.run_strategy', '--config', config],
        cwd: REPO,
        env: makeEnv({ PYTHONPATH: experimentsPythonPath() }),
        publicParams: { config },
      };
    },
  },
  bundle_release: {
    id: 'bundle_release',
    label: 'Bundle Latest BOCPDMS Run',
    description: 'Package the most recent demo run into a release tarball.',
    group: 'BOCPDMS Track',
    approach: 'track_b',
    requiredScripts: ['scripts/bundle_release.py'],
    details: [
      'Tars up resolved config, metrics, and plots for distribution.',
      'Writes artifacts/releases/pipeline_demo.tar.gz.',
    ],
    fields: [
      { name: 'run', label: 'Run Dir (or latest)', type: 'text', default: 'latest' },
      { name: 'output', label: 'Output Tarball', type: 'text', default: 'artifacts/releases/pipeline_demo.tar.gz' },
    ],
    buildCommand(params = {}) {
      const runParam = params.run || 'latest';
      const output = params.output || 'artifacts/releases/pipeline_demo.tar.gz';
      let resolvedRun;
      try {
        resolvedRun = resolveTrackBBundleRun(runParam);
      } catch (err) {
        throw Object.assign(new Error(`bundle_release: ${err.message}`), { statusCode: 400 });
      }
      return {
        cmd: [BUILD_PY, 'scripts/bundle_release.py', '--run', resolvedRun, '--output', output],
        cwd: REPO,
        env: makeEnv({ PYTHONPATH: experimentsPythonPath() }),
        publicParams: { run: runParam, output },
      };
    },
  },
  track_b_pipeline: {
    id: 'track_b_pipeline',
    label: 'BOCPDMS Pipeline Runner',
    description: 'Execute the Track B pipeline script for single steps or ranges.',
    group: 'BOCPDMS Track',
    approach: 'track_b',
    requiredScripts: [path.join('..', 'bocpdms-track', 'scripts', 'track_b_pipeline.py')],
    details: [
      'Calls bocpdms-track/scripts/track_b_pipeline.py run ...',
      'Supports --only/--start/--end to target specific steps.',
    ],
    fields: [
      {
        name: 'mode',
        label: 'Mode',
        type: 'select',
        options: [
          { value: 'full', label: 'Full Pipeline' },
          { value: 'only', label: 'Single Step' },
          { value: 'range', label: 'Step Range' },
        ],
        default: 'full',
      },
      { name: 'only_step', label: 'Only Step', type: 'text', default: 'synth-data' },
      { name: 'start_step', label: 'Start Step', type: 'text', default: 'synth-data' },
      { name: 'end_step', label: 'End Step', type: 'text', default: 'bundle-release' },
      {
        name: 'force',
        label: 'Force Re-run',
        type: 'select',
        options: [
          { value: 'false', label: 'No' },
          { value: 'true', label: 'Yes' },
        ],
        default: 'false',
      },
      {
        name: 'dry_run',
        label: 'Dry Run',
        type: 'select',
        options: [
          { value: 'false', label: 'No' },
          { value: 'true', label: 'Yes' },
        ],
        default: 'false',
      },
    ],
    buildCommand(params = {}) {
      const mode = params.mode || 'full';
      const cmd = [BUILD_PY, TRACK_B_PIPELINE_SCRIPT, 'run'];
      const safeOnly = normaliseTrackBStep(params.only_step) || 'synth-data';
      const safeStart = normaliseTrackBStep(params.start_step) || 'synth-data';
      const safeEnd = normaliseTrackBStep(params.end_step) || 'bundle-release';
      if (mode === 'only') {
        cmd.push('--only', safeOnly);
      } else if (mode === 'range') {
        cmd.push('--start', safeStart);
        if (safeEnd) cmd.push('--end', safeEnd);
      }
      if (String(params.force).toLowerCase() === 'true') cmd.push('--force');
      if (String(params.dry_run).toLowerCase() === 'true') cmd.push('--dry-run');
      const publicParams = {
        mode,
        only_step: safeOnly,
        start_step: safeStart,
        end_step: safeEnd,
        force: String(params.force || 'false') === 'true',
        dry_run: String(params.dry_run || 'false') === 'true',
      };
      return {
        cmd,
        cwd: TRACK_B_REPO,
        env: makeEnv({
          ALPHA_EXPERIMENTS_ROOT: REPO,
          PYTHONPATH: experimentsPythonPath(),
        }),
        publicParams,
      };
    },
  },
  eval_metrics: {
    id: 'eval_metrics',
    label: 'Detection Quality Audit',
    description: 'Compute headline BOCPD detection metrics over MES labels parquet',
    group: 'Evaluation & Metrics',
    approach: 'bocpd',
    requiredScripts: ['scripts/detection_metrics.py'],
    details: [
      'Reads BOCPD label parquet and evaluates hit rate, delay, and false alarms.',
      'Writes JSON summary under artifacts/reports/.',
    ],
    fields: [
      { name: 'labels', label: 'Labels Parquet', type: 'text', default: 'artifacts/bocpd_discovery/mes_1s/bocpd_signals.parquet' },
      { name: 'thr', label: 'Detection Threshold', type: 'number', default: 0.1 },
      { name: 'max_delay', label: 'Max Delay (bars)', type: 'number', default: 50 },
      { name: 'out', label: 'Output JSON', type: 'text', default: 'artifacts/reports/detection_metrics.json' },
    ],
    buildCommand(params = {}) {
      const labelsPath = params.labels || 'artifacts/bocpd_discovery/mes_1s/bocpd_signals.parquet';
      const thr = Number.isFinite(Number(params.thr)) ? String(params.thr) : '0.1';
      const maxDelay = Number.isFinite(Number(params.max_delay)) ? String(params.max_delay) : '50';
      const outPath = params.out || 'artifacts/reports/detection_metrics.json';
      return {
        cmd: [
          BUILD_PY,
          'scripts/detection_metrics.py',
          '--labels',
          labelsPath,
          '--thr',
          thr,
          '--max-delay',
          maxDelay,
          '--out',
          outPath,
        ],
        cwd: REPO,
        env: makeEnv({ PYTHONPATH: '/home/jason/ml/sparrow/src' }),
        publicParams: {
          labels: labelsPath,
          thr: Number(thr),
          max_delay: Number(maxDelay),
          out: outPath,
        },
      };
    },
  },
  wfo_bocpd: {
    id: 'wfo_bocpd',
    label: 'Walk-Forward Optimization (BOCPD)',
    description: 'Grid cp_thr and break_k via scripts/run_wfo.py',
    group: 'Detection Pipelines',
    approach: 'bocpd',
    requiredScripts: ['scripts/run_wfo.py'],
    details: [
      'Runs purged walk-forward optimisation over cp_thr & break_k grids.',
      'Writes tolerance metrics per step to artifacts/wfo/.',
      'Use run ID to differentiate scenarios.'
    ],
    fields: [
      { name: 'base_config', label: 'Base Config', type: 'text', default: 'configs/wfo_mes.yaml' },
      { name: 'out_run_id', label: 'Run ID', type: 'text', default: 'wfo_bocpd_gated' },
      { name: 'cp_thr', label: 'cp_thr grid', type: 'text', default: '0.05,0.08,0.12' },
      { name: 'break_k', label: 'break_k grid', type: 'text', default: '0.03,0.05' },
    ],
    buildCommand(params = {}) {
      const baseConfig = params.base_config || 'configs/wfo_mes.yaml';
      const outRun = params.out_run_id || 'wfo_bocpd_gated';
      const cpThr = params.cp_thr || '0.05,0.08,0.12';
      const breakK = params.break_k || '0.03,0.05';
      return {
        cmd: [BUILD_PY, 'scripts/run_wfo.py', '--base-config', baseConfig, '--out-run-id', outRun, '--cp-thr', cpThr, '--break-k', breakK],
        cwd: REPO,
        env: makeEnv({ PYTHONPATH: experimentsPythonPath() }),
        publicParams: { base_config: baseConfig, out_run_id: outRun, cp_thr: cpThr, break_k: breakK },
      };
    },
  },
  autotune: {
    id: 'autotune',
    label: 'Autotune & Run Strategy',
    description: 'Full sweep + blend + run via scripts/autotune_and_run.py',
    group: 'Discovery & Tuning',
    approach: 'bocpd',
    requiredScripts: ['scripts/autotune_and_run.py'],
    details: [
      'Sweeps cp_thr/break_k, blends predictive & economic scores.',
      'Patches config with the winner and runs strategy/report generation.'
    ],
    fields: [
      { name: 'base_config', label: 'Base Config', type: 'text', default: 'configs/_local.bocpd_gated.yaml' },
      { name: 'cp_thr', label: 'cp_thr grid', type: 'text', default: '0.05,0.08,0.12' },
      { name: 'break_k', label: 'break_k grid', type: 'text', default: '0.03,0.05' },
      { name: 'out', label: 'Sweep Output Dir', type: 'text', default: 'artifacts/sweeps/cp_thr' },
    ],
    buildCommand(params = {}) {
      const baseConfig = params.base_config || 'configs/_local.bocpd_gated.yaml';
      const cpThr = params.cp_thr || '0.05,0.08,0.12';
      const breakK = params.break_k || '0.03,0.05';
      const outDir = params.out || 'artifacts/sweeps/cp_thr';
      return {
        cmd: [BUILD_PY, 'scripts/autotune_and_run.py', '--base-config', baseConfig, '--cp-thr', cpThr, '--break-k', breakK, '--out', outDir],
        cwd: REPO,
        env: makeEnv({ PYTHONPATH: '/home/jason/ml/sparrow/src' }),
        publicParams: { base_config: baseConfig, cp_thr: cpThr, break_k: breakK, out: outDir },
      };
    },
  },
  validate_candidates: {
    id: 'validate_candidates',
    label: 'Validate Promoted Candidates',
    description: 'Run Purged-CV and WFO for promoted candidates',
    group: 'Detection Pipelines',
    approach: 'bocpd',
    requiredScripts: ['scripts/validate_candidates.py'],
    details: [
      'Loads promoted candidates from runs/promoted_candidates.jsonl.',
      'Executes Purged-CV + WFO validation; writes cv/wfo metrics per run.'
    ],
    fields: [
      { name: 'base_config', label: 'Base Config', type: 'text', default: 'configs/validate_mes.yaml' },
      { name: 'candidates', label: 'Candidates JSONL', type: 'text', default: 'runs/promoted_candidates.jsonl' },
      { name: 'summary', label: 'Summary Output', type: 'text', default: 'runs/validated_candidates/summary.json' },
      { name: 'limit', label: 'Limit (optional)', type: 'number', default: 0 },
    ],
    buildCommand(params = {}) {
      const baseConfig = params.base_config || 'configs/validate_mes.yaml';
      const candidates = params.candidates || 'runs/promoted_candidates.jsonl';
      const summary = params.summary || 'runs/validated_candidates/summary.json';
      const limit = Number.isFinite(Number(params.limit)) && Number(params.limit) > 0 ? String(Number(params.limit)) : null;
      const cmd = [
        BUILD_PY,
        'scripts/validate_candidates.py',
        '--base-config',
        baseConfig,
        '--candidates',
        candidates,
        '--out-dir',
        'runs/validated_candidates',
        '--summary',
        summary,
      ];
      if (limit) cmd.push('--limit', limit);
      return {
        cmd,
        cwd: REPO,
        env: makeEnv({ PYTHONPATH: experimentsPythonPath() }),
        publicParams: {
          base_config: baseConfig,
          candidates,
          summary,
          limit: limit ? Number(limit) : undefined,
        },
      };
    },
  },
  export_candidates: {
    id: 'export_candidates',
    label: 'Export Sweep Candidates',
    description: 'Collect best cp_thr sweeps into promoted_candidates.jsonl',
    group: 'Discovery & Tuning',
    approach: 'bocpd',
    requiredScripts: ['scripts/export_candidates_from_sweeps.py'],
    details: [
      'Scans artifacts/sweeps/** for best cp_thr combinations.',
      'Appends results into runs/promoted_candidates.jsonl.'
    ],
    fields: [],
    buildCommand() {
      return {
        cmd: [BUILD_PY, 'scripts/export_candidates_from_sweeps.py'],
        cwd: REPO,
        env: makeEnv({ PYTHONPATH: experimentsPythonPath() }),
        publicParams: {},
      };
    },
  },
  promote_candidates: {
    id: 'promote_candidates',
    label: 'Promote Config to Production',
    description: 'Copy strategy config to bocpd_production.yaml',
    group: 'Promotion',
    approach: 'bocpd',
    requiredScripts: ['scripts/promote_candidates.py'],
    details: [
      'Copies selected config to bocpd_production.yaml with backup snapshot.',
      'Records promotion metadata under artifacts/promotions/. '
    ],
    fields: [
      { name: 'source', label: 'Source Config', type: 'text', default: 'configs/run_mes_bocpd_breakout.yaml' },
      { name: 'dest', label: 'Destination', type: 'text', default: 'bocpd_production.yaml' },
      { name: 'note', label: 'Note', type: 'text', default: '' },
      { name: 'extra', label: 'Extra Metadata JSON', type: 'text', default: '' },
    ],
    buildCommand(params = {}) {
      const src = params.source || 'configs/run_mes_bocpd_breakout.yaml';
      const dest = params.dest || 'bocpd_production.yaml';
      const note = params.note ? String(params.note) : null;
      const extra = params.extra ? String(params.extra) : null;
      const cmd = [BUILD_PY, 'scripts/promote_candidates.py', '--source', src, '--dest', dest];
      if (note) cmd.push('--note', note);
      if (extra) cmd.push('--extra', extra);
      return {
        cmd,
        cwd: REPO,
        env: makeEnv({ PYTHONPATH: experimentsPythonPath() }),
        publicParams: { source: src, dest, note: note || undefined, extra: extra || undefined },
      };
    },
  },
  promote_best_validated: {
    id: 'promote_best_validated',
    label: 'Promote Best Validated',
    description: 'Promote best validated WFO run to production config',
    group: 'Promotion',
    approach: 'bocpd',
    requiredScripts: ['scripts/promote_best_validated.py'],
    details: [
      'Scans validate-* runs for best out-of-sample Sharpe.',
      'Promotes parameter mode into prod config with audit trail.'
    ],
    fields: [
      { name: 'summary', label: 'Validated Summary', type: 'text', default: 'runs/validated_candidates/summary.json' },
      { name: 'base_config', label: 'Base Config', type: 'text', default: 'configs/run_mes_bocpd_breakout.yaml' },
      { name: 'dest', label: 'Destination', type: 'text', default: 'bocpd_production.yaml' },
      { name: 'note', label: 'Note', type: 'text', default: '' },
    ],
    buildCommand(params = {}) {
      const summary = params.summary || 'runs/validated_candidates/summary.json';
      const baseConfig = params.base_config || 'configs/run_mes_bocpd_breakout.yaml';
      const dest = params.dest || 'bocpd_production.yaml';
      const note = params.note ? String(params.note) : null;
      const cmd = [
        BUILD_PY,
        'scripts/promote_best_validated.py',
        '--summary',
        summary,
        '--base-config',
        baseConfig,
        '--dest',
        dest,
      ];
      if (note) cmd.push('--note', note);
      return {
        cmd,
        cwd: REPO,
        env: makeEnv({ PYTHONPATH: experimentsPythonPath() }),
        publicParams: { summary, base_config: baseConfig, dest, note: note || undefined },
      };
    },
  },
};

function serializeDefinitions() {
  return Object.values(BUILD_DEFINITIONS).map((def) => {
    const availability = definitionAvailability(def);
    return {
      id: def.id,
      label: def.label,
      description: def.description,
      fields: def.fields,
      group: def.group || 'Other',
      details: def.details || [],
      approach: def.approach || 'other',
      enabled: availability.enabled,
      disabled_reason: availability.reason,
      missing_scripts: availability.missingScripts,
    };
  });
}

function formatStamp(date = new Date()) {
  const pad = (n) => String(n).padStart(2, '0');
  return `${date.getUTCFullYear()}${pad(date.getUTCMonth() + 1)}${pad(date.getUTCDate())}_${pad(date.getUTCHours())}${pad(date.getUTCMinutes())}${pad(date.getUTCSeconds())}`;
}

function formatLine(date = new Date()) {
  const pad = (n) => String(n).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

app.get('/api', (req, res) => {
  res.json({
    name: 'back_to_the_dashboard',
    version: '1.0.0',
    generated_at: new Date().toISOString(),
    endpoints: API_ENDPOINTS,
  });
});

app.get('/api/health', async (req, res) => {
  const definitions = serializeDefinitions();
  const disabledJobs = definitions
    .filter((def) => def.enabled === false)
    .map((def) => ({
      id: def.id,
      label: def.label,
      reason: def.disabled_reason,
      missing_scripts: def.missing_scripts,
      approach: def.approach,
    }));
  try {
    const st = await fsp.stat(REPO);
    const runsOk = await fsp.stat(readers.RUNS_ROOT).then((s) => s.isDirectory()).catch(() => false);
    res.json({
      ok: st.isDirectory() && runsOk,
      repo: REPO,
      runs_root: readers.RUNS_ROOT,
      runs_ok: runsOk,
      kanban: KANBAN,
      disabled_jobs: disabledJobs,
    });
  } catch (e) {
    const runsOk = await fsp.stat(readers.RUNS_ROOT).then((s) => s.isDirectory()).catch(() => false);
    res.json({
      ok: false,
      repo: REPO,
      runs_root: readers.RUNS_ROOT,
      runs_ok: runsOk,
      error: String(e),
      kanban: KANBAN,
      disabled_jobs: disabledJobs,
    });
  }
});

async function readJsonSafe(p) {
  try { return JSON.parse(await fsp.readFile(p, 'utf8')); } catch { return null; }
}

app.get('/api/sweeps', async (req, res) => {
  try {
    const sweepsRoot = path.join(REPO, 'artifacts', 'wfo', 'lambda_sweep');
    const entries = await fsp.readdir(sweepsRoot, { withFileTypes: true }).catch(() => []);
    const out = [];
    for (const ent of entries) {
      if (!ent.isDirectory()) continue;
      const dir = path.join(sweepsRoot, ent.name);
      const manifestPath = path.join(dir, 'manifest.json');
      const manifest = await readJsonSafe(manifestPath);
      if (manifest) {
        const st = await fsp.stat(manifestPath).catch(() => null);
        const mtime = st ? st.mtimeMs : 0;
        out.push({ key: ent.name, manifest, mtime });
      }
    }
    // Prefer mes_* keys, then most recent
    const score = (name) => name.startsWith('mes_') ? 0 : 1;
    out.sort((a, b) => (score(a.key) - score(b.key)) || (b.mtime - a.mtime));
    res.json(out);
  } catch (e) {
    res.status(500).json({ error: 'failed to list sweeps', detail: String(e) });
  }
});

app.get('/api/runs', async (req, res) => {
  try {
    const ids = await readers.listRuns();
    const items = await Promise.all(
      ids.map(async (id) => {
        try {
          const summary = await readers.readSummary(id);
          return { id, summary };
        } catch (err) {
          return { id, summary: null };
        }
      })
    );
    res.json(items);
  } catch (e) {
    res.status(500).json({ error: 'failed to list runs', detail: String(e) });
  }
});

app.get('/api/runs/:runId/summary', async (req, res) => {
  try {
    const summary = await readers.readSummary(req.params.runId);
    res.json(summary);
  } catch (e) {
    res.status(404).json({ error: 'summary not found', detail: String(e) });
  }
});

app.get('/api/runs/:runId/equity', async (req, res) => {
  try {
    const equity = await readers.readEquity(req.params.runId);
    res.json(equity);
  } catch (e) {
    res.status(404).json({ error: 'equity not found', detail: String(e) });
  }
});

app.get('/api/runs/:runId/trades', async (req, res) => {
  try {
    const limit = Number.parseInt(req.query.limit, 10) || 200;
    const offset = Number.parseInt(req.query.offset, 10) || 0;
    const trades = await readers.readTrades(req.params.runId, { limit, offset });
    res.json({ trades, limit, offset });
  } catch (e) {
    res.status(404).json({ error: 'trades not found', detail: String(e) });
  }
});

app.get('/api/runs/:runId/roundtrips', async (req, res) => {
  try {
    const trades = await readers.readTrades(req.params.runId, { limit: 1000, offset: 0 });
    const trips = readers.deriveRoundTrips(trades);
    res.json(trips);
  } catch (e) {
    res.status(404).json({ error: 'roundtrips unavailable', detail: String(e) });
  }
});

app.get('/api/runs/:runId/cpd/series', async (req, res) => {
  try {
    const downsample = Number.parseInt(req.query.downsample, 10);
    const series = await readers.readCpdSeries(req.params.runId, { downsample });
    res.json(series);
  } catch (e) {
    res.status(404).json({ error: 'cpd series unavailable', detail: String(e) });
  }
});

app.get('/api/runs/:runId/cpd/models', async (req, res) => {
  try {
    const downsample = Number.parseInt(req.query.downsample, 10);
    const models = await readers.readCpdModels(req.params.runId, { downsample });
    res.json(models);
  } catch (e) {
    res.status(404).json({ error: 'cpd models unavailable', detail: String(e) });
  }
});

app.get('/api/runs/:runId/cpd/summary', async (req, res) => {
  try {
    const summary = await readers.readCpdSummary(req.params.runId);
    res.json(summary);
  } catch (e) {
    res.status(404).json({ error: 'cpd summary unavailable', detail: String(e) });
  }
});

app.get('/api/latest-run', async (req, res) => {
  try {
    const runsRoot = path.join(REPO, 'runs');
    const entries = await fsp.readdir(runsRoot, { withFileTypes: true }).catch(() => []);
    let latest = null;
    let latestMtime = 0;
    for (const ent of entries) {
      if (!ent.isDirectory()) continue;
      const dir = path.join(runsRoot, ent.name);
      const st = await fsp.stat(dir).catch(() => null);
      if (st && st.mtimeMs > latestMtime) { latestMtime = st.mtimeMs; latest = { name: ent.name, dir }; }
    }
    if (!latest) return res.json({});
    const summary = await readJsonSafe(path.join(latest.dir, 'summary.json'));
    // Parse a few fields from resolved.yaml without adding a YAML dep
    let params = {};
    try {
      const txt = await fsp.readFile(path.join(latest.dir, 'resolved.yaml'), 'utf8');
      const rx = /\b(cp_lambda|cp_thr|break_k_atr|lookback_n|min_edge_pts):\s*([0-9.]+)/g;
      let m;
      while ((m = rx.exec(txt))) { params[m[1]] = Number(m[2]); }
    } catch {}
    res.json({ run: latest.name, summary, params });
  } catch (e) {
    res.status(500).json({ error: 'failed to read latest run', detail: String(e) });
  }
});

app.get('/api/logs', async (req, res) => {
  try {
    const logsRoot = path.join(REPO, 'logs', 'pipeline');
    const entries = await fsp.readdir(logsRoot, { withFileTypes: true }).catch(() => []);
    const files = entries.filter(e => e.isFile() && e.name.endsWith('.log')).map(e => e.name).sort();
    const file = req.query.file ? path.basename(String(req.query.file)) : null;
    if (!file) return res.json({ files });
    const p = path.join(logsRoot, file);
    const txt = await fsp.readFile(p, 'utf8').catch(() => '');
    res.type('text/plain').send(txt);
  } catch (e) {
    res.status(500).json({ error: 'failed to read logs', detail: String(e) });
  }
});

function parseTimestampStamp(stamp) {
  const m = /^(\d{8})_(\d{6})$/.exec(stamp);
  if (!m) return null;
  const [_, d, t] = m;
  const year = Number(d.slice(0, 4));
  const month = Number(d.slice(4, 6));
  const day = Number(d.slice(6, 8));
  const hour = Number(t.slice(0, 2));
  const minute = Number(t.slice(2, 4));
  const second = Number(t.slice(4, 6));
  const date = new Date(Date.UTC(year, month - 1, day, hour, minute, second));
  return Number.isFinite(date.getTime()) ? date : null;
}

function detectLogStatus(content, mtimeMs) {
  const lowered = content.toLowerCase();
  if (lowered.includes('traceback') || lowered.includes('error:') || lowered.includes('exception')) {
    return 'failed';
  }
  const now = Date.now();
  if (now - mtimeMs < 2 * 60 * 1000) {
    return 'running';
  }
  return 'success';
}

function extractLogTimestamps(content) {
  const lines = content.split(/\r?\n/);
  const regex = /^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:,\d{3})?/;
  const firstLine = lines.find(line => regex.test(line));
  const lastLine = [...lines].reverse().find(line => regex.test(line));
  const parse = (line) => {
    if (!line) return null;
    const match = regex.exec(line);
    if (!match) return null;
    const ts = match[1].replace(' ', 'T');
    const date = new Date(ts);
    return Number.isFinite(date.getTime()) ? date : null;
  };
  return { start: parse(firstLine), end: parse(lastLine) };
}

async function parseBuildLog(logsRoot, entry) {
  const name = entry.name;
  const filePath = path.join(logsRoot, name);
  const stat = await fsp.stat(filePath).catch(() => null);
  if (!stat) return null;

  let content = '';
  try {
    content = await fsp.readFile(filePath, 'utf8');
  } catch (e) {
    content = '';
  }
  if (content.length > 500_000) {
    content = content.slice(-500_000);
  }

  let job = name;
  let startedAt = null;
  const match = /^([0-9]{8}_[0-9]{6})\.(.+)\.log$/.exec(name);
  if (match) {
    const [, stamp, rest] = match;
    job = rest;
    const ts = parseTimestampStamp(stamp);
    if (ts) {
      startedAt = ts;
    }
  }

  let status = detectLogStatus(content, stat.mtimeMs);
  const { start, end } = extractLogTimestamps(content);
  const started = start || startedAt || null;
  const finished = end || (status === 'success' ? new Date(stat.mtimeMs) : null);
  let duration_s = null;
  if (started && finished) {
    const diff = (finished.getTime() - started.getTime()) / 1000;
    duration_s = Number.isFinite(diff) && diff >= 0 ? diff : null;
  }

  const info = {
    id: name,
    job,
    logfile: name,
    status,
    started_at: started ? started.toISOString() : null,
    finished_at: finished ? finished.toISOString() : null,
    duration_s,
    size_bytes: stat.size,
    updated_at: new Date(stat.mtimeMs).toISOString(),
  };

  if (BUILD_DEFINITIONS[job]) {
    info.job_label = BUILD_DEFINITIONS[job].label;
  }

  const activeEntry = activeBuilds.get(name);
  if (activeEntry) {
    const meta = activeEntry.meta || activeEntry;
    status = 'running';
    info.status = 'running';
    info.active = true;
    info.params = meta?.params || {};
    info.command = meta?.command;
    info.pid = meta?.pid;
    if (meta?.kill_requested_at) {
      info.kill_requested_at = meta.kill_requested_at;
    }
  }

  return info;
}

function summarizeBuildsByJob(builds) {
  const pick = (build) => ({
    id: build.id,
    job: build.job,
    job_label: build.job_label,
    status: build.status,
    started_at: build.started_at,
    finished_at: build.finished_at,
    duration_s: build.duration_s,
    logfile: build.logfile,
    updated_at: build.updated_at,
    active: Boolean(build.active),
    params: build.params || {},
  });

  const result = {};
  for (const build of builds) {
    const jobId = build.job;
    if (!jobId) continue;
    if (!result[jobId]) {
      result[jobId] = {
        job: jobId,
        latest: null,
        last_success: null,
        last_failed: null,
        counts: { total: 0, success: 0, failed: 0, running: 0 },
      };
    }
    const bucket = result[jobId];
    bucket.counts.total += 1;
    if (build.status === 'success') {
      bucket.counts.success += 1;
      if (!bucket.last_success) bucket.last_success = pick(build);
    } else if (build.status === 'failed') {
      bucket.counts.failed += 1;
      if (!bucket.last_failed) bucket.last_failed = pick(build);
    } else if (build.status === 'running') {
      bucket.counts.running += 1;
    }
    if (!bucket.latest) bucket.latest = pick(build);
  }
  return result;
}

function summarizeApproaches(definitions, jobSummary) {
  const map = {};
  for (const def of definitions) {
    const key = def.approach || 'other';
    if (!map[key]) {
      map[key] = {
        approach: key,
        definition_ids: [],
        counts: { total: 0, success: 0, failed: 0, running: 0 },
      };
    }
    map[key].definition_ids.push(def.id);
    const job = jobSummary[def.id];
    if (job) {
      const counts = map[key].counts;
      counts.total += job.counts.total;
      counts.success += job.counts.success;
      counts.failed += job.counts.failed;
      counts.running += job.counts.running;
    }
  }
  return map;
}

app.get('/api/builds', async (req, res) => {
  try {
    const logsRoot = path.join(REPO, 'logs', 'pipeline');
    const entries = await fsp.readdir(logsRoot, { withFileTypes: true }).catch(() => []);
    const logs = entries.filter(e => e.isFile() && e.name.endsWith('.log'));
    const builds = [];
    for (const entry of logs) {
      const info = await parseBuildLog(logsRoot, entry);
      if (info) builds.push(info);
    }
    builds.sort((a, b) => {
      const ta = a.started_at ? Date.parse(a.started_at) : 0;
      const tb = b.started_at ? Date.parse(b.started_at) : 0;
      return tb - ta;
    });
    const definitions = serializeDefinitions();
    const jobSummary = summarizeBuildsByJob(builds);
    res.json({
      builds,
      definitions,
      overview: {
        generated_at: new Date().toISOString(),
        jobs: jobSummary,
        approaches: summarizeApproaches(definitions, jobSummary),
      },
      approaches: Object.values(APPROACHES),
      active: Array.from(activeBuilds.values()).map((entry) => entry.meta || entry),
    });
  } catch (e) {
    res.status(500).json({ error: 'failed to enumerate builds', detail: String(e) });
  }
});

app.post('/api/builds/start', async (req, res) => {
  try {
    const { job_id: jobId, params = {} } = req.body || {};
    if (!jobId || !BUILD_DEFINITIONS[jobId]) {
      return res.status(400).json({ ok: false, error: 'Unknown build job' });
    }
    const def = BUILD_DEFINITIONS[jobId];
    const availability = definitionAvailability(def);
    if (!availability.enabled) {
      return res.status(400).json({
        ok: false,
        error: availability.reason || 'Job not enabled yet',
        missing_scripts: availability.missingScripts,
      });
    }
    const details = def.buildCommand ? def.buildCommand(params) : null;
    if (!details || !Array.isArray(details.cmd) || details.cmd.length === 0) {
      return res.status(400).json({ ok: false, error: 'Invalid build command' });
    }

    const logsRoot = path.join(REPO, 'logs', 'pipeline');
    await fsp.mkdir(logsRoot, { recursive: true });
    const stamp = formatStamp(new Date());
    const logName = `${stamp}.${jobId}.log`;
    const logPath = path.join(logsRoot, logName);
    const logStream = fs.createWriteStream(logPath, { flags: 'a' });
    const startLine = `${formatLine()} START job=${jobId} cmd=${details.cmd.join(' ')}\n`;
    logStream.write(startLine);

    let child;
    try {
      child = spawn(details.cmd[0], details.cmd.slice(1), {
        cwd: details.cwd || REPO,
        env: details.env || makeEnv(),
        stdio: ['ignore', 'pipe', 'pipe'],
      });
    } catch (err) {
      logStream.write(`${formatLine()} ERROR spawn ${err.message}\n`);
      logStream.end();
      return res.status(500).json({ ok: false, error: `Failed to spawn build: ${err.message}` });
    }

    child.stdout.on('data', (chunk) => logStream.write(chunk));
    child.stderr.on('data', (chunk) => logStream.write(chunk));
    child.on('close', (code) => {
      const exitLine = `${formatLine()} EXIT code=${code}\n`;
      logStream.write(exitLine);
      logStream.end();
      activeBuilds.delete(logName);
    });
    child.on('error', (err) => {
      logStream.write(`${formatLine()} ERROR child ${err.message}\n`);
      logStream.end();
      activeBuilds.delete(logName);
    });

    const commandString = details.cmd.join(' ');
    const meta = {
      job_id: jobId,
      logfile: logName,
      params: details.publicParams || params,
      pid: child.pid,
      started_at: new Date().toISOString(),
      command: commandString,
    };
    activeBuilds.set(logName, { child, meta });

    res.json({ ok: true, build: { job_id: jobId, logfile: logName, pid: child.pid } });
  } catch (e) {
    res.status(500).json({ ok: false, error: String(e) });
  }
});

app.post('/api/builds/kill', async (req, res) => {
  try {
    const { logfile } = req.body || {};
    if (!logfile || typeof logfile !== 'string') {
      return res.status(400).json({ ok: false, error: 'Missing logfile identifier' });
    }
    const entry = activeBuilds.get(logfile);
    if (!entry || !entry.child || entry.child.killed) {
      return res.status(404).json({ ok: false, error: 'No active build for specified logfile' });
    }

    const logsRoot = path.join(REPO, 'logs', 'pipeline');
    const logPath = path.join(logsRoot, logfile);
    try {
      const cancelLine = `${formatLine()} CANCEL requested via dashboard\n`;
      await fsp.appendFile(logPath, cancelLine, 'utf8');
    } catch (err) {
      // best effort; ignore append errors
    }

    if (!entry.meta) {
      entry.meta = {};
    }
    entry.meta.kill_requested_at = new Date().toISOString();

    const terminated = entry.child.kill('SIGTERM');
    if (!terminated) {
      entry.child.kill('SIGKILL');
    }

    res.json({ ok: true, logfile });
  } catch (e) {
    res.status(500).json({ ok: false, error: String(e) });
  }
});

// --- Autotune scan: parse logs/pipeline/*autotune_lam*.log into rows
function parseAutotuneLogsFromText(text, logfileName) {
  // Returns array of rows parsed from one file
  const rows = [];
  const lamMatch = /autotune_lam(\d+)\.log$/.exec(logfileName || '');
  const lam = lamMatch ? Number(lamMatch[1]) : null;
  const cmdRe = /--cp-thr\s+([0-9.,]+).*?--break-k\s+([0-9.,]+)/;
  const startRe = /^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*Running single config: .*cp_sweep_cp([0-9.]+)_k([0-9.]+)\.yaml/;
  const loadedRe = /loaded bars=(\d+) quotes=(\d+) feats=(\d+) dates=(\d+)/;
  const wroteRe = /wrote\s+(runs\/[\w\-./]+)/;
  const summaryRe = /summary\s+run_id=([^\s]+)\s+sharpe=([\-0-9.]+)\s+t=([\-0-9.]+)\s+tail=([\-0-9.]+)\s+trades=(\d+)\s+ntg=([\-0-9.]+)\s+net=([\-0-9.]+)/;
  let gridCp = null, gridK = null;
  let current = null;
  const lines = text.split(/\r?\n/);
  for (const line of lines) {
    if (gridCp === null || gridK === null) {
      const m = cmdRe.exec(line);
      if (m) {
        try { gridCp = m[1].split(',').filter(Boolean).map(Number); } catch { gridCp = null; }
        try { gridK = m[2].split(',').filter(Boolean).map(Number); } catch { gridK = null; }
      }
    }
    let ms = startRe.exec(line);
    if (ms) {
      const ts = ms[1];
      const cp = Number(ms[2]);
      const k = Number(ms[3]);
      current = {
        lam, cp, k,
        start_ts: ts.replace(',', '.000').replace(/,(\d{3})$/, '.$1'),
        start_dt: ts,
        logfile: logfileName,
        grid_cp: gridCp,
        grid_k: gridK,
      };
      continue;
    }
    if (current) {
      const mload = loadedRe.exec(line);
      if (mload) {
        current.bars = Number(mload[1]);
        current.quotes = Number(mload[2]);
        current.feats = Number(mload[3]);
        current.dates = Number(mload[4]); // trading sessions loaded
      }
      const mw = wroteRe.exec(line);
      if (mw) {
        current.run_path = mw[1]; // may include a file; still useful
      }
      const msum = summaryRe.exec(line);
      if (msum) {
        const tsE = /^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})/.exec(line);
        const endTs = tsE ? tsE[1] : null;
        const row = {
          ...current,
          run_id: msum[1],
          sharpe: Number(msum[2]),
          t: Number(msum[3]),
          tail: Number(msum[4]),
          trades: Number(msum[5]),
          ntg: Number(msum[6]),
          net: Number(msum[7]),
          end_ts: endTs ? endTs.replace(',', '.000').replace(/,(\d{3})$/, '.$1') : null,
        };
        // parse window from run_id if present
        try {
          const m = /^(.*?)-(\w+)-(\d{4}-\d{2}-\d{2})-(\d{4}-\d{2}-\d{2})-/.exec(row.run_id || '');
          if (m) { row.window_start = m[3]; row.window_end = m[4]; }
        } catch {}
        // duration (best effort)
        try {
          const start = row.start_dt;
          if (start && endTs) {
            const toMs = (s) => {
              // 'YYYY-MM-DD HH:MM:SS,mmm'
              const [ymd, hms] = s.split(' ');
              const [Y, M, D] = ymd.split('-').map(Number);
              const [H, min, Sms] = hms.split(':');
              const [S, ms] = Sms.split(',');
              return new Date(Y, M-1, D, Number(H), Number(min), Number(S), Number(ms)).getTime();
            };
            const dur = (toMs(endTs) - toMs(start)) / 1000;
            row.duration_s = Number.isFinite(dur) ? dur : null;
          }
        } catch {}
        delete row.start_dt;
        rows.push(row);
        current = null;
      }
    }
  }
  return rows;
}

app.get('/api/autotune', async (req, res) => {
  try {
    const logsRoot = path.join(REPO, 'logs', 'pipeline');
    const entries = await fsp.readdir(logsRoot, { withFileTypes: true }).catch(() => []);
    const logs = entries
      .filter(e => e.isFile() && /autotune_lam\d+\.log$/.test(e.name))
      .map(e => e.name)
      .sort();
    let rows = [];
    for (const name of logs) {
      const p = path.join(logsRoot, name);
      const text = await fsp.readFile(p, 'utf8').catch(() => '');
      const r = parseAutotuneLogsFromText(text, name);
      rows = rows.concat(r);
    }
    // sort by end_ts then start_ts
    rows.sort((a, b) => String(a.end_ts||a.start_ts||'').localeCompare(String(b.end_ts||b.start_ts||'')));
    res.json({ logs, rows });
  } catch (e) {
    res.status(500).json({ error: 'failed to parse autotune logs', detail: String(e) });
  }
});

app.get('/api/kanban/tasks', async (req, res) => {
  try {
    const board = req.query.board || 'bocpd';
    // Allow overriding the base via query for local dev (e.g. 3000 proxy)
    const base = req.query.base || KANBAN;
    const r = await fetch(`${base}/api/boards/${encodeURIComponent(board)}/tasks`);
    const js = await r.json();
    res.json(js);
  } catch (e) {
    res.status(500).json({ error: 'kanban unavailable', detail: String(e) });
  }
});

// Update Kanban task (proxy to KANBAN service)
app.patch('/api/kanban/tasks/:id', async (req, res) => {
  try {
    const id = String(req.params.id || '');
    const base = req.query.base || KANBAN;
    const r = await fetch(`${base}/api/boards/bocpd/tasks/${encodeURIComponent(id)}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(req.body||{}) });
    const js = await r.json();
    res.status(r.status).json(js);
  } catch (e) {
    res.status(500).json({ error: 'kanban unavailable', detail: String(e) });
  }
});

// --- Runtime monitors: lightweight endpoint that serves the latest runtime metrics
// Looks for JSON or JSONL files under artifacts/monitors or runs/
async function readLastJsonl(p) {
  try {
    const txt = await fsp.readFile(p, 'utf8');
    const lines = txt.trim().split(/\r?\n/).filter(Boolean);
    for (let i = lines.length - 1; i >= 0; i--) {
      try { return JSON.parse(lines[i]); } catch {}
    }
  } catch {}
  return null;
}

app.get('/api/monitors/runtime', async (req, res) => {
  try {
    const candidates = [
      path.join(REPO, 'artifacts', 'monitors', 'runtime.json'),
      path.join(REPO, 'artifacts', 'monitors', 'runtime.jsonl'),
      path.join(REPO, 'runs', 'runtime_metrics.json'),
      path.join(REPO, 'runs', 'runtime.jsonl'),
    ];
    let data = null; let source = null;
    for (const p of candidates) {
      if (!fs.existsSync(p)) continue;
      if (p.endsWith('.json')) {
        data = await readJsonSafe(p);
        source = p; if (data) break;
      } else {
        data = await readLastJsonl(p);
        source = p; if (data) break;
      }
    }
    res.json({ ok: !!data, source, data: data || {} });
  } catch (e) {
    res.status(500).json({ error: 'failed to read runtime monitors', detail: String(e) });
  }
});

// Sweep results: reads blended.csv (preferred) or metrics.csv and returns {rows, best}
// List available cp-threshold sweep directories under artifacts/sweeps (e.g., cp_thr, cp_thr_lam40, cp_thr_lam60)
app.get('/api/cp_sweeps', async (req, res) => {
  try {
    const sweepsRoot = path.join(REPO, 'artifacts', 'sweeps');
    const entries = await fsp.readdir(sweepsRoot, { withFileTypes: true }).catch(() => []);
    const out = [];
    for (const ent of entries) {
      if (!ent.isDirectory()) continue;
      if (!ent.name.startsWith('cp_thr')) continue;
      const dir = path.join(sweepsRoot, ent.name);
      const blended = path.join(dir, 'blended.csv');
      const metrics = path.join(dir, 'metrics.csv');
      const p = fs.existsSync(blended) ? blended : (fs.existsSync(metrics) ? metrics : null);
      if (!p) continue;
      const st = await fsp.stat(p).catch(() => null);
      out.push({ key: ent.name, path: p, mtime: st ? st.mtimeMs : 0 });
    }
    out.sort((a,b) => b.mtime - a.mtime || String(a.key).localeCompare(String(b.key)));
    res.json(out);
  } catch (e) {
    res.status(500).json({ error: 'failed to list cp sweeps', detail: String(e) });
  }
});

// Read a specific cp-threshold sweep CSV (supports ?key=cp_thr_lam60). Falls back to cp_thr or newest available.
app.get('/api/cp_sweep', async (req, res) => {
  try {
    const key = String(req.query.key || 'cp_thr');
    const sweepsRoot = path.join(REPO, 'artifacts', 'sweeps');
    async function resolvePath(k) {
      const root = path.join(sweepsRoot, k);
      const blended = path.join(root, 'blended.csv');
      const metrics = path.join(root, 'metrics.csv');
      if (fs.existsSync(blended)) return blended;
      if (fs.existsSync(metrics)) return metrics;
      return null;
    }
    let p = await resolvePath(key);
    if (!p) {
      // fallback: try base cp_thr
      p = await resolvePath('cp_thr');
    }
    if (!p) {
      // fallback: pick newest cp_thr* directory with a usable file
      const entries = await fsp.readdir(sweepsRoot, { withFileTypes: true }).catch(() => []);
      const candidates = [];
      for (const ent of entries) {
        if (!ent.isDirectory()) continue;
        if (!ent.name.startsWith('cp_thr')) continue;
        const alt = await resolvePath(ent.name);
        if (alt) {
          const st = await fsp.stat(alt).catch(() => null);
          candidates.push({ key: ent.name, path: alt, mtime: st ? st.mtimeMs : 0 });
        }
      }
      candidates.sort((a,b) => b.mtime - a.mtime);
      if (candidates.length) p = candidates[0].path;
    }
    if (!p) return res.json({ rows: [], best: null, key: null, path: null });
    const text = await fsp.readFile(p, 'utf8');
    const lines = text.trim().split(/\r?\n/);
    const header = lines[0].split(',');
    const rows = lines.slice(1).map(l => {
      const vals = l.split(',');
      const obj = {}; header.forEach((h, i) => obj[h] = vals[i]);
      // cast selected fields
      ['after_cost_sharpe','net','trade_count','blended','cp_thr','break_k_atr'].forEach(k => {
        if (obj[k] !== undefined && obj[k] !== '') obj[k] = Number(obj[k]);
      });
      return obj;
    });
    let best = null;
    if (rows.length) {
      if (rows[0].blended !== undefined) {
        best = rows.reduce((a, b) => (Number(a.blended||-1e9) >= Number(b.blended||-1e9) ? a : b));
      } else {
        // fallback: pick by after_cost_sharpe then net
        best = rows.reduce((a, b) => (Number(a.after_cost_sharpe||-1e9) > Number(b.after_cost_sharpe||-1e9) ? a : (Number(a.after_cost_sharpe||-1e9) < Number(b.after_cost_sharpe||-1e9) ? b : (Number(a.net||-1e9) >= Number(b.net||-1e9) ? a : b))));
      }
    }
    res.json({ rows, best, path: p, key });
  } catch (e) {
    res.status(500).json({ error: 'failed to read sweep', detail: String(e) });
  }
});

// Dataset validation summary
app.get('/api/dataset/validation', async (req, res) => {
  try {
    const p = path.join(REPO, 'artifacts', 'validation', 'dataset.json');
    if (!fs.existsSync(p)) return res.json({ ok: false, files: 0, results: [] });
    const js = JSON.parse(await fsp.readFile(p, 'utf8'));
    res.json(js);
  } catch (e) {
    res.status(500).json({ error: 'failed to read dataset validation', detail: String(e) });
  }
});

// --- Candidates (from sweeps) ---
function readJsonl(path) {
  try {
    const txt = fs.readFileSync(path, 'utf8');
    const out = [];
    for (const line of txt.split(/\r?\n/)) {
      const s = line.trim(); if (!s) continue;
      try { out.push(JSON.parse(s)); } catch {}
    }
    return out;
  } catch { return []; }
}

app.get('/api/candidates/promoted', async (req, res) => {
  try {
    const p = path.join(REPO, 'runs', 'promoted_candidates.jsonl');
    const rows = readJsonl(p);
    res.json({ path: p, count: rows.length, rows });
  } catch (e) {
    res.status(500).json({ error: 'failed to read candidates', detail: String(e) });
  }
});

app.get('/api/candidates/best', async (req, res) => {
  try {
    const p = path.join(REPO, 'runs', 'promoted_candidates.jsonl');
    const rows = readJsonl(p);
    // best-by-lambda (cp_lambda) using after_cost_sharpe then net, with trade_count guard
    const byLam = new Map();
    function score(r){ const m=r.metrics||{}; const tc=Number(m.trade_count||0); const acs=Number(m.after_cost_sharpe||0); const net=Number(m.net||0); return (tc<5?-1e9:acs)*1e6 + net; }
    for (const r of rows) {
      const lam = r.params && r.params.cp_lambda != null ? Number(r.params.cp_lambda) : null;
      const key = String(lam);
      if (!byLam.has(key) || score(r) > score(byLam.get(key))) byLam.set(key, r);
    }
    const best = Array.from(byLam.values()).sort((a,b)=> Number((b.params||{}).cp_lambda||0) - Number((a.params||{}).cp_lambda||0));
    res.json({ path: p, best });
  } catch (e) {
    res.status(500).json({ error: 'failed to compute best candidates', detail: String(e) });
  }
});

// --- Promotion status ---
function readPromotionHistory() {
  const dir = path.join(REPO, 'artifacts', 'promotions');
  try {
    const ents = fs.readdirSync(dir, { withFileTypes: true });
    const files = ents.filter(e => e.isFile() && e.name.startsWith('promotion_') && e.name.endsWith('.json'))
      .map(e => path.join(dir, e.name))
      .sort((a,b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs);
    const rows = files.slice(0, 20).map(p => { try { return JSON.parse(fs.readFileSync(p,'utf8')); } catch { return null; } }).filter(Boolean);
    return rows;
  } catch { return []; }
}

function parseProductionYaml(p) {
  // Minimal YAML field extraction: only reads a few strategy_params
  try {
    const txt = fs.readFileSync(p, 'utf8');
    const out = {};
    const grab = (key) => {
      const m = new RegExp(`\\b${key}\\s*:\\s*([0-9.]+)`).exec(txt);
      return m ? Number(m[1]) : null;
    };
    out.cp_lambda = grab('cp_lambda');
    out.cp_thr = grab('cp_thr');
    out.break_k_atr = grab('break_k_atr');
    out.trail_t_atr = grab('trail_t_atr');
    out.stop_s_atr = grab('stop_s_atr');
    out.max_hold_s = grab('max_hold_s');
    return out;
  } catch { return {}; }
}

app.get('/api/promotion/status', async (req, res) => {
  try {
    const prod = path.join(REPO, 'bocpd_production.yaml');
    const exists = fs.existsSync(prod);
    const params = exists ? parseProductionYaml(prod) : {};
    const history = readPromotionHistory();
    const last = history.length ? history[0] : null;
    res.json({ exists, path: exists ? prod : null, params, last, historyCount: history.length });
  } catch (e) {
    res.status(500).json({ error: 'failed to read promotion status', detail: String(e) });
  }
});

// --- Detection metrics (quality) ---
app.get('/api/quality/detection', async (req, res) => {
  try {
    const p = path.join(REPO, 'artifacts', 'reports', 'detection_metrics.json');
    if (!fs.existsSync(p)) return res.json({ ok: false, path: p, metrics: null });
    const txt = await fsp.readFile(p, 'utf8');
    let js = null; try { js = JSON.parse(txt); } catch {}
    const st = await fsp.stat(p).catch(() => null);
    const mtime = st ? st.mtimeMs : null;
    res.json({ ok: true, path: p, mtime, metrics: js });
  } catch (e) {
    res.status(500).json({ error: 'failed to read detection metrics', detail: String(e) });
  }
});

app.get('/api/quality/ttnc_tod', async (req, res) => {
  try {
    const p = path.join(REPO, 'artifacts', 'reports', 'ttnc_tod.json');
    if (!fs.existsSync(p)) return res.json({ ok: false, path: p, buckets: [] });
    const txt = await fsp.readFile(p, 'utf8');
    let js = null; try { js = JSON.parse(txt); } catch {}
    const buckets = (js && js.buckets) || [];
    res.json({ ok: true, path: p, buckets });
  } catch (e) {
    res.status(500).json({ error: 'failed to read ttnc_tod', detail: String(e) });
  }
});

// --- WFO endpoints: discover runs and fetch metrics ---
app.get('/api/wfo/runs', async (req, res) => {
  try {
    const runsRoot = path.join(REPO, 'runs');
    const ents = await fsp.readdir(runsRoot, { withFileTypes: true }).catch(() => []);
    const out = [];
    for (const ent of ents) {
      if (!ent.isDirectory()) continue;
      const dir = path.join(runsRoot, ent.name);
      const wfoPath = path.join(dir, 'wfo_metrics.csv');
      const oosPath = path.join(dir, 'oos_equity.csv');
      if (fs.existsSync(wfoPath)) {
        const st = await fsp.stat(wfoPath).catch(() => null);
        const mtime = st ? st.mtimeMs : 0;
        out.push({ name: ent.name, wfo_metrics: wfoPath, oos_equity: fs.existsSync(oosPath) ? oosPath : null, mtime });
      }
    }
    out.sort((a,b) => b.mtime - a.mtime || String(b.name).localeCompare(String(a.name)));
    res.json(out);
  } catch (e) {
    res.status(500).json({ error: 'failed to list WFO runs', detail: String(e) });
  }
});

app.get('/api/wfo/metrics', async (req, res) => {
  try {
    const run = String(req.query.run || '');
    if (!run) return res.status(400).json({ error: 'missing run' });
    const dir = path.join(REPO, 'runs', path.basename(run));
    const wfoPath = path.join(dir, 'wfo_metrics.csv');
    if (!fs.existsSync(wfoPath)) return res.json({ run, metrics: [], equity: [] });
    const text = await fsp.readFile(wfoPath, 'utf8');
    const lines = text.trim().split(/\r?\n/);
    const header = lines[0].split(',');
    const rows = lines.slice(1).map(l => {
      const vals = l.split(',');
      const obj = {}; header.forEach((h, i) => obj[h] = vals[i]);
      // cast
      const numCols = ['n_train','n_test','after_cost_sharpe','t_stat','tail_ratio'];
      numCols.forEach(k => { if (obj[k] !== undefined && obj[k] !== '') obj[k] = Number(obj[k]); });
      // param_* columns
      Object.keys(obj).forEach(k => {
        if (k.startsWith('param_')) {
          const v = obj[k];
          const nv = Number(v);
          if (!Number.isNaN(nv)) obj[k] = nv;
        }
      });
      return obj;
    });
    let equity = [];
    const oosPath = path.join(dir, 'oos_equity.csv');
    if (fs.existsSync(oosPath)) {
      try {
        const etxt = await fsp.readFile(oosPath, 'utf8');
        const elines = etxt.trim().split(/\r?\n/);
        const eh = elines[0].split(',');
        equity = elines.slice(1).map(l => { const v = l.split(','); const o = {}; eh.forEach((h,i)=>o[h]=v[i]); if (o.equity!==undefined) o.equity = Number(o.equity); if (o.step!==undefined) o.step = Number(o.step); return o; });
      } catch {}
    }
    res.json({ run, metrics: rows, equity });
  } catch (e) {
    res.status(500).json({ error: 'failed to read WFO metrics', detail: String(e) });
  }
});

// Serve selected run artifacts securely (report.html, summary.json, resolved.yaml, equity.csv)
app.get('/runs/:rid/:file', async (req, res) => {
  try {
    const rid = String(req.params.rid || '');
    const file = path.basename(String(req.params.file || ''));
    const ALLOWED = new Set(['report.html', 'summary.json', 'resolved.yaml', 'equity.csv']);
    if (!rid || !ALLOWED.has(file)) return res.status(400).send('Bad request');
    if (rid.includes('..') || rid.includes('/')) return res.status(400).send('Bad rid');
  const p = path.join(REPO, 'runs', rid, file);
  await fsp.access(p).catch(() => { throw new Error('not_found'); });
  res.set('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
  res.set('Pragma', 'no-cache');
  res.set('Expires', '0');
  res.sendFile(p);
  } catch (e) {
    res.status(404).send('Not found');
  }
});

// Alias for sweeps (compat)
app.get('/api/sweep', async (req, res) => {
  try {
    const sweepsRoot = path.join(REPO, 'artifacts', 'wfo', 'lambda_sweep');
    const entries = await fsp.readdir(sweepsRoot, { withFileTypes: true }).catch(() => []);
    const out = [];
    for (const ent of entries) {
      if (!ent.isDirectory()) continue;
      const dir = path.join(sweepsRoot, ent.name);
      const manifestPath = path.join(dir, 'manifest.json');
      const manifest = await readJsonSafe(manifestPath);
      if (manifest) {
        const st = await fsp.stat(manifestPath).catch(() => null);
        const mtime = st ? st.mtimeMs : 0;
        out.push({ key: ent.name, manifest, mtime });
      }
    }
    // Prefer mes_* keys, then most recent
    const score = (name) => name.startsWith('mes_') ? 0 : 1;
    out.sort((a, b) => (score(a.key) - score(b.key)) || (b.mtime - a.mtime));
    res.json(out);
  } catch (e) {
    res.status(500).json({ error: 'failed to list sweeps', detail: String(e) });
  }
});

// Provide lambda sweep metrics parsed from the manifest's metrics_csv
app.get('/api/sweep/metrics', async (req, res) => {
  try {
    const key = String(req.query.key || '');
    const sweepsRoot = path.join(REPO, 'artifacts', 'wfo', 'lambda_sweep');
    const manifestPath = path.join(sweepsRoot, key || 'mes_1s', 'manifest.json');
    const manifest = await readJsonSafe(manifestPath);
    if (!manifest) return res.json({ key, metrics: [] });
    const bucket = (manifest.buckets && manifest.buckets.all) || Object.values(manifest.buckets||{})[0];
    const metricsCsv = bucket && bucket.metrics_csv;
    if (!metricsCsv) return res.json({ key, metrics: [] });
    const p = path.join(REPO, metricsCsv);
    const text = await fsp.readFile(p, 'utf8');
    const lines = text.trim().split(/\r?\n/);
    const header = lines[0].split(',');
    const rows = lines.slice(1).map(l => {
      const vals = l.split(',');
      const obj = {}; header.forEach((h, i) => obj[h] = vals[i]);
      ['lambda','score','mean_seg','cp_count','n'].forEach(k => { if (obj[k] !== undefined) obj[k] = Number(obj[k]); });
      return obj;
    });
    res.json({ key: key || 'mes_1s', metrics: rows });
  } catch (e) {
    res.status(500).json({ error: 'failed to read sweep metrics', detail: String(e) });
  }
});

// --- Ablation results: confirmation vs baseline vs cooldown ---
app.get('/api/ablation/confirm', async (req, res) => {
  try {
    const p = path.join(REPO, 'artifacts', 'ablation', 'confirm_vs_baseline.csv');
    if (!fs.existsSync(p)) return res.json({ path: p, rows: [] });
    const text = await fsp.readFile(p, 'utf8');
    const lines = text.trim().split(/\r?\n/);
    if (!lines.length) return res.json({ path: p, rows: [] });
    const header = lines[0].split(',').map(s => s.trim());
    const rows = lines.slice(1).map(l => {
      const vals = l.split(',');
      const obj = {}; header.forEach((h, i) => obj[h] = vals[i]);
      // Best-effort numeric casting
      Object.keys(obj).forEach(k => { const v = obj[k]; const n = Number(v); if (v !== '' && !Number.isNaN(n)) obj[k] = n; });
      return obj;
    });
    res.json({ path: p, rows });
  } catch (e) {
    res.status(500).json({ error: 'failed to read ablation results', detail: String(e) });
  }
});

if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`Dashboard listening on http://localhost:${PORT} (repo=${REPO})`);
  });
}

module.exports = app;
