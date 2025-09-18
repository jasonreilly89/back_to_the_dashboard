const express = require('express');
const cors = require('cors');
const fs = require('fs');
const fsp = require('fs/promises');
const path = require('path');
const { spawn } = require('child_process');

const app = express();
const PORT = process.env.PORT || 4100;

// Where the TEST_BOCPD repo lives so we can read artifacts
const REPO = process.env.TEST_BOCPD_DIR || '/home/jason/ml/test_bocpd';
const KANBAN = process.env.KANBAN_BASE || 'http://localhost:4000';
const BUILD_PY = process.env.BUILD_PY || '/home/jason/venvs/torchbuild/bin/python';
const SEQUENCE_MODEL_DIR = process.env.SEQUENCE_MODEL_DIR || 'artifacts/models/sequence_cnn';
const BOOSTING_MODEL_DIR = process.env.BOOSTING_MODEL_DIR || 'artifacts/models/boosting';
const HMM_LABEL_DIR = process.env.HMM_LABEL_DIR || 'artifacts/labels/hmm';

const activeBuilds = new Map();

const API_ENDPOINTS = [
  { method: 'GET', path: '/api', description: 'List available REST endpoints' },
  { method: 'GET', path: '/api/health', description: 'Check repository and Kanban wiring' },
  { method: 'GET', path: '/api/builds', description: 'Enumerate build logs and pipeline metadata' },
  { method: 'POST', path: '/api/builds/start', description: 'Trigger a pipeline job' },
  { method: 'POST', path: '/api/builds/kill', description: 'Request termination of an active job' },
  { method: 'GET', path: '/api/sweeps', description: 'List lambda sweep manifests' },
  { method: 'GET', path: '/api/runs', description: 'Enumerate recent backtest runs' },
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

const APPROACHES = {
  bocpd: {
    id: 'bocpd',
    label: 'BOCPD Pipeline',
    description:
      'Bayesian online change-point detection workflow spanning discovery, tuning, validation, and promotion.',
  },
  boosting: {
    id: 'boosting',
    label: 'Boosting & Trees',
    description: 'Gradient-boosted baselines and ensemble comparators for change-point gating.',
  },
  sequence: {
    id: 'sequence',
    label: 'Sequence Models',
    description: 'Temporal CNN/RNN approaches operating on causal feature windows.',
  },
  hmm: {
    id: 'hmm',
    label: 'HMM / Regime Switching',
    description: 'Hidden Markov and probabilistic regime-switching pipelines.',
  },
};

const BUILD_DEFINITIONS = {
  eval_metrics: {
    id: 'eval_metrics',
    label: 'Detection Quality Audit',
    description: 'Compute headline BOCPD detection metrics over MES labels parquet',
    group: 'Evaluation & Metrics',
    approach: 'bocpd',
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
    details: [
      'Runs purged walk-forward optimisation over cp_thr & break_k grids.',
      'Writes tolerance metrics per step to artifacts/wfo/.',
      'Use run ID to differentiate scenarios.'
    ],
    fields: [
      { name: 'base_config', label: 'Base Config', type: 'text', default: 'configs/_local.bocpd_gated.yaml' },
      { name: 'out_run_id', label: 'Run ID', type: 'text', default: 'wfo_bocpd_gated' },
      { name: 'cp_thr', label: 'cp_thr grid', type: 'text', default: '0.05,0.08,0.12' },
      { name: 'break_k', label: 'break_k grid', type: 'text', default: '0.03,0.05' },
    ],
    buildCommand(params = {}) {
      const baseConfig = params.base_config || 'configs/_local.bocpd_gated.yaml';
      const outRun = params.out_run_id || 'wfo_bocpd_gated';
      const cpThr = params.cp_thr || '0.05,0.08,0.12';
      const breakK = params.break_k || '0.03,0.05';
      return {
        cmd: [BUILD_PY, 'scripts/run_wfo.py', '--base-config', baseConfig, '--out-run-id', outRun, '--cp-thr', cpThr, '--break-k', breakK],
        cwd: REPO,
        env: makeEnv({ PYTHONPATH: '/home/jason/ml/sparrow/src' }),
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
    details: [
      'Loads promoted candidates from runs/promoted_candidates.jsonl.',
      'Executes Purged-CV + WFO validation; writes cv/wfo metrics per run.'
    ],
    fields: [
      { name: 'base_config', label: 'Base Config', type: 'text', default: 'configs/_local.bocpd_gated.yaml' },
      { name: 'profile', label: 'Profile', type: 'select', options: [
        { value: 'production', label: 'Production' },
        { value: 'discovery', label: 'Discovery' },
      ], default: 'production' },
      { name: 'max_workers', label: 'Max Workers', type: 'number', default: 8 },
    ],
    buildCommand(params = {}) {
      const baseConfig = params.base_config || 'configs/_local.bocpd_gated.yaml';
      const profile = params.profile || 'production';
      const maxWorkers = Number.isFinite(Number(params.max_workers)) ? String(Number(params.max_workers)) : '8';
      return {
        cmd: [BUILD_PY, 'scripts/validate_candidates.py', '--base-config', baseConfig, '--profile', profile, '--max-workers', maxWorkers],
        cwd: REPO,
        env: makeEnv({ PYTHONPATH: '/home/jason/ml/sparrow/src' }),
        publicParams: { base_config: baseConfig, profile, max_workers: Number(maxWorkers) },
      };
    },
  },
  export_candidates: {
    id: 'export_candidates',
    label: 'Export Sweep Candidates',
    description: 'Collect best cp_thr sweeps into promoted_candidates.jsonl',
    group: 'Discovery & Tuning',
    approach: 'bocpd',
    details: [
      'Scans artifacts/sweeps/** for best cp_thr combinations.',
      'Appends results into runs/promoted_candidates.jsonl.'
    ],
    fields: [],
    buildCommand() {
      return {
        cmd: [BUILD_PY, 'scripts/export_candidates_from_sweeps.py'],
        cwd: REPO,
        env: makeEnv(),
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
    details: [
      'Copies selected config to bocpd_production.yaml with backup snapshot.',
      'Records promotion metadata under artifacts/promotions/. '
    ],
    fields: [
      { name: 'config', label: 'Source Config', type: 'text', default: 'configs/_local.bocpd_gated.yaml' },
      { name: 'dest', label: 'Destination', type: 'text', default: 'bocpd_production.yaml' },
    ],
    buildCommand(params = {}) {
      const src = params.config || 'configs/_local.bocpd_gated.yaml';
      const dest = params.dest || 'bocpd_production.yaml';
      return {
        cmd: [BUILD_PY, 'scripts/promote_candidates.py', '--config', src, '--dest', dest],
        cwd: REPO,
        env: makeEnv(),
        publicParams: { config: src, dest },
      };
    },
  },
  promote_best_validated: {
    id: 'promote_best_validated',
    label: 'Promote Best Validated',
    description: 'Promote best validated WFO run to production config',
    group: 'Promotion',
    approach: 'bocpd',
    details: [
      'Scans validate-* runs for best out-of-sample Sharpe.',
      'Promotes parameter mode into prod config with audit trail.'
    ],
    fields: [],
    buildCommand() {
      return {
        cmd: [BUILD_PY, 'scripts/promote_best_validated.py'],
        cwd: REPO,
        env: makeEnv(),
        publicParams: {},
      };
    },
  },
  boosting_train: {
    id: 'boosting_train',
    label: 'Boosting Model Training',
    description: 'Train gradient-boosted baseline on BOCPD labels.',
    group: 'Boosting & Trees',
    approach: 'boosting',
    enabled: true,
    details: [
      'Fits HistGradientBoosting or RandomForest baselines over discovery labels.',
      'Writes model + training metrics under artifacts/models/boosting/.',
    ],
    fields: [
      { name: 'dataset', label: 'Dataset Manifest', type: 'text', default: 'artifacts/datasets/mes_1s/manifest.json' },
      { name: 'labels', label: 'Labels Parquet', type: 'text', default: 'artifacts/bocpd_discovery/mes_1s/bocpd_signals.parquet' },
      { name: 'out', label: 'Output Dir', type: 'text', default: 'artifacts/models/boosting' },
      { name: 'model_type', label: 'Model Type', type: 'select', options: [
        { value: 'hist_gbdt', label: 'HistGradientBoosting' },
        { value: 'random_forest', label: 'Random Forest' },
      ], default: 'hist_gbdt' },
      { name: 'n_estimators', label: 'Estimators', type: 'number', default: 300 },
      { name: 'learning_rate', label: 'Learning Rate', type: 'number', default: 0.05 },
      { name: 'max_depth', label: 'Max Depth', type: 'number', default: 6 },
    ],
    buildCommand(params = {}) {
      const dataset = params.dataset || 'artifacts/datasets/mes_1s/manifest.json';
      const labels = params.labels || 'artifacts/bocpd_discovery/mes_1s/bocpd_signals.parquet';
      const outDir = params.out || 'artifacts/models/boosting';
      const modelType = params.model_type || 'hist_gbdt';
      const nEstimators = Number.isFinite(Number(params.n_estimators)) ? String(Math.max(10, Math.floor(Number(params.n_estimators)))) : '300';
      const learningRate = params.learning_rate !== undefined ? String(Number(params.learning_rate)) : '0.05';
      const maxDepth = params.max_depth !== undefined ? String(Math.floor(Number(params.max_depth))) : '6';
      return {
        cmd: [
          BUILD_PY,
          'scripts/train_boosting_model.py',
          '--dataset-manifest', dataset,
          '--labels', labels,
          '--out-dir', outDir,
          '--model-type', modelType,
          '--n-estimators', nEstimators,
          '--learning-rate', learningRate,
          '--max-depth', maxDepth,
        ],
        cwd: REPO,
        env: makeEnv({ PYTHONPATH: '/home/jason/ml/sparrow/src:/home/jason/ml/regime_detection/src' }),
        publicParams: {
          dataset,
          labels,
          out: outDir,
          model_type: modelType,
          n_estimators: Number(nEstimators),
          learning_rate: Number(learningRate),
          max_depth: Number(maxDepth),
        },
      };
    },
  },
  boosting_eval: {
    id: 'boosting_eval',
    label: 'Boosting Model Evaluation',
    description: 'Evaluate trained boosting model on holdout data.',
    group: 'Boosting & Trees',
    approach: 'boosting',
    enabled: true,
    details: [
      'Computes ROC/PR metrics and top alerts from boosting model outputs.',
      'Writes evaluation_metrics.json alongside model artifacts.',
    ],
    fields: [
      { name: 'model_dir', label: 'Model Dir', type: 'text', default: 'artifacts/models/boosting' },
      { name: 'dataset', label: 'Dataset Manifest', type: 'text', default: 'artifacts/datasets/mes_1s/manifest.json' },
      { name: 'labels', label: 'Labels Parquet', type: 'text', default: 'artifacts/bocpd_discovery/mes_1s/bocpd_signals.parquet' },
      { name: 'out', label: 'Output JSON', type: 'text', default: 'artifacts/models/boosting/evaluation_metrics.json' },
      { name: 'max_samples', label: 'Max Samples', type: 'number', default: 150000 },
    ],
    buildCommand(params = {}) {
      const modelDir = params.model_dir || 'artifacts/models/boosting';
      const dataset = params.dataset || 'artifacts/datasets/mes_1s/manifest.json';
      const labels = params.labels || 'artifacts/bocpd_discovery/mes_1s/bocpd_signals.parquet';
      const outFile = params.out || path.join(modelDir, 'evaluation_metrics.json');
      const maxSamples = Number.isFinite(Number(params.max_samples)) ? String(Math.max(0, Math.floor(Number(params.max_samples)))) : '150000';
      return {
        cmd: [
          BUILD_PY,
          'scripts/evaluate_boosting_model.py',
          '--model-dir', modelDir,
          '--dataset-manifest', dataset,
          '--labels', labels,
          '--out', outFile,
          '--max-samples', maxSamples,
        ],
        cwd: REPO,
        env: makeEnv({ PYTHONPATH: '/home/jason/ml/sparrow/src:/home/jason/ml/regime_detection/src' }),
        publicParams: {
          model_dir: modelDir,
          dataset,
          labels,
          out: outFile,
          max_samples: Number(maxSamples),
        },
      };
    },
  },
  sequence_train: {
    id: 'sequence_train',
    label: 'Sequence Model Training',
    description: 'Train causal temporal CNN using seq_models pipeline.',
    group: 'Sequence Models',
    approach: 'sequence',
    enabled: true,
    details: [
      'Consumes feature windows and emits latency comparison report.',
      'Drops trained weights under artifacts/models/sequence_cnn/.',
    ],
    fields: [
      { name: 'dataset', label: 'Dataset Manifest', type: 'text', default: 'artifacts/datasets/mes_1s/manifest.json' },
      { name: 'labels', label: 'Labels Parquet', type: 'text', default: 'artifacts/bocpd_discovery/mes_1s/bocpd_signals.parquet' },
      { name: 'out', label: 'Output Dir', type: 'text', default: 'artifacts/models/sequence_cnn' },
      { name: 'max_samples', label: 'Max Samples', type: 'number', default: 200000 },
      { name: 'window', label: 'Window', type: 'number', default: 64 },
    ],
    buildCommand(params = {}) {
      const dataset = params.dataset || 'artifacts/datasets/mes_1s/manifest.json';
      const labels = params.labels || 'artifacts/bocpd_discovery/mes_1s/bocpd_signals.parquet';
      const outDir = params.out || 'artifacts/models/sequence_cnn';
      const maxSamples = Number.isFinite(Number(params.max_samples)) ? String(Math.max(0, Math.floor(Number(params.max_samples)))) : '200000';
      const window = Number.isFinite(Number(params.window)) ? String(Math.max(2, Math.floor(Number(params.window)))) : '64';
      return {
        cmd: [
          BUILD_PY,
          'scripts/train_sequence_model.py',
          '--dataset-manifest',
          dataset,
          '--labels',
          labels,
          '--out-dir',
          outDir,
          '--max-samples',
          maxSamples,
          '--window',
          window,
        ],
        cwd: REPO,
        env: makeEnv({ PYTHONPATH: '/home/jason/ml/sparrow/src:/home/jason/ml/regime_detection/src' }),
        publicParams: {
          dataset,
          labels,
          out: outDir,
          max_samples: Number(maxSamples),
          window: Number(window),
        },
      };
    },
  },
  sequence_eval: {
    id: 'sequence_eval',
    label: 'Sequence Model Evaluation',
    description: 'Run holdout evaluation on the trained sequence model and emit metrics.',
    group: 'Sequence Models',
    approach: 'sequence',
    enabled: true,
    details: [
      'Computes ROC/PR metrics and top alerts for the latest checkpoint.',
      'Writes evaluation report alongside training artifacts.',
    ],
    fields: [
      { name: 'model_dir', label: 'Model Dir', type: 'text', default: 'artifacts/models/sequence_cnn' },
      { name: 'dataset', label: 'Dataset Manifest', type: 'text', default: 'artifacts/datasets/mes_1s/manifest.json' },
      { name: 'labels', label: 'Labels Parquet', type: 'text', default: 'artifacts/bocpd_discovery/mes_1s/bocpd_signals.parquet' },
      { name: 'out', label: 'Output JSON', type: 'text', default: 'artifacts/models/sequence_cnn/evaluation_metrics.json' },
      { name: 'max_samples', label: 'Max Samples', type: 'number', default: 150000 },
    ],
    buildCommand(params = {}) {
      const modelDir = params.model_dir || 'artifacts/models/sequence_cnn';
      const dataset = params.dataset || 'artifacts/datasets/mes_1s/manifest.json';
      const labels = params.labels || 'artifacts/bocpd_discovery/mes_1s/bocpd_signals.parquet';
      const outFile = params.out || path.join(modelDir, 'evaluation_metrics.json');
      const maxSamples = Number.isFinite(Number(params.max_samples)) ? String(Math.max(0, Math.floor(Number(params.max_samples)))) : '150000';
      return {
        cmd: [
          BUILD_PY,
          'scripts/evaluate_sequence_model.py',
          '--model-dir',
          modelDir,
          '--dataset-manifest',
          dataset,
          '--labels',
          labels,
          '--out',
          outFile,
          '--max-samples',
          maxSamples,
        ],
        cwd: REPO,
        env: makeEnv({ PYTHONPATH: '/home/jason/ml/sparrow/src:/home/jason/ml/regime_detection/src' }),
        publicParams: {
          model_dir: modelDir,
          dataset,
          labels,
          out: outFile,
          max_samples: Number(maxSamples),
        },
      };
    },
  },
  hmm_train: {
    id: 'hmm_train',
    label: 'Hidden Markov Fit',
    description: 'Run sticky HMM labeling over feature series and emit change points.',
    group: 'Probabilistic Models',
    approach: 'hmm',
    enabled: true,
    details: [
      'Derives latent regimes via sticky HMM and writes state/change-point artifacts.',
    ],
    fields: [
      { name: 'dataset', label: 'Dataset Manifest', type: 'text', default: 'artifacts/datasets/mes_1s/manifest.json' },
      { name: 'out', label: 'Output Dir', type: 'text', default: 'artifacts/labels/hmm' },
      { name: 'series_col', label: 'Series Column', type: 'text', default: 'trend_snr' },
      { name: 'compute_trend_snr', label: 'Compute SNR', type: 'select', options: [
        { value: 'true', label: 'Yes' },
        { value: 'false', label: 'No' },
      ], default: 'true' },
      { name: 'n_states', label: 'States', type: 'number', default: 3 },
    ],
    buildCommand(params = {}) {
      const dataset = params.dataset || 'artifacts/datasets/mes_1s/manifest.json';
      const outDir = params.out || 'artifacts/labels/hmm';
      const seriesCol = params.series_col || 'trend_snr';
      const computeSnr = String(params.compute_trend_snr || 'true').toLowerCase() === 'true';
      const nStates = Number.isFinite(Number(params.n_states)) ? String(Math.max(2, Math.floor(Number(params.n_states)))) : '3';
      const cmd = [
        BUILD_PY,
        'scripts/run_hmm_labeling.py',
        '--dataset-manifest', dataset,
        '--out-dir', outDir,
        '--series-col', seriesCol,
        '--n-states', nStates,
        '--min-run', '5',
      ];
      if (computeSnr) cmd.push('--compute-trend-snr');
      return {
        cmd,
        cwd: REPO,
        env: makeEnv({ PYTHONPATH: '/home/jason/ml/sparrow/src:/home/jason/ml/regime_detection/src' }),
        publicParams: {
          dataset,
          out: outDir,
          series_col: seriesCol,
          compute_trend_snr: computeSnr,
          n_states: Number(nStates),
        },
      };
    },
  },
};

function serializeDefinitions() {
  return Object.values(BUILD_DEFINITIONS).map((def) => ({
    id: def.id,
    label: def.label,
    description: def.description,
    fields: def.fields,
    group: def.group || 'Other',
    details: def.details || [],
    approach: def.approach || 'other',
    enabled: def.enabled !== false,
  }));
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
  try {
    const st = await fsp.stat(REPO);
    res.json({ ok: true, repo: REPO, exists: st.isDirectory(), kanban: KANBAN });
  } catch (e) {
    res.json({ ok: false, repo: REPO, error: String(e), kanban: KANBAN });
  }
});

async function readJsonSafe(p) {
  try { return JSON.parse(await fsp.readFile(p, 'utf8')); } catch { return null; }
}

async function resolveSequenceModelDir() {
  const candidates = [
    path.join(REPO, SEQUENCE_MODEL_DIR),
    path.join(REPO, 'artifacts', 'reports'),
  ];
  for (const dir of candidates) {
    const modelPath = path.join(dir, 'sequence_model.pt');
    try {
      const st = await fsp.stat(modelPath);
      if (st?.isFile()) return dir;
    } catch {}
  }
  return path.join(REPO, SEQUENCE_MODEL_DIR);
}

async function resolveBoostingModelDir() {
  const candidates = [
    path.join(REPO, BOOSTING_MODEL_DIR),
    path.join(REPO, 'artifacts', 'reports'),
  ];
  for (const dir of candidates) {
    const modelPath = path.join(dir, 'boosting_model.pkl');
    try {
      const st = await fsp.stat(modelPath);
      if (st?.isFile()) return dir;
    } catch {}
  }
  return path.join(REPO, BOOSTING_MODEL_DIR);
}

async function resolveHMMDir() {
  const dir = path.join(REPO, HMM_LABEL_DIR);
  try {
    const st = await fsp.stat(dir);
    if (st?.isDirectory()) return dir;
  } catch {}
  return dir;
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
    const runsRoot = path.join(REPO, 'runs');
    const entries = await fsp.readdir(runsRoot, { withFileTypes: true }).catch(() => []);
    const out = [];
    for (const ent of entries) {
      if (!ent.isDirectory()) continue;
      const dir = path.join(runsRoot, ent.name);
      const summaryPath = path.join(dir, 'summary.json');
      const summary = await readJsonSafe(summaryPath);
      // Lightly parse key params from resolved.yaml if present
      let params = {};
      try {
        const txt = await fsp.readFile(path.join(dir, 'resolved.yaml'), 'utf8');
        const rx = /(cp_lambda|cp_thr|break_k_atr|lookback_n|min_edge_pts):\s*([0-9.]+)/g;
        let m; params = {};
        while ((m = rx.exec(txt))) { params[m[1]] = Number(m[2]); }
      } catch {}
      if (summary) {
        const st = await fsp.stat(summaryPath).catch(() => null);
        const mtime = st ? st.mtimeMs : 0;
        out.push({ run: ent.name, summary, mtime, params });
      }
    }
    // sort newest first by file mtime (more reliable than lexicographic name)
    out.sort((a, b) => (b.mtime - a.mtime) || String(b.run).localeCompare(String(a.run)));
    res.json(out);
  } catch (e) {
    res.status(500).json({ error: 'failed to list runs', detail: String(e) });
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
    if (def.enabled === false) {
      return res.status(400).json({ ok: false, error: 'Job not enabled yet' });
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

app.get('/api/models/sequence', async (req, res) => {
  try {
    const dir = await resolveSequenceModelDir();
    const summary = await readJsonSafe(path.join(dir, 'summary.json'));
    const training = await readJsonSafe(path.join(dir, 'training_metrics.json'));
    const latency = await readJsonSafe(path.join(dir, 'latency_metrics.json'));
    const evaluation = await readJsonSafe(path.join(dir, 'evaluation_metrics.json'));
    let modelStat = null;
    try {
      modelStat = await fsp.stat(path.join(dir, 'sequence_model.pt'));
    } catch {}
    if (!summary && !training && !evaluation && !modelStat) {
      return res.status(404).json({ ok: false, error: 'sequence model artifacts not found' });
    }
    res.json({
      ok: true,
      model_dir: dir,
      summary,
      training,
      latency,
      evaluation,
      model_updated_at: modelStat ? modelStat.mtime.toISOString() : null,
    });
  } catch (e) {
    res.status(500).json({ ok: false, error: 'failed to load sequence model info', detail: String(e) });
  }
});

app.get('/api/models/boosting', async (req, res) => {
  try {
    const dir = await resolveBoostingModelDir();
    const summary = await readJsonSafe(path.join(dir, 'summary.json'));
    const training = await readJsonSafe(path.join(dir, 'training_metrics.json'));
    const evaluation = await readJsonSafe(path.join(dir, 'evaluation_metrics.json'));
    let modelStat = null;
    try {
      modelStat = await fsp.stat(path.join(dir, 'boosting_model.pkl'));
    } catch {}
    if (!summary && !training && !evaluation && !modelStat) {
      return res.status(404).json({ ok: false, error: 'boosting model artifacts not found' });
    }
    res.json({
      ok: true,
      model_dir: dir,
      summary,
      training,
      evaluation,
      model_updated_at: modelStat ? modelStat.mtime.toISOString() : null,
    });
  } catch (e) {
    res.status(500).json({ ok: false, error: 'failed to load boosting model info', detail: String(e) });
  }
});

app.get('/api/labels/hmm', async (req, res) => {
  try {
    const dir = await resolveHMMDir();
    const summary = await readJsonSafe(path.join(dir, 'summary.json'));
    const metrics = await readJsonSafe(path.join(dir, 'metrics.json'));
    if (!summary && !metrics) {
      return res.status(404).json({ ok: false, error: 'hmm artifacts not found' });
    }
    res.json({
      ok: true,
      label_dir: dir,
      summary,
      metrics,
    });
  } catch (e) {
    res.status(500).json({ ok: false, error: 'failed to load hmm info', detail: String(e) });
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
