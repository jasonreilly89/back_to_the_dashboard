const test = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');
const os = require('node:os');
const path = require('node:path');
const fs = require('node:fs/promises');

async function createServer(rootDir) {
  const previousEnv = {
    RUNS_DIR: process.env.RUNS_DIR,
    ALPHA_EXPERIMENTS_DIR: process.env.ALPHA_EXPERIMENTS_DIR,
    TEST_BOCPD_DIR: process.env.TEST_BOCPD_DIR,
  };

  process.env.RUNS_DIR = rootDir;
  process.env.ALPHA_EXPERIMENTS_DIR = rootDir;
  process.env.TEST_BOCPD_DIR = rootDir;

  delete require.cache[require.resolve('../lib/readers')];
  delete require.cache[require.resolve('../server')];

  const app = require('../server');
  const server = http.createServer(app);
  await new Promise((resolve) => server.listen(0, resolve));
  const address = server.address();
  return { server, port: address.port, previousEnv };
}

async function closeServer(server, previousEnv) {
  await new Promise((resolve) => server.close(resolve));
  Object.entries(previousEnv || {}).forEach(([key, value]) => {
    if (value === undefined) {
      delete process.env[key];
    } else {
      process.env[key] = value;
    }
  });
  delete require.cache[require.resolve('../lib/readers')];
  delete require.cache[require.resolve('../server')];
}

test('POST /api/runs persists submission metadata and augments summaries', async () => {
  const tmpRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'dashboard-submission-'));
  const runId = 'demo_run_001';
  const runDir = path.join(tmpRoot, runId);
  await fs.mkdir(runDir, { recursive: true });
  await fs.writeFile(path.join(runDir, 'summary.json'), JSON.stringify({ run_id: runId, ok: true }), 'utf8');

  const { server, port, previousEnv } = await createServer(tmpRoot);

  try {
    const payload = {
      name: 'Demo Run',
      path: path.join(runDir),
      tags: ['day', 'week'],
      metrics: { sharpe: 1.5 },
      summary: { trades: 42 },
      results_markdown: '# Summary\n- ok',
    };

    const response = await fetch(`http://127.0.0.1:${port}/api/runs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    assert.equal(response.status, 201);
    const body = await response.json();
    assert.equal(body.ok, true);
    assert.equal(body.run_id, runId);

    const submissionPath = path.join(runDir, 'dashboard_submission.json');
    const rawSubmission = await fs.readFile(submissionPath, 'utf8');
    const submission = JSON.parse(rawSubmission);
    assert.equal(submission.run_id, runId);
    assert.deepEqual(submission.tags, ['day', 'week']);
    assert.equal(submission.metrics.sharpe, 1.5);

    const summaryResp = await fetch(`http://127.0.0.1:${port}/api/runs/${runId}/summary`);
    assert.equal(summaryResp.status, 200);
    const summaryJson = await summaryResp.json();
    assert.ok(summaryJson.dashboard_submission);
    assert.equal(summaryJson.dashboard_submission.run_id, runId);
    assert.equal(summaryJson.dashboard_submission.metrics.sharpe, 1.5);
  } finally {
    await closeServer(server, previousEnv);
    await fs.rm(tmpRoot, { recursive: true, force: true });
  }
});
