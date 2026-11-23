import assert from 'node:assert/strict';
import { once } from 'node:events';
import { readFile, mkdtemp, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { spawn } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { createServer } from 'node:net';
import { setTimeout } from 'node:timers/promises';

const runtime = await mkdtemp(join(process.cwd(), 'runtime-smoke-'));
const portFinder = createServer().listen(0, '127.0.0.1');
await once(portFinder, 'listening');
const { port } = portFinder.address();
await new Promise(resolve => portFinder.close(resolve));
const base = 'http://127.0.0.1:' + port;
const password = randomBytes(16).toString('hex');
const env = { ...process.env, HOST: '127.0.0.1', PORT: String(port), RUNTIME_DIR: runtime,
  OPERATOR_USERNAME: 'test', OPERATOR_PASSWORD: password, JWT_SECRET: randomBytes(32).toString('hex'),
  CORS_ORIGINS: 'http://localhost:5173' };
let child;
let token;

async function request(path, expected, init = {}) {
  const headers = new Headers(init.headers);
  if (token && !headers.has('Authorization')) headers.set('Authorization', 'Bearer ' + token);
  const response = await fetch(base + path, { ...init, headers });
  assert.equal(response.status, expected, path + ' ' + (response.status === expected ? '' : await response.text()));
  return response;
}

const post = value => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(value) });

async function start() {
  child = spawn(process.execPath, ['src/server.mjs'], { env, stdio: 'inherit', windowsHide: true });
  for (let attempt = 0; attempt < 100; attempt++) {
    try {
      await request('/health', 200);
      return;
    } catch {
      await setTimeout(100);
    }
  }
  throw new Error('La API no arrancó.');
}

async function stop() {
  if (child && child.exitCode === null && child.signalCode === null) {
    const stopped = once(child, 'exit');
    child.kill();
    await stopped;
  }
}

try {
  await start();
  await request('/docs/', 200);
  const spec = await (await request('/openapi.json', 200)).json();
  assert.ok(spec.paths['/api/model/retrain/{id}']);
  await request('/api/model/metrics', 401);
  await request('/api/model/metrics', 403, { headers: { Origin: 'https://example.com' } });
  const cors = await request('/api/predictions', 204, { method: 'OPTIONS', headers: {
    Origin: 'http://localhost:5173', 'Access-Control-Request-Method': 'POST',
    'Access-Control-Request-Headers': 'Authorization,Content-Type',
  } });
  assert.equal(cors.headers.get('Access-Control-Allow-Origin'), 'http://localhost:5173');
  await request('/api/auth/login', 401, post({ username: 'test', password: 'incorrecta' }));
  ({ token } = await (await request('/api/auth/login', 200, post({ username: 'test', password }))).json());
  await request('/api/model/metrics', 401, { headers: { Authorization: 'Bearer invalido' } });
  const fields = await (await request('/api/model/features', 200)).json();
  const input = Object.fromEntries(fields.map(field => [field.name, field.example]));
  await request('/api/predictions', 422, post({}));
  const prediction = await (await request('/api/predictions', 201, post(input))).json();
  assert.deepEqual(prediction.input, input);
  assert.ok(Math.abs(Object.values(prediction.probabilities).reduce((a, b) => a + b) - 1) < 1e-9);
  const list = await (await request('/api/predictions?limit=1', 200)).json();
  assert.equal(list.items[0].id, prediction.id);
  assert.equal((await (await request('/api/predictions?before=' + prediction.id, 200)).json()).items.length, 0);
  await request('/api/predictions?limit=200', 422);
  const initial = await (await request('/api/model/metrics', 200)).json();
  assert.ok(initial.macroF1 > initial.baselineMacroF1);
  const exportResponse = await request('/api/model/export', 200);
  assert.match(exportResponse.headers.get('content-disposition'), /modelo-vial-initial\.joblib/);
  assert.deepEqual(Buffer.from(await exportResponse.arrayBuffer()), await readFile('seed/model.joblib'));
  const bad = new FormData();
  bad.append('file', new Blob(['invalid\n1\n']), 'invalido.csv');
  await request('/api/datasets', 422, { method: 'POST', body: bad });
  assert.deepEqual(await (await request('/api/datasets', 200)).json(), []);
  if (process.argv.includes('--retrain')) {
    const payload = new FormData();
    payload.append('file', new Blob([await readFile('data/train.csv')]), 'train.csv');
    const dataset = await (await request('/api/datasets', 201, { method: 'POST', body: payload })).json();
    const job = await (await request('/api/model/retrain', 202, post({ datasetId: dataset.id }))).json();
    await request('/api/model/retrain', 409, post({ datasetId: dataset.id }));
    let finished;
    for (let attempt = 0; attempt < 330; attempt++) {
      finished = await (await request('/api/model/retrain/' + job.id, 200)).json();
      if (finished.status !== 'running') break;
      await setTimeout(2000);
    }
    assert.equal(finished.status, 'completed', JSON.stringify(finished));
    const promoted = await (await request('/api/model/metrics', 200)).json();
    assert.equal(promoted.version, job.id);
    assert.equal(promoted.macroF1, initial.macroF1);
    await stop();
    await start();
    const afterRestart = await (await request('/api/model/metrics', 200)).json();
    assert.equal(afterRestart.version, job.id);
    const persisted = await (await request('/api/predictions', 200)).json();
    assert.equal(persisted.items[0].version, 'initial');
  }
  console.log('Prueba de integración aprobada' + (process.argv.includes('--retrain') ? ' con reentrenamiento y reinicio.' : '.'));
} finally {
  await stop();
  await rm(runtime, { recursive: true, force: true });
}
