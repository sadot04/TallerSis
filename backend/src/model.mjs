import { spawn } from 'node:child_process';
import { randomUUID, createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import { rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DatabaseSync } from 'node:sqlite';

export const root = fileURLToPath(new URL('../', import.meta.url));
export const fields = readJson(join(root, 'features.json'));
const runtime = process.env.RUNTIME_DIR || join(root, 'runtime');
const reference = process.env.REFERENCE_CSV || join(root, 'data', 'train.csv');
const python = process.env.PYTHON_BIN || (process.platform === 'win32' ? 'py' : 'python3');
const seed = 'initial';
const uploadPath = id => join(runtime, 'uploads', id + '.csv');
const modelDir = id => id === seed ? join(root, 'seed') : join(runtime, 'models', id);
export const modelPath = id => join(modelDir(id), 'model.joblib');
const modelReport = id => readJson(join(modelDir(id), 'report.json'));
const now = () => new Date().toISOString();
const seedReport = modelReport(seed);

function readJson(path) {
  return JSON.parse(readFileSync(path, 'utf8'));
}

export class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

mkdirSync(join(runtime, 'uploads'), { recursive: true });
const db = new DatabaseSync(join(runtime, 'traffic.sqlite'), { timeout: 5000 });
db.exec(readFileSync(join(root, 'schema.sql'), 'utf8'));
db.prepare('INSERT OR IGNORE INTO models VALUES (?, NULL, ?, ?, 1)').run(seed, JSON.stringify(seedReport.metrics), now());
db.prepare("UPDATE training_jobs SET status = 'failed', message = ?, finished_at = ? WHERE status = 'running'")
  .run('El servicio se reinició durante el entrenamiento. Puedes solicitarlo de nuevo.', now());

const activeModel = () => db.prepare('SELECT id, created_at AS createdAt FROM models WHERE active = 1').get();
const jsonModelMetrics = row => ({ version: row.id, createdAt: row.createdAt, ...modelReport(row.id).metrics });
let predictionsRunning = 0;

function runModel(args, input, timeout) {
  return new Promise((resolve, reject) => {
    const child = spawn(python, [join(root, 'ml', 'train.py'), ...args], {
      windowsHide: true,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8', OMP_NUM_THREADS: '2', OPENBLAS_NUM_THREADS: '2' },
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    let output = '';
    let diagnostics = '';
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill();
    }, timeout);
    child.stdout.setEncoding('utf8').on('data', chunk => { output += chunk; });
    child.stderr.setEncoding('utf8').on('data', chunk => { diagnostics = (diagnostics + chunk).slice(-4000); });
    child.stdin.on('error', () => {});
    child.stdin.end(input === undefined ? undefined : JSON.stringify(input));
    child.once('error', error => {
      clearTimeout(timer);
      console.error('Proceso de ML:', error.message);
      reject(new HttpError(503, 'El proceso de Python no está disponible.'));
    });
    child.once('close', code => {
      clearTimeout(timer);
      if (timedOut) return reject(new HttpError(504, 'El proceso de ML excedió el tiempo permitido.'));
      if (code === 2) {
        const message = diagnostics.trim();
        const safeMessage = /^(Columnas requeridas|Cada fila|Hay un valor inválido|Se requieren|El CSV|Después de reservar)/.test(message);
        return reject(new HttpError(422, safeMessage ? message : 'El CSV no cumple el formato esperado.'));
      }
      if (code !== 0) {
        console.error('Proceso de ML:', diagnostics);
        return reject(new HttpError(503, 'No se pudo ejecutar el proceso de ML.'));
      }
      try {
        resolve(JSON.parse(output));
      } catch {
        reject(new HttpError(503, 'El proceso de ML devolvió una respuesta inválida.'));
      }
    });
  });
}

export function getFeatures() {
  const model = activeModel();
  const { options } = modelReport(model.id);
  return fields.map(field => ({ ...field, options: options[field.name] }));
}

export const getMetrics = () => jsonModelMetrics(activeModel());
export const getExport = () => ({ version: activeModel().id });

function checkedInput(input) {
  if (!input || typeof input !== 'object' || Array.isArray(input)
    || Object.keys(input).length !== fields.length || !fields.every(field => Object.hasOwn(input, field.name))) {
    throw new HttpError(422, 'Envía los 13 campos indicados en el formulario.');
  }
  return Object.fromEntries(fields.map(({ name, type, label }) => {
    const value = input[name];
    const valid = type === 'integer'
      ? Number.isInteger(value) && value >= (name === 'Number_of_vehicles_involved' ? 1 : 0) && value <= 1000
      : typeof value === 'string' && (type === 'time'
        ? /^(?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d)?$/.test(value)
        : value.trim().length > 0 && value.trim().length <= 150);
    if (!valid) throw new HttpError(422, 'Revisa el campo «' + label + '».');
    return [name, typeof value === 'string' ? value.trim() : value];
  }));
}

export async function predict(input, operator) {
  const data = checkedInput(input);
  if (predictionsRunning >= 4) throw new HttpError(429, 'Hay predicciones en curso. Intenta de nuevo.');
  const { id: version } = activeModel();
  const createdAt = now();
  predictionsRunning++;
  try {
    const result = await runModel(['predict', '--model', modelPath(version)], data, 25000);
    const record = db.prepare('INSERT INTO predictions(model_id, operator, input_json, result_json, created_at) VALUES (?, ?, ?, ?, ?)')
      .run(version, operator, JSON.stringify(data), JSON.stringify(result), createdAt);
    return { id: Number(record.lastInsertRowid), version, createdAt, input: data, ...result };
  } finally {
    predictionsRunning--;
  }
}

export function history(operator, limit, before) {
  const rows = db.prepare(
    'SELECT * FROM predictions WHERE operator = ? AND (? IS NULL OR id < ?) ORDER BY id DESC LIMIT ?'
  ).all(operator, before, before, limit + 1);
  const items = rows.slice(0, limit).map(row => ({
    id: row.id, version: row.model_id, createdAt: row.created_at,
    input: JSON.parse(row.input_json), ...JSON.parse(row.result_json),
  }));
  return { items, nextBefore: rows.length > limit ? items.at(-1).id : null };
}

export async function upload(file) {
  if (!file || !file.originalname.toLowerCase().endsWith('.csv')) {
    throw new HttpError(422, 'Selecciona un archivo con extensión .csv.');
  }
  const id = randomUUID();
  const path = uploadPath(id);
  try {
    await writeFile(path, file.buffer, { flag: 'wx' });
    const summary = await runModel(['validate', '--source', path], undefined, 25000);
    const createdAt = now();
    db.prepare('INSERT INTO datasets VALUES (?, ?, ?, ?, ?)')
      .run(id, file.originalname, summary.rows, JSON.stringify(summary.classes), createdAt);
    return { id, filename: file.originalname, ...summary, createdAt };
  } catch (error) {
    await rm(path, { force: true });
    throw error;
  }
}

export function datasets() {
  return db.prepare('SELECT * FROM datasets ORDER BY created_at DESC LIMIT 50').all().map(row => ({
    id: row.id, filename: row.filename, rows: row.row_count, classes: JSON.parse(row.classes_json), createdAt: row.created_at,
  }));
}

export function startTraining(datasetId) {
  if (typeof datasetId !== 'string' || !db.prepare('SELECT id FROM datasets WHERE id = ?').get(datasetId)) {
    throw new HttpError(404, 'No se encontró el archivo de entrenamiento.');
  }
  if (db.prepare("SELECT id FROM training_jobs WHERE status = 'running'").get()) {
    throw new HttpError(409, 'Ya hay un reentrenamiento en curso.');
  }
  if (!existsSync(reference) || createHash('sha256').update(readFileSync(reference)).digest('hex') !== seedReport.metrics.referenceSha256) {
    throw new HttpError(409, 'Importa el train.csv original para habilitar la prueba de referencia.');
  }
  const id = randomUUID();
  db.prepare("INSERT INTO training_jobs(id, dataset_id, status, created_at) VALUES (?, ?, 'running', ?)")
    .run(id, datasetId, now());
  void retrain(id, datasetId, getMetrics().macroF1);
  return trainingStatus(id);
}

async function retrain(id, datasetId, activeF1) {
  const out = modelDir(id);
  try {
    const metrics = await runModel(
      ['train', '--source', uploadPath(datasetId), '--reference', reference, '--out', out], undefined, 600000
    );
    const qualified = metrics.macroF1 + 1e-9 >= activeF1 && metrics.macroF1 > metrics.baselineMacroF1
      && metrics.perClass['Fatal injury'].recall > 0;
    db.exec('BEGIN IMMEDIATE');
    try {
      if (qualified) {
        db.prepare('UPDATE models SET active = 0 WHERE active = 1').run();
        db.prepare('INSERT INTO models VALUES (?, ?, ?, ?, 1)').run(id, datasetId, JSON.stringify(metrics), now());
      }
      db.prepare('UPDATE training_jobs SET status = ?, model_id = ?, metrics_json = ?, message = ?, finished_at = ? WHERE id = ?')
        .run(qualified ? 'completed' : 'rejected', qualified ? id : null, JSON.stringify(metrics),
          qualified ? 'El modelo entrenado está activo.' : 'El candidato no alcanzó el criterio de promoción.', now(), id);
      db.exec('COMMIT');
    } catch (error) {
      db.exec('ROLLBACK');
      throw error;
    }
    if (!qualified) await rm(out, { recursive: true, force: true });
  } catch (error) {
    const status = trainingStatus(id).status;
    if (status !== 'running') {
      console.error('Limpieza de entrenamiento:', error.message);
      return;
    }
    db.prepare("UPDATE training_jobs SET status = 'failed', message = ?, finished_at = ? WHERE id = ?")
      .run(error instanceof HttpError ? error.message : 'No se pudo guardar el entrenamiento.', now(), id);
    await rm(out, { recursive: true, force: true }).catch(cleanup => console.error('Limpieza:', cleanup.message));
  }
}

export function trainingStatus(id) {
  const job = db.prepare('SELECT * FROM training_jobs WHERE id = ?').get(id);
  if (!job) throw new HttpError(404, 'No se encontró el entrenamiento.');
  return { id: job.id, datasetId: job.dataset_id, version: job.model_id, status: job.status,
    metrics: job.metrics_json ? JSON.parse(job.metrics_json) : null, message: job.message,
    createdAt: job.created_at, finishedAt: job.finished_at };
}
