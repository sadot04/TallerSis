import { createHash, timingSafeEqual } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import express from 'express';
import cors from 'cors';
import rateLimit from 'express-rate-limit';
import jwt from 'jsonwebtoken';
import multer from 'multer';
import swagger from 'swagger-ui-dist';
import {
  datasets, getExport, getFeatures, getMetrics, history, HttpError, modelPath,
  predict, root, startTraining, trainingStatus, upload,
} from './model.mjs';

const secret = process.env.JWT_SECRET || '';
const username = process.env.OPERATOR_USERNAME || 'operador';
const password = process.env.OPERATOR_PASSWORD || '';
if (Buffer.byteLength(secret) < 32 || password.length < 12) {
  throw new Error('Configura JWT_SECRET (32 bytes) y OPERATOR_PASSWORD (12 caracteres) en backend/.env.');
}
const origins = new Set((process.env.CORS_ORIGINS || 'http://localhost:5173,http://127.0.0.1:5173')
  .split(',').map(origin => origin.trim()).filter(Boolean));
const digest = value => createHash('sha256').update(value).digest();
const match = (given, expected) => timingSafeEqual(digest(given), digest(expected));
const issuer = 'observatorio-vial';
const audience = 'observatorio-vial-web';
const app = express();
app.disable('x-powered-by');
app.use((_req, res, next) => {
  res.set('X-Content-Type-Options', 'nosniff');
  next();
});
app.use(cors({
  origin(origin, done) {
    if (origin === undefined || origins.has(origin)) return done(null, true);
    done(new HttpError(403, 'Origen no autorizado.'));
  },
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Authorization', 'Content-Type'],
  exposedHeaders: ['Content-Disposition'],
}));
app.use(express.json({ limit: '32kb' }));

const openapi = JSON.parse(readFileSync(join(root, 'openapi.json'), 'utf8'));
app.get('/openapi.json', (_req, res) => res.json(openapi));
app.get('/docs/', (_req, res) => res.type('html').send(`<!doctype html><html lang="es"><meta charset="utf-8"><title>API · Observatorio Vial</title>
<link rel="stylesheet" href="/docs/swagger-ui.css"><div id="swagger-ui"></div>
<script src="/docs/swagger-ui-bundle.js"></script><script>SwaggerUIBundle({url:'/openapi.json',dom_id:'#swagger-ui'})</script></html>`));
app.use('/docs', express.static(swagger.getAbsoluteFSPath(), { index: false, redirect: false }));
app.get('/health', (_req, res) => res.json({ status: 'ok' }));

app.use('/api', (_req, res, next) => {
  res.set('Cache-Control', 'no-store');
  next();
});
app.post('/api/auth/login', rateLimit({
  windowMs: 15 * 60 * 1000, limit: 15, standardHeaders: 'draft-8', legacyHeaders: false,
  message: { message: 'Demasiados intentos. Vuelve a probar en unos minutos.' },
}), (req, res) => {
  const supplied = req.body || {};
  const validUser = typeof supplied.username === 'string' && match(supplied.username, username);
  const validPassword = typeof supplied.password === 'string' && match(supplied.password, password);
  if (!validUser || !validPassword) throw new HttpError(401, 'Usuario o contraseña incorrectos.');
  const token = jwt.sign({}, secret, { algorithm: 'HS256', issuer, audience, subject: username, expiresIn: '2h' });
  res.json({ token, expiresIn: 7200, username });
});

app.use('/api', (req, res, next) => {
  try {
    const auth = req.get('Authorization')?.match(/^Bearer (\S+)$/i);
    if (!auth) throw new Error();
    const claims = jwt.verify(auth[1], secret, { algorithms: ['HS256'], issuer, audience });
    if (claims.sub !== username) throw new Error();
    req.operator = claims.sub;
    next();
  } catch {
    res.set('WWW-Authenticate', 'Bearer');
    next(new HttpError(401, 'Inicia sesión con un token válido.'));
  }
});

app.get('/api/model/features', (_req, res) => res.json(getFeatures()));
app.get('/api/model/metrics', (_req, res) => res.json(getMetrics()));
app.get('/api/model/export', (_req, res, next) => {
  const { version } = getExport();
  res.download(modelPath(version), 'modelo-vial-' + version + '.joblib', next);
});
app.post('/api/predictions', async (req, res) => res.status(201).json(await predict(req.body, req.operator)));
app.get('/api/predictions', (req, res) => {
  const { limit = '20', before } = req.query;
  const size = Number(limit);
  const cursor = before === undefined ? null : Number(before);
  if (typeof limit !== 'string' || !/^\d+$/.test(limit) || !Number.isInteger(size) || size < 1 || size > 100
    || (before !== undefined && (typeof before !== 'string' || !/^\d+$/.test(before) || !Number.isSafeInteger(cursor) || cursor < 1))) {
    throw new HttpError(422, 'limit debe estar entre 1 y 100 y before debe ser un ID entero positivo.');
  }
  res.json(history(req.operator, size, cursor));
});
const receiveCsv = multer({ storage: multer.memoryStorage(), limits: { fileSize: 8 * 1024 * 1024, files: 1, fields: 0, parts: 2 } });
app.get('/api/datasets', (_req, res) => res.json(datasets()));
app.post('/api/datasets', receiveCsv.single('file'), async (req, res) => res.status(201).json(await upload(req.file)));
app.post('/api/model/retrain', (req, res) => {
  const job = startTraining(req.body?.datasetId);
  res.location('/api/model/retrain/' + job.id).status(202).json(job);
});
app.get('/api/model/retrain/:id', (req, res) => res.json(trainingStatus(req.params.id)));

app.use((_req, _res, next) => next(new HttpError(404, 'Ruta no encontrada.')));
app.use((error, _req, res, next) => {
  if (res.headersSent) return next(error);
  if (error instanceof multer.MulterError) {
    const large = error.code === 'LIMIT_FILE_SIZE';
    return res.status(large ? 413 : 422).json({ message: large ? 'El archivo supera 8 MiB.' : 'Envía un único CSV en el campo file.' });
  }
  if (error.type === 'entity.parse.failed') return res.status(400).json({ message: 'JSON inválido.' });
  if (error.status === 413) return res.status(413).json({ message: 'Solicitud demasiado grande.' });
  if (error instanceof HttpError) return res.status(error.status).json({ message: error.message });
  console.error(error.message);
  res.status(500).json({ message: 'Error interno del servicio.' });
});

app.listen(Number(process.env.PORT || 3000), process.env.HOST || '127.0.0.1', () => {
  console.log('API lista en http://' + (process.env.HOST || '127.0.0.1') + ':' + (process.env.PORT || 3000));
});
