import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Activity, ArrowDownToLine, ArrowLeft, ArrowRight, ArrowUpRight, ChartNoAxesCombined, ClipboardList,
  Database, FileSpreadsheet, FlaskConical, GitBranch, History, LockKeyhole, LogOut, Route, UploadCloud,
} from 'lucide-react';
import { request, swaggerUrl } from './api.js';
import './styles.css';

const labels = { 'Slight Injury': 'Lesión leve', 'Serious Injury': 'Lesión grave', 'Fatal injury': 'Lesión fatal' };
const colors = { 'Slight Injury': 'green', 'Serious Injury': 'amber', 'Fatal injury': 'rose' };
const optionLabels = {
  Monday: 'Lunes', Tuesday: 'Martes', Wednesday: 'Miércoles', Thursday: 'Jueves', Friday: 'Viernes',
  Saturday: 'Sábado', Sunday: 'Domingo', Unknown: 'Desconocido', 'No junction': 'Sin cruce',
  '2-5yr': '2 a 5 años', '1-2yr': '1 a 2 años', '5-10yr': '5 a 10 años', 'Above 10yr': 'Más de 10 años',
  'Below 1yr': 'Menos de 1 año', Automobile: 'Automóvil', 'Residential areas': 'Zona residencial',
  'Asphalt roads': 'Asfalto', Dry: 'Seca', Wet: 'Mojada', Daylight: 'Luz diurna', Normal: 'Normal',
  'Darkness - lights lit': 'Oscuridad con alumbrado', 'Darkness - no lighting': 'Oscuridad sin alumbrado',
  'Vehicle with vehicle collision': 'Colisión entre vehículos', Other: 'Otro', other: 'Otro',
};
const tabs = [
  ['predict', 'Predicción', ClipboardList], ['history', 'Historial', History],
  ['metrics', 'Métricas', ChartNoAxesCombined], ['training', 'Entrenamiento', Database],
];
const pageTitles = {
  predict: ['Clasificar un accidente', 'Ingresa las condiciones registradas en el reporte para estimar su gravedad.'],
  history: ['Historial de predicciones', 'Consulta los datos enviados y el resultado que devolvió cada versión.'],
  metrics: ['Rendimiento del modelo', 'Resultados del modelo activo sobre el conjunto de prueba de referencia.'],
  training: ['Entrenamiento del modelo', 'Carga un CSV etiquetado, evalúa un candidato y consulta si pasa a producción.'],
};
const groups = [
  ['Contexto del accidente', ['Time', 'Day_of_week', 'Area_accident_occured', 'Types_of_Junction']],
  ['Vía y condiciones', ['Road_surface_type', 'Road_surface_conditions', 'Light_conditions', 'Weather_conditions']],
  ['Vehículo e impacto', ['Driving_experience', 'Type_of_vehicle', 'Type_of_collision', 'Number_of_vehicles_involved', 'Number_of_casualties']],
];
const dateFormat = new Intl.DateTimeFormat('es-BO', { dateStyle: 'medium', timeStyle: 'short' });
const pct = value => new Intl.NumberFormat('es-BO', { style: 'percent', maximumFractionDigits: 1 }).format(value);
const number = value => new Intl.NumberFormat('es-BO').format(value);
const human = value => Object.hasOwn(labels, value) ? labels[value] : Object.hasOwn(optionLabels, value) ? optionLabels[value] : value;
const versionText = version => version === 'initial' ? 'Inicial' : version.slice(0, 8);
const jobKey = 'observatorio-vial.training';

function Chip({ category }) {
  return <span className={'chip ' + colors[category]}>{human(category)}</span>;
}

function Notice({ error, children }) {
  return children ? <p className={'notice' + (error ? ' error' : '')} role={error ? 'alert' : 'status'}>{children}</p> : null;
}

function Empty({ icon: Icon, title, children }) {
  return <div className="empty"><span className="empty-icon"><Icon size={24} aria-hidden="true" /></span>
    <h3>{title}</h3><p>{children}</p></div>;
}

function useResource(api, path, refresh = 0) {
  const [state, setState] = useState({ loading: true, data: null, error: '' });
  useEffect(() => {
    const controller = new AbortController();
    setState({ loading: true, data: null, error: '' });
    api(path, { signal: controller.signal }).then(data => setState({ data, loading: false, error: '' }))
      .catch(error => {
        if (error.name !== 'AbortError') setState({ data: null, loading: false, error: error.message });
      });
    return () => controller.abort();
  }, [api, path, refresh]);
  return state;
}

function Login({ onLogin, expired }) {
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  async function logIn(event) {
    event.preventDefault();
    const input = new FormData(event.currentTarget);
    setBusy(true);
    setError('');
    try {
      onLogin(await request('/api/auth/login', { data: Object.fromEntries(input) }));
    } catch (problem) {
      setError(problem.message);
    } finally {
      setBusy(false);
    }
  }
  return <main className="login">
    <div className="login-card">
      <Brand />
      <h1>Iniciar sesión</h1>
      <p className="muted">Accede a la clasificación y gestión de reportes viales.</p>
      <Notice error>{error || expired}</Notice>
      <form onSubmit={logIn} className="login-form">
        <label htmlFor="username">Usuario</label><input id="username" name="username" autoComplete="username" required autoFocus />
        <label htmlFor="password">Contraseña</label><input id="password" name="password" type="password" autoComplete="current-password" required />
        <button className="primary" disabled={busy}><LockKeyhole size={17} />{busy ? 'Verificando…' : 'Ingresar'}</button>
      </form>
      <a className="api-link" href={swaggerUrl} target="_blank" rel="noreferrer">Documentación de la API <ArrowUpRight size={15} /></a>
    </div>
  </main>;
}

function Brand() {
  return <div className="brand"><span><Route size={23} aria-hidden="true" /></span><strong>Observatorio Vial</strong></div>;
}

function Field({ field, value, onChange }) {
  const selectOptions = useMemo(() => {
    if (!field.options) return [];
    return [...new Set([field.example, ...field.options])].sort((a, b) => human(a).localeCompare(human(b), 'es'));
  }, [field]);
  return <label className="field"><span>{field.label}</span>
    {field.type === 'category'
      ? <select required value={value} onChange={event => onChange(event.target.value)}>
        <option value="">Selecciona una opción</option>
        {selectOptions.map(option => <option key={option} value={option}>{human(option)}</option>)}
      </select>
      : <input required type={field.type === 'time' ? 'time' : 'number'} min={field.name === 'Number_of_vehicles_involved' ? 1 : 0}
          max={field.type === 'integer' ? 1000 : undefined} value={value} onChange={event => onChange(event.target.value)} />}
  </label>;
}

function PredictionPanel({ api, fields }) {
  const [values, setValues] = useState({});
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const byName = Object.fromEntries((fields.data || []).map(field => [field.name, field]));
  async function classify(event) {
    event.preventDefault();
    setError('');
    setBusy(true);
    setResult(null);
    try {
      const data = Object.fromEntries(fields.data.map(field => [
        field.name, field.type === 'integer' ? Number(values[field.name]) : values[field.name],
      ]));
      setResult(await api('/api/predictions', { data }));
    } catch (problem) {
      setError(problem.message);
    } finally {
      setBusy(false);
    }
  }
  function example() {
    setValues(Object.fromEntries(fields.data.map(field => [field.name, field.example])));
  }
  return <div className="split">
    <section className="panel">
      <div className="panel-heading"><h2>Datos del accidente</h2><button className="secondary" onClick={example} disabled={!fields.data || busy}>
        <FileSpreadsheet size={16} /> Usar ejemplo
      </button></div>
      {fields.loading && <Notice>Cargando campos del modelo…</Notice>}
      <Notice error>{fields.error}</Notice>
      {fields.data && <form onSubmit={classify}>
        {groups.map(([title, names]) => <fieldset key={title}><legend>{title}</legend><div className="form-grid">
          {names.map(name => <Field key={name} field={byName[name]} value={values[name] ?? ''}
            onChange={value => setValues(previous => ({ ...previous, [name]: value }))} />)}
        </div></fieldset>)}
        <div className="form-footer"><button className="primary" disabled={busy}><Activity size={18} />
          {busy ? 'Clasificando…' : 'Clasificar accidente'}</button></div>
      </form>}
    </section>
    <aside className="panel result" aria-live="polite">
      <div className="panel-heading"><h2>Resultado</h2></div>
      <Notice error>{error}</Notice>
      {busy && <Empty icon={Activity} title="Predicción en curso">Evaluando los datos del reporte.</Empty>}
      {!busy && !result && !error && <Empty icon={Activity} title="Aún no hay una predicción">Completa el formulario y clasifica el accidente para ver su gravedad estimada.</Empty>}
      {result && <div className="result-body">
        <h3>{human(result.category)}</h3>
        <p className="muted">Probabilidad estimada por clase</p>
        <div className="bar-list">{Object.entries(labels).map(([key, label]) => <div key={key}>
          <div className="bar-label"><span>{label}</span><strong>{pct(result.probabilities[key])}</strong></div>
          <progress className={colors[key]} max="1" value={result.probabilities[key]} aria-label={label} />
        </div>)}</div>
        <p className="record-note">Guardada como #{result.id} · Versión {versionText(result.version)}</p>
      </div>}
    </aside>
  </div>;
}

function HistoryPanel({ api, fields }) {
  const [cursors, setCursors] = useState([null]);
  const [open, setOpen] = useState(null);
  const before = cursors.at(-1);
  const { data, loading, error } = useResource(api, '/api/predictions?limit=15' + (before ? '&before=' + before : ''));
  const fieldNames = Object.fromEntries((fields.data || []).map(field => [field.name, field.label]));
  return <section className="panel">
    <div className="panel-heading"><h2>Predicciones guardadas</h2></div>
    <Notice error>{error}</Notice>
    {loading && <Notice>Cargando historial…</Notice>}
    {data && (data.items.length ? <>
      <div className="table-scroll"><table><thead><tr><th>ID</th><th>Fecha y hora</th><th>Gravedad</th><th>Versión</th><th>Reporte</th></tr></thead>
        <tbody>{data.items.map(item => <React.Fragment key={item.id}>
          <tr><td>#{item.id}</td><td>{dateFormat.format(new Date(item.createdAt))}</td>
            <td><Chip category={item.category} /></td><td>{versionText(item.version)}</td>
            <td><button className="text-button" aria-expanded={open === item.id} aria-controls={'record-' + item.id}
              onClick={() => setOpen(open === item.id ? null : item.id)}>{open === item.id ? 'Cerrar' : 'Ver datos'}</button></td></tr>
          <tr hidden={open !== item.id} id={'record-' + item.id} className="details-row"><td colSpan={5}><dl>
            {Object.entries(item.input).map(([key, value]) => <div key={key}><dt>{fieldNames[key] || key}</dt><dd>{human(value)}</dd></div>)}
            {Object.entries(item.probabilities).map(([key, value]) => <div key={key}><dt>{human(key)}</dt><dd>{pct(value)}</dd></div>)}
          </dl></td></tr>
        </React.Fragment>)}</tbody>
      </table></div>
      <div className="pagination">
        <button className="secondary" disabled={cursors.length === 1} onClick={() => setCursors(cursors.slice(0, -1))}><ArrowLeft size={16} /> Anterior</button>
        <span>Página {cursors.length}</span>
        <button className="secondary" disabled={!data.nextBefore} onClick={() => setCursors([...cursors, data.nextBefore])}>Siguiente <ArrowRight size={16} /></button>
      </div>
    </> : <Empty icon={History} title="No hay predicciones guardadas">Las clasificaciones que realices aparecerán aquí.</Empty>)}
  </section>;
}

function ModelMetrics({ data }) {
  const primary = [
    ['F1 macro', data.macroF1, 'Equilibrio entre precisión y sensibilidad de las tres clases.'],
    ['Exactitud', data.accuracy, 'Porcentaje de reportes clasificados correctamente.'],
    ['Exactitud balanceada', data.balancedAccuracy, 'Promedio de la sensibilidad por clase.'],
    ['F1 macro de referencia', data.baselineMacroF1, 'Clasificador que siempre predice la clase más frecuente.'],
  ];
  return <>
    <div className="stat-grid">{primary.map(([title, value, description]) => <div className="stat" key={title}>
      <span>{title}</span><strong>{pct(value)}</strong><p>{description}</p>
    </div>)}</div>
    <div className="panel">
      <div className="panel-heading"><div><h2>Desempeño por categoría</h2><p className="muted">{number(data.testRows)} reportes de prueba · {number(data.trainRows)} de entrenamiento</p></div></div>
      <div className="table-scroll"><table><thead><tr><th>Gravedad</th><th>Precisión</th><th>Sensibilidad</th><th>F1</th><th>Casos de prueba</th></tr></thead>
        <tbody>{data.labels.map(key => <tr key={key}><td><Chip category={key} /></td>
          <td>{pct(data.perClass[key].precision)}</td><td>{pct(data.perClass[key].recall)}</td>
          <td>{pct(data.perClass[key]['f1-score'])}</td><td>{number(data.perClass[key].support)}</td></tr>)}</tbody>
      </table></div>
    </div>
    <div className="panel">
      <div className="panel-heading"><div><h2>Matriz de confusión</h2><p className="muted">Filas de gravedad registrada y columnas de gravedad predicha.</p></div></div>
      <div className="table-scroll"><table><thead><tr><th scope="col">Gravedad registrada</th>{data.labels.map(label => <th scope="col" key={label}>{human(label)}</th>)}</tr></thead>
        <tbody>{data.confusionMatrix.map((row, i) => <tr key={data.labels[i]}><th scope="row">{human(data.labels[i])}</th>
          {row.map((count, j) => <td key={data.labels[j]} className={i === j ? 'diagonal' : ''}>{number(count)}</td>)}</tr>)}</tbody>
      </table></div>
    </div>
    <div className="panel"><div className="panel-heading"><h2>Selección de algoritmo</h2></div>
      <div className="table-scroll"><table><thead><tr><th>Algoritmo</th><th>F1 macro en validación</th></tr></thead>
        <tbody>{Object.entries(data.validationMacroF1).map(([name, value]) => <tr key={name}><td>{name}{name === data.algorithm && <small className="selected-algorithm">Seleccionado</small>}</td>
          <td>{pct(value)}</td></tr>)}</tbody>
      </table></div>
    </div>
  </>;
}

function MetricsPanel({ api }) {
  const { data, error, loading } = useResource(api, '/api/model/metrics');
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState('');
  async function exportModel() {
    setExportError('');
    setExporting(true);
    try {
      const response = await api('/api/model/export', { download: true });
      const href = URL.createObjectURL(await response.blob());
      const anchor = document.createElement('a');
      anchor.href = href;
      const disposition = response.headers.get('Content-Disposition');
      const filename = disposition?.match(/\bfilename=(?:"([^"]+)"|([^;\s]+))/i);
      anchor.download = filename?.[1] || filename?.[2] || 'modelo-vial.joblib';
      anchor.click();
      setTimeout(() => URL.revokeObjectURL(href), 1000);
    } catch (problem) {
      setExportError(problem.message);
    } finally {
      setExporting(false);
    }
  }
  return <div className="stack">
    <Notice error>{error || exportError}</Notice>
    {loading && <Notice>Cargando métricas…</Notice>}
    {data && <>
      <section className="panel model-banner"><div className="model-id"><span className="empty-icon"><GitBranch size={23} /></span>
        <div><h2>{data.algorithm}</h2><p className="muted">Versión {versionText(data.version)} · scikit-learn {data.sklearnVersion}</p></div></div>
        <button className="secondary" onClick={exportModel} disabled={exporting}><ArrowDownToLine size={17} />
          {exporting ? 'Descargando…' : 'Exportar modelo'}</button>
      </section>
      <ModelMetrics data={data} />
    </>}
  </div>;
}

const jobLabels = { running: 'En entrenamiento', completed: 'Publicado', rejected: 'No publicado', failed: 'Fallido' };

function TrainingPanel({ api, onPublished }) {
  const [refresh, setRefresh] = useState(0);
  const { data, loading, error } = useResource(api, '/api/datasets', refresh);
  const [file, setFile] = useState(null);
  const [selected, setSelected] = useState('');
  const [jobId, setJobId] = useState(() => sessionStorage.getItem(jobKey) || '');
  const [job, setJob] = useState(null);
  const [pollAttempt, setPollAttempt] = useState(0);
  const [pollError, setPollError] = useState('');
  const [message, setMessage] = useState('');
  const [problem, setProblem] = useState('');
  const [busy, setBusy] = useState('');
  const waiting = Boolean(jobId && (!job || job.status === 'running'));

  useEffect(() => {
    if (!jobId) return;
    const controller = new AbortController();
    let timer;
    async function poll() {
      try {
        const next = await api('/api/model/retrain/' + jobId, { signal: controller.signal });
        setJob(next);
        if (next.status === 'running') timer = setTimeout(poll, 1600);
        if (next.status === 'completed') onPublished();
      } catch (failure) {
        if (failure.name !== 'AbortError') {
          if (failure.status === 404) {
            sessionStorage.removeItem(jobKey);
            setJobId('');
          }
          setPollError(failure.message);
        }
      }
    }
    setPollError('');
    poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [api, jobId, onPublished, pollAttempt]);

  async function upload(event) {
    event.preventDefault();
    const form = event.currentTarget;
    setBusy('upload'); setProblem(''); setMessage('');
    try {
      const record = await api('/api/datasets', { file });
      setFile(null);
      form.reset();
      setSelected(record.id);
      setRefresh(previous => previous + 1);
      setMessage('CSV validado con ' + number(record.rows) + ' registros distintos.');
    } catch (failure) {
      setProblem(failure.message);
    } finally {
      setBusy('');
    }
  }

  async function train(event) {
    event.preventDefault();
    setBusy('train'); setProblem(''); setMessage('');
    try {
      const next = await api('/api/model/retrain', { data: { datasetId: selected } });
      sessionStorage.setItem(jobKey, next.id);
      setJob(next);
      setJobId(next.id);
    } catch (failure) {
      setProblem(failure.message);
    } finally {
      setBusy('');
    }
  }

  return <div className="stack">
    <Notice error>{error || problem}</Notice><Notice>{message}</Notice>
    <div className="split training-split">
      <section className="panel">
        <div className="panel-heading"><h2>Cargar datos de entrenamiento</h2></div>
        <form className="panel-body stack" onSubmit={upload}>
          <p className="muted">El CSV debe incluir los 13 campos del formulario y la columna <code>Accident_severity</code>. Tamaño máximo de 8 MiB.</p>
          <label className="file-picker"><UploadCloud size={22} /><span>Archivo CSV</span>
            <input type="file" accept=".csv,text/csv" onChange={event => setFile(event.target.files[0] || null)} required />
          </label>
          <button className="primary align-start" disabled={!file || Boolean(busy)}><UploadCloud size={17} />
            {busy === 'upload' ? 'Validando…' : 'Cargar y validar'}</button>
        </form>
      </section>
      <section className="panel">
        <div className="panel-heading"><h2>Solicitar reentrenamiento</h2></div>
        <form className="panel-body stack" onSubmit={train}>
          <label className="field"><span>Archivo validado</span>
            <select required value={selected} onChange={event => setSelected(event.target.value)} disabled={loading || !data?.length}>
              <option value="">{loading ? 'Consultando archivos…' : 'Selecciona un archivo'}</option>
              {data?.map(record => <option value={record.id} key={record.id}>{record.filename} · {number(record.rows)} filas · {dateFormat.format(new Date(record.createdAt))}</option>)}
            </select>
          </label>
          {data?.find(record => record.id === selected) && <div className="class-counts">
            {Object.entries(data.find(record => record.id === selected).classes).map(([name, count]) => <span key={name}>{human(name)} <strong>{number(count)}</strong></span>)}
          </div>}
          <button className="primary align-start" disabled={!selected || loading || waiting || Boolean(busy)}><FlaskConical size={17} />
            {busy === 'train' ? 'Solicitando…' : 'Reentrenar modelo'}</button>
        </form>
      </section>
    </div>
    <section className="panel">
      <div className="panel-heading"><h2>Estado del entrenamiento</h2></div>
      <Notice error>{pollError}</Notice>
      {pollError && jobId && <div className="panel-body"><button className="secondary" onClick={() => setPollAttempt(attempt => attempt + 1)}>Reintentar consulta</button></div>}
      {!job && !jobId && <Empty icon={FlaskConical} title="Todavía no hay un entrenamiento">El estado y el resultado de evaluación aparecerán aquí.</Empty>}
      {!job && jobId && !pollError && <Notice>Consultando entrenamiento…</Notice>}
      {job && <div className="panel-body job">
        <div aria-live="polite"><strong>{jobLabels[job.status]}</strong><p className="muted">
          {job.message || 'El candidato se evalúa con la prueba de referencia. El modelo activo sigue disponible.'}</p></div>
        {job.metrics && <div className="job-score"><span>F1 macro del candidato</span><strong>{pct(job.metrics.macroF1)}</strong></div>}
        {job.version && <p className="muted">Versión {versionText(job.version)}</p>}
      </div>}
    </section>
  </div>;
}

function App() {
  const [session, setSession] = useState(null);
  const [expired, setExpired] = useState('');
  const [tab, setTab] = useState('predict');
  const [service, setService] = useState(null);
  const [refresh, setRefresh] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setService(null);
    request('/health', { signal: controller.signal }).then(() => setService(true)).catch(error => {
      if (error.name !== 'AbortError') setService(false);
    });
    return () => controller.abort();
  }, [session]);

  const api = useCallback(async (path, options = {}) => {
    try {
      return await request(path, { ...options, token: session?.token });
    } catch (error) {
      if (error.status === 401) {
        setExpired('La sesión terminó. Inicia sesión nuevamente.');
        setSession(null);
      }
      throw error;
    }
  }, [session?.token]);
  const modelPublished = useCallback(() => setRefresh(i => i + 1), []);

  function logIn(next) {
    setExpired('');
    setSession(next);
  }
  function logOut() {
    setSession(null);
    setTab('predict');
    setExpired('');
  }
  if (!session) return <Login onLogin={logIn} expired={expired} />;
  return <Workspace {...{ api, tab, setTab, refresh, service, session, logOut, modelPublished }} />;
}

function Workspace({ api, tab, setTab, refresh, service, session, logOut, modelPublished }) {
  const fields = useResource(api, '/api/model/features', refresh);
  const [title, description] = pageTitles[tab];
  return <>
    <a className="skip" href="#main">Saltar al contenido</a>
    <header className="topbar">
      <div className="shell topbar-inner">
        <Brand />
        <div className="session">
          <span className="user-name">{session.username}</span>
          <button className="text-button logout" type="button" onClick={logOut}><LogOut size={17} /> Salir</button>
        </div>
      </div>
    </header>
    <div className="navigation"><div className="shell nav-inner">
      <nav aria-label="Secciones">{tabs.map(([id, label, Icon]) => <button key={id} onClick={() => setTab(id)}
        aria-current={tab === id ? 'page' : undefined}><Icon size={18} aria-hidden="true" />{label}</button>)}</nav>
      <a className="api-link" href={swaggerUrl} target="_blank" rel="noreferrer">Swagger <ArrowUpRight size={15} /></a>
    </div></div>
    <main className="shell workspace" id="main">
      <div className="page-heading"><div><h1>{title}</h1><p>{description}</p></div>
        <span className={'service' + (service === false ? ' offline' : '')}><i aria-hidden="true" />
          {service === null ? 'Comprobando API' : service ? 'API conectada' : 'API sin conexión'}</span>
      </div>
      {tab === 'predict' && <PredictionPanel api={api} fields={fields} />}
      {tab === 'history' && <HistoryPanel api={api} fields={fields} />}
      {tab === 'metrics' && <MetricsPanel api={api} />}
      {tab === 'training' && <TrainingPanel api={api} onPublished={modelPublished} />}
    </main>
    <footer className="shell footer"><span>Observatorio Vial</span><span>Proyecto académico de clasificación supervisada</span></footer>
  </>;
}

createRoot(document.getElementById('root')).render(<App />);
