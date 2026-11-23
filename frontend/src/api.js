export const baseUrl = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:3000').replace(/\/$/, '');
export const swaggerUrl = new URL('/docs/', baseUrl).href;

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

export async function request(path, { token, data, file, signal, download = false } = {}) {
  const headers = {};
  if (token) headers.Authorization = 'Bearer ' + token;
  let body;
  if (file) {
    body = new FormData();
    body.append('file', file);
  } else if (data !== undefined) {
    headers['Content-Type'] = 'application/json';
    body = JSON.stringify(data);
  }
  let response;
  try {
    response = await fetch(baseUrl + path, { method: body ? 'POST' : 'GET', headers, body, signal });
  } catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new ApiError(0, 'No se pudo conectar con la API. Comprueba que el servicio esté en ejecución.');
  }
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new ApiError(response.status, error.message || 'La solicitud no se pudo completar.');
  }
  return download ? response : response.json();
}
