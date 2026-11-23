# Observatorio Vial

Proyecto académico de clasificación supervisada de gravedad de accidentes reportados. El modelo estima lesión leve, grave o fatal a partir de 13 datos del reporte. El sistema tiene dos servicios independientes. React con Vite consume la API REST de Express. Express ejecuta scikit-learn como un proceso local y guarda el historial y las versiones en SQLite.

## Requisitos y configuración

- Node.js 24 y npm.
- Python 3.11 a 3.14. Si usas un entorno virtual, configura su intérprete en `PYTHON_BIN`.
- ZIP de la competencia [Road Traffic Accidents](https://www.kaggle.com/competitions/road-traffic-accidents) con `train.csv` y `test.csv`. El modelo inicial se incluye en el repositorio. Se requiere el ZIP original para reproducir el entrenamiento y ejecutar reentrenamientos con la misma prueba de referencia.

Desde la raíz en PowerShell:

```powershell
py -m pip install -r backend/ml/requirements.txt
py backend/ml/import_data.py "C:\ruta\road-traffic-accidents.zip"
Copy-Item backend/.env.example backend/.env
Copy-Item frontend/.env.example frontend/.env
node -e "console.log(require('node:crypto').randomBytes(32).toString('hex'))"
```

Copia el secreto generado en `JWT_SECRET` y configura `OPERATOR_PASSWORD` con al menos 12 caracteres. El usuario por defecto es `operador`. `CORS_ORIGINS` admite una lista de orígenes exactos separados por coma. Configura `VITE_API_URL` con la URL base del backend. En Linux, usa `python3` y el equivalente de `Copy-Item` (`cp`).

## Ejecutar los dos servicios

Terminal del backend:

```powershell
cd backend
npm ci
npm start
```

Terminal del frontend:

```powershell
cd frontend
npm ci
npm run dev
```

Abre el [cliente web](http://127.0.0.1:5173) y entra con las credenciales del backend. La [API interactiva](http://127.0.0.1:3000/docs/) sirve [OpenAPI](backend/openapi.json). Puedes iniciar sesión en Swagger con `POST /api/auth/login`, copiar el token y pegarlo en **Authorize**.

## Usar y actualizar el modelo

1. En **Predicción**, completa los campos o usa el ejemplo y pulsa **Clasificar accidente**.
2. En **Historial**, abre el reporte para consultar los datos enviados, las probabilidades y la versión.
3. En **Métricas**, consulta el resultado por clase, la matriz de confusión y exporta el pipeline `.joblib`.
4. En **Entrenamiento**, carga un CSV con [los 13 campos](backend/features.json) y la etiqueta `Accident_severity`. Usa `Slight Injury`, `Serious Injury` o `Fatal injury`. El archivo admite un máximo de 8 MiB y 100 000 filas. Debe contener al menos 150 filas distintas y 15 por clase.
5. Selecciona el CSV y solicita el entrenamiento. Si el candidato cumple los criterios documentados, la API lo activa y conserva las predicciones anteriores con su versión.

Para clasificar localmente con el archivo exportado, desde la raíz y con las dependencias Python instaladas:

```powershell
$fields = Get-Content backend/features.json -Raw | ConvertFrom-Json
$data = @{}
$fields | ForEach-Object { $data[$_.name] = $_.example }
$data | ConvertTo-Json -Compress | py backend/ml/train.py predict --model "C:\ruta\modelo-vial-initial.joblib"
```

Joblib carga objetos Python serializados. Usa únicamente archivos de este servicio o de orígenes de confianza.

## Datos, base y pruebas

La aplicación inicializa `backend/runtime/traffic.sqlite` al arrancar con [schema.sql](backend/schema.sql). Los CSV que se cargan y las versiones entrenadas se guardan en `backend/runtime/`. Mantén ese directorio en un volumen persistente y ejecuta una instancia del backend. Para un despliegue remoto, termina HTTPS en un proxy inverso y permite solo el origen real del frontend.

```powershell
cd backend
npm run check
npm test
npm test -- --retrain
```

La última prueba requiere el archivo de referencia y tarda lo que dure el entrenamiento. Comprueba que el modelo promovido y el historial persisten después de reiniciar.

Para entrenar desde la raíz y reproducir las métricas publicadas:

```powershell
py backend/ml/train.py train --source backend/data/train.csv --reference backend/data/train.csv --out backend/runtime/reproducido
```

Para repetir la exploración, abre el [notebook](backend/ml/exploracion.ipynb) desde `backend/ml`.

## Entregables

- [Informe CRISP-ML(Q) y cronograma](output/pdf/informe-crisp-mlq.pdf)
- [Diagrama de base de datos](docs/base-de-datos.svg) y [script SQL](backend/schema.sql)
- [Especificación Swagger/OpenAPI](backend/openapi.json)
- [Modelo inicial](backend/seed/model.joblib) y [métricas](backend/seed/report.json)
- [Repositorio público TallerSis](https://github.com/sadot04/TallerSis)

Se usa `main` para la entrega y `develop` para integrar `feat/modelo-crisp`, `feat/api-rest`, `feat/cliente-react` y `feat/entrega`.
