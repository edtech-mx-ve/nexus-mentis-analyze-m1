# NEXUS MENTIS — NEXUS ANALYZE
**Módulo 1 · Práctica 1: análisis, implementación y evaluación de técnicas de IA sobre un dataset de redes sociales**

Aplicación académica y didáctica en Python para clasificar publicaciones de **Twitter (actualmente X)** como **positivas o negativas** usando NLP, TF-IDF y Machine Learning. No recopila tweets de X en tiempo real.

## Contexto académico

- **Institución:** Instituto Internacional de Aguascalientes
- **Programa:** Maestría en Inteligencia Artificial para la Transformación Digital
- **Asignatura:** Tópicos de IA para la Transformación Digital
- **Alumno:** Antonio Nicolás Toro González
- **Tutor:** Dr. Francisco Javier Luna Rosas

## Funcionalidades

La interfaz mantiene cinco etapas: **Datos → NLP → Modelos → Evaluación → Laboratorio**. Además incluye **Acerca de, Dataset, Trivia, Ayuda e Institucional**.

- Preparación documentada del dataset y explicación del tratamiento del texto.
- Tokenización, n-gramas y representación TF-IDF.
- Comparación académica de Naive Bayes, SVM lineal y regresión logística.
- Visualización de métricas y matriz de confusión.
- Inferencia en inglés y en español (traducción **local ES→EN mediante Argos Translate**).
- Predicción **binaria**: positivo / negativo, sin clase neutral.

## Dataset y metodología

**Sentiment140** contiene 1.600.000 tweets históricos en inglés. Para este estudio se muestrearon **200.000**, con **195.997 registros preparados** después del filtrado.

| Partición | Tweets | Finalidad |
|---|---:|---|
| Entrenamiento | 137.197 | Ajuste de los clasificadores |
| Validación | 29.400 | Selección del algoritmo |
| Prueba | 29.400 | Evaluación final |

Flujo: `tweet → validación y limpieza → normalización → TF-IDF → clasificador → sentimiento`.

**Big Data:** se estudia un corpus amplio de información textual de redes sociales, usando una muestra reproducible. Este módulo **no** realiza procesamiento distribuido ni streaming.

## Modelo y resultados

Modelo activo: **regresión logística**, entrenada con TF-IDF de palabras y evaluada sobre tweets en inglés.

| Métrica | Resultado |
|---|---:|
| F1 macro en validación | 80,44 % |
| F1 macro en prueba | 79,97 % |
| Exactitud en prueba | 79,97 % |
| Tweets de prueba | 29.400 |
| Predicciones correctas | 23.512 |

La matriz de confusión y las métricas de precisión y recall están disponibles en la interfaz.

**No confundir evaluaciones:** se realizaron **10 comprobaciones manuales (10/10)** con textos previamente seleccionados. Son pruebas **funcionales** del laboratorio, no estiman el rendimiento general ni la exactitud en español.

## Archivos esenciales para la nube

```text
.
├── README.md
├── Dockerfile
├── requirements-cloud.txt
├── cloud_entry.py
├── cloud_preflight.py
├── .gitignore
├── .dockerignore
├── MANIFEST_SHA256.txt
├── nexus_mentis/                  # Código FastAPI, NLP e interfaz
│   └── web/                      # HTML, CSS, JS e imágenes necesarias
└── artifacts/
    └── sentiment140_real_200k/
        ├── model.joblib          # Modelo original persistido
        └── report.json           # Reporte de entrenamiento y evaluación
```

La carpeta de modelo puede incluir pequeñas gráficas agregadas. El reporte exportado puede tener **rutas locales anonimizadas**; las métricas y etiquetas se conservan.

**Excluidos:** dataset CSV de 1,6 millones, `.venv`, respaldos, historiales de entrenamiento, experimentos exploratorios, casos de evaluación privados y secretos.

## Ejecutar en Docker (sin reentrenar)

Desde la raíz del repositorio:

```bash
docker build -t nexus-analyze-m1 .
docker run --rm -p 10000:10000 nexus-analyze-m1
```

Abrir `http://localhost:10000`. El build instala dependencias y el traductor ES→EN una sola vez; requiere Internet. La app usa un solo proceso Uvicorn y los artefactos existentes.

Comprobaciones:

- `GET /healthz`: HTTP 200 y `status=ready` solo con modelo y traductor disponibles.
- `GET /api/health`: `status=ready`, `demo=false`, traducción ES→EN preparada.
- `GET /api/metrics`: métricas del reporte real.
- Laboratorio: clasificar una publicación en español y otra en inglés.
- Revisar gráficas, cinco etapas y cinco pestañas en escritorio y móvil.

El entorno Docker **debe probarse realmente** antes de considerar terminado el despliegue. El exportador no prueba el servicio Cloud.

## Publicación con GitHub y Render

1. Generar la carpeta con el exportador de despliegue y publicar su contenido en un **repositorio GitHub privado**.
2. Conectar ese repositorio desde **Render → New → Web Service → Docker**.
3. Definir **Health Check Path**: `/healthz`. No requiere base de datos.
4. Dimensionar RAM para el modelo y Argos Translate; una instancia de 512 MB puede ser insuficiente.
5. Verificar `/api/health`, `/api/metrics`, la interfaz y el laboratorio en la URL asignada por Render.

El repositorio GitHub no implica que la aplicación esté ya desplegada. No se presupone una URL pública.

## Reproducibilidad, seguridad y limitaciones

- Semilla y parámetros del experimento conservados en el reporte original.
- Modelo persistido con `joblib`: **cargar solo artefactos propios y confiables**.
- El servidor no requiere datos de entrenamiento en producción.
- Los textos de usuario se validan. No introducir datos personales.
- Los resultados son predicciones estadísticas; no garantizan interpretación correcta de sarcasmo, ironía o contrastes.
- El corpus es histórico y mayoritariamente en inglés; los resultados formales no miden el rendimiento en español.
- No está diseñado para decisiones de alto impacto, análisis masivo en tiempo real ni el Módulo 2.
- No publicar `.env`, contraseñas, respaldos, corpus completo ni entornos virtuales.
- El código y los datos del repositorio no incluyen una licencia de reutilización implícita de terceros.

## Fuentes

- [Stanford — Sentiment140 (archivo académico original)](https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip)
- [Go, Bhayani y Huang (2009), *Twitter Sentiment Classification using Distant Supervision*](https://cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf)
- [Kaggle — Sentiment140, referencia complementaria](https://www.kaggle.com/datasets/kazanova/sentiment140)
- [Documentación oficial de FastAPI](https://fastapi.tiangolo.com/)
- [Documentación oficial de Render](https://render.com/docs/web-services)

**Alcance congelado: NEXUS ANALYZE — Módulo 1. No se cambia el modelo.**
