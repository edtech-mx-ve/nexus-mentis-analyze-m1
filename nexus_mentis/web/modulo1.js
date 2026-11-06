/* NEXUS ANALYZE · Módulo 1 · Métricas reales, sin dependencias externas */
'use strict';

const $ = id => document.getElementById(id);
const esNumber = value => value !== null && value !== undefined &&
  value !== '' && Number.isFinite(Number(value));
const entero = value => esNumber(value) ?
  Math.round(Number(value)).toString().replace(/\B(?=(\d{3})+(?!\d))/g, '.') : '—';
const pct = value => esNumber(value) ?
  `${(Number(value) * 100).toLocaleString('es-ES', {minimumFractionDigits:2, maximumFractionDigits:2})} %` : '—';

function setText(id, value) {
  const node = $(id);
  if (node) node.textContent = String(value);
}
function node(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = String(text);
  return el;
}
function cell(row, tag, value) { row.appendChild(node(tag, '', value)); }
function clearWithMessage(targetId, message) {
  const host = $(targetId);
  host.replaceChildren(node('p', 'loading', message));
}
function validFraction(value) {
  const num = Number(value);
  return esNumber(value) && num >= 0 && num <= 1 ? num : null;
}
function setBars(targetId, rows) {
  const host = $(targetId);
  host.replaceChildren();
  if (!Array.isArray(rows) || rows.length === 0) {
    clearWithMessage(targetId, 'No hay datos suficientes en el reporte.');
    return;
  }
  for (const group of rows) {
    const block = node('div', 'bar-group');
    block.appendChild(node('div', 'bar-group-label', group.label));
    for (const m of group.metrics) {
      const fraction = validFraction(m.value);
      if (fraction === null) continue;
      const line = node('div', 'bar-row');
      line.appendChild(node('span', 'bar-key', m.name));
      const track = node('div', 'bar-track');
      const bar = node('span', `bar-fill ${m.variant}`);
      bar.style.width = `${(fraction * 100).toFixed(4)}%`;
      track.appendChild(bar);
      line.appendChild(track);
      line.appendChild(node('span', 'bar-value', pct(fraction)));
      block.appendChild(line);
    }
    host.appendChild(block);
  }
}
function renderModelTable(items) {
  const host = $('tabla-modelos');
  host.replaceChildren();
  if (!Array.isArray(items) || !items.length) {
    const tr = node('tr');
    const td = node('td', '', 'No hay comparación de validación disponible.');
    td.colSpan = 3;
    tr.appendChild(td);
    host.appendChild(tr);
    return;
  }
  for (const item of items) {
    const tr = node('tr');
    cell(tr, 'td', item.model ?? 'Modelo');
    cell(tr, 'td', pct(item.f1_macro));
    cell(tr, 'td', pct(item.accuracy));
    host.appendChild(tr);
  }
}
function renderModels(metrics) {
  const items = Array.isArray(metrics.validation_comparison) ?
    metrics.validation_comparison.filter(row => row && typeof row.model === 'string') : [];
  renderModelTable(items);
  setBars('comparison-bars', items.map(item => ({
    label: item.model,
    metrics: [
      {name:'F1 macro', value:item.f1_macro, variant:'f1'},
      {name:'Exactitud', value:item.accuracy, variant:'accuracy'}
    ]
  })));
  setText('selection-note', `Modelo seleccionado: ${metrics.model || 'no disponible'}. La selección utiliza validación.`);
  // Matching the selected model avoids presenting a different algorithm's validation as comparable.
  const normalize = value => String(value || '').toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g,'').trim();
  const chosen = items.find(item => normalize(item.model) === normalize(metrics.model));
  if (!chosen) {
    clearWithMessage('comparison-splits', 'Sin valores de validación identificables para el modelo activo.');
    return;
  }
  setBars('comparison-splits', [
    {label:'F1 macro', metrics:[
      {name:'Validación', value:chosen.f1_macro, variant:'validation'},
      {name:'Prueba', value:metrics.test_f1_macro, variant:'testing'}]},
    {label:'Exactitud', metrics:[
      {name:'Validación', value:chosen.accuracy, variant:'validation'},
      {name:'Prueba', value:metrics.test_accuracy, variant:'testing'}]}
  ]);
}
function renderPreparation(metrics) {
  const audit = metrics.audit || {};
  const source = metrics.audited_source_rows;
  const selected = audit.sampled_valid_rows;
  const clean = metrics.clean_rows;
  const removed = esNumber(selected) && esNumber(clean) && Number(selected) >= Number(clean) ?
    Number(selected) - Number(clean) : null;
  setText('origen-total', entero(source));
  setText('muestra-total', entero(selected));
  setText('descartados-total', entero(removed));
  setText('limpios-total', entero(clean));

  const train = Number(metrics.train_rows);
  const validation = Number(metrics.validation_rows);
  const test = Number(metrics.test_rows);
  const total = train + validation + test;
  const bar = $('split-bar');
  const legend = $('split-legend');
  bar.replaceChildren();
  legend.replaceChildren();
  if (![train, validation, test].every(v => Number.isInteger(v) && v >= 0) ||
      total <= 0 || !esNumber(clean) || total !== Number(clean)) {
    bar.setAttribute('aria-label', 'Particiones no disponibles o incoherentes con los registros preparados');
    legend.appendChild(node('p','micro','La distribución no coincide con el reporte; se requiere revisión.'));
    return;
  }
  const options = [
    {label:'Entrenamiento', count:train, css:'split-train', purpose:'Aprender patrones'},
    {label:'Validación', count:validation, css:'split-validation', purpose:'Elegir algoritmo'},
    {label:'Prueba', count:test, css:'split-test', purpose:'Medir rendimiento final'}
  ];
  for (const entry of options) {
    const segment = node('span', entry.css);
    segment.style.width = `${(entry.count / total * 100).toFixed(4)}%`;
    bar.appendChild(segment);
    const legendEntry = node('div');
    legendEntry.appendChild(node('i', `key ${entry.css}`));
    const content = node('span', '', `${entry.label}: ${entero(entry.count)} (${(entry.count / total * 100).toLocaleString('es-ES',{maximumFractionDigits:1})} %) · ${entry.purpose}`);
    legendEntry.appendChild(content);
    legend.appendChild(legendEntry);
  }
  bar.setAttribute('aria-label', options.map(e => `${e.label} ${entero(e.count)}`).join('; '));
  const duplicates = audit.sampled_duplicates_removed;
  const conflicts = audit.sampled_conflicting_records_removed;
  let note = `Preparación: ${entero(selected)} seleccionados; ${entero(clean)} utilizables.`;
  if (removed !== null) note += ` ${entero(removed)} registros no conservados.`;
  if (esNumber(duplicates) && esNumber(conflicts))
    note += ` Entre ellos, ${entero(duplicates)} duplicados y ${entero(conflicts)} etiquetas contradictorias, según la auditoría.`;
  setText('limpieza-resumen', note);
}
function classMetrics(tp, fp, fn) {
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  const f1 = precision + recall ? 2 * precision * recall / (precision + recall) : 0;
  return {precision, recall, f1};
}
function readConfusion(matrix) {
  if (!Array.isArray(matrix) || matrix.length !== 2 ||
      matrix.some(row => !Array.isArray(row) || row.length !== 2 ||
        row.some(v => typeof v !== 'number' || !Number.isSafeInteger(v) || v < 0))) return null;
  const [tn, fp, fn, tp] = [matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]].map(Number);
  const total = tn + fp + fn + tp;
  return total > 0 ? {tn, fp, fn, tp, total} : null;
}
function renderMatrix(matrix, testRows) {
  const host = $('matriz');
  const classHost = $('metricas-clase');
  host.replaceChildren();
  classHost.replaceChildren();
  const m = readConfusion(matrix);
  if (!m) {
    clearWithMessage('matriz','Matriz no disponible en el reporte.');
    setText('matriz-summary', 'No es posible calcular las métricas por clase.');
    return;
  }
  if (esNumber(testRows) && Number(testRows) !== m.total) {
    clearWithMessage('matriz','La matriz no coincide con el total de prueba.');
    setText('matriz-summary', 'Revisar consistencia del reporte antes de visualizar.');
    return;
  }
  const labels = [
    {text:'Real / Predicción', css:'axis'},
    {text:'Pred. negativo', css:'axis'},
    {text:'Pred. positivo', css:'axis'},
    {text:'Real negativo', css:'axis'},
    {number:m.tn, correct:true}, {number:m.fp,correct:false},
    {text:'Real positivo', css:'axis'},
    {number:m.fn, correct:false}, {number:m.tp,correct:true}
  ];
  for (const item of labels) {
    const el = node('div', item.text !== undefined ? item.css : `cell ${item.correct?'correct':'incorrect'}`);
    if (item.text !== undefined) {
      el.textContent = item.text;
    } else {
      el.appendChild(node('strong','', entero(item.number)));
      el.appendChild(node('small','', item.correct ? 'Aciertos' : 'Errores'));
    }
    host.appendChild(el);
  }
  setText('matriz-summary',
    `${entero(m.tn + m.tp)} clasificaciones correctas y ${entero(m.fp + m.fn)} incorrectas de ${entero(m.total)} tweets de prueba.`);
  const negative = classMetrics(m.tn, m.fn, m.fp);
  const positive = classMetrics(m.tp, m.fp, m.fn);
  for (const item of [
    {label:'Negativo', data:negative, count:m.tn + m.fp},
    {label:'Positivo', data:positive, count:m.fn + m.tp}
  ]) {
    const tr = node('tr');
    [item.label,pct(item.data.precision),pct(item.data.recall),pct(item.data.f1),entero(item.count)]
      .forEach(value => cell(tr,'td',value));
    classHost.appendChild(tr);
  }
}
function renderEvaluation(m) {
  setText('accuracy-prueba', pct(m.test_accuracy));
  setText('precision-prueba', pct(m.test_precision_macro));
  setText('recall-prueba', pct(m.test_recall_macro));
  setText('f1-prueba', pct(m.test_f1_macro));
  renderMatrix(m.confusion_matrix, m.test_rows);
}
async function fetchJson(url, options = {}) {
  const response = await fetch(url, {...options, cache:'no-store'});
  if (!response.ok) {
    let detail = `Error HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (typeof body.detail === 'string') detail = body.detail;
    } catch (_ignored) {}
    throw new Error(detail);
  }
  return response.json();
}
async function initialize() {
  const predictButton = $('clasificar');
  predictButton.disabled = true;
  try {
    const health = await fetchJson('/api/health');
    if (health.status !== 'ready' || health.module !== 'NEXUS ANALYZE') throw new Error('Modelo local no disponible.');
    setText('estado-api', 'Modelo local listo. Los resultados se basan en Sentiment140 y la prueba en inglés.');
    setText('modelo-activo', health.model || '—');
    predictButton.disabled = false;
    try {
      const data = await fetchJson('/api/metrics');
      renderPreparation(data);
      renderModels(data);
      renderEvaluation(data);
      setText('modelo-activo', data.model ?? health.model ?? '—');
    } catch (error) {
      setText('limpieza-resumen', `Métricas no disponibles: ${error.message}`);
      clearWithMessage('comparison-bars','Resultados de validación no disponibles.');
      clearWithMessage('comparison-splits','Comparación no disponible.');
      clearWithMessage('matriz','Matriz no disponible.');
      renderModelTable([]);
    }
  } catch (error) {
    setText('estado-api', `Sin modelo: ${error.message}`);
    setText('limpieza-resumen', 'La auditoría se mostrará al cargar el modelo.');
    clearWithMessage('comparison-bars','El modelo no está disponible.');
    clearWithMessage('comparison-splits','El modelo no está disponible.');
    clearWithMessage('matriz','El modelo no está disponible.');
    renderModelTable([]);
  }
}
function renderTrace(trace) {
  const panel = $('traza-detalle');
  panel.hidden = true;
  $('pesos-reales').replaceChildren();
  if (!trace || trace.disponible !== true) return;
  setText('texto-normalizado', trace.texto_normalizado ?? '—');
  setText('tokens-reales', Array.isArray(trace.tokens) ? trace.tokens.join(' · ') : '—');
  setText('ngrams-reales', Array.isArray(trace.ngrams) ? trace.ngrams.slice(0,12).join(' · ') : '—');
  setText('vector-resumen', `Vector de ${entero(trace.dimension_vector)} dimensiones; ${entero(trace.caracteristicas_no_cero)} características activas.`);
  for (const weight of (Array.isArray(trace.pesos_principales) ? trace.pesos_principales : []).slice(0,6)) {
    const row = node('tr');
    cell(row, 'td', weight.termino ?? '');
    cell(row, 'td', esNumber(weight.peso_tfidf) ? Number(weight.peso_tfidf).toLocaleString('es-ES',{maximumFractionDigits:6}) : '—');
    $('pesos-reales').appendChild(row);
  }
  panel.hidden = false;
}
async function classify(event) {
  event.preventDefault();
  const text = $('tweet').value.trim();
  if (text.length < 3 || text.length > 1000) {
    setText('resultado-error','Escribe un texto de 3 a 1000 caracteres.');
    $('resultado-error').hidden = false;
    return;
  }
  const button = $('clasificar');
  button.disabled = true;
  setText('resultado-sentimiento','Analizando…');
  $('resultado-sentimiento').className='result-main';
  $('resultado-error').hidden = true;
  $('traza-detalle').hidden = true;
  try {
    const data = await fetchJson('/api/predict',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text,language:$('idioma').value})
    });
    if (!['positivo','negativo'].includes(data.sentiment)) throw new Error('El modelo devolvió una etiqueta no válida.');
    setText('resultado-sentimiento',data.sentiment==='positivo'?'Sentimiento positivo':'Sentimiento negativo');
    $('resultado-sentimiento').className = 'result-main ' + (data.sentiment==='positivo'?'positive':'negative');
    setText('resultado-nota',data.translation_applied ?
      'Traducción local español → inglés. Es una predicción, no una certeza.' :
      'Clasificación del modelo entrenado. Es una predicción, no una certeza.');
    renderTrace(data.nlp_trace);
  } catch (error) {
    setText('resultado-sentimiento','No se pudo clasificar');
    setText('resultado-error', error.message);
    $('resultado-error').hidden = false;
  } finally {button.disabled = false;}
}
document.addEventListener('DOMContentLoaded', () => {
  $('prediction-form').addEventListener('submit', classify);
  initialize();
});
