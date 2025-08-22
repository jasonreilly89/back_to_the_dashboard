let chart;
let timer;
let glossary = {};

function loadGlossary() {
  fetch(GLOSSARY_URL)
    .then(resp => resp.json())
    .then(data => {
      glossary = data;
      applyTooltips();
    })
    .catch(err => console.error('Failed to load glossary', err));
}

function applyTooltips() {
  document.querySelectorAll('[data-term]').forEach(el => {
    const term = el.getAttribute('data-term');
    if (glossary[term]) {
      el.setAttribute('title', glossary[term]);
    }
  });
}

loadGlossary();

function buildChart(data) {
  const ctx = document.getElementById('metricsChart').getContext('2d');
  const labels = data.map(r => r.day);
  const apDataset = {
    label: 'APμ',
    data: data.map(r => ({x: r.day, y: r.ap_micro, pos_total: r.pos_total})),
    parsing: false,
    tension: 0,
    borderColor: 'rgb(54, 162, 235)',
    backgroundColor: 'rgb(54, 162, 235)',
    pointRadius: 4,
    pointBackgroundColor: ctx => ctx.raw.pos_total === 0 ? '#ffffff' : 'rgb(54,162,235)',
    pointBorderColor: 'rgb(54,162,235)',
  };
  const prevDataset = {
    label: 'Prevμ',
    data: data.map(r => ({x: r.day, y: r.prev_micro})),
    parsing: false,
    tension: 0,
    borderColor: 'rgb(255, 159, 64)',
    backgroundColor: 'rgb(255, 159, 64)',
    pointRadius: 4,
  };
  if (chart) {
    chart.data.labels = labels;
    chart.data.datasets[0] = apDataset;
    chart.data.datasets[1] = prevDataset;
    chart.update();
  } else {
    chart = new Chart(ctx, {
      type: 'line',
      data: {labels, datasets: [apDataset, prevDataset]},
      options: {
        responsive: true,
        scales: { y: {min: 0, max: 1} },
        plugins: {
          tooltip: {
            callbacks: {
              label: function(context) {
                const val = context.parsed.y.toFixed(4);
                if (context.raw.pos_total === 0) {
                  return context.dataset.label + ': ' + val + ' (zero positives)';
                }
                return context.dataset.label + ': ' + val;
              }
            }
          }
        }
      }
    });
  }
}

function populateTable(data) {
  const tbody = document.querySelector('#metricsTable tbody');
  tbody.innerHTML = '';
  for (const r of data) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${r.day}</td>
      <td>${r.epoch}</td>
      <td>${r.ap_micro.toFixed(4)}</td>
      <td>${r.ap_macro.toFixed(4)}</td>
      <td>${r.prev_micro.toFixed(4)}</td>
      <td>${r.pos_total}</td>
      <td>${r.neg_total}</td>
      <td>${r.train_loss.toFixed(4)}</td>
      <td>${r.time_s.toFixed(2)}</td>`;
    tbody.appendChild(tr);
  }
}

function updateKpis(summary) {
  document.getElementById('kpiMean').textContent = summary.ap_micro_mean.toFixed(4);
  document.getElementById('kpiMedian').textContent = summary.ap_micro_median.toFixed(4);
  document.getElementById('kpiPrev').textContent = summary.prev_micro_mean.toFixed(4);
  document.getElementById('kpiDays').textContent = summary.days;
}

function showError(msg) {
  const el = document.getElementById('error');
  el.textContent = msg;
  el.classList.remove('d-none');
}

function clearError() {
  const el = document.getElementById('error');
  el.classList.add('d-none');
}

function fetchMetrics() {
  clearError();
  const startDay = document.getElementById('startDay').value.trim();
  const endDay = document.getElementById('endDay').value.trim();
  const limit = document.getElementById('limit').value.trim();
  let url = '/api/metrics?aggregate=true';
  if (limit) url += `&limit=${limit}`;
  if (startDay) url += `&since_day=${startDay}`;
  url += `&t=${Date.now()}`;
  fetch(url, {cache: 'no-store'})
    .then(resp => resp.json())
    .then(data => {
      let rows = data.best_per_day;
      if (endDay) {
        rows = rows.filter(r => r.day <= endDay);
      }
      buildChart(rows);
      populateTable(rows);
      updateKpis(data.summary);
    })
    .catch(err => {
      showError('Failed to load metrics');
      console.error(err);
    });
}

function setupRefresh() {
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
  const auto = document.getElementById('autoRefresh').checked;
  const interval = parseInt(document.getElementById('refreshInterval').value || '5', 10) * 1000;
  if (auto) {
    timer = setInterval(fetchMetrics, interval);
  }
}

document.getElementById('refreshBtn').addEventListener('click', fetchMetrics);

['autoRefresh', 'refreshInterval'].forEach(id => {
  document.getElementById(id).addEventListener('change', setupRefresh);
});

fetchMetrics();
setupRefresh();
