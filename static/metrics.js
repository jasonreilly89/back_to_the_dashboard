let chart;
let timer;
let glossary = {};
let allRows = [];
let currentPage = 1;
const PAGE_SIZE = 20;

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
    data: data.map(r => ({ ...r, x: r.day, y: r.ap_micro })),
    parsing: false,
    tension: 0,
    borderColor: 'rgb(54, 162, 235)',
    backgroundColor: 'rgb(54, 162, 235)',
    pointRadius: 4,
    pointBackgroundColor: ctx => ctx.raw.eligible === false ? '#ffffff' : 'rgb(54,162,235)',
    pointBorderColor: 'rgb(54,162,235)',
  };
  const prevDataset = {
    label: 'Prevμ',
    data: data.map(r => ({ ...r, x: r.day, y: r.prev_micro })),
    parsing: false,
    tension: 0,
    borderColor: 'rgb(255, 159, 64)',
    backgroundColor: 'rgb(255, 159, 64)',
    pointRadius: 4,
  };
  if (chart) {
    chart.data.labels = labels;
    chart.data.datasets[0].data = apDataset.data;
    chart.data.datasets[1].data = prevDataset.data;
    chart.update();
  } else {
    chart = new Chart(ctx, {
      type: 'line',
      data: {labels, datasets: [apDataset, prevDataset]},
      options: {
        responsive: true,
        animation: {
          duration: 300,
          easing: 'linear'
        },
        scales: { y: {min: 0, max: 1} },
        plugins: {
          tooltip: {
            callbacks: {
              label: function(context) {
                const val = context.parsed.y.toFixed(4);
                let label = context.dataset.label + ': ' + val;
                if (context.raw.eligible === false) {
                  label += ' (ineligible)';
                }
                return label;
              },
              afterBody: function(contexts) {
                const r = contexts[0].raw;
                const lines = [
                  `APμ: ${r.ap_micro.toFixed(4)}`,
                  `AP̄: ${r.ap_macro.toFixed(4)}`,
                  `Prevμ: ${r.prev_micro.toFixed(4)}`,
                  `Pos: ${r.pos_total}`,
                  `Neg: ${r.neg_total}`,
                  `TrainLoss: ${r.train_loss.toFixed(4)}`,
                  `Time(s): ${r.time_s.toFixed(2)}`
                ];
                if (r.eligible === false) {
                  lines.push('eligible: false');
                }
                return lines;
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
      <td>${r.eligible ? '' : '<span class="badge bg-secondary">ineligible</span>'}</td>
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

function renderPagination() {
  const totalPages = Math.ceil(allRows.length / PAGE_SIZE) || 1;
  const pagination = document.getElementById('pagination');
  pagination.innerHTML = '';

  const prevLi = document.createElement('li');
  prevLi.className = 'page-item' + (currentPage === 1 ? ' disabled' : '');
  const prevLink = document.createElement('a');
  prevLink.className = 'page-link';
  prevLink.href = '#';
  prevLink.textContent = 'Previous';
  prevLink.addEventListener('click', e => {
    e.preventDefault();
    if (currentPage > 1) {
      currentPage--;
      renderTable();
    }
  });
  prevLi.appendChild(prevLink);
  pagination.appendChild(prevLi);

  for (let i = 1; i <= totalPages; i++) {
    const li = document.createElement('li');
    li.className = 'page-item' + (i === currentPage ? ' active' : '');
    const a = document.createElement('a');
    a.className = 'page-link';
    a.href = '#';
    a.textContent = i;
    a.addEventListener('click', e => {
      e.preventDefault();
      currentPage = i;
      renderTable();
    });
    li.appendChild(a);
    pagination.appendChild(li);
  }

  const nextLi = document.createElement('li');
  nextLi.className = 'page-item' + (currentPage === totalPages ? ' disabled' : '');
  const nextLink = document.createElement('a');
  nextLink.className = 'page-link';
  nextLink.href = '#';
  nextLink.textContent = 'Next';
  nextLink.addEventListener('click', e => {
    e.preventDefault();
    if (currentPage < totalPages) {
      currentPage++;
      renderTable();
    }
  });
  nextLi.appendChild(nextLink);
  pagination.appendChild(nextLi);
}

function renderTable() {
  const start = (currentPage - 1) * PAGE_SIZE;
  const pageData = allRows.slice(start, start + PAGE_SIZE);
  populateTable(pageData);
  renderPagination();
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
      // Use aggregated rows for both the chart and table so the x-axis
      // shows one point per day instead of multiple points stacked on
      // the same day, which previously produced vertical lines.
      let chartRows = data.best_per_day;
      let tableRows = data.best_per_day;
      if (endDay) {
        chartRows = chartRows.filter(r => r.day <= endDay);
        tableRows = tableRows.filter(r => r.day <= endDay);
      }
      buildChart(chartRows);
      allRows = tableRows;
      currentPage = 1;
      renderTable();
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
