/**
 * main.js — Frontend logic for Fake News Detector
 * Linked to: app.py (/predict endpoint), index.html
 */

// ── DOM refs ──
const newsInput     = document.getElementById('newsInput');
const analyseBtn    = document.getElementById('analyseBtn');
const clearBtn      = document.getElementById('clearBtn');
const clearHistBtn  = document.getElementById('clearHistoryBtn');
const charCount     = document.getElementById('charCount');
const resultBox     = document.getElementById('resultBox');
const loadingBox    = document.getElementById('loadingBox');
const errorBox      = document.getElementById('errorBox');
const resultIcon    = document.getElementById('resultIcon');
const resultLabel   = document.getElementById('resultLabel');
const confBar       = document.getElementById('confBar');
const confValue     = document.getElementById('confValue');
const resultNote    = document.getElementById('resultNote');
const historyList   = document.getElementById('historyList');

// ── Stat elements ──
const sTotal = document.getElementById('s-total');
const sFake  = document.getElementById('s-fake');
const sReal  = document.getElementById('s-real');
const sConf  = document.getElementById('s-conf');


// ── Character counter ──
newsInput.addEventListener('input', () => {
  charCount.textContent = newsInput.value.length;
});


// ── Analyse button click ──
analyseBtn.addEventListener('click', analyseText);

// ── Allow Ctrl+Enter to submit ──
newsInput.addEventListener('keydown', (e) => {
  if (e.ctrlKey && e.key === 'Enter') analyseText();
});


async function analyseText() {
  const text = newsInput.value.trim();
  if (text.length < 10) {
    showError('Please enter at least 10 characters of text.');
    return;
  }

  // ── Show loading ──
  hideAll();
  loadingBox.classList.remove('hidden');
  analyseBtn.disabled = true;

  try {
    // ── POST to Flask /predict  ← calls app.py → predictor.py → preprocess.py ──
    const response = await fetch('/predict', {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify({ text }),
    });

    const data = await response.json();
    loadingBox.classList.add('hidden');

    if (!response.ok || data.error) {
      showError(data.error || 'Server error. Please try again.');
      return;
    }

    // ── Render result ──
    showResult(data);

    // ── Update stats from server ──
    await updateStats();

    // ── Prepend to history list ──
    addHistoryItem(data, text);

  } catch (err) {
    loadingBox.classList.add('hidden');
    showError('Network error. Is the Flask server running?');
  } finally {
    analyseBtn.disabled = false;
  }
}


function showResult(data) {
  const isReal = data.label === 'REAL';

  resultIcon.textContent  = isReal ? '✅' : '❌';
  resultLabel.textContent = data.label;
  resultLabel.className   = 'result-label ' + data.label.toLowerCase();

  // Confidence bar
  confBar.style.width = data.confidence + '%';
  confBar.className   = 'conf-bar ' + data.label.toLowerCase();
  confValue.textContent = data.confidence + '%';

  // Note
  if (isReal) {
    resultNote.textContent = `This article appears to be legitimate news with ${data.confidence}% confidence.`;
  } else {
    resultNote.textContent = `⚠ This content shows strong indicators of misinformation (${data.confidence}% confidence).`;
  }

  resultBox.className = 'result-box ' + data.label.toLowerCase();
  resultBox.classList.remove('hidden');
}


function showError(msg) {
  hideAll();
  errorBox.textContent = '⚠ ' + msg;
  errorBox.classList.remove('hidden');
}

function hideAll() {
  resultBox.classList.add('hidden');
  loadingBox.classList.add('hidden');
  errorBox.classList.add('hidden');
}


// ── Clear text ──
clearBtn.addEventListener('click', () => {
  newsInput.value     = '';
  charCount.textContent = '0';
  hideAll();
});


// ── Update stats from /stats endpoint ──
async function updateStats() {
  try {
    const res  = await fetch('/stats');
    const data = await res.json();
    sTotal.textContent = data.total;
    sFake.textContent  = data.fake;
    sReal.textContent  = data.real;
    sConf.textContent  = data.avg_conf + '%';
  } catch (_) { /* silent */ }
}


// ── Add a row to history list (client-side, instant) ──
function addHistoryItem(data, text) {
  // Remove empty placeholder
  const empty = historyList.querySelector('.empty-msg');
  if (empty) empty.remove();

  const item    = document.createElement('div');
  const label_l = data.label.toLowerCase();
  const snippet = text.length > 100 ? text.slice(0, 100) + '...' : text;
  const now     = new Date().toLocaleTimeString('en-IN', { hour12: false });

  item.className = `history-item ${label_l}`;
  item.innerHTML = `
    <span class="h-badge ${label_l}">${data.label}</span>
    <span class="h-text">${escapeHtml(snippet)}</span>
    <span class="h-conf">${data.confidence}%</span>
    <span class="h-time">${now}</span>
  `;

  historyList.insertBefore(item, historyList.firstChild);

  // Keep max 10 visible
  const items = historyList.querySelectorAll('.history-item');
  if (items.length > 10) items[items.length - 1].remove();
}


// ── Clear history ──
clearHistBtn.addEventListener('click', async () => {
  await fetch('/clear_history', { method: 'POST' });
  historyList.innerHTML = '<p class="empty-msg">No checks yet. Analyse something above!</p>';
  await updateStats();
});


// ── Utility ──
function escapeHtml(text) {
  const d = document.createElement('div');
  d.appendChild(document.createTextNode(text));
  return d.innerHTML;
}

// ── Nav pill active state on scroll ──
const sections = document.querySelectorAll('section[id]');
const pills    = document.querySelectorAll('.pill');
window.addEventListener('scroll', () => {
  let current = '';
  sections.forEach(sec => {
    if (window.scrollY >= sec.offsetTop - 100) current = sec.id;
  });
  pills.forEach(p => {
    p.classList.toggle('active', p.getAttribute('href') === '#' + current);
  });
});
