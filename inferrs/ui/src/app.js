// ── App ─────────────────────────────────────────────────────────────────────
(function() {
'use strict';

// ── Defaults ────────────────────────────────────────────────────────────────
var DEFAULT_SETTINGS = {
  // Basic
  temperature: 0.7,
  maxTokens: 2048,
  topP: 0.95,
  systemPrompt: '',
  // Sampler (advanced)
  topK: 0,                  // 0 = disabled
  minP: 0,                  // 0 = disabled
  repetitionPenalty: 1.0,   // 1 = disabled
  frequencyPenalty: 0,
  presencePenalty: 0,
  seed: null,               // null = auto
  stopSequences: [],
  // Behavior
  reasoningEffort: '',      // '' | 'low' | 'medium' | 'high'
  showStats: false,
  showReasoning: false,
  streamRender: true
};

// ── State ───────────────────────────────────────────────────────────────────
var state = {
  conversations: [],   // [{id, title, messages:[{role,content,stats?}], created}]
  activeId: null,
  settings: Object.assign({}, DEFAULT_SETTINGS),
  models: [],
  activeModel: null,
  serverState: 'unknown', // 'ok' | 'loading' | 'error' | 'offline' | 'unknown'
  streaming: false,
  abortController: null,
  loadingModel: null,  // model id of in-flight load (so the stop button can also cancel server-side)
  lastResponseStats: null,  // {ttftMs, totalMs, tokens, tps, finish, reasoningChunks}
  modelMetaCache: {}   // model id → ModelMeta from /api/show
};

function saveToStorage() {
  try {
    localStorage.setItem('inferrs_convos', JSON.stringify(state.conversations));
    localStorage.setItem('inferrs_settings', JSON.stringify(state.settings));
    if (state.activeModel) localStorage.setItem('inferrs_model', state.activeModel);
  } catch(e) {}
}
function loadFromStorage() {
  try {
    var c = localStorage.getItem('inferrs_convos');
    if (c) state.conversations = JSON.parse(c);
    var s = localStorage.getItem('inferrs_settings');
    if (s) state.settings = Object.assign({}, DEFAULT_SETTINGS, JSON.parse(s));
    var m = localStorage.getItem('inferrs_model');
    if (m) state.activeModel = m;
  } catch(e) {}
}

// ── Markdown renderer setup ─────────────────────────────────────────────────
marked.setOptions({
  gfm: true,
  breaks: true,
  highlight: function(code, lang) {
    if (lang && hljs.getLanguage(lang)) {
      try { return hljs.highlight(code, {language: lang}).value; } catch(e) {}
    }
    try { return hljs.highlightAuto(code).value; } catch(e) {}
    return code;
  }
});

function renderMarkdown(text) {
  // Extract <think>…</think> blocks
  var thinking = [];
  var cleaned = text.replace(/<think>([\s\S]*?)<\/think>/g, function(_, t) {
    thinking.push(t.trim());
    return '';
  });
  var html = '';
  if (thinking.length) {
    var openAttr = state.settings.showReasoning ? ' open' : '';
    html += '<details class="thinking-block"' + openAttr + '><summary>Reasoning ('
      + thinking.length + ' block' + (thinking.length > 1 ? 's' : '') + ')</summary><div>'
      + escHtml(thinking.join('\n\n')) + '</div></details>';
  }
  html += marked.parse(cleaned);
  return html;
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── DOM refs ────────────────────────────────────────────────────────────────
var $ = function(id) { return document.getElementById(id); };
var $messages = $('messages');
var $emptyState = $('empty-state');
var $prompt = $('prompt');
var $sendBtn = $('send-btn');
var $stopBtn = $('stop-btn');
var $historyList = $('history-list');
var $modelSelect = $('model-select');
var $newChatBtn = $('new-chat-btn');
var $settingsBtn = $('settings-btn');
var $settingsOverlay = $('settings-overlay');
var $settingsCancel = $('settings-cancel');
var $settingsSave = $('settings-save');
var $settingsReset = $('settings-reset');
var $sidebar = $('sidebar');
var $sidebarToggle = $('sidebar-toggle');
var $statsPanel = $('stats-panel');
var $statsToggle = $('stats-toggle');
var $statsClose = $('stats-close');
var $serverStatus = $('server-status');
var $statusDot = $serverStatus.querySelector('.status-dot');
var $statusText = $serverStatus.querySelector('.status-text');
var $samplingSummary = $('sampling-summary');
var $liveStats = $('live-stats');
var $clearHistoryBtn = $('clear-history-btn');
var $deviceBadge = $('device-badge');
var $deviceText = $deviceBadge ? $deviceBadge.querySelector('.dp-text') : null;

// ── Models ──────────────────────────────────────────────────────────────────
async function fetchModels() {
  try {
    var res = await fetch('/v1/models');
    if (!res.ok) return;
    var data = await res.json();
    state.models = (data.data || []).map(function(m) { return m.id; });
    renderModelSelect();
    renderStatsPanel();
  } catch(e) {
    state.models = [];
    renderModelSelect();
    renderStatsPanel();
  }
}

function renderModelSelect() {
  $modelSelect.innerHTML = '';
  if (state.models.length === 0) {
    var opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'No models loaded';
    $modelSelect.appendChild(opt);
    return;
  }
  state.models.forEach(function(m) {
    var opt = document.createElement('option');
    opt.value = m;
    opt.textContent = m.split('/').pop();
    opt.title = m;
    if (m === state.activeModel) opt.selected = true;
    $modelSelect.appendChild(opt);
  });
  if (!state.activeModel || state.models.indexOf(state.activeModel) === -1) {
    state.activeModel = state.models[0];
  }
  $modelSelect.value = state.activeModel;
}

$modelSelect.addEventListener('change', function() {
  state.activeModel = $modelSelect.value;
  saveToStorage();
  renderStatsPanel();
  // Don't actively pull /api/show here — that triggers a model load on the
  // server.  Wait for the user to send a real message; once the model is
  // loaded, we'll refresh the meta.
});

// ── Server health polling ───────────────────────────────────────────────────
async function pollHealth() {
  try {
    var res = await fetch('/health', { cache: 'no-store' });
    if (res.ok) {
      state.serverState = 'ok';
    } else if (res.status === 503) {
      state.serverState = 'loading';
    } else {
      state.serverState = 'error';
    }
  } catch(e) {
    state.serverState = 'offline';
  }
  updateStatusBadge();
  renderStatsPanel();
}

function updateStatusBadge() {
  var labels = {
    ok: 'ready',
    loading: 'loading',
    error: 'error',
    offline: 'offline',
    unknown: 'checking…'
  };
  $statusDot.dataset.state = state.serverState;
  $statusText.textContent = labels[state.serverState] || state.serverState;
}

// ── Conversations ───────────────────────────────────────────────────────────
function genId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
}

function newConversation() {
  var id = genId();
  var convo = { id: id, title: 'New conversation', messages: [], created: Date.now() };
  state.conversations.unshift(convo);
  state.activeId = id;
  saveToStorage();
  renderHistory();
  renderMessages();
  renderStatsPanel();
  $prompt.focus();
  return convo;
}

function activeConvo() {
  for (var i = 0; i < state.conversations.length; i++) {
    if (state.conversations[i].id === state.activeId) return state.conversations[i];
  }
  return null;
}

function switchConvo(id) {
  state.activeId = id;
  renderHistory();
  renderMessages();
  renderStatsPanel();
}

function deleteConvo(id) {
  state.conversations = state.conversations.filter(function(c) { return c.id !== id; });
  if (state.activeId === id) {
    state.activeId = state.conversations.length ? state.conversations[0].id : null;
  }
  saveToStorage();
  renderHistory();
  renderMessages();
  renderStatsPanel();
}

function clearAllHistory() {
  if (state.conversations.length === 0) return;
  if (!confirm('Delete all conversations? This cannot be undone.')) return;
  state.conversations = [];
  state.activeId = null;
  saveToStorage();
  renderHistory();
  renderMessages();
  renderStatsPanel();
}

function titleFromContent(text) {
  var t = text.trim().replace(/\s+/g, ' ');
  return t.length > 50 ? t.slice(0, 48) + '…' : t;
}

// ── History sidebar ─────────────────────────────────────────────────────────
function renderHistory() {
  $historyList.innerHTML = '';
  if (state.conversations.length === 0) return;
  state.conversations.forEach(function(c) {
    var div = document.createElement('div');
    div.className = 'history-item' + (c.id === state.activeId ? ' active' : '');
    div.innerHTML = escHtml(c.title) + '<span class="hi-date">' + fmtDate(c.created) + '</span>';
    div.addEventListener('click', function() { switchConvo(c.id); });

    var del = document.createElement('button');
    del.className = 'hi-del';
    del.title = 'Delete';
    del.innerHTML = '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
    del.addEventListener('click', function(e) {
      e.stopPropagation();
      if (confirm('Delete this conversation?')) deleteConvo(c.id);
    });
    div.appendChild(del);

    $historyList.appendChild(div);
  });
}

function fmtDate(ts) {
  var d = new Date(ts);
  var now = new Date();
  if (d.toDateString() === now.toDateString()) {
    return d.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
  }
  return d.toLocaleDateString([], {month:'short',day:'numeric'});
}

// ── Message rendering ───────────────────────────────────────────────────────
function renderMessages() {
  var convo = activeConvo();
  if (!convo || convo.messages.length === 0) {
    $messages.innerHTML = '';
    $messages.appendChild($emptyState);
    $emptyState.style.display = 'flex';
    return;
  }
  $emptyState.style.display = 'none';
  $messages.innerHTML = '';
  convo.messages.forEach(function(msg) {
    var body = appendMessageEl(msg.role, msg.content, false);
    if (msg.stats && msg.role === 'assistant') {
      attachMessageStats(body.parentElement, msg.stats);
    }
  });
  scrollToBottom();
}

function appendMessageEl(role, content, isStreaming) {
  $emptyState.style.display = 'none';
  var wrap = document.createElement('div');
  wrap.className = 'msg-wrap';
  var roleEl = document.createElement('div');
  roleEl.className = 'msg-role ' + role;
  var roleSpan = document.createElement('span');
  roleSpan.textContent = role === 'user' ? 'You' : 'Assistant';
  roleEl.appendChild(roleSpan);
  var body = document.createElement('div');
  body.className = 'msg-body';
  if (isStreaming) {
    body.innerHTML = '<div class="typing-dot"><span></span><span></span><span></span></div>';
    body.dataset.streaming = '1';
    body.dataset.raw = '';
  } else {
    body.innerHTML = role === 'user' ? '<p>' + escHtml(content) + '</p>' : renderMarkdown(content);
    addCopyButtons(body);
  }
  wrap.appendChild(roleEl);
  wrap.appendChild(body);
  $messages.appendChild(wrap);
  scrollToBottom();
  return body;
}

function attachMessageStats(wrap, stats) {
  if (!wrap) return;
  var roleEl = wrap.querySelector('.msg-role');
  if (!roleEl) return;
  var existing = roleEl.querySelector('.msg-stats');
  if (existing) existing.remove();
  var statsEl = document.createElement('span');
  statsEl.className = 'msg-stats';
  if (stats.error) {
    // Errors are surfaced via .error-notice now; nothing to show inline.
    return;
  } else {
    var parts = [];
    if (stats.ttftMs != null) parts.push('TTFT <span>' + fmtMs(stats.ttftMs) + '</span>');
    if (stats.tokens != null) parts.push('<span>' + stats.tokens + '</span> tok');
    if (stats.tps != null && isFinite(stats.tps)) parts.push('<span>' + stats.tps.toFixed(1) + '</span> tok/s');
    if (stats.totalMs != null) parts.push('<span>' + fmtMs(stats.totalMs) + '</span>');
    if (stats.finish && stats.finish !== 'stop') parts.push('<span>' + escHtml(stats.finish) + '</span>');
    statsEl.innerHTML = parts.join(' · ');
  }
  roleEl.appendChild(statsEl);
}

function fmtMs(ms) {
  if (ms == null) return '—';
  if (ms < 1000) return Math.round(ms) + 'ms';
  return (ms / 1000).toFixed(2) + 's';
}

function addCopyButtons(container) {
  container.querySelectorAll('pre').forEach(function(pre) {
    if (pre.parentNode.classList && pre.parentNode.classList.contains('code-block-wrap')) return;
    var wrap = document.createElement('div');
    wrap.className = 'code-block-wrap';
    pre.parentNode.insertBefore(wrap, pre);
    wrap.appendChild(pre);
    var btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.textContent = 'Copy';
    btn.addEventListener('click', function() {
      var code = pre.querySelector('code');
      var text = code ? code.textContent : pre.textContent;
      navigator.clipboard.writeText(text).then(function() {
        btn.textContent = 'Copied!';
        btn.classList.add('copied');
        setTimeout(function() {
          btn.textContent = 'Copy';
          btn.classList.remove('copied');
        }, 2000);
      }).catch(function() {});
    });
    wrap.appendChild(btn);
  });
}

function scrollToBottom() {
  $messages.scrollTop = $messages.scrollHeight;
}

function appendErrorNotice(message) {
  $emptyState.style.display = 'none';
  var notice = document.createElement('div');
  notice.className = 'error-notice';
  notice.innerHTML =
    '<div class="en-icon">' +
      '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">' +
        '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>' +
      '</svg>' +
    '</div>' +
    '<div class="en-body">' +
      '<div class="en-title">Request failed</div>' +
      '<div class="en-message"></div>' +
    '</div>';
  notice.querySelector('.en-message').textContent = message;
  $messages.appendChild(notice);
  scrollToBottom();
  return notice;
}

// Try to extract just the inner `message` from a JSON error body produced by
// the server, falling back to the raw error text when parsing fails.
function parseErrorMessage(err) {
  var raw = err && err.message ? err.message : String(err);
  // Common shape: "Server error 500: {\"error\":{\"message\":\"…\",\"type\":\"…\"}}"
  var braceIdx = raw.indexOf('{');
  if (braceIdx >= 0) {
    var jsonPart = raw.slice(braceIdx);
    try {
      var parsed = JSON.parse(jsonPart);
      var inner = parsed && parsed.error && parsed.error.message;
      if (inner) {
        var prefix = raw.slice(0, braceIdx).trim();
        return prefix ? prefix + ' — ' + inner : inner;
      }
    } catch(_) { /* not JSON, fall through */ }
  }
  return raw;
}

// ── Stats panel ─────────────────────────────────────────────────────────────
function renderStatsPanel() {
  var convo = activeConvo();
  setText('stat-server-state', state.serverState);
  setText('stat-endpoint', location.host || '—');
  setText('stat-models-count', String(state.models.length));
  var activeEl = $('stat-active-model');
  if (activeEl) {
    activeEl.textContent = state.activeModel || '—';
    activeEl.title = state.activeModel || '';
  }
  var s = state.lastResponseStats;
  setText('stat-ttft', s ? fmtMs(s.ttftMs) : '—');
  setText('stat-total-time', s ? fmtMs(s.totalMs) : '—');
  setText('stat-tokens', s && s.tokens != null ? String(s.tokens) : '—');
  setText('stat-tps', s && s.tps != null && isFinite(s.tps) ? s.tps.toFixed(1) : '—');
  setText('stat-finish', s && s.finish ? s.finish : '—');
  setText('stat-reasoning', s && s.reasoningChunks != null ? String(s.reasoningChunks) : '—');
  setText('stat-convo-count', String(state.conversations.length));
  setText('stat-msg-count', convo ? String(convo.messages.length) : '0');
  renderModelMeta();
}

function renderModelMeta() {
  var meta = state.activeModel ? state.modelMetaCache[state.activeModel] : null;
  updateDeviceBadge(meta);
  setText('stat-mm-format', meta && meta.format ? meta.format : '—');
  setText('stat-mm-quant', meta && meta.quantization ? meta.quantization : '—');
  setText('stat-mm-family', meta && meta.family ? meta.family : '—');
  setText('stat-mm-arch', meta && meta.architecture ? meta.architecture : '—');
  setText('stat-mm-ctx', meta && meta.context_length ? fmtCount(meta.context_length) : '—');
  setText('stat-mm-vocab', meta && meta.vocab_size ? fmtCount(meta.vocab_size) : '—');
  var shape = meta && meta.num_layers
    ? meta.num_layers + ' · ' + (meta.num_attention_heads || '?') + ' · ' + (meta.num_kv_heads || '?')
    : '—';
  setText('stat-mm-shape', shape);
  setText('stat-mm-hidden', meta && meta.hidden_size ? String(meta.hidden_size) : '—');
  var mods = [];
  if (meta && meta.has_audio) mods.push('audio');
  if (meta && meta.has_vision) mods.push('vision');
  setText('stat-mm-modalities', mods.length ? mods.join(', ') : (meta ? 'text' : '—'));
  setText('stat-rt-device', meta && meta.device ? meta.device : '—');
  setText('stat-rt-dtype', meta && meta.dtype ? meta.dtype : '—');
  var tq = meta && meta.turbo_quant_bits != null
    ? meta.turbo_quant_bits + '-bit'
    : (meta ? 'off' : '—');
  setText('stat-rt-tq', tq);
}

function fmtCount(n) {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + 'M';
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K';
  return String(n);
}

// Update the navbar device badge.  Hidden until we have meta for the active
// model.  Color is set via [data-device] in CSS.
function updateDeviceBadge(meta) {
  if (!$deviceBadge || !$deviceText) return;
  if (!meta || !meta.device) {
    $deviceBadge.hidden = true;
    return;
  }
  var dev = String(meta.device).toLowerCase().split(':')[0];
  $deviceBadge.dataset.device = dev;
  var label = dev.toUpperCase();
  if (meta.dtype) label += ' · ' + meta.dtype;
  $deviceText.textContent = label;
  var tip = 'Inference device: ' + meta.device;
  if (meta.dtype) tip += '\nCompute dtype: ' + meta.dtype;
  if (dev === 'cpu') tip += '\n\nCPU inference is significantly slower than GPU. Build with --features cuda or --features metal for GPU acceleration.';
  $deviceBadge.title = tip;
  $deviceBadge.hidden = false;
}

// Fetch metadata via /api/show.  Cached per model id; also a no-op when the
// request fails (the model may not be loaded yet — we'll try again on the
// next render after the user actually sends a message).
async function fetchModelMeta(modelId, opts) {
  if (!modelId) return;
  if (!opts || !opts.force) {
    if (state.modelMetaCache[modelId]) {
      renderModelMeta();
      return;
    }
  }
  try {
    var res = await fetch('/api/show', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: modelId })
    });
    if (!res.ok) return;
    var data = await res.json();
    var meta = data.inferrs_meta || null;
    if (!meta && data.details) {
      meta = {
        format: data.details.format,
        quantization: data.details.quantization_level,
        family: data.details.family
      };
    }
    if (meta) {
      state.modelMetaCache[modelId] = meta;
      if (modelId === state.activeModel) renderModelMeta();
    }
  } catch(e) {}
}

function setText(id, text) {
  var el = $(id);
  if (el) el.textContent = text;
}

// ── Sampling summary (next to input) ────────────────────────────────────────
function renderSamplingSummary() {
  var s = state.settings;
  var tags = [];
  tags.push(tag('temp', s.temperature.toFixed(2)));
  tags.push(tag('max', s.maxTokens));
  if (s.topP < 1) tags.push(tag('top-p', s.topP.toFixed(2)));
  if (s.topK > 0) tags.push(tag('top-k', s.topK));
  if (s.minP > 0) tags.push(tag('min-p', s.minP.toFixed(2)));
  if (s.repetitionPenalty !== 1) tags.push(tag('rep', s.repetitionPenalty.toFixed(2)));
  if (s.frequencyPenalty !== 0) tags.push(tag('freq', s.frequencyPenalty.toFixed(2)));
  if (s.presencePenalty !== 0) tags.push(tag('pres', s.presencePenalty.toFixed(2)));
  if (s.seed != null) tags.push(tag('seed', s.seed));
  if (s.reasoningEffort) tags.push(tag('reason', s.reasoningEffort));
  if (s.systemPrompt) tags.push(tag('sys', '✓'));
  $samplingSummary.innerHTML = tags.join('');
}

function tag(label, value) {
  return '<span class="ss-tag"><b>' + label + '</b> ' + escHtml(String(value)) + '</span>';
}

// ── Send message ────────────────────────────────────────────────────────────
async function sendMessage() {
  var text = $prompt.value.trim();
  if (!text || state.streaming) return;

  var model = state.activeModel || (state.models[0] || null);
  if (!model) {
    alert('No model selected. Please ensure a model is loaded.');
    return;
  }

  var convo = activeConvo() || newConversation();
  convo.messages.push({ role: 'user', content: text });
  if (convo.title === 'New conversation') {
    convo.title = titleFromContent(text);
  }
  $prompt.value = '';
  autoResizeTextarea();
  renderHistory();

  appendMessageEl('user', text, false);
  var assistantBody = appendMessageEl('assistant', '', true);
  var assistantWrap = assistantBody.parentElement;

  setStreamingUI(true);

  // Build request body from settings.  Strip locally-stored fields like
  // `stats` from the message history before shipping — the OpenAI spec only
  // allows {role, content}, and strict servers will 400 on extras.
  var s = state.settings;
  var messages = [];
  if (s.systemPrompt) messages.push({ role: 'system', content: s.systemPrompt });
  for (var i = 0; i < convo.messages.length; i++) {
    var m = convo.messages[i];
    messages.push({ role: m.role, content: m.content });
  }

  var body = {
    model: model,
    messages: messages,
    stream: true,
    temperature: s.temperature,
    max_tokens: s.maxTokens,
    top_p: s.topP
  };
  if (s.topK > 0) body.top_k = s.topK;
  if (s.minP > 0) body.min_p = s.minP;
  if (s.repetitionPenalty !== 1) body.repetition_penalty = s.repetitionPenalty;
  if (s.frequencyPenalty !== 0) body.frequency_penalty = s.frequencyPenalty;
  if (s.presencePenalty !== 0) body.presence_penalty = s.presencePenalty;
  if (s.seed != null) body.seed = s.seed;
  if (s.stopSequences && s.stopSequences.length) body.stop = s.stopSequences;
  if (s.reasoningEffort) body.reasoning_effort = s.reasoningEffort;

  var accumulated = '';
  var accumulatedReasoning = '';
  var contentChunks = 0;
  var reasoningChunks = 0;
  var startedAt = performance.now();
  var firstTokenAt = null;
  var finishReason = null;
  var liveTimer = null;
  var modelReadyAt = null;        // performance.now() when /api/ps first reported model loaded
  var psCheckInFlight = false;
  state.abortController = new AbortController();
  state.loadingModel = model;

  function updateLive() {
    var now = performance.now();
    var elapsed = now - startedAt;
    var tokens = contentChunks;
    var tps = firstTokenAt != null && now > firstTokenAt
      ? tokens / ((now - firstTokenAt) / 1000)
      : 0;
    $liveStats.hidden = false;
    $liveStats.textContent = tokens + ' tok · '
      + (isFinite(tps) ? tps.toFixed(1) : '0') + ' tok/s · '
      + fmtMs(elapsed);

    // While we wait for the first token, the placeholder bubble surfaces what
    // the server is most likely doing.  We poll /api/ps to detect the moment
    // the model becomes ready — that lets us distinguish "still loading"
    // from "model loaded, generating first token (slow)".
    if (firstTokenAt == null && assistantBody && assistantBody.dataset.streaming === '1') {
      // Trigger an /api/ps probe at most once per 1.5s (and only after the
      // model has had a chance to start loading).
      if (modelReadyAt == null && elapsed > 1500 && !psCheckInFlight) {
        psCheckInFlight = true;
        fetch('/api/ps', { cache: 'no-store' }).then(function(r) {
          return r.ok ? r.json() : null;
        }).then(function(data) {
          if (data && Array.isArray(data.models)) {
            var found = data.models.some(function(m) {
              return m && (m.model === model || m.name === model);
            });
            if (found && modelReadyAt == null) modelReadyAt = performance.now();
          }
        }).catch(function() {}).finally(function() {
          psCheckInFlight = false;
        });
      }

      var phase, hint, showCancel = false;
      if (elapsed < 800) {
        phase = 'sending'; hint = 'Sending request…';
      } else if (modelReadyAt != null) {
        // Model is loaded server-side — we're now waiting on actual inference.
        var genElapsed = (performance.now() - modelReadyAt) / 1000;
        phase = 'generating_first';
        hint = 'Model loaded · generating first token… ' + genElapsed.toFixed(0) + 's';
        if (genElapsed > 15) {
          hint += '<br><span class="ss-sub">Inference is slow on the current device — typical for CPU runs of larger models.</span>';
        }
        showCancel = true;
      } else if (elapsed < 8000) {
        phase = 'loading';
        hint = 'Loading model <b>' + escHtml(model.split('/').pop()) + '</b> · '
          + (elapsed / 1000).toFixed(0) + 's';
        showCancel = true;
      } else if (elapsed < 30000) {
        phase = 'long_load';
        hint = 'Still loading <b>' + escHtml(model.split('/').pop()) + '</b> · '
          + (elapsed / 1000).toFixed(0) + 's<br><span class="ss-sub">First request to a fresh model can take a while (download + initialize).</span>';
        showCancel = true;
      } else {
        phase = 'very_long_load';
        hint = 'Still loading <b>' + escHtml(model.split('/').pop()) + '</b> · '
          + (elapsed / 1000).toFixed(0) + 's<br><span class="ss-sub">If this is a large model, weights may still be downloading. Check the server log for progress.</span>';
        showCancel = true;
      }
      var cancelHtml = showCancel
        ? '<button type="button" class="ss-cancel" data-cancel-load="1">Cancel load</button>'
        : '';
      if (assistantBody.dataset.phase !== phase) {
        assistantBody.dataset.phase = phase;
        assistantBody.innerHTML =
          '<div class="streaming-status">' +
            '<div class="ss-text">' + hint + '</div>' +
            '<div class="typing-dot"><span></span><span></span><span></span></div>' +
          '</div>' + cancelHtml;
        var cancelBtn = assistantBody.querySelector('[data-cancel-load]');
        if (cancelBtn) cancelBtn.addEventListener('click', cancelCurrentLoad);
      } else {
        // Same phase — just update the elapsed seconds in place, no flicker.
        var textEl = assistantBody.querySelector('.streaming-status .ss-text');
        if (textEl) textEl.innerHTML = hint;
      }
    }
  }
  liveTimer = setInterval(updateLive, 250);
  updateLive();

  try {
    var res = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: state.abortController.signal,
      body: JSON.stringify(body)
    });

    if (!res.ok) {
      var errTxt = await res.text();
      throw new Error('Server error ' + res.status + ': ' + errTxt);
    }

    var reader = res.body.getReader();
    var decoder = new TextDecoder();
    var buf = '';

    while (true) {
      var chunk = await reader.read();
      if (chunk.done) break;
      buf += decoder.decode(chunk.value, { stream: true });

      var lines = buf.split('\n');
      buf = lines.pop();

      for (var i = 0; i < lines.length; i++) {
        var line = lines[i].trim();
        if (!line || line === 'data: [DONE]') continue;
        if (line.indexOf('data: ') !== 0) continue;
        try {
          var json = JSON.parse(line.slice(6));
          var choice = json.choices && json.choices[0];
          if (!choice) continue;
          var delta = choice.delta || {};
          if (delta.content) {
            if (firstTokenAt == null) {
              firstTokenAt = performance.now();
              // Once tokens are flowing, the model is loaded — clear the
              // load-cancel flag so a later stop click only aborts the
              // generation without unloading the model server-side.
              state.loadingModel = null;
            }
            accumulated += delta.content;
            contentChunks++;
          }
          if (delta.reasoning_content) {
            if (firstTokenAt == null) {
              firstTokenAt = performance.now();
              state.loadingModel = null;
            }
            accumulatedReasoning += delta.reasoning_content;
            reasoningChunks++;
          }
          if (choice.finish_reason) finishReason = choice.finish_reason;

          if (s.streamRender) {
            var display = accumulatedReasoning
              ? '<think>' + accumulatedReasoning + '</think>' + accumulated
              : accumulated;
            if (display) {
              assistantBody.innerHTML = renderMarkdown(display);
              addCopyButtons(assistantBody);
              scrollToBottom();
            }
          }
        } catch(e) { /* ignore parse errors on partial lines */ }
      }
    }

    var endedAt = performance.now();
    var finalContent = accumulatedReasoning
      ? '<think>' + accumulatedReasoning + '</think>' + accumulated
      : accumulated;
    var stats = computeStats(startedAt, firstTokenAt, endedAt, contentChunks, reasoningChunks, finishReason);
    convo.messages.push({ role: 'assistant', content: finalContent, stats: stats });
    state.lastResponseStats = stats;
    attachMessageStats(assistantWrap, stats);
    // The model is definitely loaded now — refresh metadata for the panel.
    fetchModelMeta(model, { force: true });
  } catch(e) {
    var endedAt2 = performance.now();
    var stats2;
    if (e.name === 'AbortError') {
      var finalContent2 = accumulatedReasoning
        ? '<think>' + accumulatedReasoning + '</think>' + accumulated
        : accumulated;
      stats2 = computeStats(startedAt, firstTokenAt, endedAt2, contentChunks, reasoningChunks, 'aborted');
      if (finalContent2) {
        convo.messages.push({ role: 'assistant', content: finalContent2, stats: stats2 });
        attachMessageStats(assistantWrap, stats2);
      } else if (assistantWrap && assistantWrap.parentNode) {
        assistantWrap.parentNode.removeChild(assistantWrap);
      }
      state.lastResponseStats = stats2;
    } else {
      // Network/server error — render as a distinct system error notice rather
      // than dressing it up as an assistant turn. Drop the placeholder bubble
      // entirely so the conversation history isn't polluted with error text.
      if (assistantWrap && assistantWrap.parentNode) {
        assistantWrap.parentNode.removeChild(assistantWrap);
      }
      assistantBody = null;
      appendErrorNotice(parseErrorMessage(e));
      // Don't overwrite lastResponseStats — keep the previous successful run's
      // stats visible in the panel rather than blanking everything to "—".
    }
  } finally {
    clearInterval(liveTimer);
    $liveStats.hidden = true;
    $liveStats.textContent = '';
    setStreamingUI(false);
    state.loadingModel = null;

    // Final render in case streamRender is off
    var finalDisplay = accumulatedReasoning
      ? '<think>' + accumulatedReasoning + '</think>' + accumulated
      : accumulated;
    if (finalDisplay && assistantBody && assistantBody.dataset.streaming === '1') {
      assistantBody.innerHTML = renderMarkdown(finalDisplay);
      addCopyButtons(assistantBody);
      delete assistantBody.dataset.streaming;
      scrollToBottom();
    }
    saveToStorage();
    renderHistory();
    renderStatsPanel();
  }
}

// Tell the server to stop loading / unload the currently-pending model and
// abort the in-flight chat request.  Used by the inline "Cancel load" button
// and by the global stop button when no first token has arrived yet.
function cancelCurrentLoad() {
  var modelId = state.loadingModel;
  if (state.abortController) state.abortController.abort();
  if (modelId) {
    fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: modelId, prompt: '', keep_alive: 0 })
    }).catch(function() {});
  }
}

function computeStats(startedAt, firstTokenAt, endedAt, contentChunks, reasoningChunks, finishReason) {
  var totalMs = endedAt - startedAt;
  var ttftMs = firstTokenAt != null ? firstTokenAt - startedAt : null;
  var genMs = firstTokenAt != null ? endedAt - firstTokenAt : 0;
  var tokens = contentChunks; // SSE chunk count is a close proxy for token count
  var tps = genMs > 0 ? tokens / (genMs / 1000) : null;
  return {
    ttftMs: ttftMs,
    totalMs: totalMs,
    tokens: tokens,
    tps: tps,
    finish: finishReason || 'stop',
    reasoningChunks: reasoningChunks
  };
}

function setStreamingUI(active) {
  state.streaming = active;
  if (!active) state.abortController = null;
  $sendBtn.disabled = active;
  if (active) {
    $stopBtn.dataset.active = '1';
    $sendBtn.style.display = 'none';
  } else {
    delete $stopBtn.dataset.active;
    $sendBtn.style.display = 'flex';
  }
}

// ── Input handling ──────────────────────────────────────────────────────────
function autoResizeTextarea() {
  $prompt.style.height = 'auto';
  $prompt.style.height = Math.min($prompt.scrollHeight, 200) + 'px';
}

$prompt.addEventListener('input', autoResizeTextarea);
$prompt.addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

$sendBtn.addEventListener('click', sendMessage);
$stopBtn.addEventListener('click', function() {
  // If we never received a first token, this stop is most likely the user
  // changing their mind about the model load — also tell the server to
  // unload, otherwise the worker keeps loading in the background.
  cancelCurrentLoad();
});

$newChatBtn.addEventListener('click', function() {
  if (state.streaming && state.abortController) state.abortController.abort();
  newConversation();
});

$sidebarToggle.addEventListener('click', function() {
  $sidebar.classList.toggle('collapsed');
});

$statsToggle.addEventListener('click', toggleStatsPanel);
$statsClose.addEventListener('click', function() { setStatsVisible(false); });
$clearHistoryBtn.addEventListener('click', clearAllHistory);

function toggleStatsPanel() {
  setStatsVisible($statsPanel.hidden);
}
function setStatsVisible(visible) {
  $statsPanel.hidden = !visible;
  $statsToggle.classList.toggle('active', visible);
  if (visible) renderStatsPanel();
}

// ── Settings ────────────────────────────────────────────────────────────────
var SETTING_FIELDS = [
  ['temperature', 's-temperature', 'temp-val', parseFloat, function(v){return v.toFixed(2);}],
  ['maxTokens', 's-max-tokens', 'maxtok-val', function(v){return parseInt(v,10);}, function(v){return String(v);}],
  ['topP', 's-top-p', 'topp-val', parseFloat, function(v){return v.toFixed(2);}],
  ['topK', 's-top-k', 'topk-val', function(v){return parseInt(v,10);}, function(v){return v === 0 ? '0 (disabled)' : String(v);}],
  ['minP', 's-min-p', 'minp-val', parseFloat, function(v){return v === 0 ? '0 (disabled)' : v.toFixed(2);}],
  ['repetitionPenalty', 's-repetition-penalty', 'reppen-val', parseFloat, function(v){return v.toFixed(2);}],
  ['frequencyPenalty', 's-frequency-penalty', 'freqpen-val', parseFloat, function(v){return v.toFixed(2);}],
  ['presencePenalty', 's-presence-penalty', 'prespen-val', parseFloat, function(v){return v.toFixed(2);}]
];

function loadSettingsIntoForm() {
  var s = state.settings;
  SETTING_FIELDS.forEach(function(f) {
    var el = $(f[1]);
    if (!el) return;
    el.value = s[f[0]];
    var lbl = $(f[2]);
    if (lbl) lbl.textContent = f[4](s[f[0]]);
  });
  $('s-system').value = s.systemPrompt || '';
  $('s-seed').value = s.seed == null ? '' : s.seed;
  $('s-stop').value = (s.stopSequences || []).join('\n');
  $('s-reasoning-effort').value = s.reasoningEffort || '';
  $('s-show-stats').checked = !!s.showStats;
  $('s-show-reasoning').checked = !!s.showReasoning;
  $('s-stream-render').checked = !!s.streamRender;
}

function saveSettingsFromForm() {
  var ns = {};
  SETTING_FIELDS.forEach(function(f) {
    var el = $(f[1]);
    if (el) ns[f[0]] = f[3](el.value);
  });
  ns.systemPrompt = $('s-system').value.trim();
  var seedRaw = $('s-seed').value.trim();
  ns.seed = seedRaw === '' ? null : parseInt(seedRaw, 10);
  if (isNaN(ns.seed)) ns.seed = null;
  ns.stopSequences = $('s-stop').value.split('\n').filter(function(s){return s.length > 0;});
  ns.reasoningEffort = $('s-reasoning-effort').value;
  ns.showStats = $('s-show-stats').checked;
  ns.showReasoning = $('s-show-reasoning').checked;
  ns.streamRender = $('s-stream-render').checked;
  state.settings = Object.assign({}, state.settings, ns);
  saveToStorage();
  renderSamplingSummary();
  setStatsVisible(state.settings.showStats);
  renderMessages();
}

function openSettings() {
  loadSettingsIntoForm();
  $settingsOverlay.classList.add('open');
}
function closeSettings() { $settingsOverlay.classList.remove('open'); }

SETTING_FIELDS.forEach(function(f) {
  var el = $(f[1]);
  var lbl = $(f[2]);
  if (el && lbl) {
    el.addEventListener('input', function() {
      lbl.textContent = f[4](f[3](el.value));
    });
  }
});

$settingsBtn.addEventListener('click', openSettings);
$settingsCancel.addEventListener('click', closeSettings);
$settingsOverlay.addEventListener('click', function(e) {
  if (e.target === $settingsOverlay) closeSettings();
});
$settingsSave.addEventListener('click', function() {
  saveSettingsFromForm();
  closeSettings();
});
$settingsReset.addEventListener('click', function() {
  if (!confirm('Reset all settings to defaults?')) return;
  state.settings = Object.assign({}, DEFAULT_SETTINGS);
  loadSettingsIntoForm();
});

$('seed-random').addEventListener('click', function() {
  var v = Math.floor(Math.random() * 0x7FFFFFFF);
  $('s-seed').value = v;
});
$('seed-clear').addEventListener('click', function() {
  $('s-seed').value = '';
});

document.querySelectorAll('.tab').forEach(function(tab) {
  tab.addEventListener('click', function() {
    var name = tab.dataset.tab;
    document.querySelectorAll('.tab').forEach(function(t) { t.classList.toggle('active', t === tab); });
    document.querySelectorAll('.tab-pane').forEach(function(p) { p.hidden = p.dataset.pane !== name; });
  });
});

document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape' && $settingsOverlay.classList.contains('open')) closeSettings();
});

// ── Init ────────────────────────────────────────────────────────────────────
loadFromStorage();
renderHistory();
renderSamplingSummary();
updateStatusBadge();
setStatsVisible(!!state.settings.showStats);

if (state.conversations.length > 0) {
  if (!state.activeId) state.activeId = state.conversations[0].id;
  renderMessages();
} else {
  $messages.innerHTML = '';
  $messages.appendChild($emptyState);
  $emptyState.style.display = 'flex';
}

renderStatsPanel();

fetchModels().then(function() {
  setInterval(fetchModels, 10000);
});

pollHealth();
setInterval(pollHealth, 5000);

$prompt.focus();
})();
