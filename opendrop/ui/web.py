"""Embedded Web UI for OpenDrop.

Serves a self-contained single-page chat interface + model management panel
from the FastAPI app with zero build step.  The full HTML/JS is embedded as
a Python string constant.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# ---------------------------------------------------------------------------
# Embedded HTML/JS/CSS Web UI
# ---------------------------------------------------------------------------

_WEB_UI = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OpenDrop</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #0f1117; --surface: #1a1d26; --border: #2d3145;
    --accent: #6c8ef5; --green: #4ade80; --red: #f87171;
    --text: #e2e8f0; --dim: #64748b;
    --radius: 8px; --font: 'Inter', system-ui, sans-serif;
  }
  body { background: var(--bg); color: var(--text); font-family: var(--font);
         display: flex; height: 100vh; overflow: hidden; }
  /* Sidebar */
  #sidebar { width: 280px; background: var(--surface); border-right: 1px solid var(--border);
             display: flex; flex-direction: column; padding: 16px; gap: 12px; overflow-y: auto; }
  #sidebar h1 { font-size: 1.2rem; font-weight: 700; color: var(--accent); }
  #sidebar h2 { font-size: 0.75rem; text-transform: uppercase; letter-spacing: .1em;
                color: var(--dim); margin-top: 8px; }
  .model-item { padding: 10px 12px; border-radius: var(--radius); cursor: pointer;
                border: 1px solid var(--border); transition: border-color .2s; }
  .model-item:hover { border-color: var(--accent); }
  .model-item.active { border-color: var(--accent); background: rgba(108,142,245,.1); }
  .model-item .name { font-weight: 600; font-size: .9rem; }
  .model-item .meta { font-size: .75rem; color: var(--dim); margin-top: 2px; }
  .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                margin-right: 4px; }
  .status-dot.running { background: var(--green); }
  .status-dot.idle { background: var(--dim); }
  #pull-form { display: flex; flex-direction: column; gap: 6px; }
  #pull-form input { background: var(--bg); border: 1px solid var(--border); color: var(--text);
                     padding: 8px 10px; border-radius: var(--radius); font-size: .85rem; }
  #pull-form button, .btn { background: var(--accent); color: #fff; border: none;
                            padding: 8px 14px; border-radius: var(--radius); cursor: pointer;
                            font-size: .85rem; font-weight: 600; transition: opacity .2s; }
  .btn:hover, #pull-form button:hover { opacity: .85; }
  .btn.danger { background: var(--red); }
  /* Main chat */
  #main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  #chat-header { padding: 12px 20px; border-bottom: 1px solid var(--border);
                 display: flex; align-items: center; gap: 10px; }
  #chat-header .model-label { font-weight: 600; }
  #chat-header .hw-badge { margin-left: auto; font-size: .75rem; color: var(--dim);
                           background: var(--surface); padding: 4px 10px;
                           border-radius: 99px; border: 1px solid var(--border); }
  #messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
  .msg { max-width: 75%; padding: 12px 16px; border-radius: var(--radius); line-height: 1.5; }
  .msg.user { align-self: flex-end; background: var(--accent); color: #fff; }
  .msg.assistant { align-self: flex-start; background: var(--surface);
                   border: 1px solid var(--border); white-space: pre-wrap; }
  .msg.system { align-self: center; background: transparent; color: var(--dim);
                font-size: .8rem; font-style: italic; }
  #input-area { padding: 16px 20px; border-top: 1px solid var(--border);
                display: flex; gap: 10px; align-items: flex-end; }
  #msg-input { flex: 1; background: var(--surface); border: 1px solid var(--border);
               color: var(--text); padding: 10px 14px; border-radius: var(--radius);
               font-size: .9rem; resize: none; font-family: var(--font);
               min-height: 44px; max-height: 200px; }
  #msg-input:focus { outline: none; border-color: var(--accent); }
  #send-btn { padding: 10px 18px; }
  /* Settings panel */
  #settings { padding: 8px; display: flex; flex-direction: column; gap: 6px;
              border-top: 1px solid var(--border); margin-top: auto; }
  #settings label { font-size: .75rem; color: var(--dim); }
  #settings input[type=range] { width: 100%; }
  #settings .setting-row { display: flex; justify-content: space-between; align-items: center; }
  /* Spinner */
  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid var(--dim);
             border-top-color: var(--accent); border-radius: 50%; animation: spin .6s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  /* Notifications */
  #toast { position: fixed; bottom: 20px; right: 20px; background: var(--surface);
           border: 1px solid var(--border); padding: 10px 16px; border-radius: var(--radius);
           font-size: .85rem; opacity: 0; transition: opacity .3s; pointer-events: none; }
  #toast.show { opacity: 1; }
</style>
</head>
<body>
<!-- Sidebar -->
<div id="sidebar">
  <h1>⬇ OpenDrop</h1>
  <h2>Pull Model</h2>
  <div id="pull-form">
    <input id="pull-url" type="text" placeholder="HuggingFace URL or org/model" />
    <button onclick="pullModel()">Pull</button>
  </div>
  <h2>Models</h2>
  <div id="model-list"></div>
  <div id="settings">
    <div class="setting-row">
      <label>Temperature</label>
      <span id="temp-val">0.7</span>
    </div>
    <input type="range" id="temperature" min="0" max="2" step="0.05" value="0.7"
           oninput="document.getElementById('temp-val').textContent=this.value">
    <div class="setting-row">
      <label>Max tokens</label>
      <span id="maxtok-val">1024</span>
    </div>
    <input type="range" id="max-tokens" min="64" max="8192" step="64" value="1024"
           oninput="document.getElementById('maxtok-val').textContent=this.value">
  </div>
</div>

<!-- Main chat -->
<div id="main">
  <div id="chat-header">
    <span class="model-label" id="active-model-label">Select a model →</span>
    <span class="hw-badge" id="hw-badge">Loading hardware…</span>
    <button class="btn danger" style="margin-left:8px;padding:4px 10px;font-size:.75rem"
            onclick="clearChat()">Clear</button>
  </div>
  <div id="messages">
    <div class="msg system">Welcome to OpenDrop. Select or pull a model to begin.</div>
  </div>
  <div id="input-area">
    <textarea id="msg-input" placeholder="Type a message… (Enter to send, Shift+Enter for newline)"
              rows="1" onkeydown="handleKey(event)"></textarea>
    <button class="btn" id="send-btn" onclick="sendMessage()">Send</button>
  </div>
</div>

<div id="toast"></div>

<script>
const API = '';  // same origin
let activeModel = null;
let messages = [];
let streaming = false;

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
async function boot() {
  await loadModels();
  await loadHardware();
}

async function loadModels() {
  try {
    const r = await fetch(`${API}/v1/models`);
    const data = await r.json();
    renderModels(data.data || []);
  } catch(e) { showToast('Cannot reach OpenDrop server', 'error'); }
}

function renderModels(models) {
  const el = document.getElementById('model-list');
  if (!models.length) {
    el.innerHTML = '<div style="color:var(--dim);font-size:.8rem">No models yet. Pull one above.</div>';
    return;
  }
  el.innerHTML = models.map(m => `
    <div class="model-item ${m.id===activeModel?'active':''}" onclick="selectModel('${m.id}')">
      <div class="name">
        <span class="status-dot ${m.status==='running'?'running':'idle'}"></span>${m.id}
      </div>
      <div class="meta">${m.status}</div>
    </div>
  `).join('');
}

async function loadHardware() {
  // OpenDrop doesn't expose hardware via API yet; show placeholder
  document.getElementById('hw-badge').textContent = 'OpenDrop local';
}

// ---------------------------------------------------------------------------
// Model management
// ---------------------------------------------------------------------------
function selectModel(id) {
  activeModel = id;
  messages = [];
  document.getElementById('active-model-label').textContent = id;
  document.getElementById('messages').innerHTML = '';
  addSystemMsg(`Model: ${id}`);
  loadModels();
}

async function pullModel() {
  const url = document.getElementById('pull-url').value.trim();
  if (!url) return;
  showToast('Pull started — check server logs for progress');
  // Pull is a CLI operation; the UI can show a message directing user to CLI
  addSystemMsg(`Run in terminal: opendrop pull "${url}"`);
  document.getElementById('pull-url').value = '';
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------
function addMsg(role, content) {
  const el = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.textContent = content;
  div.dataset.role = role;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
  return div;
}

function addSystemMsg(text) {
  const el = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'msg system';
  div.textContent = text;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
}

async function sendMessage() {
  if (streaming) return;
  if (!activeModel) { showToast('Select a model first'); return; }
  const input = document.getElementById('msg-input');
  const text = input.value.trim();
  if (!text) return;

  input.value = '';
  messages.push({ role: 'user', content: text });
  addMsg('user', text);

  const assistantDiv = addMsg('assistant', '');
  const spinner = document.createElement('span');
  spinner.className = 'spinner';
  assistantDiv.appendChild(spinner);

  const payload = {
    model: activeModel,
    messages: [...messages],
    temperature: parseFloat(document.getElementById('temperature').value),
    max_tokens: parseInt(document.getElementById('max-tokens').value),
    stream: true,
  };

  streaming = true;
  let fullText = '';
  try {
    const resp = await fetch(`${API}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    spinner.remove();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value);
      for (const line of chunk.split('\n')) {
        if (!line.startsWith('data:')) continue;
        const raw = line.slice(5).trim();
        if (raw === '[DONE]') break;
        try {
          const obj = JSON.parse(raw);
          const delta = obj.choices?.[0]?.delta?.content || '';
          fullText += delta;
          assistantDiv.textContent = fullText;
          document.getElementById('messages').scrollTop = 999999;
        } catch(_) {}
      }
    }
  } catch(e) {
    spinner.remove();
    assistantDiv.textContent = `Error: ${e.message}`;
  } finally {
    streaming = false;
    if (fullText) messages.push({ role: 'assistant', content: fullText });
  }
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
  // Auto-resize textarea
  const ta = document.getElementById('msg-input');
  ta.style.height = 'auto';
  ta.style.height = Math.min(ta.scrollHeight, 200) + 'px';
}

function clearChat() {
  messages = [];
  document.getElementById('messages').innerHTML = '';
  if (activeModel) addSystemMsg(`Model: ${activeModel}`);
}

// ---------------------------------------------------------------------------
// Toast
// ---------------------------------------------------------------------------
function showToast(msg, _type) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 3000);
}

// ---------------------------------------------------------------------------
// Auto-refresh model list every 5 s
// ---------------------------------------------------------------------------
setInterval(loadModels, 5000);
boot();
</script>
</body>
</html>
"""


def mount_web_ui(app: FastAPI) -> None:
    """Mount the embedded Web UI on the root route of the given FastAPI app."""

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def web_ui() -> str:
        return _WEB_UI

    @app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
    async def web_ui_alias() -> str:
        return _WEB_UI
