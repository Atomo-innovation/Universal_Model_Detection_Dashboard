/**
 * ASNN Detection Dashboard Server
 * Runs on aarch64 device, accessible from any browser on the network
 *
 * Features:
 *  - Serves dashboard UI over HTTP
 *  - Scans /models directory for available models (reads data.yaml)
 *  - Accepts file uploads (video/image)
 *  - Spawns Python inference processes
 *    · person model  → person.py --json-stream  (YOLO26s, RTSP-only)
 *    · all others    → detect.py                (generic YOLOv3-style)
 *  - Streams output frames + logs via WebSocket
 */

const express    = require('express');
const http       = require('http');
const WebSocket  = require('ws');
const multer     = require('multer');
const path       = require('path');
const fs         = require('fs');
const { spawn, execSync }  = require('child_process');
const { v4: uuidv4 } = require('uuid');
const chokidar   = require('chokidar');

// ── Try to parse YAML (data.yaml for model metadata) ────────────
let YAML;
try { YAML = require('yaml'); } catch(e) { YAML = null; }

const app    = express();
const server = http.createServer(app);
const wss    = new WebSocket.Server({ server });

const PORT          = 8050;
const MODELS_DIR    = process.env.MODELS_DIR  || path.join(__dirname, 'models');
const UPLOADS_DIR   = process.env.UPLOADS_DIR || path.join(__dirname, 'uploads');
const PUBLIC_DIR    = path.join(__dirname, 'public');
const DETECT_SCRIPT = process.env.DETECT_SCRIPT || path.join(__dirname, 'detect.py');
const PERSON_SCRIPT = process.env.PERSON_SCRIPT || path.join(__dirname, 'person.py');

// Ensure dirs exist
[MODELS_DIR, UPLOADS_DIR, PUBLIC_DIR].forEach(d => fs.mkdirSync(d, { recursive: true }));

// ── Active inference sessions ────────────────────────────────────
const sessions = new Map();  // sessionId -> { proc, ws, type }

// ── Multer upload config ─────────────────────────────────────────
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOADS_DIR),
  filename:    (req, file, cb) => {
    const ext = path.extname(file.originalname);
    cb(null, `${Date.now()}-${uuidv4().slice(0,8)}${ext}`);
  }
});
const upload = multer({ storage, limits: { fileSize: 2 * 1024 * 1024 * 1024 } }); // 2GB

// ── Express middleware ───────────────────────────────────────────
app.use(express.json());
app.use(express.static(PUBLIC_DIR));
app.use('/public',  express.static(PUBLIC_DIR));
app.use('/uploads', express.static(UPLOADS_DIR));

// ── API: List available models ───────────────────────────────────
app.get('/api/models', (req, res) => {
  res.json({ models: scanModels() });
});

// ── API: Upload file ─────────────────────────────────────────────
app.post('/api/upload', upload.single('file'), (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No file uploaded' });
  res.json({
    filename:     req.file.filename,
    originalname: req.file.originalname,
    path:         req.file.path,
    size:         req.file.size
  });
});

// ── API: Delete uploaded file ────────────────────────────────────
app.delete('/api/upload/:filename', (req, res) => {
  const fp = path.join(UPLOADS_DIR, path.basename(req.params.filename));
  try { fs.unlinkSync(fp); res.json({ ok: true }); }
  catch(e) { res.status(404).json({ error: 'File not found' }); }
});

// ── API: System info ─────────────────────────────────────────────
app.get('/api/system', (req, res) => {
  let arch = 'unknown', hostname = 'unknown', ip = [];
  try { arch     = execSync('uname -m').toString().trim(); } catch(e){}
  try { hostname = execSync('hostname').toString().trim();  } catch(e){}
  try {
    const raw = execSync("hostname -I 2>/dev/null || ip addr show | grep 'inet ' | awk '{print $2}' | cut -d/ -f1").toString().trim();
    ip = raw.split(/\s+/).filter(Boolean);
  } catch(e){}
  res.json({ arch, hostname, ip, port: PORT, uptime: process.uptime() });
});

// ── API: Start inference session ─────────────────────────────────
app.post('/api/inference/start', (req, res) => {
  const { modelName, inputType, inputValue, objThresh, nmsThresh,
          platform, logLevel, sessionId: existingId } = req.body;

  const models = scanModels();
  const model  = models.find(m => m.name === modelName);
  if (!model) return res.status(404).json({ error: `Model '${modelName}' not found` });

  const sid = existingId || uuidv4();
  if (sessions.has(sid)) stopSession(sid);

  const isPerson = isPersonModel(model);
  const args     = isPerson
    ? buildPersonArgs({ model, inputType, inputValue, objThresh, nmsThresh, logLevel })
    : buildDetectArgs({ model, inputType, inputValue, objThresh, nmsThresh, platform, logLevel });

  sessions.set(sid, {
    model, args, inputType, inputValue,
    isPerson,
    status: 'pending', proc: null, ws: null
  });

  const script = isPerson ? 'person.py' : 'detect.py';
  res.json({ sessionId: sid, command: `python3 ${script} ${args.join(' ')}` });
});

// ── API: Stop inference session ──────────────────────────────────
app.post('/api/inference/stop/:sid', (req, res) => {
  stopSession(req.params.sid);
  res.json({ ok: true });
});

// ── API: List active sessions ────────────────────────────────────
app.get('/api/inference/sessions', (req, res) => {
  const list = [];
  sessions.forEach((v, k) => list.push({
    id: k, status: v.status, model: v.model?.name,
    inputType: v.inputType, isPerson: v.isPerson
  }));
  res.json({ sessions: list });
});

// ── Serve main dashboard ─────────────────────────────────────────
app.get('/', (req, res) => res.sendFile(path.join(PUBLIC_DIR, 'index.html')));

app.get('*', (req, res) => {
  const ext = path.extname(req.path);
  if (ext && ext !== '.html') return res.status(404).send('Not found');
  res.sendFile(path.join(PUBLIC_DIR, 'index.html'));
});

// ── WebSocket: real-time inference stream ────────────────────────
wss.on('connection', (ws, req) => {
  console.log(`[WS] Client connected from ${req.socket.remoteAddress}`);

  ws.on('message', (raw) => {
    let msg;
    try { msg = JSON.parse(raw); } catch(e) { return; }
    switch(msg.type) {
      case 'attach': handleAttach(ws, msg); break;
      case 'start':  handleStart(ws, msg);  break;
      case 'stop':   handleStop(ws, msg);   break;
      case 'ping':   ws.send(JSON.stringify({ type: 'pong' })); break;
    }
  });

  ws.on('close', () => console.log('[WS] Client disconnected'));
  ws.on('error', (e) => console.error('[WS] Error:', e.message));
});

// ── WS handlers ─────────────────────────────────────────────────
function handleAttach(ws, msg) {
  const session = sessions.get(msg.sessionId);
  if (!session) { ws.send(JSON.stringify({ type: 'error', message: 'Session not found' })); return; }
  session.ws = ws;
  wsend(ws, { type: 'attached', sessionId: msg.sessionId, status: session.status });
}

function handleStart(ws, msg) {
  const session = sessions.get(msg.sessionId);
  if (!session) { wsend(ws, { type: 'error', message: 'Call /api/inference/start first' }); return; }
  session.ws     = ws;
  session.status = 'running';
  sessions.set(msg.sessionId, session);
  wsend(ws, { type: 'status', status: 'starting', message: 'Spawning inference process...' });
  spawnInference(msg.sessionId, session);
}

function handleStop(ws, msg) {
  stopSession(msg.sessionId);
  wsend(ws, { type: 'status', status: 'stopped' });
}

// ── Spawn inference process ──────────────────────────────────────
function spawnInference(sid, session) {
  const { args, isPerson, ws } = session;

  // Pick the right script
  const scriptPath = isPerson ? PERSON_SCRIPT : DETECT_SCRIPT;

  if (!fs.existsSync(scriptPath)) {
    wsend(ws, {
      type: 'log', level: 'warn',
      message: `${path.basename(scriptPath)} not found at ${scriptPath} — using simulation mode`
    });
    startSimulation(sid, session);
    return;
  }

  let proc;
  try {
    proc = spawn('python3', [scriptPath, ...args], {
      cwd: path.dirname(scriptPath),
      env: { ...process.env, PYTHONUNBUFFERED: '1' }
    });
  } catch(e) {
    wsend(ws, { type: 'error', message: `Failed to spawn: ${e.message}` });
    return;
  }

  session.proc = proc;
  sessions.set(sid, session);
  wsend(ws, { type: 'status', status: 'running', pid: proc.pid });

  // Stream stdout — JSON lines from both detect.py and person.py --json-stream
  let buf = '';
  proc.stdout.on('data', (chunk) => {
    buf += chunk.toString();
    const lines = buf.split('\n');
    buf = lines.pop();
    lines.forEach(line => {
      if (!line.trim()) return;
      try {
        const data = JSON.parse(line);
        if (data.type === 'log') {
          // Explicit log line: {type:'log', level, message}
          wsend(ws, data);
        } else if (data.frame !== undefined) {
          // Inference frame: {frame, fps, inference_ms, detections, jpeg}
          wsend(ws, { type: 'inference', ...data });
        } else {
          // Other structured output
          wsend(ws, { type: data.type || 'log', ...data });
        }
      } catch(e) {
        // Plain text line → log entry
        wsend(ws, { type: 'log', level: 'info', message: line });
      }
    });
  });

  proc.stderr.on('data', (chunk) => {
    wsend(ws, { type: 'log', level: 'stderr', message: chunk.toString() });
  });

  proc.on('close', (code) => {
    session.status = 'stopped';
    session.proc   = null;
    wsend(ws, { type: 'status', status: 'stopped', exitCode: code });
    console.log(`[${sid}] Process exited with code ${code}`);
  });

  proc.on('error', (e) => {
    wsend(ws, { type: 'error', message: e.message });
    session.status = 'error';
  });
}

// ── Simulation mode (no script available) ───────────────────────
function startSimulation(sid, session) {
  const { model, ws } = session;
  let frame = 0;
  const classes = model.classes || ['Object'];

  wsend(ws, { type: 'status', status: 'running', simulated: true });

  const interval = setInterval(() => {
    const s = sessions.get(sid);
    if (!s || s.status !== 'running') { clearInterval(interval); return; }
    frame++;
    const dets = [];
    const count = Math.random() > 0.4 ? Math.floor(Math.random() * 3) + 1 : 0;
    for (let i = 0; i < count; i++) {
      const cls = Math.floor(Math.random() * classes.length);
      const x1 = Math.random() * 0.6, y1 = Math.random() * 0.6;
      dets.push({
        class_id:   cls,
        class_name: classes[cls],
        score:      parseFloat((0.4 + Math.random() * 0.55).toFixed(3)),
        box: [
          parseFloat(x1.toFixed(4)), parseFloat(y1.toFixed(4)),
          parseFloat(Math.min(x1 + 0.1 + Math.random() * 0.25, 1).toFixed(4)),
          parseFloat(Math.min(y1 + 0.1 + Math.random() * 0.25, 1).toFixed(4))
        ]
      });
    }
    wsend(ws, {
      type: 'inference', frame,
      fps: parseFloat((15 + Math.random() * 10).toFixed(1)),
      inference_ms: parseFloat((8 + Math.random() * 12).toFixed(1)),
      detections: dets, simulated: true
    });
    if (frame % 30 === 0) {
      wsend(ws, { type: 'log', level: 'info', message: `[SIM] Frame ${frame} | ${dets.length} detections` });
    }
  }, 66);

  session.simInterval = interval;
  sessions.set(sid, session);
}

// ── Stop a session ───────────────────────────────────────────────
function stopSession(sid) {
  const session = sessions.get(sid);
  if (!session) return;
  if (session.proc) {
    try { session.proc.kill('SIGTERM'); } catch(e) {}
    session.proc = null;
  }
  if (session.simInterval) { clearInterval(session.simInterval); session.simInterval = null; }
  session.status = 'stopped';
  sessions.set(sid, session);
  console.log(`[${sid}] Session stopped`);
}

// ── Detect whether this model should use person.py ───────────────
function isPersonModel(model) {
  // Match if the model folder name is exactly "person" (case-insensitive)
  // or the model's single class is "person"
  if (model.name.toLowerCase() === 'person') return true;
  if (model.classes && model.classes.length === 1 &&
      model.classes[0].toLowerCase() === 'person') return true;
  return false;
}

// ── Scan /models directory ───────────────────────────────────────
function scanModels() {
  const models = [];
  if (!fs.existsSync(MODELS_DIR)) return models;

  const dirs = fs.readdirSync(MODELS_DIR, { withFileTypes: true })
    .filter(d => d.isDirectory())
    .map(d => d.name);

  for (const name of dirs) {
    const dir   = path.join(MODELS_DIR, name);
    const files = fs.readdirSync(dir);

    const nbFile   = files.find(f => f.endsWith('.nb'));
    const soFile   = files.find(f => f.endsWith('.so'));
    const yamlFile = files.find(f => f === 'data.yaml' || f === 'dataset.yaml' || f.endsWith('.yaml'));

    if (!nbFile || !soFile) continue;

    const model = {
      name,
      dir,
      nb:       nbFile,
      lib:      soFile,
      nb_path:  path.join(dir, nbFile),
      lib_path: path.join(dir, soFile),
      classes:  [name.charAt(0).toUpperCase() + name.slice(1)],
      num_cls:  1,
      listsize: 65,
      yaml:     yamlFile || null
    };

    if (yamlFile && YAML) {
      try {
        const raw    = fs.readFileSync(path.join(dir, yamlFile), 'utf8');
        const parsed = YAML.parse(raw);
        if (parsed.names) {
          const names = Array.isArray(parsed.names)
            ? parsed.names : Object.values(parsed.names);
          model.classes  = names;
          model.num_cls  = names.length;
          model.listsize = model.num_cls + 64;
        }
        if (parsed.nc) model.num_cls = parsed.nc;
      } catch(e) {
        console.warn(`[models] Failed to parse ${yamlFile} in ${name}: ${e.message}`);
      }
    }

    models.push(model);
  }

  return models;
}

// ── Build CLI args for detect.py ─────────────────────────────────
function buildDetectArgs({ model, inputType, inputValue, objThresh, nmsThresh, platform, logLevel }) {
  const args = [
    '--model',   model.nb_path,
    '--library', model.lib_path,
    '--level',   String(logLevel || 0),
  ];
  if (inputType === 'rtsp') {
    args.push('--type', 'rtsp', '--device', inputValue);
  } else if (inputType === 'webcam') {
    const [capType, devNum] = (inputValue || 'usb:0').split(':');
    args.push('--type', capType || 'usb', '--device', devNum || '0');
  } else if (inputType === 'video') {
    args.push('--type', 'video', '--device', inputValue);
  } else if (inputType === 'image') {
    args.push('--type', 'image', '--device', inputValue);
  }
  if (objThresh) args.push('--obj-thresh', String(objThresh));
  if (nmsThresh) args.push('--nms-thresh', String(nmsThresh));
  if (platform)  args.push('--platform',   platform);
  return args;
}

// ── Build CLI args for person.py ─────────────────────────────────
// person.py uses the same --type / --device interface as detect.py.
function buildPersonArgs({ model, inputType, inputValue, objThresh, nmsThresh, logLevel }) {
  const args = [
    '--model',        model.nb_path,
    '--library',      model.lib_path,
    '--level',        String(logLevel || 0),
    '--json-stream',                // emit dashboard-compatible JSON lines on stdout
    '--jpeg-quality', '75',
  ];

  if (inputType === 'rtsp') {
    args.push('--type', 'rtsp', '--device', inputValue);
  } else if (inputType === 'webcam') {
    const [capType, devNum] = (inputValue || 'usb:0').split(':');
    args.push('--type', capType || 'usb', '--device', devNum || '0');
  } else if (inputType === 'video') {
    args.push('--type', 'video', '--device', inputValue);
  } else if (inputType === 'image') {
    args.push('--type', 'image', '--device', inputValue);
  }

  if (objThresh) args.push('--conf', String(objThresh));
  if (nmsThresh) args.push('--nms',  String(nmsThresh));
  return args;
}

// ── Watch models dir for changes ─────────────────────────────────
chokidar.watch(MODELS_DIR, { depth: 1, ignoreInitial: true })
  .on('addDir', () => broadcastModels())
  .on('add',    (p) => {
    if (p.endsWith('.nb') || p.endsWith('.so') || p.endsWith('.yaml')) broadcastModels();
  });

function broadcastModels() {
  const msg = JSON.stringify({ type: 'models_updated', models: scanModels() });
  wss.clients.forEach(c => { if (c.readyState === WebSocket.OPEN) c.send(msg); });
}

// ── Helper: safe WS send ─────────────────────────────────────────
function wsend(ws, data) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    try { ws.send(JSON.stringify(data)); } catch(e) {}
  }
}

// ── Start server ─────────────────────────────────────────────────
server.listen(PORT, '0.0.0.0', () => {
  console.log('\n╔══════════════════════════════════════════════════╗');
  console.log('║         ASNN DETECTION DASHBOARD SERVER          ║');
  console.log('╠══════════════════════════════════════════════════╣');
  console.log(`║  HTTP   : http://0.0.0.0:${PORT}                   ║`);
  console.log(`║  Models : ${MODELS_DIR}`);
  console.log(`║  Uploads: ${UPLOADS_DIR}`);
  console.log(`║  Scripts: detect.py + person.py`);
  console.log('╠══════════════════════════════════════════════════╣');
  try {
    const ips = execSync("hostname -I 2>/dev/null || ip addr show | grep 'inet ' | awk '{print $2}' | cut -d/ -f1")
      .toString().trim().split(/\s+/);
    ips.forEach(ip => { if (ip) console.log(`║  Access : http://${ip}:${PORT}`); });
  } catch(e) {}
  console.log('╚══════════════════════════════════════════════════╝\n');
  console.log(`Models found: ${scanModels().length}`);
  console.log('Press Ctrl+C to stop\n');
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\nShutting down...');
  sessions.forEach((_, sid) => stopSession(sid));
  server.close(() => process.exit(0));
});
