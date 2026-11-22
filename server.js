#!/usr/bin/env node
'use strict';

/*
 * server.js — NVIDIA NIM / OpenAI-compatible proxy
 *
 * Usage:
 *   NVIDIA_API_KEY=nvapi-xxx node server.js
 *   node server.js                       # key must then come from the request
 *
 * Clients call the proxy exactly like the OpenAI API:
 *   POST http://localhost:3000/v1/chat/completions
 *   {
 *     "model": "meta/llama-3.1-70b-instruct",   // required (or DEFAULT_MODEL env)
 *     "messages": [...],
 *     "stream": true,
 *     "tools": [...]
 *     // "api_key": "nvapi-..."  <-- optional: key can be passed per request
 *   }
 *
 * API key resolution (first match wins, unless IGNORE_CLIENT_AUTH=1):
 *   1. body.api_key / body.apiKey
 *   2. ?api_key= / ?apiKey= query parameter
 *   3. x-api-key / x-nvidia-api-key request header
 *   4. authorization: Bearer <key>
 *   5. env: NVIDIA_API_KEY | NIM_API_KEY | NVIDIA_NIM_API_KEY | NIM_TOKEN
 *
 * Model resolution (first match wins):
 *   body.model  >  ?model=  >  x-model header  >  DEFAULT_MODEL / NIM_MODEL env
 *
 * Features:
 *   - Streams SSE with backpressure, idle timeout, client-disconnect abort
 *   - Reasoning/thinking handling:
 *       * reasoning_content / reasoning / thinking fields from NIM are
 *         normalized to `reasoning_content` on both stream and non-stream
 *       * <thinking>...</thinking> and <think></thinking> blocks embedded in
 *         content are stripped and re-emitted as reasoning_content
 *   - Tool-call handling + fallbacks:
 *       * normalizes malformed tool_calls (missing id/index/type,
 *         object arguments, legacy function_call, tool_name/args shapes)
 *       * repairs truncated/unbalanced JSON arguments
 *       * if the model dumps the call as text (fenced ```tool_call/json```,
 *         <function_call>, plain JSON) the proxy extracts it into real
 *         tool_calls and strips it from content
 *       * streaming: content is held back only when an explicit call
 *         marker is seen, so nothing is lost or duplicated
 *   - Automatic retries (429/5xx/network) with backoff + Retry-After
 *   - Proxy-only fields (api_key ...) are stripped before forwarding
 *   - CORS, /health, /v1/models, generic /v1/* passthrough (embeddings,
 *     completions, rerank, ...)
 *
 * Env:
 *   PORT, HOST, NIM_BASE_URL, DEFAULT_MODEL, MAX_RETRIES,
 *   REQUEST_TIMEOUT_MS, CONNECT_TIMEOUT_MS, STREAM_IDLE_TIMEOUT_MS,
 *   STREAM_MAX_MS, MAX_BODY_BYTES, MAX_UPSTREAM_BODY_BYTES,
 *   CORS_ORIGIN, LOG_LEVEL, LOG_FORMAT, IGNORE_CLIENT_AUTH,
 *   PROXY_TOKEN (door auth for public deployments), RATE_LIMIT_RPM,
 *   TRUST_PROXY
 *
 * Production (public deployment):
 *   Set PROXY_TOKEN so only your clients can use the proxy; set
 *   IGNORE_CLIENT_AUTH=1 so the NIM key only ever lives on the server;
 *   optionally RATE_LIMIT_RPM=60 and LOG_FORMAT=json.
 *
 * Requires Node.js 18+ (uses global fetch). No dependencies.
 */

const http = require('node:http');
const crypto = require('node:crypto');
const { Readable } = require('node:stream');

/* ------------------------------------------------------------------ */
/* Configuration                                                       */
/* ------------------------------------------------------------------ */

function envInt(name, def) {
  const v = parseInt(process.env[name], 10);
  return Number.isFinite(v) && v >= 0 ? v : def;
}

const CONFIG = {
  host: process.env.HOST || '0.0.0.0',
  port: envInt('PORT', 3000),
  baseUrl: (process.env.NIM_BASE_URL || process.env.NIM_ENDPOINT ||
    'https://integrate.api.nvidia.com/v1').trim().replace(/\/+$/, ''),
  defaultModel: (process.env.DEFAULT_MODEL || process.env.NIM_MODEL || '').trim(),
  requestTimeoutMs: envInt('REQUEST_TIMEOUT_MS', 120_000),
  connectTimeoutMs: envInt('CONNECT_TIMEOUT_MS', 30_000),
  streamIdleTimeoutMs: envInt('STREAM_IDLE_TIMEOUT_MS', 90_000),
  streamMaxMs: envInt('STREAM_MAX_MS', 0), // 0 = unlimited
  maxRetries: envInt('MAX_RETRIES', 2),
  maxBodyBytes: envInt('MAX_BODY_BYTES', 10 * 1024 * 1024),
  maxUpstreamBodyBytes: envInt('MAX_UPSTREAM_BODY_BYTES', 32 * 1024 * 1024),
  corsOrigin: process.env.CORS_ORIGIN || '*',
  ignoreClientAuth: /^(1|true|yes)$/i.test(process.env.IGNORE_CLIENT_AUTH || ''),
  logLevel: (process.env.LOG_LEVEL || 'info').toLowerCase(),
  logFormat: (process.env.LOG_FORMAT || 'text').toLowerCase(),
  proxyToken: process.env.PROXY_TOKEN || '',
  rateLimitRpm: envInt('RATE_LIMIT_RPM', 0), // 0 = disabled
  trustProxy: !/^(0|false|no|off)$/i.test(process.env.TRUST_PROXY || '1'),
  envApiKey: [process.env.NVIDIA_API_KEY, process.env.NIM_API_KEY,
    process.env.NVIDIA_NIM_API_KEY, process.env.NIM_TOKEN].find(Boolean) || '',
};

const LOG_LEVELS = { debug: 10, info: 20, warn: 30, error: 40, silent: 99 };

function log(level, msg, extra) {
  const threshold = LOG_LEVELS[CONFIG.logLevel] ?? LOG_LEVELS.info;
  if ((LOG_LEVELS[level] ?? 20) < threshold) return;
  const out = level === 'error' ? console.error
    : level === 'warn' ? console.warn : console.log;
  if (CONFIG.logFormat === 'json') {
    const rec = { ts: new Date().toISOString(), level, msg };
    if (extra) for (const [k, v] of Object.entries(extra)) {
      if (v !== undefined) rec[k] = v;
    }
    out(JSON.stringify(rec));
    return;
  }
  const parts = [];
  if (extra) for (const [k, v] of Object.entries(extra)) {
    if (v === undefined) continue;
    parts.push(`${k}=${typeof v === 'string' ? v : JSON.stringify(v)}`);
  }
  const line = `${new Date().toISOString()} ${level.toUpperCase().padEnd(5)} ${msg}` +
    (parts.length ? ' ' + parts.join(' ') : '');
  out(line);
}

function maskKey(key) {
  if (!key) return '(none)';
  if (key.length <= 8) return '****';
  return `${key.slice(0, 7)}****${key.slice(-4)}`;
}

/* ------------------------------------------------------------------ */
/* Small utilities                                                     */
/* ------------------------------------------------------------------ */

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const genId = (prefix) => prefix + crypto.randomBytes(9).toString('hex');
const hasKeys = (o) => !!o && typeof o === 'object' && Object.keys(o).length > 0;

function safeEqual(a, b) {
  const ba = Buffer.from(String(a || ''), 'utf8');
  const bb = Buffer.from(String(b || ''), 'utf8');
  if (ba.length === 0 || ba.length !== bb.length) return false;
  return crypto.timingSafeEqual(ba, bb);
}

/** Client IP, honoring X-Forwarded-For when behind a platform proxy. */
function clientIp(req) {
  if (CONFIG.trustProxy) {
    const xff = String(req.headers['x-forwarded-for'] || '');
    if (xff) {
      const parts = xff.split(',').map((s) => s.trim()).filter(Boolean);
      // last entry is the one appended by our own trusted proxy
      if (parts.length) return parts[parts.length - 1];
    }
    const real = req.headers['x-real-ip'];
    if (real) return String(real);
  }
  return req.socket && req.socket.remoteAddress ? req.socket.remoteAddress : 'unknown';
}

/* In-memory per-IP rate limit (sliding 60s window, no dependencies). */
const rateBuckets = new Map();
function checkRateLimit(ip) {
  if (!CONFIG.rateLimitRpm) return null;
  const now = Date.now();
  let b = rateBuckets.get(ip);
  if (!b || now >= b.resetAt) {
    b = { count: 0, resetAt: now + 60_000 };
    rateBuckets.set(ip, b);
  }
  b.count++;
  if (b.count > CONFIG.rateLimitRpm) {
    return Math.max(1, Math.ceil((b.resetAt - now) / 1000));
  }
  if (rateBuckets.size > 20_000) {
    for (const [k, v] of rateBuckets) if (now >= v.resetAt) rateBuckets.delete(k);
  }
  return null;
}
function startRateLimitSweeper() {
  const t = setInterval(() => {
    const now = Date.now();
    for (const [k, v] of rateBuckets) if (now >= v.resetAt) rateBuckets.delete(k);
  }, 60_000);
  if (t.unref) t.unref();
}

class HttpError extends Error {
  constructor(status, message, type = 'proxy_error', code = null) {
    super(message);
    this.status = status;
    this.type = type;
    this.code = code;
  }
}

function sendJSON(res, status, obj, extraHeaders) {
  if (res.writableEnded) return;
  const body = JSON.stringify(obj);
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(body),
    'access-control-allow-origin': CONFIG.corsOrigin,
    ...(extraHeaders || {}),
  });
  res.end(body);
}

function sendError(res, status, message, type = 'proxy_error', code, rid) {
  sendJSON(res, status, {
    error: { message, type, code: code ?? null, param: null },
  }, rid ? { 'x-request-id': rid } : undefined);
}

async function readBody(req, limit) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > limit) throw new HttpError(413, `request body exceeds ${limit} bytes`, 'invalid_request_error');
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString('utf8');
}

async function readAll(stream, limit) {
  const chunks = [];
  let size = 0;
  if (!stream) return '';
  for await (const chunk of stream) {
    size += chunk.length;
    if (size > limit) throw new HttpError(502, 'upstream response too large', 'upstream_error');
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString('utf8');
}

async function* iterateBody(body) {
  if (!body) return;
  if (typeof body[Symbol.asyncIterator] === 'function') {
    yield* body;
    return;
  }
  yield* Readable.fromWeb(body);
}

/* ------------------------------------------------------------------ */
/* Credentials & body sanitizing                                       */
/* ------------------------------------------------------------------ */

const PROXY_ONLY_KEYS = new Set([
  'api_key', 'apikey', 'api-key', 'x_api_key', 'nim_api_key', 'nvidia_api_key',
  'client_api_key', 'proxy_api_key', '_proxy', '__proxy',
]);

function firstString(...vals) {
  for (const v of vals) if (typeof v === 'string' && v.trim()) return v.trim();
  return '';
}

function resolveApiKey(req, url, body) {
  if (!CONFIG.ignoreClientAuth) {
    const fromBody = body && firstString(body.api_key, body.apiKey, body.nim_api_key);
    const fromQuery = firstString(url.searchParams.get('api_key'), url.searchParams.get('apiKey'), url.searchParams.get('nim_api_key'));
    const fromHeader = firstString(req.headers['x-api-key'], req.headers['x-nvidia-api-key'], req.headers['nvidia-api-key']);
    const auth = String(req.headers.authorization || '');
    const fromAuth = /^bearer\s+/i.test(auth) ? auth.replace(/^bearer\s+/i, '').trim() : '';
    const found = firstString(fromBody, fromQuery, fromHeader, fromAuth);
    if (found) return found;
  }
  return CONFIG.envApiKey;
}

function sanitizeBody(body) {
  if (!body || typeof body !== 'object' || Array.isArray(body)) return body;
  const out = { ...body };
  for (const key of Object.keys(out)) {
    if (PROXY_ONLY_KEYS.has(key.toLowerCase())) delete out[key];
  }
  return out;
}

function resolveModel(req, url, body) {
  const fromBody = body && typeof body.model === 'string' ? body.model.trim() : '';
  return firstString(
    fromBody,
    url.searchParams.get('model'),
    req.headers['x-model'],
    CONFIG.defaultModel,
  );
}

/* --- thinking_effort -> reasoning_effort compatibility shim ---------------
 * Some client platforms send their reasoning knob as `thinking_effort`
 * (or nested `thinking.effort`) instead of the OpenAI/NIM standard
 * `reasoning_effort`. Without this mapping the value never reaches the
 * model and it silently runs in non-thinking mode. The original field is
 * left untouched (upstreams ignore it), and a reasoning_effort the client
 * already sent is never overwritten. This is additive only: bodies without
 * either field pass through unchanged.
 */
function translateThinkingEffort(body) {
  if (!body || typeof body !== 'object' || Array.isArray(body)) return body;
  if (body.reasoning_effort != null) return body; // client already sent the standard field
  let effort;
  if (body.thinking_effort != null) effort = body.thinking_effort;
  else if (body.thinking && typeof body.thinking === 'object' && !Array.isArray(body.thinking) && body.thinking.effort != null) effort = body.thinking.effort;
  if (effort == null) return body;
  body.reasoning_effort = effort;
  log('debug', 'mapped thinking_effort -> reasoning_effort', { effort });
  return body;
}

/* ------------------------------------------------------------------ */
/* JSON repair for tool-call arguments                                 */
/* ------------------------------------------------------------------ */

function tryParse(text) {
  if (typeof text !== 'string' || !text.trim()) return null;
  try { return JSON.parse(text); } catch { return null; }
}

/** Close unbalanced braces/quotes, strip fences & trailing commas. */
function repairJSONString(raw) {
  if (typeof raw !== 'string') return raw;
  let s = raw.trim();
  if (!s) return s;
  let parsed = tryParse(s);
  if (parsed !== null) return JSON.stringify(parsed);

  // strip markdown fences
  const fence = s.match(/^```[a-zA-Z0-9_+-]*\s*\n?([\s\S]*?)\n?```$/);
  if (fence) s = fence[1].trim();
  parsed = tryParse(s);
  if (parsed !== null) return JSON.stringify(parsed);

  // trailing commas
  let t = s.replace(/,(\s*[}\]])/g, '$1');
  parsed = tryParse(t);
  if (parsed !== null) return JSON.stringify(parsed);

  // balance braces/brackets/quotes
  const stack = [];
  let inStr = false;
  let esc = false;
  for (let i = 0; i < t.length; i++) {
    const ch = t[i];
    if (inStr) {
      if (esc) esc = false;
      else if (ch === '\\') esc = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') inStr = true;
    else if (ch === '{' || ch === '[') stack.push(ch);
    else if (ch === '}' || ch === ']') stack.pop();
  }
  let fixed = t;
  if (inStr) fixed += '"';
  while (stack.length) fixed += stack.pop() === '{' ? '}' : ']';
  parsed = tryParse(fixed);
  if (parsed !== null) return JSON.stringify(parsed);

  // truncation: take the longest prefix ending at } or ] that parses
  for (let i = t.length - 1; i > 0; i--) {
    if (t[i] === '}' || t[i] === ']') {
      const cand = t.slice(0, i + 1);
      parsed = tryParse(cand);
      if (parsed !== null) return JSON.stringify(parsed);
    }
  }
  return raw; // give up: keep original string, client decides
}

/* ------------------------------------------------------------------ */
/* Tool-call extraction from free text (fallback)                      */
/* ------------------------------------------------------------------ */

function sliceBalanced(text, openIdx) {
  let depth = 0;
  let inStr = false;
  let esc = false;
  for (let i = openIdx; i < text.length; i++) {
    const ch = text[i];
    if (inStr) {
      if (esc) esc = false;
      else if (ch === '\\') esc = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') inStr = true;
    else if (ch === '{' || ch === '[') depth++;
    else if (ch === '}' || ch === ']') {
      depth--;
      if (depth === 0) return i + 1;
    }
  }
  return -1;
}

function pickName(obj) {
  for (const k of ['name', 'function_name', 'tool_name', 'func']) {
    const v = obj[k];
    if (typeof v === 'string' && v.trim()) return v.trim();
  }
  return '';
}

function pickArgs(obj) {
  for (const k of ['arguments', 'args', 'parameters', 'input', 'params', 'tool_input']) {
    if (obj[k] !== undefined) return obj[k];
  }
  return undefined;
}

function makeCall(name, args) {
  const clean = String(name).trim().replace(/^(?:functions?|tools?)\./i, '');
  if (!/^[\w.-]{1,256}$/.test(clean)) return null;
  let argStr;
  if (typeof args === 'string') argStr = repairJSONString(args);
  else {
    try { argStr = JSON.stringify(args ?? {}); } catch { argStr = '{}'; }
    if (argStr === undefined) argStr = '{}';
  }
  return { id: genId('call_'), type: 'function', function: { name: clean, arguments: argStr } };
}

/** Try to interpret a parsed JSON value as one or more tool calls. */
function callsFromParsed(p, allowEmptyArgs) {
  if (p == null || typeof p !== 'object') return null;
  if (Array.isArray(p)) {
    const out = [];
    for (const item of p) {
      const c = callsFromParsed(item, allowEmptyArgs);
      if (c) out.push(...c);
    }
    return out.length ? out : null;
  }
  if (Array.isArray(p.tool_calls)) {
    const out = [];
    for (const item of p.tool_calls) {
      const c = callsFromParsed(item, false);
      if (c) out.push(...c);
    }
    return out.length ? out : null;
  }
  for (const key of ['function_call', 'function', 'tool', 'tool_call']) {
    const v = p[key];
    if (v && typeof v === 'object' && !Array.isArray(v)) {
      const c = callsFromParsed(v, true);
      if (c) return c;
    }
  }
  const name = pickName(p);
  if (name) {
    const args = pickArgs(p);
    if (args !== undefined) return [makeCall(name, args)].filter(Boolean);
    if (allowEmptyArgs) return [makeCall(name, {})].filter(Boolean);
  }
  return null;
}

function removeSpans(text, spans) {
  if (!spans || !spans.length) return text;
  const sorted = [...spans].sort((a, b) => b[0] - a[0]);
  let out = text;
  for (const [start, end] of sorted) out = out.slice(0, start) + out.slice(end);
  return out;
}


/**
 * Find tool calls hidden inside free-form model text.
 * Returns { calls, spans } so callers can strip raw call text from content.
 */
function extractToolCalls(text) {
  const calls = [];
  const spans = [];
  if (typeof text !== 'string' || !text.trim() || text.length > 400000) return { calls, spans };
  const seen = new Set();

  const overlaps = (s, e) => spans.some(([as, ae]) => s < ae && e > as);
  const add = (list, start, end) => {
    let added = false;
    for (const c of list) {
      const sig = c.function.name + '|' + c.function.arguments;
      if (seen.has(sig)) continue;
      seen.add(sig);
      calls.push(c);
      added = true;
    }
    if (added && end > start && !overlaps(start, end)) spans.push([start, end]);
    return added;
  };

  // 1) fenced code blocks: ```json / ```tool_call / ```function / bare ```
  const fenceRe = /```[ \t]*([^\n`]*)\r?\n?([\s\S]*?)```/g;
  let m;
  while ((m = fenceRe.exec(text))) {
    const lang = (m[1] || '').trim().toLowerCase();
    if (lang && !['json', 'js', 'javascript', 'tool_call', 'tool_calls',
      'function', 'functions', 'tool', 'tools'].includes(lang)) continue;
    const parsed = tryParse(m[2].trim());
    if (parsed == null) continue;
    const c = callsFromParsed(parsed, false);
    if (c) add(c, m.index, m.index + m[0].length);
  }

  // 2) XML-ish wrappers: <function_call>...</function_call>, <invoke>...</invoke>
  const xmlRe = /<(function_call|tool_call|invoke)>([\s\S]*?)<\/\1>/gi;
  while ((m = xmlRe.exec(text))) {
    const parsed = tryParse(m[2].trim());
    if (parsed != null) {
      const c = callsFromParsed(parsed, false);
      if (c) add(c, m.index, m.index + m[0].length);
    }
  }

  // 3) Anthropic-style invoke blocks with parameter children
  const invokeRe = /<invoke\s+name="?([^"\s]+)"?>([\s\S]*?)<\/invoke>/gi;
  while ((m = invokeRe.exec(text))) {
    const args = {};
    const paramRe = /<parameter\s+name="?([^"\s]+)"?>([\s\S]*?)<\/parameter>/gi;
    let p;
    while ((p = paramRe.exec(m[2]))) args[p[1]] = p[2].trim();
    const c = [makeCall(m[1], args)].filter(Boolean);
    if (c.length) add(c, m.index, m.index + m[0].length);
  }

  return { calls, spans };
}


/** Remaining extraction strategies, appended to extractToolCalls' results. */
function extractToolCallsMore(text, acc) {
  const { calls, spans } = acc;
  const seen = new Set(calls.map((c) => c.function.name + '|' + c.function.arguments));
  const overlaps = (s, e) => spans.some(([as, ae]) => s < ae && e > as);
  const add = (list, start, end) => {
    let added = false;
    for (const c of list) {
      const sig = c.function.name + '|' + c.function.arguments;
      if (seen.has(sig)) continue;
      seen.add(sig);
      calls.push(c);
      added = true;
    }
    if (added && end > start && !overlaps(start, end)) spans.push([start, end]);
  };

  // 4) whole text is a JSON object/array describing a call
  const trimmed = text.trim();
  if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
    const parsed = tryParse(trimmed);
    if (parsed != null) {
      const c = callsFromParsed(parsed, false);
      if (c) {
        const start = text.indexOf(trimmed.charAt(0));
        add(c, start, start + trimmed.length);
      }
    }
  }

  // 5) scan for {"name"|"function"|"tool_calls"...} objects inside prose
  const openRe = /\{\s*"(?:name|function|tool_calls|tool_name|function_name|function_call)"\s*:/g;
  let guard = 0;
  let m;
  while (guard++ < 50 && (m = openRe.exec(text))) {
    const end = sliceBalanced(text, m.index);
    if (end < 0) continue;
    const parsed = tryParse(text.slice(m.index, end));
    if (parsed == null) continue;
    const c = callsFromParsed(parsed, false);
    if (c) add(c, m.index, end);
    openRe.lastIndex = m.index + 2;
  }
  return acc;
}

/** null | 'soft' | 'explicit' - how strongly text looks like a raw tool call. */
function classifyCallLike(text) {
  if (typeof text !== 'string' || !text.trim()) return null;
  if (/```(?:tool_call|tool_calls|function|functions|tool|tools)\b/i.test(text)) return 'explicit';
  if (/<(?:\/)?(?:function_calls|function_call|tool_calls|tool_call|invoke)\b/i.test(text)) return 'explicit';
  if (/^\s*(?:tool_call|function_call)\s*[:=]/im.test(text)) return 'explicit';
  const t = text.trimStart();
  if (t.startsWith('{') && /"(?:name|arguments|function|tool_calls|function_call)"\s*:/.test(t)) return 'soft';
  if (/"(?:tool_calls|function_call)"\s*:/.test(text)) return 'soft';
  return null;
}


/* ------------------------------------------------------------------ */
/* Tool-call normalization                                             */
/* ------------------------------------------------------------------ */

function normalizeToolCallEntry(tc, index, idMap) {
  if (!tc || typeof tc !== 'object') return null;
  const fn = tc.function && typeof tc.function === 'object' ? tc.function : null;
  const name = (fn && typeof fn.name === 'string' && fn.name) ||
    (typeof tc.name === 'string' ? tc.name : '') ||
    (typeof tc.function_name === 'string' ? tc.function_name : '') ||
    (typeof tc.tool_name === 'string' ? tc.tool_name : '');
  let args = fn ? fn.arguments : undefined;
  if (args === undefined) args = tc.arguments ?? tc.args ?? tc.parameters ?? tc.input;
  let id = typeof tc.id === 'string' && tc.id ? tc.id : '';
  const key = Number.isInteger(tc.index) ? tc.index : index;
  if (!id) {
    if (!idMap[key]) idMap[key] = genId('call_');
    id = idMap[key];
  } else {
    idMap[key] = id;
  }
  if (typeof args === 'object' && args !== null) {
    try { args = JSON.stringify(args); } catch { args = ''; }
  }
  if (args == null) args = '';
  if (typeof args !== 'string') args = String(args);
  const clean = name.trim().replace(/^(?:functions?|tools?)\./i, '');
  if (!clean && !args) return null;
  const out = { id, type: tc.type === 'function' || !tc.type ? 'function' : tc.type };
  const fnOut = {};
  if (clean) fnOut.name = clean;
  fnOut.arguments = args;
  out.function = fnOut;
  return out;
}

/** Streaming fragments: keep ids stable per index, always emit index. */
function normalizeDeltaToolCalls(raw, st) {
  const arr = Array.isArray(raw) ? raw : [raw];
  const out = [];
  for (let i = 0; i < arr.length; i++) {
    const tc = arr[i];
    if (!tc || typeof tc !== 'object') continue;
    const pos = Number.isInteger(tc.index) ? tc.index : (Number.isInteger(tc.i) ? tc.i : i);
    const n = normalizeToolCallEntry(tc, pos, st.toolIds);
    if (!n) continue;
    const fn = tc.function && typeof tc.function === 'object' ? tc.function : {};
    const rawName = typeof fn.name === 'string' && fn.name ? fn.name :
      (typeof tc.name === 'string' ? tc.name : '');
    if (rawName) st.toolNames[pos] = n.function.name;
    else if (st.toolNames[pos]) n.function.name = st.toolNames[pos];
    n.index = pos;
    out.push(n);
  }
  return out;
}

/** Non-streaming: full normalization, no index field (OpenAI style). */
function normalizeMessageToolCalls(msg) {
  let raw = msg.tool_calls;
  if (!raw && msg.function_call) raw = [{ function: msg.function_call }];
  if (!raw) return null;
  const arr = Array.isArray(raw) ? raw : [raw];
  const idMap = {};
  const out = [];
  arr.forEach((tc, i) => {
    const n = normalizeToolCallEntry(tc, i, idMap);
    if (n && n.function.name) out.push(n);
  });
  return out.length ? out : null;
}

/* ------------------------------------------------------------------ */
/* Thinking / reasoning handling                                       */
/* ------------------------------------------------------------------ */

const REASONING_KEYS = ['reasoning_content', 'reasoning', 'thinking', 'thinking_content', 'analysis_content', 'thought'];

function extractThinkingTags(content) {
  if (typeof content !== 'string' || !/<think/i.test(content)) return null;
  let reasoning = '';
  let out = content;
  const pairs = [
    /<thinking>([\s\S]*?)<\/thinking>/gi,
    /<think>([\s\S]*?)<\/think>/gi,
  ];
  for (const re of pairs) {
    out = out.replace(re, (_full, inner) => {
      reasoning += (reasoning ? '\n' : '') + String(inner).trim();
      return '';
    });
  }
  // unclosed block (stream cut off mid-thinking)
  out = out.replace(/<think(?:ing)?>([\s\S]*)$/i, (_full, inner) => {
    reasoning += (reasoning ? '\n' : '') + String(inner).trim();
    return '';
  });
  if (!reasoning) return null;
  out = out.replace(/^(?:[ \t]*\r?\n)+/, '');
  out = out.replace(/[ \t]+\n/g, '\n').replace(/\n{3,}/g, '\n\n');
  return { reasoning, content: out };
}

/**
 * Streaming think-tag splitter. State per choice: st.inThink, st.tagPending.
 * Thinking text is emitted as reasoning, everything else as content. A short
 * tail that could be a partially-received tag is held back until the next
 * fragment arrives so tags split across chunks still work.
 */
function processThinkTags(st, text) {
  let s = (st.tagPending || '') + text;
  st.tagPending = '';
  let outContent = '';
  let outReasoning = '';
  let pos = 0;
  const OPEN = /<think(?:ing)?>/i;
  const CLOSE = /<\/think(?:ing)?>/i;
  const consumeNL = () => {
    if (!st.tagSkipNL) return;
    st.tagSkipNL = false;
    if (s[pos] === '\r' && s[pos + 1] === '\n') pos += 2;
    else if (s[pos] === '\n') pos += 1;
  };

  while (pos < s.length) {
    consumeNL();
    if (pos >= s.length) break;
    if (!st.inThink) {
      const open = OPEN.exec(s.slice(pos));
      if (open) {
        outContent += s.slice(pos, pos + open.index);
        st.inThink = true;
        pos = pos + open.index + open[0].length;
        st.tagSkipNL = true;
        continue;
      }
      // no opening tag: hold back a tail that could still become one
      const from = Math.max(pos, s.length - 16);
      const tail = s.slice(from);
      const part = /<(?:t(?:h(?:i(?:n(?:k(?:i(?:n(?:g)?)?)?)?)?)?)?)?$/.exec(tail);
      let emitEnd = s.length;
      if (part && part[0].length > 1) emitEnd = from + part.index;
      outContent += s.slice(pos, emitEnd);
      st.tagPending = s.slice(emitEnd);
      break;
    }
    const close = CLOSE.exec(s.slice(pos));
    if (close) {
      outReasoning += s.slice(pos, pos + close.index);
      st.inThink = false;
      pos = pos + close.index + close[0].length;
      st.tagSkipNL = true;
      continue;
    }
    // inside thinking with no close yet: hold back possible partial close tag
    const from = Math.max(pos, s.length - 16);
    const tail = s.slice(from);
    const part = /<\/(?:t(?:h(?:i(?:n(?:k(?:i(?:n(?:g)?)?)?)?)?)?)?)?$/.exec(tail);
    let emitEnd = s.length;
    if (part && part[0].length > 1) emitEnd = from + part.index;
    outReasoning += s.slice(pos, emitEnd);
    st.tagPending = s.slice(emitEnd);
    break;
  }
  return { content: outContent, reasoning: outReasoning };
}


/* ------------------------------------------------------------------ */
/* SSE parsing                                                         */
/* ------------------------------------------------------------------ */

function createSSEParser() {
  let buf = '';
  let fields = null;
  const events = [];
  const dispatch = () => {
    if (fields && fields.data.length) {
      events.push({
        data: fields.data.join('\n'),
        event: fields.event || '',
        id: fields.id || '',
        retry: fields.retry || '',
      });
    }
    fields = null;
  };
  const handleLine = (line) => {
    if (line === '') { dispatch(); return; }
    if (line.startsWith(':')) return;
    if (!fields) fields = { data: [], event: '', id: '', retry: '' };
    const colon = line.indexOf(':');
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? '' : line.slice(colon + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'data') fields.data.push(value);
    else if (field === 'event') fields.event = value;
    else if (field === 'id') fields.id = value;
    else if (field === 'retry') fields.retry = value;
  };
  return {
    push(chunk) {
      buf += chunk;
      buf = buf.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) handleLine(line);
      return events.splice(0, events.length);
    },
    flush() {
      if (buf) { handleLine(buf); buf = ''; }
      return events.splice(0, events.length);
    },
  };
}

function serializeSSE(ev) {
  let out = '';
  if (ev.event) out += 'event: ' + ev.event + '\n';
  if (ev.id) out += 'id: ' + ev.id + '\n';
  for (const line of String(ev.data).split('\n')) out += 'data: ' + line + '\n';
  return out + '\n';
}

/* ------------------------------------------------------------------ */
/* Non-streaming chat.completion normalization                         */
/* ------------------------------------------------------------------ */

function normalizeChatResponse(resp, meta, opts) {
  if (!resp || typeof resp !== 'object') return resp;
  if (!resp.id) resp.id = meta.id;
  if (!resp.object) resp.object = 'chat.completion';
  if (!resp.created) resp.created = meta.created;
  if (!resp.model) resp.model = meta.model;
  if (!Array.isArray(resp.choices)) return resp;

  const toolsPresent = !!(opts && opts.toolsPresent);

  resp.choices.forEach((choice, pos) => {
    if (!Number.isInteger(choice.index)) choice.index = pos;
    const msg = choice.message || (choice.delta ? choice.delta : null);
    if (!msg || typeof msg !== 'object') return;
    if (!msg.role) msg.role = 'assistant';

    // --- reasoning: normalize any known field to reasoning_content ---
    let reasoning = typeof msg.reasoning_content === 'string' ? msg.reasoning_content : '';
    if (!reasoning) {
      for (const k of REASONING_KEYS) {
        if (typeof msg[k] === 'string' && msg[k]) { reasoning = msg[k]; break; }
      }
    }
    if (!reasoning && typeof choice.reasoning === 'string') reasoning = choice.reasoning;

    // thinking tags embedded in content move to reasoning
    if (typeof msg.content === 'string') {
      const tagged = extractThinkingTags(msg.content);
      if (tagged) {
        reasoning = reasoning ? reasoning + '\n' + tagged.reasoning : tagged.reasoning;
        msg.content = tagged.content;
      }
    }
    if (reasoning) msg.reasoning_content = reasoning;

    // --- tool calls: normalize known-good shapes ---
    let calls = normalizeMessageToolCalls(msg);
    if (calls) {
      msg.tool_calls = calls;
      delete msg.function_call;
    }

    // --- fallback: model dumped the call as text ---
    const finish = choice.finish_reason;
    const finishIsTool = finish === 'tool_calls' || finish === 'function_call';
    if (!calls && typeof msg.content === 'string' && msg.content) {
      const cls = classifyCallLike(msg.content);
      const forced = !!(opts && opts.toolChoiceForcesCall);
      if (finishIsTool || cls === 'explicit' || (cls === 'soft' && (toolsPresent || forced))) {
        const r = extractToolCalls(msg.content);
        extractToolCallsMore(msg.content, r);
        if (r.calls.length) {
          msg.tool_calls = r.calls;
          calls = r.calls;
          msg.content = removeSpans(msg.content, r.spans).trim();
          if (!msg.content) msg.content = null;
        }
      }
    }

    // --- finish_reason consistency ---
    if (choice.finish_reason === 'function_call') choice.finish_reason = 'tool_calls';
    if (calls && (choice.finish_reason == null || choice.finish_reason === 'stop')) {
      choice.finish_reason = 'tool_calls';
    }
    if (choice.finish_reason === 'tool_calls' && !calls) choice.finish_reason = 'stop';
  });

  return resp;
}


/* ------------------------------------------------------------------ */
/* Streaming chat.completion assembler                                 */
/* ------------------------------------------------------------------ */

class ChatStreamAssembler {
  constructor(meta, opts) {
    this.meta = meta;              // { id, created, model }
    this.opts = opts || {};        // { toolsPresent }
    this.states = new Map();       // index -> per-choice state
    this.errored = false;
  }

  state(index) {
    if (!this.states.has(index)) {
      this.states.set(index, {
        index,
        content: '',
        held: false,
        heldBuf: '',
        holdReason: null,          // 'explicit' | 'soft'
        inThink: false,
        tagPending: '',
        reasoning: '',
        sawToolCalls: false,
        toolIds: {},
        toolNames: {},
        finish: null,
        finishDetails: null,
        flushed: false,
      });
    }
    return this.states.get(index);
  }

  fill(chunk) {
    if (!chunk.id) chunk.id = this.meta.id;
    if (!chunk.object) chunk.object = 'chat.completion.chunk';
    if (!chunk.created) chunk.created = this.meta.created;
    if (!chunk.model) chunk.model = this.meta.model;
    return chunk;
  }

  chunkFor(st, delta) {
    return this.fill({
      choices: [{ index: st.index, delta }],
    });
  }

  /**
   * Feed one upstream chunk (object or JSON string).
   * Returns an array of chunks to forward to the client (may be empty).
   */
  push(raw) {
    let chunk = raw;
    if (typeof chunk === 'string') {
      try { chunk = JSON.parse(chunk); } catch { return []; }
    }
    if (!chunk || typeof chunk !== 'object') return [];
    if (chunk.error) { this.errored = true; return [chunk]; }

    if (typeof chunk.id === 'string' && chunk.id) this.meta.id = chunk.id;
    if (typeof chunk.model === 'string' && chunk.model) this.meta.model = chunk.model;
    if (typeof chunk.created === 'number') this.meta.created = chunk.created;

    const out = [];
    const choices = Array.isArray(chunk.choices) ? chunk.choices : null;

    // sideband chunk (usage etc.) - flush finished choices first so that
    // content -> finish -> usage ordering is preserved
    if (!choices || choices.length === 0) {
      out.push(...this.flushAll());
      out.push(this.fill(chunk));
      return out;
    }

    const emitted = [];
    let anyFinish = false;

    for (let pos = 0; pos < choices.length; pos++) {
      const c = choices[pos] || {};
      const index = Number.isInteger(c.index) ? c.index : pos;
      const st = this.state(index);

      if (c.finish_reason != null || c.finish_details != null) {
        if (c.finish_reason != null) st.finish = c.finish_reason;
        if (c.finish_details != null) st.finishDetails = c.finish_details;
        anyFinish = true;
      }

      const ec = { ...c };
      delete ec.finish_reason;
      delete ec.finish_details;

      if (ec.delta && typeof ec.delta === 'object') {
        const d = { ...ec.delta };

        // --- reasoning fields ---
        for (const k of REASONING_KEYS) {
          const v = d[k];
          if (v != null && v !== '') {
            const piece = typeof v === 'string' ? v : JSON.stringify(v);
            st.reasoning += piece;
            if (k !== 'reasoning_content') {
              const prev = typeof d.reasoning_content === 'string' ? d.reasoning_content : '';
              d.reasoning_content = prev + piece;
            }
          }
        }

        // --- tool call fragments ---
        if (d.tool_calls != null) {
          st.sawToolCalls = true;
          d.tool_calls = normalizeDeltaToolCalls(d.tool_calls, st);
        }
        if (d.function_call != null) {
          st.sawToolCalls = true;
          const conv = normalizeDeltaToolCalls([d.function_call], st);
          delete d.function_call;
          d.tool_calls = d.tool_calls ? d.tool_calls.concat(conv) : conv;
        }

        // --- content + think-tag splitting + explicit-call holding ---
        if (typeof d.content === 'string' && d.content.length) {
          const t = processThinkTags(st, d.content);
          if (t.reasoning) st.reasoning += t.reasoning;
          let cOut = t.content;
          if (cOut) {
            st.content += cOut;
            const cls = classifyCallLike(st.content);
            if (cls) st.holdReason = st.holdReason === 'explicit' ? 'explicit' : cls;
            if (st.holdReason === 'explicit') st.held = true;
            if (st.held) { st.heldBuf += cOut; cOut = ''; }
          }
          delete d.content;
          if (cOut) d.content = cOut;
          if (t.reasoning) {
            const prev = typeof d.reasoning_content === 'string' ? d.reasoning_content : '';
            d.reasoning_content = prev + t.reasoning;
          }
        }

        if (hasKeys(d)) {
          ec.delta = d;
          emitted.push(ec);
        }
      } else if (hasKeys(ec)) {
        emitted.push(ec);
      }
    }

    if (emitted.length) out.push(this.fill({ ...chunk, choices: emitted }));
    if (anyFinish) out.push(...this.flushFinished());
    return out;
  }

  /** Flush choices that have received a finish_reason. */
  flushFinished() {
    const out = [];
    for (const st of this.states.values()) {
      if (!st.flushed && st.finish != null) out.push(...this.flushChoice(st));
    }
    return out;
  }

  /** Flush every unflushed choice (usage chunk / end of stream). */
  flushAll() {
    const out = [];
    for (const st of this.states.values()) {
      if (!st.flushed) out.push(...this.flushChoice(st));
    }
    return out;
  }

  flushChoice(st) {
    st.flushed = true;
    const out = [];

    // resolve a pending think-tag tail
    let pendingReasoning = '';
    if (st.tagPending) {
      if (st.inThink) pendingReasoning = st.tagPending;
      else { st.held = true; st.heldBuf += st.tagPending; }
      st.tagPending = '';
    }

    const finishIsTool = st.finish === 'tool_calls' || st.finish === 'function_call';
    const wantExtract = !st.sawToolCalls && (finishIsTool || st.holdReason === 'explicit');

    // held content goes through extraction so raw call text never reaches
    // the client; already-streamed content can only get a synthetic chunk
    if (wantExtract) {
      const source = st.held ? st.heldBuf : st.content;
      if (source) {
        const r = extractToolCalls(source);
        extractToolCallsMore(source, r);
        if (r.calls.length) {
          st.sawToolCalls = true;
          if (st.held) st.heldBuf = removeSpans(st.heldBuf, r.spans);
          out.push(this.chunkFor(st, {
            tool_calls: r.calls.map((c, i) => Object.assign({ index: i }, c)),
          }));
        }
      }
    }

    if (pendingReasoning) out.push(this.chunkFor(st, { reasoning_content: pendingReasoning }));
    if (st.held && st.heldBuf) out.push(this.chunkFor(st, { content: st.heldBuf }));

    let finish = st.finish;
    if (st.sawToolCalls && (finish == null || finish === 'stop')) finish = 'tool_calls';
    if ((finish === 'tool_calls' || finish === 'function_call') && !st.sawToolCalls) finish = 'stop';
    if (finish == null) finish = 'stop';
    if (finish === 'function_call') finish = 'tool_calls';

    const fc = { index: st.index, finish_reason: finish };
    if (st.finishDetails) fc.finish_details = st.finishDetails;
    out.push(this.fill({ choices: [fc] }));
    return out;
  }
}


/* ------------------------------------------------------------------ */
/* Upstream calls                                                      */
/* ------------------------------------------------------------------ */

const RETRYABLE_STATUS = new Set([408, 429, 500, 502, 503, 504]);

function anySignal(signals) {
  const ctrl = new AbortController();
  for (const s of signals) {
    if (!s) continue;
    if (s.aborted) { ctrl.abort(s.reason); break; }
    s.addEventListener('abort', () => ctrl.abort(s.reason), { once: true });
  }
  return ctrl;
}

function backoffDelay(attempt, retryAfter) {
  if (retryAfter) {
    const secs = parseFloat(retryAfter);
    if (Number.isFinite(secs)) return Math.min(secs * 1000, 30000);
    const when = Date.parse(retryAfter);
    if (Number.isFinite(when)) return Math.min(Math.max(when - Date.now(), 0), 30000);
  }
  const base = Math.min(500 * 2 ** attempt, 8000);
  return base + Math.floor(Math.random() * 250);
}

function buildUpstreamPath(path, url) {
  let p = path;
  const baseEndsWithV1 = /\/v1$/.test(CONFIG.baseUrl);
  if (baseEndsWithV1 && p.startsWith('/v1/')) p = p.slice(3);
  if (!baseEndsWithV1 && !p.startsWith('/v1/')) p = '/v1' + p;
  const params = new URLSearchParams(url.searchParams);
  for (const k of ['api_key', 'apiKey', 'nim_api_key', 'model']) params.delete(k);
  const qs = params.toString();
  return qs ? p + '?' + qs : p;
}

function upstreamHeaders(apiKey, rid, clientHeaders, stream) {
  const headers = {
    'content-type': 'application/json',
    accept: stream ? 'text/event-stream, application/json' : 'application/json',
    'user-agent': 'nim-proxy/1.0',
    'x-request-id': rid,
  };
  if (apiKey) {
    headers.authorization = 'Bearer ' + apiKey;
    headers['x-api-key'] = apiKey;
  }
  const ua = clientHeaders['user-agent'];
  if (ua) headers['user-agent'] = ua;
  // forward NVIDIA-specific extras the client may set
  for (const [k, v] of Object.entries(clientHeaders)) {
    if (k.startsWith('x-nvidia-') && !headers[k]) headers[k] = v;
  }
  return headers;
}

/**
 * Fetch the upstream. Resolves as soon as headers arrive.
 * The returned object carries an AbortController usable for the body phase.
 */
async function fetchUpstream(url, init, opts) {
  const ctrl = anySignal([init.signal]);
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    ctrl.abort(new Error('connect timeout'));
  }, opts.connectTimeoutMs);
  if (timer.unref) timer.unref();
  try {
    const res = await fetch(url, { ...init, signal: ctrl.signal });
    return { res, ctrl, isTimeout: () => timedOut };
  } catch (err) {
    if (timedOut) throw new HttpError(504, 'upstream connect timeout', 'upstream_error');
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

function toText(piece) {
  if (typeof piece === 'string') return piece;
  if (Buffer.isBuffer(piece)) return piece.toString('utf8');
  if (piece instanceof Uint8Array) {
    return Buffer.from(piece.buffer, piece.byteOffset, piece.byteLength).toString('utf8');
  }
  return String(piece);
}

function relayUpstreamError(clientRes, status, bodyText, rid, upstreamRes) {
  let payload = null;
  try { payload = JSON.parse(bodyText); } catch { /* not json */ }
  if (!payload || typeof payload !== 'object' || !payload.error) {
    payload = {
      error: {
        message: (bodyText && bodyText.slice(0, 2000)) ||
          (`upstream returned HTTP ${status}`),
        type: 'upstream_error',
        code: status,
        param: null,
      },
    };
  }
  const headers = { 'x-request-id': rid };
  const uh = upstreamRes && typeof upstreamRes.headers?.get === 'function'
    ? upstreamRes.headers.get('x-request-id') : null;
  if (uh) headers['x-upstream-request-id'] = uh;
  sendJSON(clientRes, status, payload, headers);
}

/* ------------------------------------------------------------------ */
/* Response piping                                                      */
/* ------------------------------------------------------------------ */

function sseHeaders(rid) {
  return {
    'content-type': 'text/event-stream; charset=utf-8',
    'cache-control': 'no-cache, no-transform',
    connection: 'keep-alive',
    'x-accel-buffering': 'no',
    'access-control-allow-origin': CONFIG.corsOrigin,
    'x-request-id': rid,
  };
}

function writeChunk(res, str) {
  if (res.writableEnded || res.destroyed) return Promise.resolve(false);
  if (res.write(str)) return Promise.resolve(true);
  return new Promise((resolve) => {
    const onDrain = () => { cleanup(); resolve(true); };
    const onClose = () => { cleanup(); resolve(false); };
    const cleanup = () => {
      res.off('drain', onDrain);
      res.off('close', onClose);
      res.off('error', onClose);
    };
    res.once('drain', onDrain);
    res.once('close', onClose);
    res.once('error', onClose);
  });
}

/**
 * Pipe an upstream SSE response to the client.
 * `transform` (optional) maps a parsed event's data string to an array of
 * strings to write as `data:` lines; it may return 'DONE' markers.
 */
async function pipeEventStream(upstreamRes, clientRes, opts) {
  const { transform, idleTimeoutMs, maxMs, onDone } = opts;
  clientRes.writeHead(200, sseHeaders(opts.rid));
  if (clientRes.socket) clientRes.socket.setNoDelay(true);

  let idleTimer = null;
  let maxTimer = null;
  let finished = false;
  let sentDone = false;
  let timedOutMsg = null;

  const clearTimers = () => {
    if (idleTimer) clearTimeout(idleTimer);
    if (maxTimer) clearTimeout(maxTimer);
    idleTimer = maxTimer = null;
  };
  const armIdle = () => {
    if (!idleTimeoutMs) return;
    if (idleTimer) clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      timedOutMsg = 'upstream stream idle timeout';
      if (opts.abortCtrl) opts.abortCtrl.abort(new Error(timedOutMsg));
    }, idleTimeoutMs);
    if (idleTimer.unref) idleTimer.unref();
  };
  if (maxMs) {
    maxTimer = setTimeout(() => {
      timedOutMsg = 'upstream stream exceeded max duration';
      if (opts.abortCtrl) opts.abortCtrl.abort(new Error(timedOutMsg));
    }, maxMs);
    if (maxTimer.unref) maxTimer.unref();
  }

  const writeData = async (data) => {
    if (data === 'DONE') {
      if (sentDone) return true;
      sentDone = true;
      return writeChunk(clientRes, 'data: [DONE]\n\n');
    }
    return writeChunk(clientRes, 'data: ' + String(data).split('\n').join('\ndata: ') + '\n\n');
  };

  try {
    armIdle();
    const parser = createSSEParser();
    for await (const piece of iterateBody(upstreamRes.body)) {
      armIdle();
      const text = toText(piece);
      for (const ev of parser.push(text)) {
        if (ev.data === '[DONE]') {
          if (onDone) for (const s of await onDone()) if (!(await writeData(s))) return;
          await writeData('DONE');
          clearTimers();
          finished = true;
          clientRes.end();
          return;
        }
        if (transform) {
          for (const s of transform(ev)) if (!(await writeData(s))) return;
        } else if (!(await writeData(ev.data))) return;
      }
    }
    for (const ev of parser.flush()) {
      if (ev.data === '[DONE]') continue;
      if (transform) {
        for (const s of transform(ev)) if (!(await writeData(s))) return;
      } else if (!(await writeData(ev.data))) return;
    }
    if (onDone) for (const s of await onDone()) if (!(await writeData(s))) return;
    if (!sentDone) await writeData('DONE');
    clearTimers();
    finished = true;
    clientRes.end();
  } catch (err) {
    clearTimers();
    if (!clientRes.headersSent) throw err;
    if (!finished && !clientRes.writableEnded) {
      const msg = timedOutMsg || (err && err.message) || 'stream error';
      try {
        await writeChunk(clientRes, 'data: ' + JSON.stringify({
          error: { message: msg, type: 'upstream_error', code: null, param: null },
        }) + '\n\n');
        if (!sentDone) await writeData('DONE');
      } catch { /* client gone */ }
      clientRes.end();
    }
    log('warn', 'stream terminated early', { err: msg2(err), rid: opts.rid });
  }
}

function msg2(err) {
  if (!err) return '';
  if (err.name === 'AbortError' || err.name === 'TimeoutError') return err.message || err.name;
  return err.message || String(err);
}


/* ------------------------------------------------------------------ */
/* Chat completions handler                                            */
/* ------------------------------------------------------------------ */

function makeMeta(body, rid) {
  return {
    id: 'chatcmpl-' + rid.replace(/^req_/, ''),
    created: Math.floor(Date.now() / 1000),
    model: (body && body.model) || 'unknown',
  };
}

async function handleChatCompletions(req, res, url, body, ctx) {
  const stream = body.stream === true;
  const meta = makeMeta(body, ctx.rid);
  const upstreamUrl = CONFIG.baseUrl + buildUpstreamPath('/v1/chat/completions', url);
  const headers = upstreamHeaders(ctx.apiKey, ctx.rid, req.headers, stream);
  const initBase = {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
  };

  let lastErr = null;
  for (let attempt = 0; attempt <= CONFIG.maxRetries; attempt++) {
    if (attempt > 0) {
      const delay = backoffDelay(attempt - 1, lastErr && lastErr.retryAfter);
      log('debug', 'retrying chat request', { attempt, delay, rid: ctx.rid });
      await sleep(delay);
    }
    let fetched;
    try {
      fetched = await fetchUpstream(upstreamUrl, { ...initBase, signal: req.__abortSignal },
        { connectTimeoutMs: CONFIG.connectTimeoutMs });
    } catch (err) {
      lastErr = err instanceof HttpError ? err : { retryAfter: null };
      if (err instanceof HttpError && err.status !== 504) throw err;
      if (attempt >= CONFIG.maxRetries) {
        throw new HttpError(504, 'upstream unreachable: ' + msg2(err), 'upstream_error');
      }
      continue;
    }

    const { res: up, ctrl, isTimeout } = fetched;

    if (!up.ok) {
      const bodyText = await readAll(up.body, 1024 * 1024).catch(() => '');
      if (RETRYABLE_STATUS.has(up.status) && attempt < CONFIG.maxRetries) {
        lastErr = { retryAfter: up.headers.get('retry-after') };
        log('debug', 'upstream error, may retry', { status: up.status, attempt, rid: ctx.rid });
        continue;
      }
      log('warn', 'upstream error', { status: up.status, rid: ctx.rid, model: body.model });
      relayUpstreamError(res, up.status, bodyText, ctx.rid, up);
      return;
    }

    const ct = String(up.headers.get('content-type') || '');

    if (!stream) {
      // non-streaming: read, normalize, return
      const text = await (async () => {
        const timer = setTimeout(() => ctrl.abort(new Error('read timeout')), CONFIG.requestTimeoutMs);
        if (timer.unref) timer.unref();
        try { return await readAll(up.body, CONFIG.maxUpstreamBodyBytes); }
        finally { clearTimeout(timer); }
      })().catch((err) => {
        if (isTimeout() || /timeout/.test(msg2(err))) {
          throw new HttpError(504, 'upstream read timeout', 'upstream_error');
        }
        throw new HttpError(502, 'upstream read failed: ' + msg2(err), 'upstream_error');
      });
      let parsed;
      try { parsed = JSON.parse(text); }
      catch { throw new HttpError(502, 'upstream returned invalid JSON', 'upstream_error'); }
      normalizeChatResponse(parsed, meta, {
        toolsPresent: Array.isArray(body.tools) && body.tools.length > 0,
        toolChoiceForcesCall: !!(body.tool_choice && typeof body.tool_choice === 'object'),
      });
      log('info', 'chat completion', {
        model: body.model, stream: false, rid: ctx.rid, toolCalls:
          (parsed.choices || []).some((c) => c.message && c.message.tool_calls),
      });
      sendJSON(res, 200, parsed, { 'x-request-id': ctx.rid });
      return;
    }

    // streaming
    if (!/text\/event-stream/i.test(ct)) {
      // upstream ignored stream=true and returned JSON
      const text = await readAll(up.body, CONFIG.maxUpstreamBodyBytes).catch(() => '');
      let parsed = null;
      try { parsed = JSON.parse(text); } catch { /* fallthrough */ }
      res.writeHead(200, sseHeaders(ctx.rid));
      if (parsed) {
        normalizeChatResponse(parsed, meta, {
          toolsPresent: Array.isArray(body.tools) && body.tools.length > 0,
          toolChoiceForcesCall: !!(body.tool_choice && typeof body.tool_choice === 'object'),
        });
        const choices = Array.isArray(parsed.choices) ? parsed.choices : [];
        for (const c of choices) {
          const delta = {};
          if (c.message && c.message.role) delta.role = c.message.role;
          if (c.message && typeof c.message.content === 'string') delta.content = c.message.content;
          if (c.message && c.message.reasoning_content) delta.reasoning_content = c.message.reasoning_content;
          if (c.message && c.message.tool_calls) delta.tool_calls = c.message.tool_calls;
          await writeChunk(res, 'data: ' + JSON.stringify(this_fill(parsed, meta, c, delta)) + '\n\n');
        }
        await writeChunk(res, 'data: ' + JSON.stringify(this_fill(parsed, meta, { index: 0, finish_reason: (choices[0] || {}).finish_reason || 'stop' }, {})) + '\n\n');
        if (parsed.usage) {
          await writeChunk(res, 'data: ' + JSON.stringify({ id: meta.id, object: 'chat.completion.chunk', created: meta.created, model: meta.model, choices: [], usage: parsed.usage }) + '\n\n');
        }
      }
      await writeChunk(res, 'data: [DONE]\n\n');
      res.end();
      return;
    }

    const asm = new ChatStreamAssembler(meta, {
      toolsPresent: Array.isArray(body.tools) && body.tools.length > 0,
    });
    // re-run extraction decision on finish using opts
    asm.opts.toolChoiceForcesCall = !!(body.tool_choice && typeof body.tool_choice === 'object');

    log('info', 'chat stream start', { model: body.model, rid: ctx.rid });

    await pipeEventStream(up, res, {
      rid: ctx.rid,
      idleTimeoutMs: CONFIG.streamIdleTimeoutMs,
      maxMs: CONFIG.streamMaxMs,
      abortCtrl: ctrl,
      transform: (ev) => asm.push(ev.data).map((c) => JSON.stringify(c)),
      onDone: async () => asm.flushAll().map((c) => JSON.stringify(c)),
    });
    log('info', 'chat stream end', { model: body.model, rid: ctx.rid, reasoning: asm.meta.model });
    return;
  }
}

function this_fill(parsed, meta, choice, delta) {
  return {
    id: parsed.id || meta.id,
    object: 'chat.completion.chunk',
    created: parsed.created || meta.created,
    model: parsed.model || meta.model,
    choices: [{ index: choice.index || 0, delta, finish_reason: choice.finish_reason || null }],
  };
}

/* ------------------------------------------------------------------ */
/* Generic passthrough (embeddings, completions, rerank, models, ...)  */
/* ------------------------------------------------------------------ */

async function handlePassthrough(req, res, url, path, ctx, bodyText) {
  const stream = bodyText ? (() => {
    try { return JSON.parse(bodyText).stream === true; } catch { return false; }
  })() : false;
  const upstreamUrl = CONFIG.baseUrl + buildUpstreamPath(path, url);
  const headers = upstreamHeaders(ctx.apiKey, ctx.rid, req.headers, stream);
  if (bodyText) headers['content-length'] = Buffer.byteLength(bodyText);

  let lastErr = null;
  for (let attempt = 0; attempt <= CONFIG.maxRetries; attempt++) {
    if (attempt > 0) await sleep(backoffDelay(attempt - 1, lastErr));
    let fetched;
    try {
      fetched = await fetchUpstream(upstreamUrl, {
        method: req.method,
        headers,
        body: bodyText || undefined,
        signal: req.__abortSignal,
      }, { connectTimeoutMs: CONFIG.connectTimeoutMs });
    } catch (err) {
      lastErr = null;
      if (attempt >= CONFIG.maxRetries) {
        throw new HttpError(504, 'upstream unreachable: ' + msg2(err), 'upstream_error');
      }
      continue;
    }
    const { res: up, ctrl } = fetched;
    if (!up.ok) {
      const errText = await readAll(up.body, 1024 * 1024).catch(() => '');
      if (RETRYABLE_STATUS.has(up.status) && attempt < CONFIG.maxRetries) {
        lastErr = up.headers.get('retry-after');
        continue;
      }
      relayUpstreamError(res, up.status, errText, ctx.rid, up);
      return;
    }
    const ct = String(up.headers.get('content-type') || '');
    if (/text\/event-stream/i.test(ct)) {
      await pipeEventStream(up, res, {
        rid: ctx.rid,
        idleTimeoutMs: CONFIG.streamIdleTimeoutMs,
        maxMs: CONFIG.streamMaxMs,
        abortCtrl: ctrl,
      });
      return;
    }
    const text = await readAll(up.body, CONFIG.maxUpstreamBodyBytes);
    const outHeaders = { 'x-request-id': ctx.rid };
    const uh = up.headers.get('x-request-id');
    if (uh) outHeaders['x-upstream-request-id'] = uh;
    res.writeHead(up.status, {
      'content-type': ct || 'application/json; charset=utf-8',
      'content-length': Buffer.byteLength(text),
      'access-control-allow-origin': CONFIG.corsOrigin,
      ...outHeaders,
    });
    res.end(text);
    return;
  }
}


/* ------------------------------------------------------------------ */
/* Request routing                                                     */
/* ------------------------------------------------------------------ */

const KNOWN_POST_PATHS = new Set([
  '/v1/chat/completions', '/v1/completions', '/v1/embeddings',
  '/v1/rerank', '/v1/moderations', '/v1/responses',
]);

function normalizePath(pathname) {
  let p = pathname.replace(/\/+$/, '') || '/';
  if (p === '/v1') p = '/';
  if (!p.startsWith('/v1') && KNOWN_POST_PATHS.has('/v1' + p)) p = '/v1' + p;
  return p;
}

async function handler(req, res) {
  const rid = typeof req.headers['x-request-id'] === 'string' && req.headers['x-request-id']
    ? req.headers['x-request-id'].slice(0, 128)
    : genId('req_');
  res.setHeader('x-request-id', rid);
  const started = Date.now();

  const abort = new AbortController();
  req.__abortSignal = abort.signal;
  res.on('close', () => {
    if (!res.writableEnded) abort.abort(new Error('client disconnected'));
  });

  let url;
  try { url = new URL(req.url, 'http://localhost'); }
  catch { return sendError(res, 400, 'invalid request URL', 'invalid_request_error', null, rid); }

  const path = normalizePath(url.pathname);
  // Preflight: echo the headers/methods the browser asks for so any
  // client platform (even ones with unknown custom headers) is allowed.
  const requestedHeaders = String(req.headers['access-control-request-headers'] || '')
    .split(',').map((s) => s.trim()).filter(Boolean).join(', ');
  const corsHeaders = {
    'access-control-allow-origin': CONFIG.corsOrigin,
    'access-control-allow-methods': 'GET, POST, OPTIONS, PUT, DELETE',
    'access-control-allow-headers': requestedHeaders ||
      'authorization, content-type, x-api-key, x-proxy-token, x-proxy-key, x-request-id, x-model, x-nvidia-*',
    'access-control-expose-headers': 'x-request-id, retry-after, x-upstream-request-id',
    'access-control-max-age': '86400',
  };

  try {
    if (req.method === 'OPTIONS') {
      res.writeHead(204, corsHeaders);
      res.end();
      return;
    }

    if (req.method === 'GET' && (path === '/health' || path === '/healthz')) {
      return sendJSON(res, 200, { status: 'ok' }, corsHeaders);
    }

    // --- door auth: PROXY_TOKEN guards everything except CORS + /health ---
    if (CONFIG.proxyToken) {
      const provided = firstString(
        req.headers['x-proxy-token'],
        req.headers['x-proxy-key'],
        url.searchParams.get('token'),
      );
      if (!safeEqual(provided, CONFIG.proxyToken)) {
        log('warn', 'rejected request: bad or missing proxy token', {
          rid, ip: clientIp(req), path,
        });
        return sendJSON(res, 401, {
          error: {
            message: 'Missing or invalid proxy token. Send it in the x-proxy-token header.',
            type: 'proxy_authentication_error',
            code: 'invalid_proxy_token',
            param: null,
          },
        }, { 'x-request-id': rid, 'www-authenticate': 'Bearer' });
      }
    }

    // --- per-IP rate limit ---
    const retryAfter = checkRateLimit(clientIp(req));
    if (retryAfter != null) {
      return sendJSON(res, 429, {
        error: {
          message: `Rate limit exceeded. Retry in ${retryAfter}s.`,
          type: 'rate_limit_error',
          code: 'rate_limited',
          param: null,
        },
      }, { 'x-request-id': rid, 'retry-after': String(retryAfter) });
    }

    if (req.method === 'GET' && path === '/') {
      return sendJSON(res, 200, {
        name: 'nim-proxy',
        endpoints: ['/v1/chat/completions', '/v1/completions', '/v1/embeddings', '/v1/rerank', '/v1/models', '/health'],
        upstream: CONFIG.baseUrl,
        defaultModel: CONFIG.defaultModel || null,
        keySource: CONFIG.envApiKey ? 'env-or-request' : 'request-only',
      }, corsHeaders);
    }

    const isPost = req.method === 'POST';
    const isGet = req.method === 'GET';
    const wantsChat = isPost && path === '/v1/chat/completions';
    const wantsPassthrough = (isPost && KNOWN_POST_PATHS.has(path)) ||
      (isGet && (path === '/v1/models' || path === '/v1/models/'));

    if (!wantsChat && !wantsPassthrough) {
      return sendError(res, 404, `unknown endpoint: ${req.method} ${path}`, 'invalid_request_error', null, rid);
    }

    // read + parse body
    let raw = '';
    if (isPost) {
      raw = await readBody(req, CONFIG.maxBodyBytes);
    }
    let body = null;
    if (raw) {
      try { body = JSON.parse(raw); }
      catch { return sendError(res, 400, 'request body is not valid JSON', 'invalid_request_error', null, rid); }
      if (!body || typeof body !== 'object' || Array.isArray(body)) {
        return sendError(res, 400, 'request body must be a JSON object', 'invalid_request_error', null, rid);
      }
    } else if (isPost) {
      body = {};
    }

    const apiKey = resolveApiKey(req, url, body);
    if (!apiKey) {
      return sendError(res, 401,
        'No API key provided. Set NVIDIA_API_KEY or pass api_key in the body, ?api_key=, x-api-key or Authorization header.',
        'authentication_error', 'missing_api_key', rid);
    }

    if (wantsChat) {
      if (url.searchParams.get('stream') === 'true') body.stream = true;
      if (url.searchParams.get('stream') === 'false') body.stream = false;
      const model = resolveModel(req, url, body);
      if (!model) {
        return sendError(res, 400,
          'No model specified. Pass "model" in the body or set DEFAULT_MODEL.',
          'invalid_request_error', 'missing_model', rid);
      }
      body.model = model;
      body = sanitizeBody(body);
      body = translateThinkingEffort(body);

      const ctx = { rid, apiKey, started };
      await handleChatCompletions(req, res, url, body, ctx);
      return;
    }

    // passthrough: strip proxy-only fields from body, forward query minus secrets
    if (body) body = sanitizeBody(body);
    const ctx = { rid, apiKey, started };
    await handlePassthrough(req, res, url, path, ctx, body ? JSON.stringify(body) : (raw || undefined));
  } catch (err) {
    const status = err instanceof HttpError ? err.status : 500;
    const type = err instanceof HttpError ? err.type : 'proxy_error';
    log(status >= 500 ? 'error' : 'warn', 'request failed', {
      rid, path, status, err: msg2(err),
    });
    if (!res.headersSent) sendError(res, status, msg2(err) || 'internal error', type, null, rid);
    else if (!res.writableEnded) res.end();
  } finally {
    log('debug', 'request done', { rid, ms: Date.now() - started });
  }
}

/* ------------------------------------------------------------------ */
/* Server bootstrap                                                    */
/* ------------------------------------------------------------------ */

function createServer() {
  return http.createServer((req, res) => {
    handler(req, res).catch((err) => {
      log('error', 'unhandled', { err: msg2(err) });
      try { if (!res.headersSent) sendError(res, 500, 'internal error'); else if (!res.writableEnded) res.end(); } catch { /* ignore */ }
    });
  });
}

function startServer() {
  const server = createServer();
  server.keepAliveTimeout = 65000;
  server.headersTimeout = 70000;
  server.requestTimeout = 0; // we manage timeouts ourselves
  startRateLimitSweeper();

  // a crash in one request must not take the whole proxy down
  process.on('uncaughtException', (err) => {
    log('error', 'uncaught exception (proxy keeps running)', { err: msg2(err), stack: err && err.stack });
  });
  process.on('unhandledRejection', (err) => {
    log('error', 'unhandled rejection (proxy keeps running)', { err: msg2(err) });
  });

  server.on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
      log('error', `port ${CONFIG.port} is already in use — set PORT to another value`);
      process.exit(1);
    }
    log('error', 'server error', { err: msg2(err) });
  });

  server.listen(CONFIG.port, CONFIG.host, () => {
    log('info', 'nim-proxy listening', {
      url: `http://${CONFIG.host === '0.0.0.0' ? 'localhost' : CONFIG.host}:${CONFIG.port}`,
      upstream: CONFIG.baseUrl,
      key: maskKey(CONFIG.envApiKey),
      model: CONFIG.defaultModel || '(per-request)',
      doorAuth: CONFIG.proxyToken ? 'PROXY_TOKEN' : 'open',
      rateLimit: CONFIG.rateLimitRpm ? `${CONFIG.rateLimitRpm}/min/ip` : 'off',
      node: process.version,
    });
  });

  const shutdown = (sig) => {
    log('info', 'shutting down', { sig });
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(0), 3000).unref();
  };
  process.on('SIGINT', () => shutdown('SIGINT'));
  process.on('SIGTERM', () => shutdown('SIGTERM'));
  return server;
}

if (require.main === module) startServer();

module.exports = {
  createServer,
  startServer,
  CONFIG,
  extractToolCalls,
  extractToolCallsMore,
  classifyCallLike,
  repairJSONString,
  normalizeChatResponse,
  ChatStreamAssembler,
  createSSEParser,
  translateThinkingEffort,
};

