import react from '@vitejs/plugin-react-swc'
import { defineConfig, type Plugin } from 'vite'
import { readdirSync, existsSync, readFileSync, statSync, openSync, readSync, closeSync } from 'fs'
import { get as httpGet, type IncomingMessage, type ServerResponse } from 'http'
import { networkInterfaces } from 'os'
import { resolve, join } from 'path'

// ── Repo root where JSON experiment files live ────────────────────────────────
const DATA_ROOT          = resolve(__dirname, '../../')
const MODEL_COMPARE_DIR  = resolve(DATA_ROOT, 'legacy_by_model_sample')
const CORR_RESULTS_DIR   = resolve(DATA_ROOT, 'correlation_matching_results')
const REAL_BUNDLE_DIR    = resolve(DATA_ROOT, 'ttav_bundles_real')
const RAW_FAMILY_DIRS: Record<string, 'ce' | 'saliency'> = { raw_ce: 'ce', raw_sa: 'saliency' }
const ALL_RAW_DIRS = ['raw_ce', 'raw_sa', 'raw'] as const
const EIF_API_PORT = Number(process.env.EIF_API_PORT || 8766)
const JSONL_CHUNK = 1024 * 1024
const JSONL_LINE_CAP = 64 * 1024 * 1024

function countNonEmptyLines(filePath: string): number {
  const fd = openSync(filePath, 'r')
  const buf = Buffer.alloc(JSONL_CHUNK)
  let carry = ''
  let n = 0
  try {
    for (;;) {
      const read = readSync(fd, buf, 0, buf.length, null)
      if (read === 0) break
      carry += buf.toString('utf8', 0, read)
      let nl = carry.indexOf('\n')
      while (nl >= 0) {
        const line = carry.slice(0, nl).replace(/\r$/, '')
        carry = carry.slice(nl + 1)
        if (line.trim()) n += 1
        nl = carry.indexOf('\n')
      }
    }
    if (carry.trim()) n += 1
  } finally {
    closeSync(fd)
  }
  return n
}

function listRawJsonlFiles(): Record<string, unknown>[] {
  const out: Record<string, unknown>[] = []
  for (const [folder, family] of Object.entries(RAW_FAMILY_DIRS)) {
    const dir = join(CORR_RESULTS_DIR, folder)
    if (!existsSync(dir)) continue
    const names = readdirSync(dir).filter(f => f.toLowerCase().endsWith('.jsonl')).sort()
    for (const name of names) {
      const filePath = join(dir, name)
      let nRows = 0
      try {
        nRows = countNonEmptyLines(filePath)
      } catch {
        continue
      }
      const tag = family === 'ce' ? 'CE' : 'SA'
      out.push({
        fileName: `${folder}/${name}`,
        label: `[${tag}] ${name.replace(/\.jsonl$/i, '')}`,
        nRows,
        folder,
        reportFamily: family,
      })
    }
  }
  return out
}

function resolveRawJsonlPath(fileName: string): string | null {
  const rel = (fileName || '').replace(/\\/g, '/').replace(/^\/+/, '')
  if (!rel || rel.split('/').includes('..')) return null
  const parts = rel.split('/').filter(Boolean)
  if (parts.length === 1) {
    const name = parts[0]
    if (!name.toLowerCase().endsWith('.jsonl')) return null
    for (const folder of ALL_RAW_DIRS) {
      const cand = join(CORR_RESULTS_DIR, folder, name)
      if (existsSync(cand) && statSync(cand).isFile()) return cand
    }
    return null
  }
  if (parts.length !== 2) return null
  const folder = parts[0].toLowerCase()
  const name = parts[1]
  if (!(ALL_RAW_DIRS as readonly string[]).includes(folder)) return null
  if (!name.toLowerCase().endsWith('.jsonl')) return null
  const cand = join(CORR_RESULTS_DIR, folder, name)
  return existsSync(cand) && statSync(cand).isFile() ? cand : null
}

function listRawJsonlRows(filePath: string, previewChars = 96): Record<string, unknown>[] {
  const fd = openSync(filePath, 'r')
  const buf = Buffer.alloc(JSONL_CHUNK)
  let carry = ''
  let lineNo = 0
  const rows: Record<string, unknown>[] = []
  const consume = (raw: string) => {
    lineNo += 1
    const line = raw.replace(/\r$/, '')
    if (!line.trim()) return
    try {
      const obj = JSON.parse(line) as Record<string, unknown>
      if (!obj || typeof obj !== 'object') return
      const predict = String(obj.predict ?? obj.output ?? '')
      const label = String(obj.label ?? obj.response ?? obj.gold ?? '')
      rows.push({
        line: lineNo,
        task_id: String(obj.task_id || `row_${lineNo}`),
        predict_preview: predict.slice(0, previewChars),
        label_preview: label.slice(0, previewChars),
      })
    } catch {
      /* skip bad line */
    }
  }
  try {
    for (;;) {
      const n = readSync(fd, buf, 0, buf.length, null)
      if (n === 0) break
      carry += buf.toString('utf8', 0, n)
      let nl = carry.indexOf('\n')
      while (nl >= 0) {
        consume(carry.slice(0, nl))
        carry = carry.slice(nl + 1)
        nl = carry.indexOf('\n')
      }
      if (carry.length > JSONL_LINE_CAP) {
        throw new Error(`jsonl line exceeds ${JSONL_LINE_CAP} bytes in ${filePath}`)
      }
    }
    if (carry) consume(carry)
  } finally {
    closeSync(fd)
  }
  return rows
}

function probeEifApiServer(host: string, port: number): Promise<string | null> {
  return new Promise(resolve => {
    const req = httpGet({ host, port, path: '/api/raw-eval-files', timeout: 1500 }, res => {
      resolve(String(res.headers.server || ''))
      res.resume()
    })
    req.on('error', () => resolve(null))
    req.on('timeout', () => {
      req.destroy()
      resolve(null)
    })
  })
}

function lanIPv4s(): string[] {
  const out: string[] = []
  for (const addrs of Object.values(networkInterfaces())) {
    for (const a of addrs ?? []) {
      if (a.family !== 'IPv4' || a.internal) continue
      if (a.address.startsWith('169.254.')) continue
      out.push(a.address)
    }
  }
  return out
}

async function resolveEifApiProxyTarget(): Promise<string> {
  const loopback = await probeEifApiServer('127.0.0.1', EIF_API_PORT)
  if (loopback && loopback.includes('EIFTTAVBundleAPI')) {
    return `http://127.0.0.1:${EIF_API_PORT}`
  }
  if (loopback) {
    console.warn(
      `[eif-api] 127.0.0.1:${EIF_API_PORT} is "${loopback}", not the local EIF API ` +
        '(often VS Code / Cursor forwarding this port to a remote uvicorn). ' +
        'Looking on LAN interfaces…',
    )
  }
  for (const ip of lanIPv4s()) {
    const server = await probeEifApiServer(ip, EIF_API_PORT)
    if (server && server.includes('EIFTTAVBundleAPI')) {
      const target = `http://${ip}:${EIF_API_PORT}`
      console.warn(`[eif-api] proxying /api → ${target}`)
      return target
    }
  }
  const fallback = `http://127.0.0.1:${EIF_API_PORT}`
  console.warn(`[eif-api] no local EIFTTAVBundleAPI on :${EIF_API_PORT}; proxying ${fallback} anyway`)
  return fallback
}

/** Lightweight parse of repo-root eif_api.env (no dependency on dotenv). */
function readEifApiEnv(): Record<string, string> {
  const envPath = resolve(DATA_ROOT, 'eif_api.env')
  if (!existsSync(envPath)) return {}
  const out: Record<string, string> = {}
  for (const raw of readFileSync(envPath, 'utf-8').split(/\r?\n/)) {
    const line = raw.trim()
    if (!line || line.startsWith('#') || !line.includes('=')) continue
    const i = line.indexOf('=')
    const key = line.slice(0, i).trim()
    const value = line.slice(i + 1).trim().replace(/^['"]|['"]$/g, '')
    if (key) out[key] = value
  }
  return out
}

const EIF_ENV = readEifApiEnv()

// Prefer continue-train subset (annotation-viewer writes here), then source train.
// Env: TRAIN_GT_JSONL / VITE_TRAIN_GT_JSONL, or ANNOTATION_* from eif_api.env.
const TRAIN_GT_EDGES_JSONL = (() => {
  const fromEnv =
    process.env.TRAIN_GT_JSONL ||
    process.env.VITE_TRAIN_GT_JSONL ||
    process.env.ANNOTATION_TRAIN_DATA ||
    EIF_ENV.ANNOTATION_TRAIN_DATA ||
    EIF_ENV.EIF_TRAIN_DATA
  if (fromEnv) {
    if (fromEnv.startsWith('/') || /^[A-Za-z]:[\\/]/.test(fromEnv)) return fromEnv
    return resolve(DATA_ROOT, fromEnv)
  }
  const candidates = [
    resolve(DATA_ROOT, 'smoke_train_data.jsonl'),
    resolve(DATA_ROOT, 'smoke_train_data_oversample_llm.jsonl'),
  ]
  return candidates.find(existsSync) ?? candidates[0]
})()

const CONTINUE_GT_EDGES_JSONL = (() => {
  const fromEnv =
    process.env.ANNOTATION_CONTINUE_TRAIN_DATA ||
    process.env.VITE_CONTINUE_GT_JSONL ||
    process.env.CONTINUE_GT_JSONL ||
    EIF_ENV.ANNOTATION_CONTINUE_TRAIN_DATA
  if (fromEnv) {
    if (fromEnv.startsWith('/') || /^[A-Za-z]:[\\/]/.test(fromEnv)) return fromEnv
    return resolve(DATA_ROOT, fromEnv)
  }
  return resolve(DATA_ROOT, 'continue_annotated_subset.jsonl')
})()

/** targetIdx -> sourceIdx[] for each train sample id (line index in the jsonl). */
type TrainGtEdges = Record<string, Record<string, number[]>>

let trainGtEdgesCache: { sig: string; payload: string | null } | undefined

/** First N source-train rows used as GT underlines (reports historically use 0..4). */
const TRAIN_GT_HEAD_LINES = 5

function readJsonlHeadSync(filePath: string, maxLines: number): string[] {
  const fd = openSync(filePath, 'r')
  const buf = Buffer.alloc(JSONL_CHUNK)
  let carry = ''
  const lines: string[] = []
  try {
    while (lines.length < maxLines) {
      const n = readSync(fd, buf, 0, buf.length, null)
      if (n === 0) break
      carry += buf.toString('utf8', 0, n)
      let nl = carry.indexOf('\n')
      while (nl >= 0 && lines.length < maxLines) {
        const line = carry.slice(0, nl).replace(/\r$/, '')
        carry = carry.slice(nl + 1)
        if (line) lines.push(line)
        nl = carry.indexOf('\n')
      }
      if (carry.length > JSONL_LINE_CAP) {
        throw new Error(`jsonl line exceeds ${JSONL_LINE_CAP} bytes in ${filePath}`)
      }
    }
    if (carry.trim() && lines.length < maxLines) {
      lines.push(carry.replace(/\r$/, ''))
    }
  } finally {
    closeSync(fd)
  }
  return lines
}

function forEachJsonlLineSync(filePath: string, onLine: (line: string) => void): void {
  const fd = openSync(filePath, 'r')
  const buf = Buffer.alloc(JSONL_CHUNK)
  let carry = ''
  try {
    for (;;) {
      const n = readSync(fd, buf, 0, buf.length, null)
      if (n === 0) break
      carry += buf.toString('utf8', 0, n)
      let nl = carry.indexOf('\n')
      while (nl >= 0) {
        const line = carry.slice(0, nl).replace(/\r$/, '')
        carry = carry.slice(nl + 1)
        if (line) onLine(line)
        nl = carry.indexOf('\n')
      }
      if (carry.length > JSONL_LINE_CAP) {
        throw new Error(`jsonl line exceeds ${JSONL_LINE_CAP} bytes in ${filePath}`)
      }
    }
    if (carry.trim()) onLine(carry.replace(/\r$/, ''))
  } finally {
    closeSync(fd)
  }
}

function _edgesFromRow(row: {
  attention_edges?: { src?: unknown; dst?: unknown }[]
  qwen_annotations?: { token_i_idx?: unknown; token_j_idx?: unknown }[]
}): Record<string, number[]> {
  const byTarget: Record<string, number[]> = {}
  const edges = Array.isArray(row.attention_edges) && row.attention_edges.length > 0
    ? row.attention_edges.map(e => ({ src: e.src, dst: e.dst }))
    : (row.qwen_annotations ?? []).map(e => ({ src: e.token_i_idx, dst: e.token_j_idx }))
  for (const edge of edges) {
    const src = typeof edge.src === 'number' ? edge.src : null
    const dst = typeof edge.dst === 'number' ? edge.dst : null
    if (src === null || dst === null) continue
    const key = String(dst)
    if (!byTarget[key]) byTarget[key] = []
    byTarget[key].push(src)
  }
  for (const key of Object.keys(byTarget)) {
    byTarget[key] = [...new Set(byTarget[key])]
  }
  return byTarget
}

function buildTrainGtEdgesPayload(): string | null {
  if (!existsSync(TRAIN_GT_EDGES_JSONL)) {
    console.warn(`[train-gt-edges] missing ${TRAIN_GT_EDGES_JSONL} — red annotation underlines disabled`)
    return null
  }
  let sig = TRAIN_GT_EDGES_JSONL
  try {
    sig += `:${statSync(TRAIN_GT_EDGES_JSONL).mtimeMs}`
    if (existsSync(CONTINUE_GT_EDGES_JSONL)) {
      sig += `|${CONTINUE_GT_EDGES_JSONL}:${statSync(CONTINUE_GT_EDGES_JSONL).mtimeMs}`
    }
  } catch {
    /* ignore */
  }
  if (trainGtEdgesCache?.sig === sig) return trainGtEdgesCache.payload

  try {
    let sizeMb = 0
    try {
      sizeMb = statSync(TRAIN_GT_EDGES_JSONL).size / (1024 * 1024)
    } catch {
      /* ignore */
    }
    if (sizeMb > 64) {
      console.info(
        `[train-gt-edges] ${TRAIN_GT_EDGES_JSONL} is ${sizeMb.toFixed(0)}MB — ` +
          `streaming first ${TRAIN_GT_HEAD_LINES} lines (not readFileSync)`,
      )
    }
    const lines = readJsonlHeadSync(TRAIN_GT_EDGES_JSONL, TRAIN_GT_HEAD_LINES)
    const out: TrainGtEdges = {}
    // First five lines map to train samples 0..4 used by the current reports.
    for (let i = 0; i < Math.min(TRAIN_GT_HEAD_LINES, lines.length); i++) {
      const row = JSON.parse(lines[i]) as {
        attention_edges?: { src?: unknown; dst?: unknown }[]
        qwen_annotations?: { token_i_idx?: unknown; token_j_idx?: unknown }[]
      }
      out[String(i)] = _edgesFromRow(row)
    }

    // Overlay annotation-viewer continue subset (edits never rewrite source JSONL).
    if (existsSync(CONTINUE_GT_EDGES_JSONL)) {
      let overlaid = 0
      forEachJsonlLineSync(CONTINUE_GT_EDGES_JSONL, line => {
        try {
          const row = JSON.parse(line) as {
            source_train_index?: unknown
            attention_edges?: { src?: unknown; dst?: unknown }[]
            qwen_annotations?: { token_i_idx?: unknown; token_j_idx?: unknown }[]
          }
          const idx = typeof row.source_train_index === 'number' ? row.source_train_index : null
          if (idx === null || idx < 0 || idx > 4) return
          out[String(idx)] = _edgesFromRow(row)
          overlaid += 1
        } catch {
          /* skip bad line */
        }
      })
      console.info(
        `[train-gt-edges] loaded ${Object.keys(out).length} samples from ${TRAIN_GT_EDGES_JSONL}` +
          ` + overlay ${overlaid} from ${CONTINUE_GT_EDGES_JSONL}`,
      )
    } else {
      console.info(`[train-gt-edges] loaded ${Object.keys(out).length} samples from ${TRAIN_GT_EDGES_JSONL}`)
    }

    const payload = JSON.stringify(out)
    trainGtEdgesCache = { sig, payload }
    return payload
  } catch (err) {
    console.warn('[train-gt-edges] failed to parse GT jsonl', err)
    trainGtEdgesCache = { sig, payload: null }
    return null
  }
}

interface ModelInfo { slug: string; name: string }
interface RawModelInfo { model_slug: string; model_name?: unknown }
interface ManifestData {
  allTokensExperiments: { taskId: string; label: string; fileName: string }[]
  modelCompare: { models: ModelInfo[]; sampleIds: string[]; oursSlug: string } | null
}
interface MiddlewareServer {
  middlewares: {
    use: (fn: (req: IncomingMessage, res: ServerResponse, next: () => void) => void) => void
  }
}

function isRawModelInfo(value: unknown): value is RawModelInfo {
  return typeof value === 'object'
    && value !== null
    && typeof (value as { model_slug?: unknown }).model_slug === 'string'
}

function isSafeSegment(value: string): boolean {
  // Task / probe ids may include non-ASCII (e.g. Chinese org names in
  // Coding-CC-L2-Go_分组核心网...). Reject only path traversal / separators.
  // Old /^[\w-]+$/ treated those ids as unsafe → middleware fell through to the
  // SPA index.html, and the frontend then failed with
  // `Unexpected token '<', "<!doctype "...`.
  if (!value || value.length > 512) return false
  if (value.includes('/') || value.includes('\\') || value.includes('\0')) return false
  if (value === '.' || value === '..' || value.includes('..')) return false
  return /^[\p{L}\p{N}_.-]+$/u.test(value)
}

// ─────────────────────────────────────────────────────────────────────────────
// experimentDataPlugin
//
// Dev / Preview server:
//   GET /data/index.json                             → manifest
//   GET /data/results/<filename>.json                → correlation result files
//   GET /data/model-sample/<slug>/<sampleId>/latest_saliency.json
//
// Build:
//   emits dist/data/index.json + all data files
// ─────────────────────────────────────────────────────────────────────────────
function experimentDataPlugin(): Plugin {
  function buildManifest(): ManifestData {
    const allTokensExperiments: { taskId: string; label: string; fileName: string }[] = []

    // Read legacy_by_model_sample/manifest.json
    const modelCompare: { models: ModelInfo[]; sampleIds: string[]; oursSlug: string } | null = (() => {
      const mp = join(MODEL_COMPARE_DIR, 'manifest.json')
      if (!existsSync(mp)) return null
      try {
        const mm = JSON.parse(readFileSync(mp, 'utf-8')) as { models?: unknown[] }
        const models: ModelInfo[] = (mm.models ?? [])
          .filter((m): m is RawModelInfo => isRawModelInfo(m) && isSafeSegment(m.model_slug))
          .map(m => ({ slug: m.model_slug, name: typeof m.model_name === 'string' ? m.model_name : m.model_slug }))
        const oursSlug = models.find(m => m.slug === 'ours_graphsignal')?.slug ?? models[0]?.slug ?? ''
        const oursDir = join(MODEL_COMPARE_DIR, oursSlug)
        const sampleIds = existsSync(oursDir)
          ? readdirSync(oursDir).filter(isSafeSegment)
          : []
        return models.length > 0 && sampleIds.length > 0 ? { models, sampleIds, oursSlug } : null
      } catch { return null }
    })()

    let files: { rel: string; family: string }[] = []
    const families = ['ce', 'saliency'] as const
    for (const family of families) {
      const dir = join(CORR_RESULTS_DIR, family)
      if (!existsSync(dir)) continue
      try {
        for (const f of readdirSync(dir)) {
          if (!f.endsWith('.json') || f.includes('_prescreen')) continue
          files.push({ rel: `${family}/${f}`, family })
        }
      } catch { /* ignore */ }
    }
    // Legacy flat files still under correlation_matching_results/*.json
    try {
      for (const f of readdirSync(CORR_RESULTS_DIR)) {
        if (!f.endsWith('.json') || f.includes('_prescreen')) continue
        files.push({ rel: f, family: 'legacy' })
      }
    } catch { /* directory not present */ }

    for (const { rel, family } of files) {
      const baseName = rel.includes('/') ? rel.slice(rel.lastIndexOf('/') + 1) : rel
      const stem = baseName.slice(0, -5)
      const taskId = stem.endsWith('_all_tokens') ? stem.slice(0, -11) : stem
      const m = stem.match(/^correlation_matching_results_(.+)_all_tokens$/)
      const core = m ? m[1] : taskId
      const label = family === 'legacy' ? core : `[${family}] ${core}`
      allTokensExperiments.push({ taskId: `${family}:${taskId}`, label, fileName: rel })
    }

    allTokensExperiments.sort((a, b) => a.label.localeCompare(b.label))

    return { allTokensExperiments, modelCompare }
  }

  function readDataFile(filename: string): string | null {
    const raw = decodeURIComponent(filename).replace(/\\/g, '/').replace(/\.\./g, '')
    const parts = raw.split('/').filter(Boolean)
    if (parts.length === 1) {
      if (!parts[0].endsWith('.json')) return null
      const filePath = join(CORR_RESULTS_DIR, parts[0])
      if (!existsSync(filePath)) return null
      return readFileSync(filePath, 'utf-8')
    }
    if (parts.length === 2 && (parts[0] === 'ce' || parts[0] === 'saliency') && parts[1].endsWith('.json')) {
      const filePath = join(CORR_RESULTS_DIR, parts[0], parts[1])
      if (!existsSync(filePath)) return null
      return readFileSync(filePath, 'utf-8')
    }
    return null
  }

  function readModelSaliency(slug: string, sampleId: string): string | null {
    if (!isSafeSegment(slug) || !isSafeSegment(sampleId)) return null
    const filePath = join(MODEL_COMPARE_DIR, slug, sampleId, 'latest_saliency.json')
    if (!existsSync(filePath)) return null
    return readFileSync(filePath, 'utf-8')
  }

  function readRealBundlePayload(sampleId: string): string | null {
    if (!isSafeSegment(sampleId)) return null
    const filePath = join(REAL_BUNDLE_DIR, sampleId, 'bundle_payload.json')
    if (!existsSync(filePath)) return null
    return readFileSync(filePath, 'utf-8')
  }

  // The bundle without `embeddings`. Those vectors are ~99% of a payload — 9-10 MB
  // for a 3584-dim sample, 250 MB for a 4096-dim probe — and nothing in the
  // in-page plot reads them; they exist for TTAV's neighbour lines and refine.
  //
  // Prefer a projection.json written at build time and stream it back untouched.
  // The fallback strips the field out of the full payload, which means parsing the
  // whole document on every request: fine for a 10 MB bundle, ruinous for a 250 MB
  // one, and that cost lands on each click once probes are built on demand.
  function readRealBundleProjection(sampleId: string): string | null {
    if (!isSafeSegment(sampleId)) return null

    const slimPath = join(REAL_BUNDLE_DIR, sampleId, 'projection.json')
    if (existsSync(slimPath)) return readFileSync(slimPath, 'utf-8')

    const raw = readRealBundlePayload(sampleId)
    if (raw === null) return null
    try {
      const parsed = JSON.parse(raw) as { bundle?: Record<string, unknown> }
      if (parsed.bundle && typeof parsed.bundle === 'object') {
        delete parsed.bundle.embeddings
      }
      return JSON.stringify(parsed)
    } catch {
      return null
    }
  }

  function addMiddleware(server: MiddlewareServer) {
    server.middlewares.use((req, res, next) => {
      const reqUrl = req.url ?? ''
      const reqPath = reqUrl.split('?')[0]
      // Serve raw JSONL listing from disk so the corpus dropdown does not
      // depend on 127.0.0.1:8766 (often stolen by VS Code port-forward).
      if (reqPath === '/api/raw-eval-files') {
        res.setHeader('Content-Type', 'application/json; charset=utf-8')
        res.setHeader('Cache-Control', 'no-cache')
        res.end(JSON.stringify({ status: 'success', files: listRawJsonlFiles() }))
        return
      }
      if (reqPath === '/api/raw-eval-rows') {
        const qs = new URL(reqUrl, 'http://vite.local').searchParams
        const file = (qs.get('file') || qs.get('fileName') || '').trim()
        const filePath = resolveRawJsonlPath(file)
        res.setHeader('Content-Type', 'application/json; charset=utf-8')
        res.setHeader('Cache-Control', 'no-cache')
        if (!filePath) {
          res.statusCode = 404
          res.end(JSON.stringify({ status: 'error', message: `raw jsonl not found: ${file}` }))
          return
        }
        let rel = file.replace(/\\/g, '/').replace(/^\/+/, '')
        if (!rel.includes('/')) {
          const bits = filePath.replace(/\\/g, '/').split('/')
          rel = `${bits[bits.length - 2]}/${bits[bits.length - 1]}`
        }
        res.end(JSON.stringify({
          status: 'success',
          fileName: rel,
          rows: listRawJsonlRows(filePath),
        }))
        return
      }
      if (reqUrl === '/data/index.json') {
        res.setHeader('Content-Type', 'application/json')
        res.setHeader('Cache-Control', 'no-cache')
        res.end(JSON.stringify(buildManifest()))
        return
      }
      if (reqUrl === '/data/train-gt-edges.json') {
        const content = buildTrainGtEdgesPayload()
        if (content !== null) {
          res.setHeader('Content-Type', 'application/json')
          res.setHeader('Cache-Control', 'no-cache')
          res.end(content)
          return
        }
      }
      // Model compare: /data/model-sample/<slug>/<sampleId>/latest_saliency.json
      const modelM = reqUrl.match(/^\/data\/model-sample\/([^/?]+)\/([^/?]+)\/latest_saliency\.json/)
      if (modelM) {
        const content = readModelSaliency(decodeURIComponent(modelM[1]), decodeURIComponent(modelM[2]))
        if (content !== null) {
          res.setHeader('Content-Type', 'application/json')
          res.setHeader('Cache-Control', 'no-cache')
          res.end(content)
          return
        }
      }
      // All-tokens correlation files:
      //   /data/results/<file>.json
      //   /data/results/ce/<file>.json
      //   /data/results/saliency/<file>.json
      const resultsM = reqUrl.match(/^\/data\/results\/((?:ce|saliency)\/)?([^/?]+\.json)/)
      if (resultsM) {
        const rel = `${resultsM[1] ?? ''}${resultsM[2]}`
        const content = readDataFile(decodeURIComponent(rel))
        if (content !== null) {
          res.setHeader('Content-Type', 'application/json')
          res.setHeader('Cache-Control', 'no-cache')
          res.end(content)
          return
        }
      }
      const projectionM = reqUrl.match(/^\/data\/real-bundles\/([^/?]+)\/projection\.json/)
      if (projectionM) {
        const content = readRealBundleProjection(decodeURIComponent(projectionM[1]))
        if (content !== null) {
          res.setHeader('Content-Type', 'application/json')
          res.setHeader('Cache-Control', 'no-cache')
          res.end(content)
          return
        }
      }
      const realBundleM = (req.url as string)?.match(/^\/data\/real-bundles\/([^/?]+)\/bundle_payload\.json/)
      if (realBundleM) {
        const content = readRealBundlePayload(decodeURIComponent(realBundleM[1]))
        if (content !== null) {
          res.setHeader('Content-Type', 'application/json')
          res.setHeader('Cache-Control', 'no-cache')
          res.end(content)
          return
        }
      }
      next()
    })
  }

  return {
    name: 'experiment-data',

    configureServer:        addMiddleware,
    configurePreviewServer: addMiddleware,

    generateBundle() {
      const manifest = buildManifest()
      this.emitFile({
        type: 'asset',
        fileName: 'data/index.json',
        source: JSON.stringify(manifest, null, 2),
      })

      const trainGtEdges = buildTrainGtEdgesPayload()
      if (trainGtEdges) {
        this.emitFile({
          type: 'asset',
          fileName: 'data/train-gt-edges.json',
          source: trainGtEdges,
        })
      }

      // All-tokens experiment files
      for (const exp of manifest.allTokensExperiments) {
        const content = readDataFile(exp.fileName)
        if (content) this.emitFile({ type: 'asset', fileName: `data/results/${exp.fileName}`, source: content })
      }

      const realBundleIds = existsSync(REAL_BUNDLE_DIR)
        ? readdirSync(REAL_BUNDLE_DIR).filter(isSafeSegment)
        : []
      for (const sampleId of realBundleIds) {
        const content = readRealBundlePayload(sampleId)
        if (content) {
          this.emitFile({
            type: 'asset',
            fileName: `data/real-bundles/${sampleId}/bundle_payload.json`,
            source: content,
          })
        }
        const projection = readRealBundleProjection(sampleId)
        if (projection) {
          this.emitFile({
            type: 'asset',
            fileName: `data/real-bundles/${sampleId}/projection.json`,
            source: projection,
          })
        }
      }

      // Model compare files
      if (manifest.modelCompare) {
        for (const model of manifest.modelCompare.models) {
          for (const sampleId of manifest.modelCompare.sampleIds) {
            const content = readModelSaliency(model.slug, sampleId)
            if (content) {
              this.emitFile({
                type: 'asset',
                fileName: `data/model-sample/${model.slug}/${sampleId}/latest_saliency.json`,
                source: content,
              })
            }
          }
        }
      }
    },
  }
}

// Reverse-proxy remaining /api/* to the EIF bundle API (tokenize / prepare).
// 127.0.0.1:8766 is often a VS Code forward to a remote uvicorn — probe first.
const EIF_REPORT_PORT = Number(process.env.EIF_REPORT_PORT || 5273)

export default defineConfig(async () => {
  const eifApiTarget = await resolveEifApiProxyTarget()
  const proxy = {
    '/api': {
      target: eifApiTarget,
      changeOrigin: true,
    },
  }
  return {
    plugins: [react(), experimentDataPlugin()],
    server: { port: EIF_REPORT_PORT, strictPort: true, proxy },
    preview: { port: EIF_REPORT_PORT, strictPort: true, proxy },
  }
})
