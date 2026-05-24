import react from '@vitejs/plugin-react-swc'
import { defineConfig, type Plugin } from 'vite'
import { readdirSync, existsSync, readFileSync } from 'fs'
import { resolve, join } from 'path'

// ── Repo root where JSON experiment files live ────────────────────────────────
const DATA_ROOT   = resolve(__dirname, '../../')
const LEGACY_DIR  = resolve(DATA_ROOT, 'legacy_by_sample')

// ─────────────────────────────────────────────────────────────────────────────
// experimentDataPlugin
//
// Dev / Preview server:
//   GET /data/index.json                            → manifest (experiments + allTokensExperiments)
//   GET /data/<filename>.json                       → individual data files from DATA_ROOT
//   GET /data/marked_code_samples.md
//
// Build:
//   emits dist/data/index.json + all data files
// ─────────────────────────────────────────────────────────────────────────────
function experimentDataPlugin(): Plugin {
  function buildManifest() {
    const experiments: {
      testIdx: number
      tokIdx: number
      hasSaliency: boolean
      hasCorrelation: boolean
    }[] = []

    const allTokensExperiments: { taskId: string; label: string; fileName: string }[] = []
    const legacySamples: { sampleId: string }[] = []

    // Read legacy_by_sample/manifest.json
    const legacyManifestPath = join(LEGACY_DIR, 'manifest.json')
    if (existsSync(legacyManifestPath)) {
      try {
        const lm = JSON.parse(readFileSync(legacyManifestPath, 'utf-8'))
        for (const s of lm.samples ?? []) {
          if (typeof s.sample_id === 'string' && /^[\w\-]+$/.test(s.sample_id)) {
            legacySamples.push({ sampleId: s.sample_id })
          }
        }
      } catch { /* ignore */ }
    }

    let files: string[] = []
    try { files = readdirSync(DATA_ROOT) } catch { /* data root not accessible */ }

    for (const f of files) {
      // Single-token mode: saliency_test{N}_tok{T}.json
      const ms = f.match(/^saliency_test(\d+)_tok(\d+)\.json$/)
      if (ms) {
        const testIdx = parseInt(ms[1], 10)
        const tokIdx  = parseInt(ms[2], 10)
        const corrName = `correlation_matching_results_test${testIdx}_tok${tokIdx}.json`
        experiments.push({
          testIdx,
          tokIdx,
          hasSaliency:    true,
          hasCorrelation: existsSync(join(DATA_ROOT, corrName)),
        })
        continue
      }

      // All-tokens mode: correlation_matching_results_{taskId}_all_tokens.json
      const ma = f.match(/^correlation_matching_results_(.+)_all_tokens\.json$/)
      if (ma) {
        allTokensExperiments.push({ taskId: ma[1], label: ma[1], fileName: f })
      }
    }

    experiments.sort((a, b) => a.testIdx - b.testIdx || a.tokIdx - b.tokIdx)
    allTokensExperiments.sort((a, b) => a.taskId.localeCompare(b.taskId))

    return {
      experiments,
      allTokensExperiments,
      legacySamples,
      hasLegacySaliency:    existsSync(join(DATA_ROOT, 'latest_saliency.json')),
      hasLegacyCorrelation: existsSync(join(DATA_ROOT, 'correlation_matching_results.json')),
      hasMarkedCode:        existsSync(join(DATA_ROOT, 'marked_code_samples.md')),
    }
  }

  function readDataFile(filename: string): string | null {
    const safe = filename.replace(/[/\\]/g, '').replace(/\.\./g, '')
    const allowed =
      /^(saliency_test\d+_tok\d+|correlation_matching_results_test\d+_tok\d+|latest_saliency|correlation_matching_results)\.json$/.test(safe) ||
      /^correlation_matching_results_.+_all_tokens\.json$/.test(safe) ||
      /^marked_code_samples\.md$/.test(safe)
    if (!allowed) return null
    const filePath = join(DATA_ROOT, safe)
    if (!existsSync(filePath)) return null
    return readFileSync(filePath, 'utf-8')
  }

  function readLegacySaliency(sampleId: string): string | null {
    if (!/^[\w\-]+$/.test(sampleId)) return null
    const filePath = join(LEGACY_DIR, sampleId, 'latest_saliency.json')
    if (!existsSync(filePath)) return null
    return readFileSync(filePath, 'utf-8')
  }

  function addMiddleware(server: { middlewares: { use: (fn: (req: any, res: any, next: () => void) => void) => void } }) {
    server.middlewares.use((req: any, res: any, next: () => void) => {
      if (req.url === '/data/index.json') {
        res.setHeader('Content-Type', 'application/json')
        res.setHeader('Cache-Control', 'no-cache')
        res.end(JSON.stringify(buildManifest()))
        return
      }
      // Legacy saliency: /data/legacy/<sampleId>/latest_saliency.json
      const legacyM = (req.url as string)?.match(/^\/data\/legacy\/([^/?]+)\/latest_saliency\.json/)
      if (legacyM) {
        const content = readLegacySaliency(decodeURIComponent(legacyM[1]))
        if (content !== null) {
          res.setHeader('Content-Type', 'application/json')
          res.setHeader('Cache-Control', 'no-cache')
          res.end(content)
          return
        }
      }
      const m = (req.url as string)?.match(/^\/data\/([^?]+)/)
      if (m) {
        const content = readDataFile(decodeURIComponent(m[1]))
        if (content !== null) {
          const isJson = m[1].endsWith('.json')
          res.setHeader('Content-Type', isJson ? 'application/json' : 'text/plain; charset=utf-8')
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

      // Single-token experiment files
      for (const exp of manifest.experiments) {
        const salName  = `saliency_test${exp.testIdx}_tok${exp.tokIdx}.json`
        const corrName = `correlation_matching_results_test${exp.testIdx}_tok${exp.tokIdx}.json`
        const salContent = readDataFile(salName)
        if (salContent) this.emitFile({ type: 'asset', fileName: `data/${salName}`, source: salContent })
        if (exp.hasCorrelation) {
          const corrContent = readDataFile(corrName)
          if (corrContent) this.emitFile({ type: 'asset', fileName: `data/${corrName}`, source: corrContent })
        }
      }

      // All-tokens experiment files
      for (const exp of manifest.allTokensExperiments) {
        const name = exp.fileName
        const content = readDataFile(name)
        if (content) this.emitFile({ type: 'asset', fileName: `data/${name}`, source: content })
      }

      if (manifest.hasLegacySaliency) {
        const c = readDataFile('latest_saliency.json')
        if (c) this.emitFile({ type: 'asset', fileName: 'data/latest_saliency.json', source: c })
      }
      if (manifest.hasLegacyCorrelation) {
        const c = readDataFile('correlation_matching_results.json')
        if (c) this.emitFile({ type: 'asset', fileName: 'data/correlation_matching_results.json', source: c })
      }
      if (manifest.hasMarkedCode) {
        const c = readDataFile('marked_code_samples.md')
        if (c) this.emitFile({ type: 'asset', fileName: 'data/marked_code_samples.md', source: c })
      }

      // Legacy saliency files
      for (const s of manifest.legacySamples) {
        const content = readLegacySaliency(s.sampleId)
        if (content) {
          this.emitFile({
            type: 'asset',
            fileName: `data/legacy/${s.sampleId}/latest_saliency.json`,
            source: content,
          })
        }
      }
    },
  }
}

export default defineConfig({
  plugins: [react(), experimentDataPlugin()],
})
