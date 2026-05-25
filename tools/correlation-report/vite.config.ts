import react from '@vitejs/plugin-react-swc'
import { defineConfig, type Plugin } from 'vite'
import { readdirSync, existsSync, readFileSync } from 'fs'
import { resolve, join } from 'path'

// ── Repo root where JSON experiment files live ────────────────────────────────
const DATA_ROOT          = resolve(__dirname, '../../')
const MODEL_COMPARE_DIR  = resolve(DATA_ROOT, 'legacy_by_model_sample')
const CORR_RESULTS_DIR   = resolve(DATA_ROOT, 'correlation_matching_results')

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
  function buildManifest() {
    const allTokensExperiments: { taskId: string; label: string; fileName: string }[] = []

    // Read legacy_by_model_sample/manifest.json
    interface ModelInfo { slug: string; name: string }
    const modelCompare: { models: ModelInfo[]; sampleIds: string[]; oursSlug: string } | null = (() => {
      const mp = join(MODEL_COMPARE_DIR, 'manifest.json')
      if (!existsSync(mp)) return null
      try {
        const mm = JSON.parse(readFileSync(mp, 'utf-8'))
        const models: ModelInfo[] = (mm.models ?? [])
          .filter((m: any) => typeof m.model_slug === 'string' && /^[\w\-]+$/.test(m.model_slug))
          .map((m: any) => ({ slug: m.model_slug as string, name: m.model_name as string }))
        const oursSlug = models.find(m => m.slug === 'ours_graphsignal')?.slug ?? models[0]?.slug ?? ''
        const oursDir = join(MODEL_COMPARE_DIR, oursSlug)
        const sampleIds = existsSync(oursDir)
          ? readdirSync(oursDir).filter(d => /^[\w\-]+$/.test(d))
          : []
        return models.length > 0 && sampleIds.length > 0 ? { models, sampleIds, oursSlug } : null
      } catch { return null }
    })()

    let files: string[] = []
    try { files = readdirSync(CORR_RESULTS_DIR) } catch { /* directory not present */ }

    for (const f of files) {
      if (!f.endsWith('.json')) continue
      // Use stem as taskId; strip trailing _all_tokens if present for cleaner display
      const stem = f.slice(0, -5)
      const taskId = stem.endsWith('_all_tokens') ? stem.slice(0, -11) : stem
      allTokensExperiments.push({ taskId, label: taskId, fileName: f })
    }

    allTokensExperiments.sort((a, b) => a.taskId.localeCompare(b.taskId))

    return { allTokensExperiments, modelCompare }
  }

  function readDataFile(filename: string): string | null {
    const safe = filename.replace(/[/\\]/g, '').replace(/\.\./g, '')
    if (!safe.endsWith('.json') || safe.includes('/') || safe.includes('\\')) return null
    const filePath = join(CORR_RESULTS_DIR, safe)
    if (!existsSync(filePath)) return null
    return readFileSync(filePath, 'utf-8')
  }

  function readModelSaliency(slug: string, sampleId: string): string | null {
    if (!/^[\w\-]+$/.test(slug) || !/^[\w\-]+$/.test(sampleId)) return null
    const filePath = join(MODEL_COMPARE_DIR, slug, sampleId, 'latest_saliency.json')
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
      // Model compare: /data/model-sample/<slug>/<sampleId>/latest_saliency.json
      const modelM = (req.url as string)?.match(/^\/data\/model-sample\/([^/?]+)\/([^/?]+)\/latest_saliency\.json/)
      if (modelM) {
        const content = readModelSaliency(decodeURIComponent(modelM[1]), decodeURIComponent(modelM[2]))
        if (content !== null) {
          res.setHeader('Content-Type', 'application/json')
          res.setHeader('Cache-Control', 'no-cache')
          res.end(content)
          return
        }
      }
      // All-tokens correlation files: /data/results/<filename>.json
      const resultsM = (req.url as string)?.match(/^\/data\/results\/([^/?]+\.json)/)
      if (resultsM) {
        const content = readDataFile(decodeURIComponent(resultsM[1]))
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

      // All-tokens experiment files
      for (const exp of manifest.allTokensExperiments) {
        const content = readDataFile(exp.fileName)
        if (content) this.emitFile({ type: 'asset', fileName: `data/results/${exp.fileName}`, source: content })
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

export default defineConfig({
  plugins: [react(), experimentDataPlugin()],
})
