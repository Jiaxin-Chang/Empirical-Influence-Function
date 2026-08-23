export const SUBTYPE_COLORS: Record<string, string> = {
  route: '#6d28d9',
  bracket: '#2563eb',
  defuse: '#16a34a',
  call: '#ca8a04',
  return: '#db2777',
  type: '#7c3aed',
  dataflow: '#0891b2',
  semantic: '#ea580c',
  api: '#4f46e5',
}

export const SUBTYPE_LABELS: Record<string, string> = {
  route: '注意力路由（续训 saliency）',
  bracket: '括号/定界符配对',
  defuse: '变量声明 → 使用点',
  call: '被调函数 → 实参',
  return: 'return → 返回表达式',
  type: '类型标注 ↔ 变量名',
  dataflow: '值流（非同名绑定）',
  semantic: '语法配对关键词',
  api: '库用法配对',
}

export type Edge = {
  src: number
  dst: number
  subtype: string
  weight?: number
  contrib?: 'source' | 'user_add' | 'user_bump' | 'llm_auto' | string
  reason?: string
}

export type SampleSummary = {
  index: number
  uid: string
  language: string
  raw_id: string
  length: number
  n_edges: number
  in_continue?: boolean
}

export type SampleDetail = {
  index: number
  uid: string
  language: string
  raw_id: string
  tokens: string[]
  input_ids: number[]
  answer_start: number
  attention_edges: Edge[]
  n_continue_edges?: number
  annotation_meta: Record<string, unknown>
  subtypes: string[]
  in_continue?: boolean
  sample_key?: string
  continue_path?: string | null
  corpus_line?: number | null
  corpus_path?: string | null
  corpus_mode?: boolean
}

export type SaliencyHit = { src: number; score: number }

function corpusQuery(path?: string | null): string {
  return path?.trim() ? `?corpusPath=${encodeURIComponent(path.trim())}` : ''
}

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers || {}),
    },
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail || JSON.stringify(body)
    } catch {
      /* ignore */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () =>
    jsonFetch<{
      ok: boolean
      data_path: string | null
      n_samples: number
      continue_path: string | null
      n_continue: number
      corpus_path?: string | null
      n_corpus?: number
      write_mode: string
      subtypes: string[]
      saliency_available: boolean
    }>('/api/health'),

  open: (path: string) =>
    jsonFetch<{ path: string; n_samples: number }>('/api/open', {
      method: 'POST',
      body: JSON.stringify({ path }),
    }),

  listSamples: (q: string, offset = 0, limit = 40) =>
    jsonFetch<{ items: SampleSummary[]; total_approx: number }>(
      `/api/samples?q=${encodeURIComponent(q)}&offset=${offset}&limit=${limit}`,
    ),

  getSample: (idx: number) => jsonFetch<SampleDetail>(`/api/sample/${idx}`),

  getCorpusSample: (line: number, opts?: { corpusPath?: string | null }) =>
    jsonFetch<SampleDetail>(`/api/corpus/sample/${line}${corpusQuery(opts?.corpusPath)}`),

  saliency: (idx: number, target: number, topK = 6) =>
    jsonFetch<{
      target: number
      top: SaliencyHit[]
      available?: boolean
      message?: string
    }>(`/api/sample/${idx}/saliency/${target}?top_k=${topK}`),

  corpusSaliency: (line: number, target: number, topK = 6, opts?: { corpusPath?: string | null }) => {
    const q = new URLSearchParams({ top_k: String(topK) })
    if (opts?.corpusPath?.trim()) q.set('corpusPath', opts.corpusPath.trim())
    return jsonFetch<{
      target: number
      top: SaliencyHit[]
      available?: boolean
      message?: string
    }>(`/api/corpus/sample/${line}/saliency/${target}?${q.toString()}`)
  },

  deleteCorpusEdge: (line: number, edge: Edge, opts?: { corpusPath?: string | null }) =>
    jsonFetch<{
      ok: boolean
      n_edges: number
      action?: string
      continue_path?: string
      n_continue?: number
    }>(`/api/corpus/sample/${line}/edges/delete${corpusQuery(opts?.corpusPath)}`, {
      method: 'POST',
      body: JSON.stringify(edge),
    }),

  addCorpusEdge: (
    line: number,
    edge: Edge & { source?: string; weight?: number },
    opts?: { corpusPath?: string | null },
  ) =>
    jsonFetch<{
      ok: boolean
      n_edges: number
      action?: string
      continue_path?: string
      n_continue?: number
      edge?: Edge
    }>(`/api/corpus/sample/${line}/edges/add${corpusQuery(opts?.corpusPath)}`, {
      method: 'POST',
      body: JSON.stringify(edge),
    }),

  bumpCorpusWeight: (
    line: number,
    edge: Pick<Edge, 'src' | 'dst' | 'subtype'> & { delta?: number },
    opts?: { corpusPath?: string | null },
  ) =>
    jsonFetch<{
      ok: boolean
      n_edges: number
      n_continue_edges?: number
      action?: string
      continue_path?: string
      n_continue?: number
      edge?: Edge
      old_weight?: number
      new_weight?: number
    }>(`/api/corpus/sample/${line}/edges/bump-weight${corpusQuery(opts?.corpusPath)}`, {
      method: 'POST',
      body: JSON.stringify(edge),
    }),

  deleteEdge: (idx: number, edge: Edge) =>
    jsonFetch<{
      ok: boolean
      n_edges: number
      action?: string
      continue_path?: string
      n_continue?: number
    }>(`/api/sample/${idx}/edges/delete`, {
      method: 'POST',
      body: JSON.stringify(edge),
    }),

  addEdge: (idx: number, edge: Edge & { source?: string; weight?: number }) =>
    jsonFetch<{
      ok: boolean
      n_edges: number
      action?: string
      continue_path?: string
      n_continue?: number
      edge?: Edge
    }>(`/api/sample/${idx}/edges/add`, {
      method: 'POST',
      body: JSON.stringify(edge),
    }),

  bumpWeight: (
    idx: number,
    edge: Pick<Edge, 'src' | 'dst' | 'subtype'> & { delta?: number },
  ) =>
    jsonFetch<{
      ok: boolean
      n_edges: number
      n_continue_edges?: number
      action?: string
      continue_path?: string
      n_continue?: number
      edge?: Edge
      old_weight?: number
      new_weight?: number
    }>(`/api/sample/${idx}/edges/bump-weight`, {
      method: 'POST',
      body: JSON.stringify(edge),
    }),

  autoAnnotate: (
    idx: number,
    body: {
      probe_src_token?: string
      probe_dst_token?: string
      probe_tokens?: string[]
      probe_answer_start?: number
      probe_focus_src?: number
      probe_focus_dst?: number
      probe_mid_text?: string
      probe_fim_view?: string
      probe_id?: string
      hint_train_src?: number
      hint_train_dst?: number
      focus_src?: number
      focus_dst?: number
      focus_src_token?: string
      focus_dst_token?: string
      query_mode?: string
      mid_text?: string
      max_edges?: number
    },
  ) =>
    jsonFetch<{
      ok: boolean
      n_edges: number
      n_continue_edges?: number
      proposed?: Edge[]
      raw_preview?: string
      action?: string
      continue_path?: string
      n_continue?: number
    }>(`/api/sample/${idx}/auto-annotate`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  getProbeFocus: (probeId: string) =>
    jsonFetch<{
      ok: boolean
      probe: {
        probe_tokens: string[]
        probe_answer_start: number
        probe_focus_src: number
        probe_focus_dst: number
        probe_src_token?: string
        probe_dst_token?: string
        probe_mid_text?: string | null
        query_mode?: string
      }
    }>(`/api/probe-focus-cache/${probeId}`),
}
