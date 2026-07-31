export const SUBTYPE_COLORS: Record<string, string> = {
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
  bracket: '括号/定界符配对',
  defuse: '变量声明 → 使用点',
  call: '被调函数 → 实参',
  return: 'return → 返回表达式',
  type: '类型标注 ↔ 变量名',
  dataflow: '值流（非同名绑定）',
  semantic: '语法配对关键词',
  api: '库用法配对',
}

export type Edge = { src: number; dst: number; subtype: string }

export type SampleSummary = {
  index: number
  uid: string
  language: string
  raw_id: string
  length: number
  n_edges: number
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
  annotation_meta: Record<string, unknown>
  subtypes: string[]
}

export type SaliencyHit = { src: number; score: number }

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

  saliency: (idx: number, target: number, topK = 6) =>
    jsonFetch<{
      target: number
      top: SaliencyHit[]
      available?: boolean
      message?: string
    }>(`/api/sample/${idx}/saliency/${target}?top_k=${topK}`),

  deleteEdge: (idx: number, edge: Edge) =>
    jsonFetch<{ ok: boolean; n_edges: number }>(`/api/sample/${idx}/edges/delete`, {
      method: 'POST',
      body: JSON.stringify(edge),
    }),

  addEdge: (idx: number, edge: Edge & { source?: string }) =>
    jsonFetch<{ ok: boolean; n_edges: number }>(`/api/sample/${idx}/edges/add`, {
      method: 'POST',
      body: JSON.stringify(edge),
    }),
}
