import { useCallback, useEffect, useMemo, useState, type CSSProperties, type MouseEvent } from 'react'
import {
  SUBTYPE_COLORS,
  SUBTYPE_LABELS,
  api,
  type Edge,
  type SampleDetail,
  type SampleSummary,
  type SaliencyHit,
} from './api'

type Mode = 'inspect' | 'add'

// Prefer .env (VITE_ANNOTATION_TRAIN_DATA). After backend connects, health.data_path
// overwrites this — no machine-specific absolute path in code.
const DEFAULT_DATA =
  (import.meta.env.VITE_ANNOTATION_TRAIN_DATA as string | undefined)?.trim() || ''

function displayToken(tok: string): string {
  return tok.replace(/\r/g, '␍').replace(/\n/g, '↵\n').replace(/\t/g, '⇥')
}

export default function App() {
  const [dataPath, setDataPath] = useState(DEFAULT_DATA)
  const [nSamples, setNSamples] = useState(0)
  const [query, setQuery] = useState('')
  const [list, setList] = useState<SampleSummary[]>([])
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null)
  const [sample, setSample] = useState<SampleDetail | null>(null)
  const [target, setTarget] = useState<number | null>(null)
  const [saliency, setSaliency] = useState<SaliencyHit[]>([])
  const [saliencyMsg, setSaliencyMsg] = useState<string>('')
  const [mode, setMode] = useState<Mode>('inspect')
  const [addSubtype, setAddSubtype] = useState('defuse')
  const [addSrc, setAddSrc] = useState<number | null>(null)
  const [pendingEdge, setPendingEdge] = useState<Edge | null>(null)
  const [status, setStatus] = useState('')
  const [error, setError] = useState('')
  const [saliencyAvailable, setSaliencyAvailable] = useState(false)
  const [busy, setBusy] = useState(false)
  const [listOffset, setListOffset] = useState(0)
  const [jumpIdx, setJumpIdx] = useState('0')
  const PAGE = 100

  const refreshList = useCallback(async (q: string, offset = 0, append = false) => {
    const res = await api.listSamples(q, offset, PAGE)
    setListOffset(offset)
    setList(prev => (append ? [...prev, ...res.items] : res.items))
  }, [])

  const openData = useCallback(async () => {
    setBusy(true)
    setError('')
    try {
      const res = await api.open(dataPath)
      setNSamples(res.n_samples)
      setStatus(`已打开 ${res.path}（${res.n_samples} 条）`)
      setSelectedIdx(null)
      setSample(null)
      await refreshList(query, 0, false)
      const h = await api.health()
      setSaliencyAvailable(h.saliency_available)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }, [dataPath, query, refreshList])

  const loadSample = useCallback(async (idx: number, initialTarget?: number | null) => {
    setBusy(true)
    setError('')
    try {
      const detail = await api.getSample(idx)
      setSelectedIdx(idx)
      setSample(detail)
      setTarget(initialTarget ?? null)
      setSaliency([])
      setSaliencyMsg('')
      setPendingEdge(null)
      setAddSrc(null)
      setJumpIdx(String(idx))
      const nToTarget =
        initialTarget != null
          ? detail.attention_edges.filter(e => e.dst === initialTarget).length
          : null
      const edgeNote =
        initialTarget != null
          ? ` · target @${initialTarget} · ${nToTarget} ann edges` +
            (nToTarget === 0
              ? '（该 target 无标注边，请点其它 token，例如有边的 Type 等）'
              : '')
          : ` · ${detail.attention_edges.length} edges`
      setStatus(`样本 #${idx} · ${detail.uid}${edgeNote}`)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }, [])

  useEffect(() => {
    ;(async () => {
      try {
        const h = await api.health()
        setSaliencyAvailable(h.saliency_available)
        if (h.data_path) {
          setDataPath(h.data_path)
          setNSamples(h.n_samples)
          setStatus(
            `服务已加载 ${h.data_path}（${h.n_samples} 条）` +
              (h.saliency_available
                ? ' · saliency 已启用'
                : ' · 仅展示 annotation（未开 saliency）'),
          )
          await refreshList('', 0, false)

          // Deep link from correlation-report: ?sample=0&target=123
          const params = new URLSearchParams(window.location.search)
          const sampleRaw = params.get('sample')
          const targetRaw = params.get('target')
          if (sampleRaw != null && sampleRaw !== '') {
            const sampleIdx = Number(sampleRaw)
            const targetIdx =
              targetRaw != null && targetRaw !== '' ? Number(targetRaw) : null
            if (Number.isInteger(sampleIdx) && sampleIdx >= 0) {
              await loadSample(
                sampleIdx,
                targetIdx != null && Number.isInteger(targetIdx) ? targetIdx : null,
              )
            }
          }
        }
      } catch {
        setError('无法连接后端。请先启动: python -m server.main')
      }
    })()
  }, [refreshList, loadSample])

  const relatedEdges = useMemo(() => {
    if (!sample || target == null) return [] as Edge[]
    return sample.attention_edges.filter(e => e.dst === target)
  }, [sample, target])

  const underlineMap = useMemo(() => {
    // src -> subtypes that annotate the current target
    const map = new Map<number, string[]>()
    for (const e of relatedEdges) {
      const arr = map.get(e.src) || []
      if (!arr.includes(e.subtype)) arr.push(e.subtype)
      map.set(e.src, arr)
    }
    return map
  }, [relatedEdges])

  const saliencySet = useMemo(() => new Set(saliency.map(s => s.src)), [saliency])

  const onTokenClick = async (i: number) => {
    if (!sample || selectedIdx == null) return

    if (mode === 'add') {
      if (addSrc == null) {
        setAddSrc(i)
        setStatus(`已选 source @${i}，再点一个 token 作为 target`)
        return
      }
      if (addSrc === i) {
        setAddSrc(null)
        setStatus('已取消 source 选择')
        return
      }
      setBusy(true)
      setError('')
      try {
        await api.addEdge(selectedIdx, { src: addSrc, dst: i, subtype: addSubtype })
        setAddSrc(null)
        await loadSample(selectedIdx)
        setTarget(i)
        setStatus(`已添加 ${addSubtype}: ${addSrc} → ${i}`)
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e))
      } finally {
        setBusy(false)
      }
      return
    }

    // inspect mode: select target + fetch saliency + show related annotations
    setTarget(i)
    setPendingEdge(null)
    setSaliency([])
    setSaliencyMsg('')
    try {
      const res = await api.saliency(selectedIdx, i, 6)
      if (res.available === false) {
        setSaliencyMsg(res.message || 'Saliency 未启用（启动时加 --model）')
      } else {
        setSaliency(res.top || [])
      }
    } catch (e) {
      setSaliencyMsg(e instanceof Error ? e.message : String(e))
    }
  }

  const onUnderlinedClick = (e: MouseEvent, srcIdx: number) => {
    if (mode !== 'inspect' || target == null) return
    e.stopPropagation()
    const edges = relatedEdges.filter(x => x.src === srcIdx)
    if (edges.length >= 1) {
      setPendingEdge(edges[0])
    }
  }

  const deletePending = async () => {
    if (!pendingEdge || selectedIdx == null) return
    const edge = pendingEdge
    setBusy(true)
    setError('')
    try {
      await api.deleteEdge(selectedIdx, edge)
      setPendingEdge(null)
      await loadSample(selectedIdx)
      setTarget(edge.dst)
      setStatus(`已删除 ${edge.subtype}: ${edge.src} → ${edge.dst}`)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const underlineStyle = (subtypes: string[]): CSSProperties | undefined => {
    if (!subtypes.length) return undefined
    // CSS can only show one underline color; prefer first, tooltip lists all.
    // For multiple, use the first subtype color; title shows full list.
    return {
      textDecorationLine: 'underline',
      textDecorationColor: SUBTYPE_COLORS[subtypes[0]] || '#ef4444',
      textDecorationThickness: subtypes.length > 1 ? '3px' : '2.5px',
      textDecorationStyle: subtypes.length > 1 ? 'double' : 'solid',
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Train Annotation Viewer</h1>
        <p>
          加载带标注的 train JSONL，点 target token 看 top-6 saliency（蓝底）与相关 annotation（彩色下划线）。
          可删除 / 新增标注并写回原文件。
        </p>
      </header>

      <div className="toolbar">
        <label className="grow">
          数据文件路径
          <input
            type="text"
            value={dataPath}
            onChange={ev => setDataPath(ev.target.value)}
          />
        </label>
        <button type="button" onClick={openData} disabled={busy}>
          打开 / 刷新
        </button>
        <label>
          搜索 uid
          <input
            type="text"
            value={query}
            onChange={ev => setQuery(ev.target.value)}
            onKeyDown={ev => {
              if (ev.key === 'Enter') void refreshList(query)
            }}
          />
        </label>
        <button
          type="button"
          className="secondary"
          onClick={() => void refreshList(query, 0, false)}
          disabled={busy}
        >
          搜索
        </button>
        <label>
          跳转 index
          <input
            type="number"
            min={0}
            max={Math.max(0, nSamples - 1)}
            value={jumpIdx}
            onChange={ev => setJumpIdx(ev.target.value)}
            style={{ width: 100 }}
          />
        </label>
        <button
          type="button"
          className="secondary"
          disabled={busy || !nSamples}
          onClick={() => {
            const idx = Number(jumpIdx)
            if (!Number.isFinite(idx) || idx < 0 || idx >= nSamples) {
              setError(`index 需在 0..${nSamples - 1}`)
              return
            }
            void loadSample(idx)
          }}
        >
          打开
        </button>
        <span className="status">{nSamples ? `${nSamples} samples` : ''}</span>
      </div>

      {(status || error) && (
        <div className={`status ${error ? 'error' : ''}`}>{error || status}</div>
      )}

      <div className="layout">
        <aside className="panel">
          <h2>
            训练样本
            <span className="hint" style={{ fontWeight: 400, marginLeft: 8 }}>
              已显示 {list.length} / {nSamples || '?'}
            </span>
          </h2>
          <div className="sampleList">
            {list.map(item => (
              <button
                key={item.index}
                type="button"
                className={`sampleItem ${selectedIdx === item.index ? 'active' : ''}`}
                onClick={() => void loadSample(item.index)}
              >
                <div className="uid">#{item.index} {item.uid}</div>
                <div className="meta">
                  {item.language} · len={item.length} · edges={item.n_edges}
                </div>
              </button>
            ))}
            {!list.length && <p className="hint">打开数据文件后显示列表</p>}
          </div>
          <div className="addRow" style={{ marginTop: 8 }}>
            <button
              type="button"
              className="secondary"
              style={{
                padding: '6px 12px',
                borderRadius: 6,
                border: '1px solid var(--line)',
                background: '#fff',
                cursor: 'pointer',
              }}
              disabled={busy || (nSamples > 0 && list.length >= nSamples)}
              onClick={() => void refreshList(query, listOffset + PAGE, true)}
            >
              加载更多 (+{PAGE})
            </button>
          </div>
        </aside>

        <main className="panel">
          {!sample ? (
            <p className="hint">选择左侧一条训练数据开始查看。</p>
          ) : (
            <>
              <div className="modeBar">
                <button
                  type="button"
                  className={mode === 'inspect' ? 'active' : ''}
                  onClick={() => {
                    setMode('inspect')
                    setAddSrc(null)
                  }}
                >
                  查看 / 删除
                </button>
                <button
                  type="button"
                  className={mode === 'add' ? 'active' : ''}
                  onClick={() => {
                    setMode('add')
                    setPendingEdge(null)
                  }}
                >
                  添加标注
                </button>
                {mode === 'add' && (
                  <label style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                    类型
                    <select
                      value={addSubtype}
                      onChange={ev => setAddSubtype(ev.target.value)}
                    >
                      {Object.keys(SUBTYPE_COLORS).map(k => (
                        <option key={k} value={k}>
                          {k} — {SUBTYPE_LABELS[k]}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
              </div>

              <div className="legend">
                {Object.entries(SUBTYPE_COLORS).map(([k, c]) => (
                  <span key={k} style={{ color: c }}>
                    <i className="swatch" style={{ color: c }} />
                    {k}
                  </span>
                ))}
                <span style={{ color: '#3b82f6' }}>
                  <i className="swatch" style={{ borderBottomColor: '#3b82f6', borderBottomWidth: 8, height: 0 }} />
                  saliency top-6
                </span>
              </div>

              <p className="hint">
                {mode === 'inspect'
                  ? `点击 token 设为 target。当前 target: ${target ?? '无'}（指向它的标注边: ${relatedEdges.length}）。${
                      saliencyAvailable ? '' : '（未启用模型 saliency，不影响看标注下划线）'
                    } ${saliencyMsg}`
                  : `添加模式：先点 source${addSrc != null ? `（已选 @${addSrc}）` : ''}，再点 target，类型=${addSubtype}`}
              </p>

              <div className="codeWrap">
                <pre className="codeBlock">
                  <code>
                    {sample.tokens.map((tok, i) => {
                      const subs = underlineMap.get(i) || []
                      const isTarget = target === i
                      const isSal = saliencySet.has(i) && !isTarget
                      const isAddSrc = mode === 'add' && addSrc === i
                      const classes = [
                        'tok',
                        i >= sample.answer_start ? 'responseZone' : 'promptZone',
                        isTarget || isAddSrc ? 'target' : '',
                        isSal ? 'saliency' : '',
                      ]
                        .filter(Boolean)
                        .join(' ')
                      const titleParts = [
                        `@${i}`,
                        JSON.stringify(tok),
                        subs.length ? `ann: ${subs.join(',')}` : '',
                        isSal
                          ? `saliency=${saliency.find(s => s.src === i)?.score.toFixed(4)}`
                          : '',
                      ].filter(Boolean)
                      return (
                        <span
                          key={i}
                          className={classes}
                          style={underlineStyle(subs)}
                          title={titleParts.join(' · ')}
                          onClick={ev => {
                            // 点击已下划线的 source：弹出删除，不切换 target
                            if (subs.length && mode === 'inspect' && target != null && i !== target) {
                              onUnderlinedClick(ev, i)
                              return
                            }
                            void onTokenClick(i)
                          }}
                        >
                          {displayToken(tok)}
                        </span>
                      )
                    })}
                  </code>
                </pre>
              </div>

              <div className="sideActions">
                {target != null && (
                  <div className="card">
                    <h3>
                      Target @{target} 的标注边（{relatedEdges.length}）
                    </h3>
                    {!relatedEdges.length && <p className="hint">该 target 没有 annotation edge</p>}
                    {relatedEdges.map(e => (
                      <div className="edgeRow" key={`${e.src}-${e.dst}-${e.subtype}`}>
                        <span
                          style={{
                            color: SUBTYPE_COLORS[e.subtype] || '#333',
                            fontWeight: 600,
                          }}
                        >
                          {e.subtype}
                        </span>
                        <span>
                          src @{e.src}{' '}
                          <code>{JSON.stringify(sample.tokens[e.src] ?? '')}</code>
                          {' → '}
                          dst @{e.dst}
                        </span>
                        <button
                          type="button"
                          className="danger"
                          onClick={() => setPendingEdge(e)}
                        >
                          删除
                        </button>
                      </div>
                    ))}
                  </div>
                )}

                {pendingEdge && (
                  <div className="card">
                    <h3>确认删除标注</h3>
                    <p className="hint">
                      将从原始 JSONL 删除：{pendingEdge.subtype}{' '}
                      {pendingEdge.src} → {pendingEdge.dst}
                    </p>
                    <div className="addRow">
                      <button type="button" onClick={() => void deletePending()} disabled={busy}>
                        确认删除并写回文件
                      </button>
                      <button
                        type="button"
                        className="secondary"
                        style={{
                          alignSelf: 'end',
                          padding: '8px 14px',
                          borderRadius: 6,
                          border: '1px solid var(--line)',
                          background: '#fff',
                          cursor: 'pointer',
                        }}
                        onClick={() => setPendingEdge(null)}
                      >
                        取消
                      </button>
                    </div>
                  </div>
                )}

                {saliency.length > 0 && (
                  <div className="card">
                    <h3>Top-{saliency.length} saliency sources</h3>
                    {saliency.map(s => (
                      <div className="edgeRow" key={s.src}>
                        <span>@{s.src}</span>
                        <code>{JSON.stringify(sample.tokens[s.src] ?? '')}</code>
                        <span style={{ marginLeft: 'auto' }}>{s.score.toFixed(4)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  )
}
