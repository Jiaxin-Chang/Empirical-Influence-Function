import { useEffect, useMemo, useState } from 'react';
import { scaleToUnit } from '../../utils';
import styles from './ModelCompareView.module.css';

// ─── Types ────────────────────────────────────────────────────────────────────

interface SaliencyItem {
    index: number;
    saliency: number[];
}

interface LegacyReport {
    target_test_sample: {
        before: {
            full_tokens: string[];
            start_index: number;
            saliency_list: SaliencyItem[];
        };
    };
}

export interface ModelCompareMeta {
    models: { slug: string; name: string }[];
    sampleIds: string[];
    oursSlug: string;
}

// ─── Token decoder (GPT-2 byte map) ──────────────────────────────────────────

function buildByteDecodeMap(): Map<number, number> {
    const bs: number[] = [
        ...Array.from({ length: 0x7e - 0x21 + 1 }, (_, i) => i + 0x21),
        ...Array.from({ length: 0xac - 0xa1 + 1 }, (_, i) => i + 0xa1),
        ...Array.from({ length: 0xff - 0xae + 1 }, (_, i) => i + 0xae),
    ];
    const cs = [...bs];
    let n = 0;
    for (let b = 0; b < 256; b++) {
        if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
    }
    const map = new Map<number, number>();
    bs.forEach((b, i) => map.set(cs[i], b));
    return map;
}

const BYTE_MAP = buildByteDecodeMap();
const TEXT_DECODER = new TextDecoder('utf-8', { fatal: false });

function decodeToken(t: string): string {
    const bytes = new Uint8Array(t.length);
    for (let i = 0; i < t.length; i++) {
        const b = BYTE_MAP.get(t.charCodeAt(i));
        if (b === undefined) return t;
        bytes[i] = b;
    }
    return TEXT_DECODER.decode(bytes);
}

// ─── Saliency color ──────────────────────────────────────────────────────────

function saliencyStyle(score: number): React.CSSProperties {
    if (score < 0.01) return {};
    const a = 0.15 + score * 0.75;
    return {
        background: `rgba(251, 191, 36, ${a.toFixed(2)})`,
        color: score > 0.45 ? '#78350f' : '#92400e',
        fontWeight: score > 0.6 ? 700 : undefined,
    };
}

// ─── Single panel ─────────────────────────────────────────────────────────────

interface PanelProps {
    title: string;
    accentColor: string;
    report: LegacyReport | null;
    loading: boolean;
    loadError: boolean;
    selectedTargetIdx: number | null;
    onTargetSelect: (idx: number | null) => void;
}

function SaliencyPanel({
    title,
    accentColor,
    report,
    loading,
    loadError,
    selectedTargetIdx,
    onTargetSelect,
}: PanelProps) {
    const before = report?.target_test_sample?.before ?? null;
    const fullTokens = before?.full_tokens ?? [];
    const startIndex = before?.start_index ?? fullTokens.length;
    const saliencyList = before?.saliency_list ?? [];

    const targetSet = useMemo(() => new Set(saliencyList.map(s => s.index)), [saliencyList]);

    const saliencyMap = useMemo(() => {
        const m = new Map<number, number[]>();
        for (const item of saliencyList) {
            m.set(item.index, scaleToUnit(item.saliency, { gamma: 0.5 }));
        }
        return m;
    }, [saliencyList]);

    const activeScores = selectedTargetIdx !== null ? (saliencyMap.get(selectedTargetIdx) ?? null) : null;

    const topSources = useMemo(() => {
        if (!activeScores) return [];
        return activeScores
            .map((score, idx) => ({ idx, score, token: fullTokens[idx] ?? '' }))
            .filter(x => x.score > 0.05 && x.idx !== selectedTargetIdx)
            .sort((a, b) => b.score - a.score)
            .slice(0, 10);
    }, [activeScores, fullTokens, selectedTargetIdx]);

    return (
        <div className={styles.panel}>
            {/* Panel title */}
            <div className={styles.panelHeader} style={{ borderLeftColor: accentColor }}>
                <span className={styles.panelTitle}>{title}</span>
                {before && (
                    <span className={styles.panelMeta}>
                        {fullTokens.length} tokens · {saliencyList.length} analyzed
                    </span>
                )}
            </div>

            {loading && <div className={styles.panelState}>Loading…</div>}
            {loadError && <div className={styles.panelState}>Failed to load.</div>}
            {!loading && !loadError && !before && <div className={styles.panelState}>No data.</div>}

            {!loading && !loadError && before && (
                <>
                    {/* Token display */}
                    <pre className={styles.codeBlock}>
                        {fullTokens.map((tok, i) => {
                            const isPrompt = i < startIndex;
                            const isTarget = !isPrompt && targetSet.has(i);
                            const isSelected = i === selectedTargetIdx;
                            const score = activeScores ? (activeScores[i] ?? 0) : 0;
                            const text = decodeToken(tok).replace(/\n/g, '↵\n').replace(/\t/g, '→ ');

                            let cls = styles.token;
                            if (isSelected) cls += ' ' + styles.tokenSelected;
                            else if (isPrompt) cls += ' ' + styles.tokenPrompt;
                            else if (isTarget) cls += ' ' + styles.tokenTarget;
                            else cls += ' ' + styles.tokenResponse;

                            const style = (!isSelected && !isPrompt && !isTarget)
                                ? saliencyStyle(score) : undefined;

                            return (
                                <span
                                    key={i}
                                    className={cls}
                                    style={style}
                                    title={score > 0.01 ? `saliency: ${score.toFixed(3)}` : undefined}
                                    onClick={isTarget ? () => onTargetSelect(i === selectedTargetIdx ? null : i) : undefined}
                                >
                                    {text}
                                </span>
                            );
                        })}
                    </pre>

                    {/* Top sources */}
                    {selectedTargetIdx !== null && (
                        <div className={styles.sourcesList}>
                            <div className={styles.sourcesTitle}>
                                Top sources for{' '}
                                <code className={styles.targetChip} style={{ background: accentColor + '33', color: accentColor }}>
                                    {decodeToken(fullTokens[selectedTargetIdx] ?? '')}
                                </code>
                            </div>
                            {topSources.length === 0
                                ? <span className={styles.noSources}>No significant sources.</span>
                                : topSources.map(({ idx, score, token }) => (
                                    <div key={idx} className={styles.sourceItem}>
                                        <span className={styles.sourceToken}>{decodeToken(token)}</span>
                                        <span className={styles.sourceIdx}>@{idx}</span>
                                        <div className={styles.scoreBar}>
                                            <div
                                                className={styles.scoreBarFill}
                                                style={{ width: `${(score * 100).toFixed(1)}%`, background: accentColor }}
                                            />
                                        </div>
                                        <span className={styles.scoreVal}>{score.toFixed(3)}</span>
                                    </div>
                                ))
                            }
                        </div>
                    )}
                </>
            )}
        </div>
    );
}

// ─── Main Component ───────────────────────────────────────────────────────────

interface Props {
    meta: ModelCompareMeta;
}

const OURS_COLOR     = '#6366f1';
const BASELINE_COLOR = '#f59e0b';

export function ModelCompareView({ meta }: Props) {
    const baselineModels = meta.models.filter(m => m.slug !== meta.oursSlug);

    const [selectedSampleIdx, setSelectedSampleIdx] = useState(0);
    const [selectedBaselineSlug, setSelectedBaselineSlug] = useState(baselineModels[0]?.slug ?? '');
    const [oursTargetIdx, setOursTargetIdx]   = useState<number | null>(null);
    const [baseTargetIdx, setBaseTargetIdx]   = useState<number | null>(null);

    const [oursReport, setOursReport]         = useState<LegacyReport | null>(null);
    const [oursLoading, setOursLoading]       = useState(false);
    const [oursError, setOursError]           = useState(false);
    const [baseReport, setBaseReport]         = useState<LegacyReport | null>(null);
    const [baseLoading, setBaseLoading]       = useState(false);
    const [baseError, setBaseError]           = useState(false);

    const sampleId = meta.sampleIds[selectedSampleIdx] ?? '';

    // Fetch ours when sample changes
    useEffect(() => {
        if (!sampleId || !meta.oursSlug) return;
        setOursLoading(true); setOursError(false); setOursReport(null);
        setOursTargetIdx(null);
        fetch(`/data/model-sample/${encodeURIComponent(meta.oursSlug)}/${encodeURIComponent(sampleId)}/latest_saliency.json`)
            .then(r => { if (!r.ok) throw new Error(); return r.json(); })
            .then((d: LegacyReport) => { setOursReport(d); setOursLoading(false); })
            .catch(() => { setOursError(true); setOursLoading(false); });
    }, [sampleId, meta.oursSlug]);

    // Fetch baseline when sample or baseline model changes
    useEffect(() => {
        if (!sampleId || !selectedBaselineSlug) return;
        setBaseLoading(true); setBaseError(false); setBaseReport(null);
        setBaseTargetIdx(null);
        fetch(`/data/model-sample/${encodeURIComponent(selectedBaselineSlug)}/${encodeURIComponent(sampleId)}/latest_saliency.json`)
            .then(r => { if (!r.ok) throw new Error(); return r.json(); })
            .then((d: LegacyReport) => { setBaseReport(d); setBaseLoading(false); })
            .catch(() => { setBaseError(true); setBaseLoading(false); });
    }, [sampleId, selectedBaselineSlug]);

    const oursName = meta.models.find(m => m.slug === meta.oursSlug)?.name ?? meta.oursSlug;
    const baseName = meta.models.find(m => m.slug === selectedBaselineSlug)?.name ?? selectedBaselineSlug;

    return (
        <div className={styles.root}>

            {/* ── Top controls ── */}
            <div className={styles.controls}>
                <div className={styles.controlGroup}>
                    <span className={styles.controlLabel}>Sample</span>
                    <div className={styles.btnRow}>
                        {meta.sampleIds.map((id, i) => (
                            <button
                                key={id}
                                className={`${styles.sampleBtn} ${i === selectedSampleIdx ? styles.sampleBtnActive : ''}`}
                                onClick={() => setSelectedSampleIdx(i)}
                            >
                                {id}
                            </button>
                        ))}
                    </div>
                </div>

                <div className={styles.controlGroup}>
                    <span className={styles.controlLabel}>Baseline method</span>
                    <div className={styles.btnRow}>
                        {baselineModels.map(m => (
                            <button
                                key={m.slug}
                                className={`${styles.methodBtn} ${m.slug === selectedBaselineSlug ? styles.methodBtnActive : ''}`}
                                onClick={() => { setSelectedBaselineSlug(m.slug); setBaseTargetIdx(null); }}
                            >
                                {m.name}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {/* ── Hint ── */}
            <div className={styles.hint}>
                Click a <span style={{ color: '#a6e3a1', textDecoration: 'underline dashed' }}>response token</span> (underlined green) to highlight source tokens. Each panel is independent.
            </div>

            {/* ── Two-column panels ── */}
            <div className={styles.columns}>
                <SaliencyPanel
                    title={baseName}
                    accentColor={BASELINE_COLOR}
                    report={baseReport}
                    loading={baseLoading}
                    loadError={baseError}
                    selectedTargetIdx={baseTargetIdx}
                    onTargetSelect={setBaseTargetIdx}
                />
                <SaliencyPanel
                    title={`${oursName} (Ours)`}
                    accentColor={OURS_COLOR}
                    report={oursReport}
                    loading={oursLoading}
                    loadError={oursError}
                    selectedTargetIdx={oursTargetIdx}
                    onTargetSelect={setOursTargetIdx}
                />
            </div>
        </div>
    );
}
