import { useEffect, useMemo, useState } from 'react';
import { scaleToUnit } from '../../utils';
import styles from './LegacySaliencyView.module.css';

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

export interface LegacySampleMeta {
    sampleId: string;
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
        if (b === undefined) return t;  // already proper Unicode
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

// ─── Token span ──────────────────────────────────────────────────────────────

function TokenSpan({
    raw,
    isPrompt,
    isTarget,
    isSelected,
    saliencyScore,
    onClick,
}: {
    raw: string;
    isPrompt: boolean;
    isTarget: boolean;
    isSelected: boolean;
    saliencyScore: number;
    onClick?: () => void;
}) {
    const text = decodeToken(raw);
    const display = text.replace(/\n/g, '↵\n').replace(/\t/g, '→ ');

    let cls = styles.token;
    if (isSelected) cls += ' ' + styles.tokenSelected;
    else if (isPrompt) cls += ' ' + styles.tokenPrompt;
    else if (isTarget) cls += ' ' + styles.tokenTarget;
    else cls += ' ' + styles.tokenResponse;

    const style = (!isSelected && !isPrompt && !isTarget) ? saliencyStyle(saliencyScore) : undefined;

    return (
        <span
            className={cls}
            style={style}
            title={isTarget ? undefined : saliencyScore > 0.01 ? `saliency: ${saliencyScore.toFixed(3)}` : undefined}
            onClick={isTarget ? onClick : undefined}
        >
            {display}
        </span>
    );
}

// ─── Main Component ───────────────────────────────────────────────────────────

interface Props {
    samples: LegacySampleMeta[];
}

export function LegacySaliencyView({ samples }: Props) {
    const [selectedSampleIdx, setSelectedSampleIdx] = useState(0);
    const [report, setReport] = useState<LegacyReport | null>(null);
    const [loading, setLoading] = useState(false);
    const [loadError, setLoadError] = useState(false);
    const [selectedTargetIdx, setSelectedTargetIdx] = useState<number | null>(null);

    // Fetch report when sample changes
    useEffect(() => {
        const meta = samples[selectedSampleIdx];
        if (!meta) return;
        setLoading(true);
        setLoadError(false);
        setReport(null);
        setSelectedTargetIdx(null);
        fetch(`/data/legacy/${encodeURIComponent(meta.sampleId)}/latest_saliency.json`)
            .then(r => { if (!r.ok) throw new Error('fetch failed'); return r.json(); })
            .then((d: LegacyReport) => { setReport(d); setLoading(false); })
            .catch(() => { setLoadError(true); setLoading(false); });
    }, [samples, selectedSampleIdx]);

    const before = report?.target_test_sample?.before ?? null;
    const fullTokens = before?.full_tokens ?? [];
    const startIndex = before?.start_index ?? fullTokens.length;
    const saliencyList = before?.saliency_list ?? [];

    // Set of target token indices that have saliency data
    const targetSet = useMemo(() => new Set(saliencyList.map(s => s.index)), [saliencyList]);

    // Map targetIdx → normalized saliency scores
    const saliencyMap = useMemo(() => {
        const m = new Map<number, number[]>();
        for (const item of saliencyList) {
            const scaled = scaleToUnit(item.saliency, { gamma: 0.5 });
            m.set(item.index, scaled);
        }
        return m;
    }, [saliencyList]);

    const activeScores = selectedTargetIdx !== null ? (saliencyMap.get(selectedTargetIdx) ?? null) : null;

    // Top source tokens for the selected target
    const topSources = useMemo(() => {
        if (!activeScores) return [];
        return activeScores
            .map((score, idx) => ({ idx, score, token: fullTokens[idx] ?? '' }))
            .filter(x => x.score > 0.05 && x.idx !== selectedTargetIdx)
            .sort((a, b) => b.score - a.score)
            .slice(0, 12);
    }, [activeScores, fullTokens, selectedTargetIdx]);

    if (samples.length === 0) {
        return (
            <div className={styles.emptyState}>
                No legacy saliency samples found in <code>legacy_by_sample/</code>.
            </div>
        );
    }

    return (
        <div className={styles.root}>

            {/* ── Sample selector ── */}
            <div className={styles.selector}>
                <span className={styles.selectorLabel}>Sample:</span>
                {samples.map((s, i) => (
                    <button
                        key={s.sampleId}
                        className={`${styles.sampleBtn} ${i === selectedSampleIdx ? styles.sampleBtnActive : ''}`}
                        onClick={() => setSelectedSampleIdx(i)}
                    >
                        {s.sampleId}
                    </button>
                ))}
            </div>

            {loading && <div className={styles.emptyState}>Loading…</div>}
            {loadError && <div className={styles.emptyState}>Failed to load saliency data.</div>}

            {!loading && !loadError && before && (
                <div className={styles.body}>

                    {/* ── Instruction ── */}
                    <div className={styles.hint}>
                        Click a <span className={styles.hintTarget}>response token</span> (underlined) to see which source tokens influenced it.
                        {selectedTargetIdx !== null && (
                            <button className={styles.clearBtn} onClick={() => setSelectedTargetIdx(null)}>
                                Clear selection
                            </button>
                        )}
                    </div>

                    {/* ── Token display ── */}
                    <div className={styles.codePanel}>
                        <div className={styles.codePanelHeader}>
                            <span className={styles.badge} style={{ background: '#6366f1' }}>TOKENS</span>
                            <span className={styles.codePanelLabel}>
                                {fullTokens.length} tokens · prompt ends at {startIndex} · {saliencyList.length} response tokens analyzed
                            </span>
                        </div>
                        <pre className={styles.codeBlock}>
                            {fullTokens.map((tok, i) => {
                                const isPrompt = i < startIndex;
                                const isTarget = !isPrompt && targetSet.has(i);
                                const isSelected = i === selectedTargetIdx;
                                const score = activeScores ? (activeScores[i] ?? 0) : 0;
                                return (
                                    <TokenSpan
                                        key={i}
                                        raw={tok}
                                        isPrompt={isPrompt}
                                        isTarget={isTarget}
                                        isSelected={isSelected}
                                        saliencyScore={score}
                                        onClick={isTarget ? () => setSelectedTargetIdx(i === selectedTargetIdx ? null : i) : undefined}
                                    />
                                );
                            })}
                        </pre>
                    </div>

                    {/* ── Top sources panel ── */}
                    {selectedTargetIdx !== null && (
                        <div className={styles.sourcesPanel}>
                            <div className={styles.sourcesPanelHeader}>
                                Top source tokens for{' '}
                                <code className={styles.targetChip}>
                                    {decodeToken(fullTokens[selectedTargetIdx] ?? '')}
                                </code>
                                {' '}(idx {selectedTargetIdx})
                            </div>
                            <div className={styles.sourcesList}>
                                {topSources.length === 0
                                    ? <span className={styles.noSources}>No significant source tokens.</span>
                                    : topSources.map(({ idx, score, token }) => (
                                        <div key={idx} className={styles.sourceItem}>
                                            <span className={styles.sourceToken}>{decodeToken(token)}</span>
                                            <span className={styles.sourceIdx}>idx {idx}</span>
                                            <div className={styles.scoreBar}>
                                                <div
                                                    className={styles.scoreBarFill}
                                                    style={{ width: `${(score * 100).toFixed(1)}%` }}
                                                />
                                            </div>
                                            <span className={styles.scoreVal}>{score.toFixed(3)}</span>
                                        </div>
                                    ))
                                }
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
