import { useEffect, useMemo, useState } from 'react';
import styles from './NewView.module.css';

// ─── Types ────────────────────────────────────────────────────────────────────

interface TestCorrelation {
    source_token: string;
    source_token_index: number;
    target_token: string;
    target_token_index: number;
    saliency_score: number;
}

interface TrainCorrelation {
    source_token: string;
    source_token_index: number;
    target_token: string;
    target_token_index: number;
    saliency_score: number;
    response_token_offset: number;
}

interface CorrelationPair {
    id: string;
    cos_sim: number;
    coarse_cos_sim: number;
    train_sample_id: number;
    test_correlation: TestCorrelation;
    train_correlation: TrainCorrelation;
    train_context: { source_context: string[]; target_context: string[] };
    annotation: string | null;
}

interface PerTokenResult {
    target_token_index: number;
    target_token: string;
    top_correlations: TestCorrelation[];
    correlation_pairs: CorrelationPair[];
}

interface TrainSampleDetail {
    full_tokens: string[];
    answer_start_index: number;
    coarse_cos_sim: number;
    saliencies_by_token: Record<string, number[]>;
}

interface AllTokensReport {
    experiment_meta: {
        test_sample_index: number;
        mode: 'all_tokens';
        tokens_analyzed: number;
    };
    test_sample_baseline: {
        full_tokens: string[];
        correct_full_tokens: string[];
        prompt_len: number;
    };
    per_token_results: PerTokenResult[];
    train_sample_details: Record<string, TrainSampleDetail>;
}

// ─── Token helpers ────────────────────────────────────────────────────────────

function decodeToken(t: string): string {
    return t.replaceAll('Ċ', '\n').replaceAll('Ġ', ' ').replaceAll('ĉ', '  ');
}

function decodeTokens(tokens: string[]): string[] {
    return tokens.map(decodeToken);
}

const TRIVIAL_STRIPPED = new Set(['{', '}', '(', ')', '[', ']', ',', ';']);

function isTrivialToken(t: string): boolean {
    const stripped = decodeToken(t).trim();
    if (!stripped) return true;
    if (TRIVIAL_STRIPPED.has(stripped)) return true;
    if (stripped.length === 1 && !/[a-zA-Z0-9_]/.test(stripped)) return true;
    return false;
}

function cosSimilarityColor(s: number): { bg: string; fg: string } {
    if (s > 0.6) return { bg: '#dcfce7', fg: '#15803d' };
    if (s > 0.3) return { bg: '#fef9c3', fg: '#854d0e' };
    return { bg: '#fee2e2', fg: '#b91c1c' };
}

// ─── Token Renderer ───────────────────────────────────────────────────────────

type TokenState = 'normal' | 'response' | 'selected' | 'source-highlight' | 'analyzed';

function TokenSpan({
    token,
    state,
    onClick,
    title,
}: {
    token: string;
    state: TokenState;
    onClick?: () => void;
    title?: string;
}) {
    const display = token === '\n' ? '↵\n' : token === '  ' ? '→' : token;
    return (
        <span
            className={`${styles.token} ${styles[`token-${state}`]}`}
            onClick={onClick}
            title={title}
            style={{ cursor: onClick ? 'pointer' : 'default' }}
        >
            {display}
        </span>
    );
}

// ─── Code Panel (tokens display) ─────────────────────────────────────────────

function CodePanel({
    label,
    badge,
    badgeColor,
    tokens,
    promptLen,
    highlightSourceIndices,
    selectedTargetIndex,
    analyzedIndices,
    onTokenClick,
}: {
    label: string;
    badge: string;
    badgeColor: string;
    tokens: string[];
    promptLen: number;
    highlightSourceIndices?: Set<number>;
    selectedTargetIndex?: number;
    analyzedIndices?: Set<number>;
    onTokenClick?: (idx: number) => void;
}) {
    return (
        <div className={styles.codePanel}>
            <div className={styles.codePanelHeader}>
                <span className={styles.badge} style={{ background: badgeColor }}>{badge}</span>
                <span className={styles.codePanelLabel}>{label}</span>
            </div>
            <pre className={styles.codeBlock}>
                <code>
                    {tokens.map((tok, i) => {
                        const isResponse = i >= promptLen;
                        const isSelected = i === selectedTargetIndex;
                        const isSource   = highlightSourceIndices?.has(i) ?? false;
                        const isAnalyzed = analyzedIndices?.has(i) ?? false;

                        let state: TokenState = 'normal';
                        if (isSelected)  state = 'selected';
                        else if (isSource)   state = 'source-highlight';
                        else if (isAnalyzed && isResponse) state = 'analyzed';
                        else if (isResponse) state = 'response';

                        const clickable = isResponse && onTokenClick && !isTrivialToken(tok);
                        return (
                            <TokenSpan
                                key={i}
                                token={tok}
                                state={state}
                                onClick={clickable ? () => onTokenClick(i) : undefined}
                                title={clickable ? `Token ${i}: "${decodeToken(tok)}"` : undefined}
                            />
                        );
                    })}
                </code>
            </pre>
        </div>
    );
}

// ─── Context Chip ─────────────────────────────────────────────────────────────

function ContextChip({ tokens }: { tokens: string[] }) {
    return (
        <span className={styles.contextChip}>
            {tokens.map((tok, i) => {
                const marked = tok.startsWith('→[') && tok.endsWith(']←');
                const text   = (marked ? tok.slice(2, -2) : tok).replace(/\n/g, '↵').replace(/\t/g, '⇥') || '·';
                return (
                    <span key={i} className={marked ? styles.contextTokenMarked : styles.contextToken}>
                        {text}
                    </span>
                );
            })}
        </span>
    );
}

// ─── Train Sample Detail Viewer ───────────────────────────────────────────────

function TrainSampleViewer({
    detail,
    highlightPairs,
}: {
    detail: TrainSampleDetail;
    highlightPairs: CorrelationPair[];
}) {
    const tokens = useMemo(() => decodeTokens(detail.full_tokens), [detail]);

    // Collect all source + target indices to highlight from the pairs
    const sourceIndices = useMemo(() => new Set(highlightPairs.map(p => p.train_correlation.source_token_index)), [highlightPairs]);
    const targetIndices = useMemo(() => new Set(highlightPairs.map(p => p.train_correlation.target_token_index)), [highlightPairs]);

    return (
        <div className={styles.trainSampleViewer}>
            <pre className={styles.codeBlock} style={{ fontSize: '12px', maxHeight: '260px', overflow: 'auto' }}>
                <code>
                    {tokens.map((tok, i) => {
                        const isSrc = sourceIndices.has(i);
                        const isTgt = targetIndices.has(i);
                        let state: TokenState = i >= detail.answer_start_index ? 'response' : 'normal';
                        if (isTgt) state = 'selected';
                        else if (isSrc) state = 'source-highlight';
                        return <TokenSpan key={i} token={tok} state={state} />;
                    })}
                </code>
            </pre>
        </div>
    );
}

// ─── Correlation Pair Card ────────────────────────────────────────────────────

function PairCard({ pair, detail }: { pair: CorrelationPair; detail?: TrainSampleDetail }) {
    const [expanded, setExpanded] = useState(false);
    const { bg, fg } = cosSimilarityColor(pair.cos_sim);

    return (
        <div className={styles.pairCard}>
            <div className={styles.pairCardHeader} onClick={() => setExpanded(e => !e)}>
                <span className={styles.pairId}>{pair.id}</span>

                <span className={styles.cosSim} style={{ background: bg, color: fg }}>
                    {pair.cos_sim.toFixed(4)}
                </span>

                <span className={styles.corrTag} style={{ background: '#eff6ff', borderColor: '#bfdbfe', color: '#1d4ed8' }}>
                    <span className={styles.corrLabel}>test </span>
                    <strong>{pair.test_correlation.source_token.trim() || '·'}</strong>
                    <span className={styles.arrow}> → </span>
                    <strong>{pair.test_correlation.target_token.trim() || '·'}</strong>
                </span>

                <span className={styles.corrArrow}>⇔</span>

                <span className={styles.corrTag} style={{ background: '#fffbeb', borderColor: '#fde68a', color: '#92400e' }}>
                    <span className={styles.corrLabel}>train </span>
                    <strong>{pair.train_correlation.source_token.trim() || '·'}</strong>
                    <span className={styles.arrow}> → </span>
                    <strong>{pair.train_correlation.target_token.trim() || '·'}</strong>
                    <span className={styles.offset}>+{pair.train_correlation.response_token_offset}</span>
                </span>

                <span className={styles.trainBadge}>TRAIN #{pair.train_sample_id}</span>
                <span className={styles.expandIcon}>{expanded ? '▼' : '▶'}</span>
            </div>

            {expanded && (
                <div className={styles.pairCardBody}>
                    <div className={styles.contextRow}>
                        <div>
                            <div className={styles.contextRowLabel}>Source Context</div>
                            <ContextChip tokens={pair.train_context.source_context} />
                        </div>
                        <div className={styles.contextDivider} />
                        <div>
                            <div className={styles.contextRowLabel}>Target Context</div>
                            <ContextChip tokens={pair.train_context.target_context} />
                        </div>
                    </div>
                    {detail && (
                        <TrainSampleViewer detail={detail} highlightPairs={[pair]} />
                    )}
                </div>
            )}
        </div>
    );
}

// ─── Train Sample Group ───────────────────────────────────────────────────────

function TrainSampleGroup({
    trainIdx,
    pairs,
    detail,
}: {
    trainIdx: number;
    pairs: CorrelationPair[];
    detail?: TrainSampleDetail;
}) {
    const [collapsed, setCollapsed] = useState(false);
    const bestSim = Math.max(...pairs.map(p => p.cos_sim));
    const { bg, fg } = cosSimilarityColor(bestSim);

    return (
        <div className={styles.trainGroup}>
            <div className={styles.trainGroupHeader} onClick={() => setCollapsed(c => !c)}>
                <span className={styles.trainGroupId}>TRAIN #{trainIdx}</span>
                <span className={styles.trainGroupCoarse}>coarse {(detail?.coarse_cos_sim ?? pairs[0]?.coarse_cos_sim ?? 0).toFixed(4)}</span>
                <span className={styles.trainGroupCount}>{pairs.length} pairs</span>
                <span className={styles.cosSim} style={{ background: bg, color: fg }}>best {bestSim.toFixed(4)}</span>
                <span className={styles.expandIcon} style={{ marginLeft: 'auto' }}>{collapsed ? '▶' : '▼'}</span>
            </div>
            {!collapsed && (
                <div className={styles.trainGroupBody}>
                    {detail && (
                        <div className={styles.trainFullView}>
                            <div className={styles.subLabel}>完整训练样本 — 高亮所有相关 correlation</div>
                            <TrainSampleViewer detail={detail} highlightPairs={pairs} />
                        </div>
                    )}
                    <div className={styles.pairList}>
                        {pairs.map(pair => (
                            <PairCard key={pair.id} pair={pair} detail={detail} />
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

// ─── Main NewView Component ───────────────────────────────────────────────────

export interface AllTokensExperimentMeta {
    testIdx: number;
}

interface Props {
    metas: AllTokensExperimentMeta[];
}

export function NewView({ metas }: Props) {
    const [selectedMetaIdx, setSelectedMetaIdx] = useState(0);
    const [report, setReport]     = useState<AllTokensReport | null>(null);
    const [loading, setLoading]   = useState(false);
    const [loadError, setLoadError] = useState(false);

    // Selected output token (by absolute sequence index)
    const [selectedTokIdx, setSelectedTokIdx] = useState<number | null>(null);
    // Selected test correlation (source_token_index)
    const [selectedTestCorrIdx, setSelectedTestCorrIdx] = useState<number | null>(null);
    // cos_sim filter threshold
    const [threshold, setThreshold] = useState(0.0);
    // Whether to hide pairs with cos_sim exactly 0
    const [hideZero, setHideZero] = useState(false);

    // Load report when meta selection changes
    useEffect(() => {
        if (metas.length === 0) return;
        const meta = metas[selectedMetaIdx] ?? metas[0];
        const url  = `/data/correlation_matching_results_test${meta.testIdx}_all_tokens.json`;

        setLoading(true);
        setLoadError(false);
        setReport(null);
        setSelectedTokIdx(null);
        setSelectedTestCorrIdx(null);

        fetch(url)
            .then(r => { if (!r.ok) throw new Error('fetch failed'); return r.json(); })
            .then((data: AllTokensReport) => { setReport(data); setLoading(false); })
            .catch(() => { setLoadError(true); setLoading(false); });
    }, [metas, selectedMetaIdx]);

    // Reset test correlation state when selected token changes
    useEffect(() => { setSelectedTestCorrIdx(null); }, [selectedTokIdx]);

    const modelTokens   = useMemo(() => report ? decodeTokens(report.test_sample_baseline.full_tokens) : [], [report]);
    const correctTokens = useMemo(() => report ? decodeTokens(report.test_sample_baseline.correct_full_tokens ?? []) : [], [report]);
    const promptLen     = report?.test_sample_baseline.prompt_len ?? 0;

    // Map from token index → PerTokenResult for quick lookup
    const perTokenMap = useMemo(() => {
        const m = new Map<number, PerTokenResult>();
        report?.per_token_results.forEach(r => m.set(r.target_token_index, r));
        return m;
    }, [report]);

    // Indices of analyzed output tokens (those with per_token_results)
    const analyzedIndices = useMemo(() => new Set(perTokenMap.keys()), [perTokenMap]);

    const selectedResult = selectedTokIdx !== null ? perTokenMap.get(selectedTokIdx) ?? null : null;

    // Source token highlights for the selected token
    const sourceHighlightIndices = useMemo(() => {
        if (!selectedResult) return new Set<number>();
        if (selectedTestCorrIdx !== null) return new Set([selectedTestCorrIdx]);
        return new Set(selectedResult.top_correlations.map(c => c.source_token_index));
    }, [selectedResult, selectedTestCorrIdx]);

    // Pairs to show in the right panel
    const allDisplayPairs = useMemo(() => {
        const keep = (p: CorrelationPair) =>
            p.cos_sim >= threshold && !(hideZero && p.cos_sim === 0);

        if (selectedResult) {
            if (selectedTestCorrIdx !== null) {
                return selectedResult.correlation_pairs.filter(p => 
                    keep(p) && p.test_correlation.source_token_index === selectedTestCorrIdx
                );
            }
            return selectedResult.correlation_pairs.filter(keep);
        }
        
        // Show all pairs across all analyzed tokens if no token is selected
        const all: CorrelationPair[] = [];
        report?.per_token_results.forEach(r => {
            r.correlation_pairs.forEach(p => { if (keep(p)) all.push(p); });
        });
        all.sort((a, b) => b.cos_sim - a.cos_sim);
        return all;
    }, [selectedResult, selectedTestCorrIdx, report, threshold, hideZero]);

    // Group pairs by train_sample_id
    const trainGroups = useMemo(() => {
        const map = new Map<number, CorrelationPair[]>();
        allDisplayPairs.forEach(p => {
            if (!map.has(p.train_sample_id)) map.set(p.train_sample_id, []);
            map.get(p.train_sample_id)!.push(p);
        });
        return Array.from(map.entries())
            .map(([id, pairs]) => ({ id, pairs, bestSim: Math.max(...pairs.map(p => p.cos_sim)) }))
            .sort((a, b) => b.bestSim - a.bestSim);
    }, [allDisplayPairs]);

    // ── Early states ──────────────────────────────────────────────────────────

    if (metas.length === 0) {
        return (
            <div className={styles.emptyState}>
                No all-tokens experiment files found.<br />
                Run <code>python -m src.intervention_experiment --all-tokens</code> to generate<br />
                <code>correlation_matching_results_test&#123;N&#125;_all_tokens.json</code>.
            </div>
        );
    }

    if (loading) return <div className={styles.emptyState}>Loading experiment data…</div>;
    if (loadError) return <div className={styles.emptyState}>Failed to load experiment data.</div>;
    if (!report)   return null;

    // ── Render ────────────────────────────────────────────────────────────────

    return (
        <div className={styles.root}>

            {/* ── Experiment selector ── */}
            {metas.length > 1 && (
                <div className={styles.metaSelector}>
                    <span className={styles.metaSelectorLabel}>Test Sample:</span>
                    {metas.map((m, i) => (
                        <button
                            key={i}
                            className={`${styles.metaBtn} ${i === selectedMetaIdx ? styles.metaBtnActive : ''}`}
                            onClick={() => setSelectedMetaIdx(i)}
                        >
                            test={m.testIdx}
                        </button>
                    ))}
                </div>
            )}

            {/* ── Top: Ground Truth (Full width, scrolls normally) ── */}
            <div className={styles.topPanel}>
                <CodePanel
                    label="Correct Output (Ground Truth)"
                    badge="GT"
                    badgeColor="#16a34a"
                    tokens={correctTokens}
                    promptLen={promptLen}
                />
            </div>

            {/* ── Bottom Section: Left Sticky, Right Scroll ── */}
            <div className={styles.bottomSection}>
                {/* ── Left Column: Model Output & Correlations ── */}
                <div className={styles.bottomLeft}>
                    <CodePanel
                        label="Model Output (Incorrect)"
                        badge="MODEL"
                        badgeColor="#dc2626"
                        tokens={modelTokens}
                        promptLen={promptLen}
                        highlightSourceIndices={sourceHighlightIndices}
                        selectedTargetIndex={selectedTokIdx ?? undefined}
                        analyzedIndices={analyzedIndices}
                        onTokenClick={idx => setSelectedTokIdx(prev => prev === idx ? null : idx)}
                    />

                    {selectedResult && (
                        <div className={styles.correlationList}>
                            <div className={styles.correlationListTitle}>
                                Top Correlations for "{decodeToken(selectedResult.target_token).trim()}" @ idx {selectedResult.target_token_index}
                            </div>
                            <div className={styles.correlationListItems}>
                                {selectedResult.top_correlations.slice(0, 4).map(c => (
                                    <button
                                        key={c.source_token_index}
                                        className={`${styles.corrBtn} ${c.source_token_index === selectedTestCorrIdx ? styles.corrBtnActive : ''}`}
                                        onClick={() => setSelectedTestCorrIdx(
                                            prev => prev === c.source_token_index ? null : c.source_token_index
                                        )}
                                    >
                                        <div className={styles.corrBtnLeft}>
                                            <span className={styles.corrLabel}>source token</span>
                                            <span className={styles.corrSourceTok}>{c.source_token.trim() || '·'}</span>
                                        </div>
                                        <div className={styles.corrBtnRight}>
                                            <span className={styles.corrScoreLabel}>saliency</span>
                                            <span className={styles.sourceChipSal}>{c.saliency_score.toFixed(3)}</span>
                                        </div>
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}
                </div>

                {/* ── Right Column: Training pairs ── */}
                <div className={styles.bottomRight}>
                    <div className={styles.bottomPanel}>
                        <div className={styles.bottomPanelHeader}>
                            <div className={styles.bottomPanelTitle}>
                                {selectedResult 
                                    ? (selectedTestCorrIdx !== null 
                                        ? `Training Correlations for Selected Source Token`
                                        : `Training Correlations for Target Token`)
                                    : 'All Training Correlations'}
                                <span className={styles.pairCount}>
                                    {trainGroups.length} groups · {allDisplayPairs.length} pairs
                                </span>
                            </div>
                            <div className={styles.filterRow}>
                                <span className={styles.filterLabel}>cos_sim ≥</span>
                                <input
                                    type="range" min={0} max={0.2} step={0.001} value={threshold}
                                    onChange={e => setThreshold(parseFloat(e.target.value))}
                                    className={styles.thresholdSlider}
                                />
                                <span className={styles.thresholdVal}>{threshold.toFixed(3)}</span>
                                <button
                                    onClick={() => setHideZero(v => !v)}
                                    style={{
                                        marginLeft: '12px',
                                        padding: '3px 10px',
                                        borderRadius: '6px',
                                        fontSize: '11px',
                                        cursor: 'pointer',
                                        border: `1px solid ${hideZero ? '#ef4444' : '#d1d5db'}`,
                                        background: hideZero ? '#fef2f2' : '#fff',
                                        color: hideZero ? '#b91c1c' : '#6b7280',
                                        fontWeight: hideZero ? 700 : 500,
                                        transition: 'all 0.12s',
                                    }}
                                >
                                    {hideZero ? '✗ 已隐藏 cos=0' : '隐藏 cos_sim=0'}
                                </button>
                            </div>
                        </div>

                        {trainGroups.length === 0 ? (
                            <div className={styles.emptyState} style={{ padding: '32px 0' }}>
                                No matching pairs. Try lowering the threshold.
                            </div>
                        ) : (
                            <div className={styles.trainGroupList}>
                                {trainGroups.map(({ id, pairs }) => (
                                    <TrainSampleGroup
                                        key={id}
                                        trainIdx={id}
                                        pairs={pairs}
                                        detail={report.train_sample_details[String(id)]}
                                    />
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
