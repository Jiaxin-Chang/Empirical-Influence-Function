import { useState, useMemo } from 'react';

// ─── Types ────────────────────────────────────────────────────────────────────

type Annotation = 'correct' | 'incorrect' | 'ambiguous';
type AnnotationLabel = Annotation | null;

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

interface TrainContext {
    source_context: string[];
    target_context: string[];
}

interface CorrelationPair {
    id: string;
    cos_sim: number;
    coarse_cos_sim: number;
    train_sample_id: number;
    test_correlation: TestCorrelation;
    train_correlation: TrainCorrelation;
    train_context: TrainContext;
    annotation: AnnotationLabel;
}

interface TestBaseline {
    target_token: string;
    target_token_index: number;
    full_tokens: string[];
    top_correlations: TestCorrelation[];
}

interface ReportData {
    experiment_meta: {
        test_sample_index: number;
        target_token_index: number;
        config?: Record<string, number>;
    };
    test_sample_baseline: TestBaseline;
    correlation_pairs: CorrelationPair[];
}

interface Props {
    reportData: ReportData;
}

// ─── Annotation persistence via localStorage ──────────────────────────────────

const LS_KEY = 'eif_correlation_annotations_v2';

function loadAnnotations(): Record<string, Annotation> {
    try { return JSON.parse(localStorage.getItem(LS_KEY) || '{}'); }
    catch { return {}; }
}

function saveAnnotations(data: Record<string, Annotation>) {
    localStorage.setItem(LS_KEY, JSON.stringify(data));
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function cosSimStyle(score: number) {
    if (score > 0.6) return { bg: '#dcfce7', fg: '#15803d' };
    if (score > 0.3) return { bg: '#fef9c3', fg: '#854d0e' };
    return { bg: '#fee2e2', fg: '#991b1b' };
}

const ANNOTATION_CONFIG: Record<Annotation, { emoji: string; label: string; activeBg: string; activeFg: string; borderColor: string }> = {
    correct:   { emoji: '✓', label: 'Correct',   activeBg: '#dcfce7', activeFg: '#166534', borderColor: '#22c55e' },
    incorrect: { emoji: '✗', label: 'Incorrect', activeBg: '#fee2e2', activeFg: '#991b1b', borderColor: '#ef4444' },
    ambiguous: { emoji: '?', label: 'Ambiguous', activeBg: '#fef3c7', activeFg: '#92400e', borderColor: '#f59e0b' },
};

// ─── Sub-components ───────────────────────────────────────────────────────────

/** Renders a context window, highlighting the marked token (→[token]←) */
function ContextChip({ tokens }: { tokens: string[] }) {
    return (
        <span style={{ fontFamily: 'monospace', fontSize: '12px', display: 'inline-flex', flexWrap: 'wrap', gap: '2px', alignItems: 'center' }}>
            {tokens.map((tok, i) => {
                const isMarked = tok.startsWith('→[') && tok.endsWith(']←');
                const raw = isMarked ? tok.slice(2, -2) : tok;
                const display = raw.replace(/\n/g, '↵').replace(/\t/g, '⇥') || '·';
                return (
                    <span
                        key={i}
                        title={`token: "${raw}"`}
                        style={{
                            padding: '1px 5px',
                            borderRadius: '4px',
                            background: isMarked ? '#fde68a' : '#f1f5f9',
                            color: isMarked ? '#78350f' : '#475569',
                            fontWeight: isMarked ? 700 : 400,
                            border: isMarked ? '1.5px solid #f59e0b' : '1px solid transparent',
                        }}
                    >
                        {display}
                    </span>
                );
            })}
        </span>
    );
}

/** Annotation button with active/inactive states */
function AnnotationBtn({ type, current, onToggle }: { type: Annotation; current: AnnotationLabel; onToggle: () => void }) {
    const cfg = ANNOTATION_CONFIG[type];
    const isActive = current === type;
    return (
        <button
            onClick={onToggle}
            style={{
                padding: '4px 13px',
                borderRadius: '6px',
                border: `1.5px solid ${isActive ? cfg.borderColor : '#d1d5db'}`,
                background: isActive ? cfg.activeBg : '#fff',
                color: isActive ? cfg.activeFg : '#9ca3af',
                fontSize: '12px',
                fontWeight: isActive ? 700 : 500,
                cursor: 'pointer',
                transition: 'all 0.15s ease',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '4px',
            }}
        >
            <span>{cfg.emoji}</span>
            <span>{cfg.label}</span>
        </button>
    );
}

/** Small chip to display a correlation pair inline */
function CorrChip({ src, tgt, isTest }: { src: string; tgt: string; isTest: boolean }) {
    const s = src.trim() || '[SP]';
    const t = tgt.trim() || '[SP]';
    const [bg, fg, border] = isTest
        ? ['#eff6ff', '#1d4ed8', '#bfdbfe']
        : ['#fffbeb', '#92400e', '#fde68a'];
    return (
        <span style={{ background: bg, border: `1px solid ${border}`, borderRadius: '6px', padding: '2px 8px', fontSize: '11px', fontFamily: 'monospace', color: fg }}>
            <span style={{ opacity: 0.65 }}>{isTest ? 'test: ' : 'train: '}</span>
            <strong>{s}</strong>
            <span style={{ margin: '0 3px', opacity: 0.6 }}>→</span>
            <strong>{t}</strong>
        </span>
    );
}

/** Filter toggle button */
function FilterBtn({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
    return (
        <button
            onClick={onClick}
            style={{
                padding: '3px 10px',
                borderRadius: '6px',
                border: `1px solid ${active ? '#6366f1' : '#d1d5db'}`,
                background: active ? '#6366f1' : '#fff',
                color: active ? '#fff' : '#6b7280',
                fontSize: '11px',
                fontWeight: active ? 700 : 500,
                cursor: 'pointer',
                transition: 'all 0.12s',
            }}
        >
            {children}
        </button>
    );
}

// ─── Main Component ───────────────────────────────────────────────────────────

export function CausalInterventionSection({ reportData }: Props) {
    if (!reportData?.experiment_meta) return null;

    const baseline = reportData.test_sample_baseline;
    const allPairs = reportData.correlation_pairs || [];
    const testCorrs = baseline.top_correlations || [];

    // ── Annotation state ──────────────────────────────────────────────────────
    const [annotations, setAnnotations] = useState<Record<string, Annotation>>(loadAnnotations);

    const toggleAnnotation = (id: string, label: Annotation) => {
        setAnnotations(prev => {
            const next = { ...prev };
            if (next[id] === label) delete next[id]; // click same → clear
            else next[id] = label;
            saveAnnotations(next);
            return next;
        });
    };

    const clearAnnotation = (id: string) => {
        setAnnotations(prev => {
            const next = { ...prev };
            delete next[id];
            saveAnnotations(next);
            return next;
        });
    };

    // ── Filter state ──────────────────────────────────────────────────────────
    const [threshold, setThreshold] = useState(0.0);
    const [testSrcFilter, setTestSrcFilter] = useState<number | 'all'>('all');
    const [annFilter, setAnnFilter] = useState<'all' | 'unannotated' | Annotation>('all');

    // ── Derived data ──────────────────────────────────────────────────────────
    const effectiveAnn = (pair: CorrelationPair): AnnotationLabel =>
        annotations[pair.id] !== undefined ? annotations[pair.id] : pair.annotation;

    const displayPairs = useMemo(() => allPairs.filter(p => {
        if (p.cos_sim < threshold) return false;
        if (testSrcFilter !== 'all' && p.test_correlation.source_token_index !== testSrcFilter) return false;
        const ann = effectiveAnn(p);
        if (annFilter === 'unannotated') return ann === null || ann === undefined;
        if (annFilter !== 'all') return ann === annFilter;
        return true;
    }), [allPairs, threshold, testSrcFilter, annFilter, annotations]);

    const stats = useMemo(() => {
        const aboveThreshold = allPairs.filter(p => p.cos_sim >= threshold).length;
        const annCounts = { correct: 0, incorrect: 0, ambiguous: 0, total: 0 };
        allPairs.forEach(p => {
            const a = effectiveAnn(p);
            if (a) { annCounts[a]++; annCounts.total++; }
        });
        return { total: allPairs.length, aboveThreshold, ...annCounts };
    }, [allPairs, threshold, annotations]);

    // ── Export ────────────────────────────────────────────────────────────────
    const exportJSON = () => {
        const out = {
            ...reportData,
            correlation_pairs: allPairs.map(p => ({
                ...p,
                annotation: effectiveAnn(p),
            })),
        };
        const blob = new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = Object.assign(document.createElement('a'), { href: url, download: 'correlation_annotated.json' });
        a.click();
        URL.revokeObjectURL(url);
    };

    // ─────────────────────────────────────────────────────────────────────────
    return (
        <section className="analysis-section" style={{ marginTop: '40px', borderTop: '2px dashed #c7d2fe', paddingTop: '40px' }}>

            {/* ── Section Header ── */}
            <div className="section-header">
                <h2>Section 3: Correlation Pair Annotation</h2>
                <p className="section-desc">
                    <strong>目标:</strong> 对每个 (train_corr ↔ test_corr) 匹配对，判断该训练样本中的 source→target 关联是否在语义上合理。<br />
                    <strong>标注方案:</strong>
                    &nbsp;<span style={{ background: '#dcfce7', color: '#166534', padding: '1px 7px', borderRadius: '4px', fontSize: '11px', fontWeight: 600 }}>✓ Correct</span> 推理路径语义合理
                    &nbsp;<span style={{ background: '#fee2e2', color: '#991b1b', padding: '1px 7px', borderRadius: '4px', fontSize: '11px', fontWeight: 600 }}>✗ Incorrect</span> Spurious 关联（错误推理路径）
                    &nbsp;<span style={{ background: '#fef3c7', color: '#92400e', padding: '1px 7px', borderRadius: '4px', fontSize: '11px', fontWeight: 600 }}>? Ambiguous</span> 难以判断<br />
                    <strong>Target Token:</strong>&nbsp;
                    <code style={{ background: '#fee2e2', padding: '2px 7px', borderRadius: '4px' }}>{baseline.target_token}</code>
                    &nbsp;@ index {baseline.target_token_index}
                </p>
            </div>

            {/* ── Stats Bar ── */}
            <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', alignItems: 'stretch', marginBottom: '18px' }}>
                {[
                    { label: 'Total Pairs',      value: stats.total,          color: '#6b7280', bg: '#f9fafb' },
                    { label: 'Above Threshold',  value: stats.aboveThreshold, color: '#2563eb', bg: '#eff6ff' },
                    { label: 'Annotated',        value: stats.total,          color: '#7c3aed', bg: '#f5f3ff' },
                    { label: '✓ Correct',        value: stats.correct,        color: '#15803d', bg: '#f0fdf4' },
                    { label: '✗ Incorrect',      value: stats.incorrect,      color: '#b91c1c', bg: '#fef2f2' },
                    { label: '? Ambiguous',      value: stats.ambiguous,      color: '#b45309', bg: '#fffbeb' },
                ].map(({ label, value, color, bg }) => (
                    <div key={label} style={{ background: bg, border: '1px solid #e5e7eb', borderRadius: '8px', padding: '8px 14px', minWidth: '90px', textAlign: 'center', flex: '0 0 auto' }}>
                        <div style={{ fontSize: '22px', fontWeight: 700, color, lineHeight: 1.2 }}>{value}</div>
                        <div style={{ fontSize: '10px', color: '#9ca3af', marginTop: '3px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.04em' }}>{label}</div>
                    </div>
                ))}
                <button
                    onClick={exportJSON}
                    style={{ marginLeft: 'auto', padding: '0 20px', background: 'linear-gradient(135deg, #1d4ed8, #6366f1)', color: '#fff', border: 'none', borderRadius: '8px', fontWeight: 700, fontSize: '13px', cursor: 'pointer', minHeight: '60px', boxShadow: '0 2px 8px rgba(99,102,241,0.3)', transition: 'opacity 0.15s' }}
                    onMouseEnter={e => (e.currentTarget.style.opacity = '0.88')}
                    onMouseLeave={e => (e.currentTarget.style.opacity = '1')}
                >
                    ↓ Export JSON
                </button>
            </div>

            {/* ── Filters ── */}
            <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '10px', padding: '14px 18px', marginBottom: '18px', display: 'flex', flexDirection: 'column', gap: '12px' }}>

                {/* cos_sim threshold slider */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
                    <span style={{ fontSize: '11px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', minWidth: '110px' }}>cos_sim ≥</span>
                    <input
                        type="range" min={0} max={1} step={0.01}
                        value={threshold}
                        onChange={e => setThreshold(parseFloat(e.target.value))}
                        style={{ width: '160px', accentColor: '#6366f1' }}
                    />
                    <span style={{ fontWeight: 700, fontSize: '14px', color: '#4338ca', minWidth: '36px' }}>{threshold.toFixed(2)}</span>
                    <span style={{ fontSize: '11px', color: '#94a3b8' }}>({stats.aboveThreshold} pairs pass)</span>
                </div>

                {/* Test correlation filter */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                    <span style={{ fontSize: '11px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', minWidth: '110px' }}>Test Correlation</span>
                    <FilterBtn active={testSrcFilter === 'all'} onClick={() => setTestSrcFilter('all')}>All</FilterBtn>
                    {testCorrs.map(c => (
                        <button
                            key={c.source_token_index}
                            onClick={() => setTestSrcFilter(c.source_token_index)}
                            title={`saliency: ${c.saliency_score?.toFixed(5)}`}
                            style={{
                                padding: '3px 10px', borderRadius: '6px', fontSize: '11px', fontWeight: 600, fontFamily: 'monospace', cursor: 'pointer',
                                background: testSrcFilter === c.source_token_index ? '#3b82f6' : '#fff',
                                color: testSrcFilter === c.source_token_index ? '#fff' : '#1d4ed8',
                                border: `1px solid ${testSrcFilter === c.source_token_index ? '#3b82f6' : '#bfdbfe'}`,
                                transition: 'all 0.12s',
                            }}
                        >
                            {c.source_token.trim() || '[SP]'} → {c.target_token.trim() || '[SP]'}
                        </button>
                    ))}
                </div>

                {/* Annotation status filter */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                    <span style={{ fontSize: '11px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', minWidth: '110px' }}>Annotation</span>
                    {(['all', 'unannotated', 'correct', 'incorrect', 'ambiguous'] as const).map(f => (
                        <FilterBtn key={f} active={annFilter === f} onClick={() => setAnnFilter(f)}>
                            {f.charAt(0).toUpperCase() + f.slice(1)}
                        </FilterBtn>
                    ))}
                </div>
            </div>

            {/* ── Result count ── */}
            <div style={{ marginBottom: '12px', fontSize: '13px', color: '#64748b' }}>
                显示 <strong style={{ color: '#1e293b' }}>{displayPairs.length}</strong> 条&nbsp;/&nbsp;共 {stats.total} 条
            </div>

            {/* ── Pair List ── */}
            {displayPairs.length === 0 ? (
                <div className="no-data">无匹配记录，尝试调低 cos_sim 阈值或切换筛选条件。</div>
            ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    {displayPairs.map(pair => {
                        const ann = effectiveAnn(pair);
                        const { bg, fg } = cosSimStyle(pair.cos_sim);
                        const annCfg = ann ? ANNOTATION_CONFIG[ann] : null;

                        return (
                            <div
                                key={pair.id}
                                style={{
                                    border: ann ? `2px solid ${annCfg!.borderColor}` : '1px solid #e5e7eb',
                                    borderRadius: '10px',
                                    background: '#fff',
                                    padding: '14px 16px',
                                    display: 'flex',
                                    flexDirection: 'column',
                                    gap: '10px',
                                    transition: 'border-color 0.2s, box-shadow 0.2s',
                                    boxShadow: ann ? `0 0 0 3px ${annCfg!.activeBg}` : 'none',
                                }}
                            >
                                {/* ── Row 1: metadata + cos_sim badge + correlation tags ── */}
                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                                    <span style={{ fontSize: '10px', fontFamily: 'monospace', color: '#c4b5fd', letterSpacing: '0.03em' }}>
                                        {pair.id}
                                    </span>
                                    <span style={{ background: bg, color: fg, fontWeight: 700, fontSize: '12px', padding: '2px 10px', borderRadius: '10px', flexShrink: 0 }}>
                                        {pair.cos_sim.toFixed(4)}
                                    </span>
                                    <span style={{ fontSize: '11px', color: '#a5b4fc' }}>
                                        coarse {pair.coarse_cos_sim.toFixed(3)}
                                    </span>
                                    <span style={{ fontSize: '11px', color: '#9ca3af', background: '#f3f4f6', borderRadius: '4px', padding: '1px 6px' }}>
                                        train #{pair.train_sample_id}
                                    </span>

                                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginLeft: '4px', flexWrap: 'wrap' }}>
                                        <CorrChip src={pair.test_correlation.source_token} tgt={pair.test_correlation.target_token} isTest={true} />
                                        <span style={{ color: '#c4b5fd', fontSize: '16px', lineHeight: 1 }}>⇔</span>
                                        <CorrChip src={pair.train_correlation.source_token} tgt={pair.train_correlation.target_token} isTest={false} />
                                        <span style={{ fontSize: '10px', color: '#94a3b8', fontFamily: 'monospace' }}>
                                            +{pair.train_correlation.response_token_offset}
                                        </span>
                                    </div>

                                    {/* Inline annotation badge */}
                                    {ann && (
                                        <span style={{ marginLeft: 'auto', background: annCfg!.activeBg, color: annCfg!.activeFg, border: `1px solid ${annCfg!.borderColor}`, borderRadius: '6px', padding: '2px 10px', fontSize: '11px', fontWeight: 700 }}>
                                            {ANNOTATION_CONFIG[ann].emoji} {ANNOTATION_CONFIG[ann].label}
                                        </span>
                                    )}
                                </div>

                                {/* ── Row 2: Context windows ── */}
                                <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', background: '#f8fafc', borderRadius: '6px', padding: '8px 12px', border: '1px solid #f1f5f9' }}>
                                    <div style={{ flex: '1 1 200px' }}>
                                        <div style={{ fontSize: '10px', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '5px' }}>
                                            Source Context
                                        </div>
                                        <ContextChip tokens={pair.train_context.source_context} />
                                    </div>
                                    <div style={{ width: '1px', background: '#e2e8f0', flexShrink: 0 }} />
                                    <div style={{ flex: '1 1 200px' }}>
                                        <div style={{ fontSize: '10px', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '5px' }}>
                                            Target Context
                                        </div>
                                        <ContextChip tokens={pair.train_context.target_context} />
                                    </div>
                                </div>

                                {/* ── Row 3: Annotation controls ── */}
                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                                    <span style={{ fontSize: '10px', color: '#94a3b8', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                                        标注:
                                    </span>
                                    {(['correct', 'incorrect', 'ambiguous'] as Annotation[]).map(label => (
                                        <AnnotationBtn
                                            key={label}
                                            type={label}
                                            current={ann}
                                            onToggle={() => toggleAnnotation(pair.id, label)}
                                        />
                                    ))}
                                    {ann && (
                                        <button
                                            onClick={() => clearAnnotation(pair.id)}
                                            style={{ fontSize: '11px', color: '#9ca3af', background: 'none', border: 'none', cursor: 'pointer', textDecoration: 'underline', padding: '4px 6px' }}
                                        >
                                            clear
                                        </button>
                                    )}

                                    {/* Extra info on hover */}
                                    <span style={{ marginLeft: 'auto', fontSize: '10px', color: '#c4b5fd', fontFamily: 'monospace' }}>
                                        src_sal {pair.train_correlation.saliency_score.toFixed(4)} &middot; test_sal {pair.test_correlation.saliency_score.toFixed(4)}
                                    </span>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}
        </section>
    );
}
