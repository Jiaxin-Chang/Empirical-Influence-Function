import { useState, useMemo } from 'react';
import { SwitchTokenCodeBlock } from '../SwitchTokenCodeBlock';

interface Correlation {
    source_token: string;
    source_token_index: number;
    target_token: string;
    target_token_index: number;
    saliency_score?: number;
}

interface CorrelationMatch {
    test_correlation: Correlation;
    train_correlation: Correlation;
    cos_sim: number;
}

interface TrainContext {
    target_token: string;
    target_token_index: number;
    full_tokens: string[];
    saliency_list: number[];
}

interface Intervention {
    train_sample_id: number;
    coarse_cos_sim: number;
    train_context: TrainContext;
    correlation_matches: CorrelationMatch[];
}

interface TestBaseline {
    target_token: string;
    target_token_index: number;
    full_tokens: string[];
    top_correlations: Correlation[];
}

interface ReportData {
    experiment_meta: { test_sample_index: number; target_token_index: number };
    test_sample_baseline: TestBaseline;
    interventions: Intervention[];
}

interface Props {
    reportData: ReportData;
}

function convertTokens(tokens: string[]) {
    return tokens.map(t =>
        t.replaceAll('Ċ', '\n').replaceAll('Ġ', ' ').replaceAll('ĉ', '  ')
    );
}

function cosSimColor(score: number): string {
    // green at 1.0, yellow at 0.5, red at 0.0
    const t = Math.max(0, Math.min(1, score));
    const r = Math.round(255 * (1 - t));
    const g = Math.round(200 * t);
    return `rgb(${r},${g},50)`;
}

function CosineBadge({ score }: { score: number }) {
    return (
        <span style={{
            display: 'inline-block',
            padding: '3px 10px',
            borderRadius: '12px',
            background: cosSimColor(score),
            color: '#fff',
            fontWeight: 'bold',
            fontSize: '13px',
            marginLeft: '8px',
        }}>
            cos_sim: {score.toFixed(4)}
        </span>
    );
}

function CorrelationTag({ corr, isTest, isHighlight, onClick }:
    { corr: Correlation; isTest: boolean; isHighlight?: boolean; onClick?: () => void }) {
    const bg = isTest
        ? (isHighlight ? '#1d4ed8' : '#3b82f6')
        : '#d97706';
    return (
        <span
            onClick={onClick}
            title={`index: ${corr.source_token_index} -> ${corr.target_token_index}`}
            style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '4px',
                padding: '4px 10px',
                borderRadius: '8px',
                background: bg,
                color: '#fff',
                fontSize: '12px',
                fontFamily: 'monospace',
                cursor: onClick ? 'pointer' : 'default',
                margin: '3px',
                outline: isHighlight ? '2px solid #93c5fd' : 'none',
                fontWeight: isHighlight ? 'bold' : 'normal',
            }}
        >
            <span style={{ opacity: 0.8 }}>{corr.source_token.trim() || '[SPACE]'}</span>
            <span>→</span>
            <span>{corr.target_token.trim() || '[SPACE]'}</span>
        </span>
    );
}

export function CausalInterventionSection({ reportData }: Props) {
    if (!reportData?.experiment_meta) return null;

    const baseline = reportData.test_sample_baseline;
    const interventions = reportData.interventions || [];
    const testCorrelations = baseline.top_correlations || [];

    // Which test correlation is currently selected (by source_token_index)
    const [selectedTestCorrIdx, setSelectedTestCorrIdx] = useState<number>(
        testCorrelations[0]?.source_token_index ?? -1
    );

    // Selected test token's full tokens for display
    const testTokens = useMemo(() => convertTokens(baseline.full_tokens || []), [baseline]);

    // For the selected test correlation, filter & sort interventions by cos_sim
    const filteredInterventions = useMemo(() => {
        return interventions
            .map(iv => {
                const match = iv.correlation_matches.find(
                    m => m.test_correlation.source_token_index === selectedTestCorrIdx
                );
                return match ? { intervention: iv, match } : null;
            })
            .filter((x): x is { intervention: Intervention; match: CorrelationMatch } => x !== null)
            .sort((a, b) => b.match.cos_sim - a.match.cos_sim);
    }, [interventions, selectedTestCorrIdx]);

    const selectedTestCorr = testCorrelations.find(c => c.source_token_index === selectedTestCorrIdx);

    const [expandedTrainIdx, setExpandedTrainIdx] = useState<number | null>(null);

    return (
        <section className="analysis-section" style={{ marginTop: '40px', borderTop: '2px dashed #ccc', paddingTop: '40px' }}>
            <div className="section-header">
                <h2>Section 3: Correlation Matching Verification</h2>
                <p className="section-desc">
                    <strong>实验目标:</strong> 追溯 Test Sample 的微观 Correlation（某个 source token 导致了 target token 的预测）
                    起源于哪些 Training Sample 的哪个内部 Correlation。<br />
                    <strong>方法:</strong> 提取 <code>Saliency Loss</code> 的二阶梯度特征指纹，用 Cosine Similarity 匹配测试集与训练集的 Correlation 签名。<br />
                    <strong>Target Token:</strong> <code style={{ background: '#fee2e2', padding: '2px 6px', borderRadius: '4px' }}>{baseline.target_token}</code>
                    &nbsp;@ index {baseline.target_token_index}
                </p>
            </div>

            {/* ── STEP 1: Test Sample Correlations selector ── */}
            <div style={{ background: '#f0f9ff', border: '1px solid #bae6fd', borderRadius: '10px', padding: '16px', marginBottom: '24px' }}>
                <div style={{ fontWeight: 'bold', fontSize: '13px', color: '#0369a1', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                    Test Sample — 点击选择一个 Correlation 查看匹配的训练样本
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginBottom: '12px' }}>
                    {testCorrelations.map(corr => (
                        <CorrelationTag
                            key={corr.source_token_index}
                            corr={corr}
                            isTest={true}
                            isHighlight={corr.source_token_index === selectedTestCorrIdx}
                            onClick={() => {
                                setSelectedTestCorrIdx(corr.source_token_index);
                                setExpandedTrainIdx(null);
                            }}
                        />
                    ))}
                </div>
                {selectedTestCorr && (
                    <div style={{ fontSize: '12px', color: '#64748b' }}>
                        当前选中: <strong>'{selectedTestCorr.source_token.trim()}'</strong> [index {selectedTestCorr.source_token_index}]
                        → <strong>'{selectedTestCorr.target_token}'</strong> [index {selectedTestCorr.target_token_index}]
                        &nbsp;| saliency: {selectedTestCorr.saliency_score?.toFixed(5)}
                    </div>
                )}
            </div>

            {/* ── STEP 2: Matched Train Samples list ── */}
            <div style={{ marginBottom: '12px', fontWeight: 'bold', color: '#374151' }}>
                匹配到的训练样本 (共 {filteredInterventions.length} 条，按 cos_sim 降序)
            </div>

            {filteredInterventions.length === 0 && (
                <div style={{ color: '#9ca3af', fontStyle: 'italic' }}>未找到匹配结果，请先运行实验。</div>
            )}

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {filteredInterventions.map(({ intervention, match }, idx) => {
                    const isExpanded = expandedTrainIdx === idx;
                    const trainTokens = convertTokens(intervention.train_context.full_tokens || []);

                    // Build saliency map for train display
                    const trainSalMap: { [k: number]: number[] } = {};
                    if (Array.isArray(intervention.train_context.saliency_list)) {
                        trainSalMap[intervention.train_context.target_token_index] =
                            intervention.train_context.saliency_list;
                    }

                    return (
                        <div
                            key={intervention.train_sample_id}
                            style={{
                                border: `1px solid ${isExpanded ? '#6366f1' : '#e5e7eb'}`,
                                borderRadius: '10px',
                                overflow: 'hidden',
                                boxShadow: isExpanded ? '0 0 0 2px #c7d2fe' : 'none',
                                transition: 'box-shadow 0.2s',
                            }}
                        >
                            {/* Card Header */}
                            <div
                                onClick={() => setExpandedTrainIdx(isExpanded ? null : idx)}
                                style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '12px',
                                    padding: '12px 16px',
                                    cursor: 'pointer',
                                    background: isExpanded ? '#eef2ff' : '#f9fafb',
                                    borderBottom: isExpanded ? '1px solid #e0e7ff' : 'none',
                                }}
                            >
                                <span style={{ fontWeight: 'bold', color: '#6366f1', minWidth: '28px' }}>#{idx + 1}</span>
                                <span style={{ fontSize: '12px', color: '#6b7280' }}>
                                    Train ID: {intervention.train_sample_id}
                                </span>
                                <span style={{ fontSize: '12px', color: '#6b7280' }}>
                                    Coarse Score: {intervention.coarse_cos_sim.toFixed(4)}
                                </span>

                                {/* Show the matched correlation pair */}
                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flex: 1 }}>
                                    <CorrelationTag corr={match.test_correlation} isTest={true} />
                                    <span style={{ color: '#9ca3af', fontSize: '18px' }}>⇔</span>
                                    <CorrelationTag corr={match.train_correlation} isTest={false} />
                                </div>

                                <CosineBadge score={match.cos_sim} />
                                <span style={{ color: '#9ca3af', fontSize: '18px' }}>{isExpanded ? '▲' : '▼'}</span>
                            </div>

                            {/* Expanded: show all correlation_matches + train token block */}
                            {isExpanded && (
                                <div style={{ padding: '16px', background: '#fff' }}>
                                    {/* All correlations table */}
                                    <div style={{ marginBottom: '16px' }}>
                                        <div style={{ fontSize: '12px', fontWeight: 'bold', color: '#374151', marginBottom: '8px', textTransform: 'uppercase' }}>
                                            全部 Correlation 对比打分
                                        </div>
                                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
                                            <thead>
                                                <tr style={{ borderBottom: '1px solid #e5e7eb', textAlign: 'left', color: '#6b7280' }}>
                                                    <th style={{ padding: '6px 8px' }}>Test Correlation</th>
                                                    <th style={{ padding: '6px 8px' }}>Train Correlation (Best Match)</th>
                                                    <th style={{ padding: '6px 8px' }}>Cosine Sim</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {intervention.correlation_matches.map((m, mi) => (
                                                    <tr
                                                        key={mi}
                                                        style={{
                                                            borderBottom: '1px solid #f3f4f6',
                                                            background: m.test_correlation.source_token_index === selectedTestCorrIdx
                                                                ? '#eff6ff' : 'transparent'
                                                        }}
                                                    >
                                                        <td style={{ padding: '6px 8px' }}>
                                                            <CorrelationTag corr={m.test_correlation} isTest={true} />
                                                        </td>
                                                        <td style={{ padding: '6px 8px' }}>
                                                            <CorrelationTag corr={m.train_correlation} isTest={false} />
                                                        </td>
                                                        <td style={{ padding: '6px 8px' }}>
                                                            <CosineBadge score={m.cos_sim} />
                                                        </td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>

                                    {/* Train code block with saliency highlight */}
                                    <div>
                                        <div style={{ fontSize: '12px', fontWeight: 'bold', color: '#374151', marginBottom: '8px', textTransform: 'uppercase' }}>
                                            Train Sample — Target Token: '{intervention.train_context.target_token}' [index {intervention.train_context.target_token_index}]
                                        </div>
                                        <SwitchTokenCodeBlock
                                            tokens={trainTokens}
                                            salienciesByToken={trainSalMap}
                                            answerStartIndex={intervention.train_context.target_token_index}
                                        />
                                    </div>
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>
        </section>
    );
}
