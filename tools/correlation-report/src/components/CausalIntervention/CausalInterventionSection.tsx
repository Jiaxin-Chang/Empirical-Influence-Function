import { useMemo, useState } from 'react';
import { SwitchTokenCodeBlock } from '../SwitchTokenCodeBlock';

interface CausalInterventionSectionProps {
    reportData: any;
}

export function CausalInterventionSection({ reportData }: CausalInterventionSectionProps) {
    if (!reportData || !reportData.experiment_meta) return null;

    const [sampleIndex, setSampleIndex] = useState(0);
    const meta = reportData.experiment_meta;
    const testBaseline = reportData.test_sample_baseline;
    const interventions = reportData.interventions || [];

    if (interventions.length === 0) return null;

    const intervention = interventions[sampleIndex];
    const maxCount = interventions.length;

    const goPrev = () => setSampleIndex(((sampleIndex - 1) + maxCount) % maxCount);
    const goNext = () => setSampleIndex((sampleIndex + 1) % maxCount);

    const testSalAfter = useMemo(() => convertRawSaliencyToObject(intervention.test_after_intervention.saliency_list), [intervention]);

    // Train data
    const trainContext = intervention.train_context;

    const testTokens = useMemo(() => convertTokens(testBaseline.full_tokens || []), [testBaseline]);
    const trainTokens = useMemo(() => convertTokens(trainContext.full_tokens || []), [trainContext]);

    // Test context
    const testAfter = intervention.test_after_intervention;
    const isPositive = testAfter.conclusion === "POSITIVE_CORRELATION";
    const ProbChange = () => {
        const p1 = testBaseline.target_token_prob * 100;
        const p2 = testAfter.target_token_prob * 100;
        return (
            <span style={{ marginLeft: '12px' }}>
                Prob: {p1.toFixed(1)}% ➔ <strong>{p2.toFixed(1)}%</strong>
                {p2 > p1 ? ' 📈' : p2 < p1 ? ' 📉' : ' ➖'}
            </span>
        )
    };

    return (
        <section className="analysis-section" style={{ marginTop: '40px', borderTop: '2px dashed #ccc', paddingTop: '40px' }}>
            <div className="section-header">
                <h2>Section 3: Causal Intervention Verification</h2>
                <p className="section-desc">
                    <strong>实验目标:</strong> 验证 Train Sample 中的 Correlation 是否真实导致了 Test Sample 预测错误。<br />
                    <strong>干预方式:</strong> 对 Train Sample 的关键 Token (boost_indices) 进行 {meta.boost_coef}倍注意力放大微调 ({meta.intervention_epochs} epochs)。<br />
                    左侧展示 Train Sample 的病灶干预点，右侧为强力干预后 Test Sample `{testBaseline.target_token}` Target 预测的反应。
                </p>
            </div>

            <div className="train-nav" style={{ backgroundColor: isPositive ? '#e6f4ea' : '#fce8e6' }}>
                <span className="badge badge-train">VERIFICATION</span>
                <span>Intervention Rank {sampleIndex + 1} / {maxCount} (Train ID: {intervention.train_sample_id})</span>
                <ProbChange />
                <span style={{ marginLeft: 'auto', fontWeight: 'bold', color: isPositive ? 'green' : 'red' }}>
                    {isPositive ? '✅ POSITIVE_CORRELATION' : '❌ UNRELATED (OR NEGATIVE)'}
                </span>
                <span className="nav-buttons" style={{ marginLeft: '16px' }}>
                    <button onClick={goPrev}>← prev</button>
                    <button onClick={goNext}>next →</button>
                </span>
            </div>

            <div className="two-panel-row">
                {/* LEFT PANEL: TRAIN SAMPLE */}
                <div className="panel" style={{ flex: '1 0 0' }}>
                    <div className="panel-title">
                        <span className="badge badge-train" style={{ background: '#d4a373' }}>INTERVENTION TARGET</span>
                        Train Sample
                    </div>
                    <div className="panel-columns">
                        <div className="panel-col">
                            <div className="col-label">Target Token: [{trainContext.first_valid_token_index}] '{trainContext.first_valid_token}'</div>
                            <SwitchTokenCodeBlock
                                tokens={trainTokens}
                                salienciesByToken={convertRawSaliencyToObject(trainContext.saliency_list)}
                                answerStartIndex={trainContext.first_valid_token_index}
                                // Highlight the boost indices as pseudo-GPT marks so they show up clearly
                                gptAnnotationIndices={new Set(trainContext.boost_indices)}
                                gptOpacity={1.0}
                            />
                        </div>
                    </div>
                </div>

                {/* RIGHT PANEL: TEST SAMPLE SHIFTS */}
                <div className="panel" style={{ flex: '1 0 0' }}>
                    <div className="panel-title">
                        <span className="badge badge-test" style={{ background: '#457b9d' }}>OBSERVATION</span>
                        Test Sample (`{testBaseline.target_token}`) Before/After
                    </div>

                    <div className="shifts-table-container" style={{ margin: '16px 0', padding: '12px', background: '#f8f9fa', borderRadius: '8px' }}>
                        <h4 style={{ margin: '0 0 8px 0', fontSize: '13px', textTransform: 'uppercase', color: '#555' }}>
                            Top Correlation Shifts for `{testBaseline.target_token}`
                        </h4>

                        {/* Only table if prediction didn't change entirely, or if it changed we display a notice */}
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
                            <thead>
                                <tr style={{ borderBottom: '1px solid #ddd', textAlign: 'left' }}>
                                    <th style={{ padding: '4px' }}>Token</th>
                                    <th style={{ padding: '4px' }}>Before</th>
                                    <th style={{ padding: '4px' }}>After</th>
                                    <th style={{ padding: '4px' }}>Delta</th>
                                </tr>
                            </thead>
                            <tbody>
                                {testAfter.correlation_shifts.map((shift: any, idx: number) => (
                                    <tr key={idx} style={{ borderBottom: '1px solid #eee' }}>
                                        <td style={{ padding: '4px', fontFamily: 'monospace' }}>
                                            <span style={{ background: '#eee', padding: '2px 4px', borderRadius: '4px' }}>
                                                {shift.prompt_token.replace(/\n/g, '↩').trim() || '[SPACE]'}
                                            </span>
                                        </td>
                                        <td style={{ padding: '4px' }}>{shift.saliency_before.toFixed(4)}</td>
                                        <td style={{ padding: '4px' }}>{shift.saliency_after.toFixed(4)}</td>
                                        <td style={{ padding: '4px', color: shift.delta > 0 ? 'green' : 'red', fontWeight: 'bold' }}>
                                            {shift.delta > 0 ? '+' : ''}{shift.delta.toFixed(4)}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                        {!isPositive && (
                            <div style={{ marginTop: '12px', color: '#666', fontStyle: 'italic', fontSize: '13px' }}>
                                * 结论为无关，由于干预未引发明显的测试关联度提升，或预测直接改变。
                            </div>
                        )}
                    </div>

                    <div className="panel-columns">
                        <div className="panel-col">
                            <div className="col-label">Test Saliency After Intervention</div>
                            <SwitchTokenCodeBlock
                                tokens={testTokens}
                                salienciesByToken={testSalAfter}
                                answerStartIndex={meta.target_token_index}
                            />
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
}

function convertRawSaliencyToObject(saliencyList: any[]): { [key: number]: number[] } {
    // intervention_results dumps raw array inside test_sample_baseline.saliency_list
    // but occasionally depending on format it might be an array of numbers, or old format list
    // the new NIF output returns an array of numbers directly if it's single target. Let's handle both.
    if (!saliencyList) return {};

    // If it's the raw array of floats for 1 target token
    if (typeof saliencyList[0] === 'number') {
        return { 0: saliencyList };
    }

    // Otherwise fallback if it's the NIF format
    const converted: { [key: number]: number[] } = {};
    saliencyList.forEach((x: any) => { converted[x.index] = x.saliency; });
    return converted;
}

function convertTokens(tokens: string[]) {
    return tokens.map(t => t.replaceAll('Ċ', '\n').replaceAll('Ġ', ' ').replaceAll('ĉ', '  '));
}
