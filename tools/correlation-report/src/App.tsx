import { useEffect, useMemo, useState } from 'react';
import './App.css';
import { SwitchTokenCodeBlock } from './components/SwitchTokenCodeBlock';
import { CausalInterventionSection } from './components/CausalIntervention';
import { NewView } from './components/NewView';

/* ------------------------------------------------------------------ */
/* Manifest types (matches vite.config.ts plugin output)              */
/* ------------------------------------------------------------------ */

interface ExperimentMeta {
    testIdx: number;
    tokIdx: number;
    hasSaliency: boolean;
    hasCorrelation: boolean;
}

interface AllTokensMeta {
    taskId: string;
    label: string;
    fileName: string;
}

interface Manifest {
    experiments: ExperimentMeta[];
    allTokensExperiments: AllTokensMeta[];
    hasLegacySaliency: boolean;
    hasLegacyCorrelation: boolean;
    hasMarkedCode: boolean;
}

/* ------------------------------------------------------------------ */
/* Helpers                                                            */
/* ------------------------------------------------------------------ */

const BOOST_COEFS = [1, 10, 100, 1000, 10000];

function convertRawSaliencyToObject(saliency: any[]): Record<number, number[]> {
    const out: Record<number, number[]> = {};
    saliency.forEach((x: any) => { out[x['index']] = x['saliency']; });
    return out;
}

function convertTokens(tokens: string[]): string[] {
    return tokens.map(t =>
        t.replaceAll('Ċ', '\n').replaceAll('Ġ', ' ').replaceAll('ĉ', '  ')
    );
}

function extractFencedCodeBlocks(text: string): string[] {
    const re = /```[^\n]*\n(.*?)\n```/gs;
    const blocks: string[] = [];
    let m;
    while ((m = re.exec(text)) !== null) blocks.push(m[1]);
    return blocks;
}

function extractAttnSpans(text: string): { cleanedText: string; spans: [number, number][] } {
    const TAG_RE = /<ATTN>(.*?)<\/ATTN>/gs;
    const spans: [number, number][] = [];
    const parts: string[] = [];
    let cursor = 0, outLen = 0, m;
    while ((m = TAG_RE.exec(text)) !== null) {
        const pre = text.slice(cursor, m.index);
        parts.push(pre);
        outLen += pre.length;
        const content = m[1];
        spans.push([outLen, outLen + content.length]);
        parts.push(content);
        outLen += content.length;
        cursor = m.index + m[0].length;
    }
    parts.push(text.slice(cursor));
    return { cleanedText: parts.join(''), spans };
}

async function fetchText(url: string): Promise<string | null> {
    try {
        const res = await fetch(url);
        if (!res.ok) return null;
        return res.text();
    } catch {
        return null;
    }
}

/* ------------------------------------------------------------------ */
/* Experiment entry & data types                                      */
/* ------------------------------------------------------------------ */

interface ExperimentEntry {
    label: string;
    testIdx: number;
    tokIdx: number;
    saliencyText: string;
    correlationText: string | null;
}

interface ExperimentData {
    saliency: any;
    overfitResults: any[];
    trainSamples: any[];
    correlationData: any | null;
}

function parseExperimentData(entry: ExperimentEntry): ExperimentData {
    let saliency: any = {};
    try { saliency = JSON.parse(entry.saliencyText); } catch { saliency = {}; }

    const hasNewFormat = 'overfit_test_results' in saliency;
    const overfitResults: any[] = hasNewFormat
        ? (saliency['overfit_test_results'] ?? [])
        : (saliency['related_train_samples'] ?? []);
    const trainSamples: any[] = hasNewFormat ? (saliency['related_train_samples'] ?? []) : [];

    let correlationData: any = null;
    if (entry.correlationText) {
        try { correlationData = JSON.parse(entry.correlationText); } catch { correlationData = null; }
    }

    return { saliency, overfitResults, trainSamples, correlationData };
}

/* ------------------------------------------------------------------ */
/* ExperimentSelector                                                 */
/* ------------------------------------------------------------------ */

function ExperimentSelector({
    metas,
    currentIndex,
    loading,
    onChange,
}: {
    metas: ExperimentMeta[];
    currentIndex: number;
    loading: boolean;
    onChange: (idx: number) => void;
}) {
    if (metas.length === 0) return null;
    if (metas.length === 1) {
        const m = metas[0];
        return (
            <div className="experiment-selector single">
                <span className="exp-label">Experiment:</span>
                <span className="exp-pill active">test={m.testIdx} tok={m.tokIdx}</span>
            </div>
        );
    }
    return (
        <div className="experiment-selector">
            <span className="exp-label">Experiment:</span>
            {metas.map((m, i) => (
                <button
                    key={i}
                    className={`exp-pill${i === currentIndex ? ' active' : ''}`}
                    onClick={() => onChange(i)}
                    disabled={loading}
                >
                    test={m.testIdx} tok={m.tokIdx}
                </button>
            ))}
            {loading && <span className="exp-loading">Loading…</span>}
        </div>
    );
}

/* ------------------------------------------------------------------ */
/* App                                                                */
/* ------------------------------------------------------------------ */

type ViewMode = 'new' | 'legacy';

function App() {
    const [viewMode, setViewMode] = useState<ViewMode>('new');

    const [manifest, setManifest] = useState<Manifest | null>(null);
    const [manifestError, setManifestError] = useState(false);

    const [experimentIndex, setExperimentIndex] = useState(0);
    const [experimentEntry, setExperimentEntry] = useState<ExperimentEntry | null>(null);
    const [loadingExp, setLoadingExp] = useState(false);

    const [markedCodeSamplesText, setMarkedCodeSamplesText] = useState('');

    // 1) Fetch the manifest on mount — tiny JSON, instant
    useEffect(() => {
        fetch('/data/index.json')
            .then(r => {
                if (!r.ok) throw new Error('manifest not found');
                return r.json();
            })
            .then((m: Manifest) => {
                setManifest(m);
                if (m.hasMarkedCode) {
                    fetchText('/data/marked_code_samples.md').then(t => {
                        if (t) setMarkedCodeSamplesText(t);
                    });
                }
            })
            .catch(() => setManifestError(true));
    }, []);

    // 2) Fetch only the selected experiment's data when the user picks one
    useEffect(() => {
        if (!manifest) return;

        const snap = manifest;
        const metas = snap.experiments;

        async function load() {
            setLoadingExp(true);
            setExperimentEntry(null);

            if (metas.length > 0) {
                const m = metas[experimentIndex] ?? metas[0];
                const salUrl  = `/data/saliency_test${m.testIdx}_tok${m.tokIdx}.json`;
                const corrUrl = `/data/correlation_matching_results_test${m.testIdx}_tok${m.tokIdx}.json`;

                const [salText, corrText] = await Promise.all([
                    fetchText(salUrl),
                    m.hasCorrelation ? fetchText(corrUrl) : Promise.resolve(null),
                ]);

                if (salText) {
                    setExperimentEntry({
                        label: `test=${m.testIdx}  tok=${m.tokIdx}`,
                        testIdx: m.testIdx,
                        tokIdx: m.tokIdx,
                        saliencyText: salText,
                        correlationText: corrText,
                    });
                }
            } else if (snap.hasLegacySaliency) {
                const [salText, corrText] = await Promise.all([
                    fetchText('/data/latest_saliency.json'),
                    snap.hasLegacyCorrelation
                        ? fetchText('/data/correlation_matching_results.json')
                        : Promise.resolve(null),
                ]);
                if (salText) {
                    setExperimentEntry({
                        label: 'legacy (latest_saliency)',
                        testIdx: -1,
                        tokIdx: -1,
                        saliencyText: salText,
                        correlationText: corrText,
                    });
                }
            }

            setLoadingExp(false);
        }

        load();
    }, [manifest, experimentIndex]);

    const gptBlocks = useMemo(
        () => extractFencedCodeBlocks(markedCodeSamplesText).map(b => extractAttnSpans(b)),
        [markedCodeSamplesText]
    );

    // ── Render states ──────────────────────────────────────────────────────────

    if (manifestError) {
        return (
            <div className="app-root">
                <header className="app-header"><h1>Attribution Analysis</h1></header>
                <section className="analysis-section">
                    <p className="no-data" style={{ marginTop: '16px' }}>
                        Failed to load <code>/data/index.json</code>. Make sure the Vite dev server is running
                        (<code>pnpm dev</code>) or run <code>pnpm preview</code> after building.
                    </p>
                </section>
            </div>
        );
    }

    if (!manifest) {
        return (
            <div className="app-root">
                <header className="app-header"><h1>Attribution Analysis</h1></header>
                <section className="analysis-section">
                    <p className="no-data" style={{ marginTop: '16px' }}>Loading experiment list…</p>
                </section>
            </div>
        );
    }

    const allTokensMetas = manifest.allTokensExperiments ?? [];
    const hasLegacyData = manifest.experiments.length > 0 || manifest.hasLegacySaliency;

    return (
        <div className="app-root">
            <header className="app-header">
                <h1>Attribution Analysis</h1>
                <div className="view-toggle">
                    <button
                        className={`view-toggle-btn${viewMode === 'new' ? ' active' : ''}`}
                        onClick={() => setViewMode('new')}
                    >
                        New View
                    </button>
                    <button
                        className={`view-toggle-btn${viewMode === 'legacy' ? ' active' : ''}`}
                        onClick={() => setViewMode('legacy')}
                    >
                        Legacy View
                    </button>
                </div>
            </header>

            {/* ── New View ── */}
            {viewMode === 'new' && (
                <NewView metas={allTokensMetas} />
            )}

            {/* ── Legacy View ── */}
            {viewMode === 'legacy' && (
                !hasLegacyData ? (
                    <section className="analysis-section">
                        <div className="section-header">
                            <p className="no-data" style={{ marginTop: '16px' }}>
                                No experiment files found. Run <code>NIF.py</code> to generate{' '}
                                <code>saliency_test&#123;N&#125;_tok&#123;tok&#125;.json</code>.
                            </p>
                        </div>
                    </section>
                ) : (
                    <>
                        <ExperimentSelector
                            metas={manifest.experiments}
                            currentIndex={experimentIndex}
                            loading={loadingExp}
                            onChange={idx => setExperimentIndex(idx)}
                        />

                        {loadingExp && (
                            <section className="analysis-section">
                                <p className="no-data" style={{ marginTop: '16px' }}>Loading experiment data…</p>
                            </section>
                        )}

                        {!loadingExp && experimentEntry && (() => {
                            const expData = parseExperimentData(experimentEntry);
                            return (
                                <>
                                    <OverfitSection key={`overfit-${experimentIndex}`} expData={expData} />
                                    <TrainSampleSection
                                        key={`train-${experimentIndex}`}
                                        expData={expData}
                                        gptBlocks={gptBlocks}
                                    />
                                    {expData.correlationData && (
                                        <CausalInterventionSection reportData={expData.correlationData} />
                                    )}
                                </>
                            );
                        })()}
                    </>
                )
            )}
        </div>
    );
}

/* ------------------------------------------------------------------ */
/* Section 1: Overfit Experiment                                      */
/* ------------------------------------------------------------------ */

function OverfitSection({ expData }: { expData: ExperimentData }) {
    const [coefIndex, setCoefIndex] = useState(0);
    const coef = BOOST_COEFS[coefIndex] ?? '?';

    const { saliency, overfitResults } = expData;
    const testSample = saliency['target_test_sample'];

    if (!testSample || !testSample['before']) {
        return (
            <section className="analysis-section">
                <div className="section-header">
                    <h2>Section 1: Overfit Experiment</h2>
                    <p className="no-data" style={{ marginTop: '16px' }}>
                        No saliency data found. Please run <code>NIF.py</code>.
                    </p>
                </div>
            </section>
        );
    }

    const testTokensBefore = useMemo(() => convertTokens(testSample['before']['full_tokens']), [testSample]);
    const testTokensAfter  = useMemo(() => convertTokens(testSample['after']['full_tokens']),  [testSample]);
    const testSalBefore    = useMemo(() => convertRawSaliencyToObject(testSample['before']['saliency_list']), [testSample]);
    const testSalAfter     = useMemo(() => convertRawSaliencyToObject(testSample['after']['saliency_list']),  [testSample]);

    const overfitSample = overfitResults[coefIndex];
    const overfitTokens = useMemo(
        () => overfitSample ? convertTokens(overfitSample['before_original']['full_tokens']) : [],
        [overfitSample]
    );
    const overfitSal = useMemo(
        () => overfitSample ? convertRawSaliencyToObject(overfitSample['before_original']['saliency_list']) : {},
        [overfitSample]
    );

    const goPrev = () => setCoefIndex(((coefIndex - 1) + BOOST_COEFS.length) % BOOST_COEFS.length);
    const goNext = () => setCoefIndex((coefIndex + 1) % BOOST_COEFS.length);

    return (
        <section className="analysis-section">
            <div className="section-header">
                <h2>Section 1: Overfit Experiment</h2>
                <p className="section-desc">
                    对同一个 test sample，使用 GPT 标注的 attention token 以不同强度（boost_coef）过拟合后，
                    观察模型 saliency 分布的变化。左侧为 test sample 的推理结果，右侧为过拟合后 test sample 的 saliency 变化。
                </p>
            </div>

            <div className="two-panel-row">
                <div className="panel" style={{ flex: '2 0 0' }}>
                    <div className="panel-title">
                        <span className="badge badge-test">TEST</span>
                        Target Test Sample
                    </div>
                    <div className="panel-columns">
                        <div className="panel-col">
                            <div className="col-label">Ground Truth</div>
                            <SwitchTokenCodeBlock
                                tokens={testTokensBefore}
                                salienciesByToken={testSalBefore}
                                answerStartIndex={testSample['before']['start_index']}
                            />
                        </div>
                        <div className="panel-col">
                            <div className="col-label">Prediction</div>
                            <SwitchTokenCodeBlock
                                tokens={testTokensAfter}
                                salienciesByToken={testSalAfter}
                                answerStartIndex={testSample['after']['start_index']}
                            />
                        </div>
                    </div>
                </div>

                <div className="panel" style={{ flex: '1 0 0' }}>
                    <div className="panel-title">
                        <span className="badge badge-overfit">OVERFIT</span>
                        <span>Overfit Result</span>
                        <span className="coef-label">boost_coef = <strong>{coef}</strong></span>
                        <span className="nav-buttons">
                            <button onClick={goPrev}>← prev</button>
                            <button onClick={goNext}>next →</button>
                        </span>
                    </div>
                    {overfitSample ? (
                        <div className="panel-columns">
                            <div className="panel-col">
                                <div className="col-label">Saliency after overfit (coef={coef})</div>
                                <SwitchTokenCodeBlock
                                    key={coefIndex}
                                    tokens={overfitTokens}
                                    salienciesByToken={overfitSal}
                                    answerStartIndex={overfitSample['before_original']['start_index']}
                                />
                            </div>
                        </div>
                    ) : (
                        <div className="no-data">No overfit data available</div>
                    )}
                </div>
            </div>
        </section>
    );
}

/* ------------------------------------------------------------------ */
/* Section 2: Related Train Samples                                   */
/* ------------------------------------------------------------------ */

type LayerMode = 'both' | 'saliency' | 'gpt';

function TrainSampleSection({
    expData,
    gptBlocks,
}: {
    expData: ExperimentData;
    gptBlocks: { cleanedText: string; spans: [number, number][] }[];
}) {
    const [sampleIndex, setSampleIndex] = useState(0);
    const [layerMode, setLayerMode] = useState<LayerMode>('both');

    const { trainSamples } = expData;
    const maxCount = Math.max(trainSamples.length, gptBlocks.length);

    if (maxCount === 0) {
        return (
            <section className="analysis-section">
                <div className="section-header">
                    <h2>Section 2: Related Train Samples</h2>
                    <p className="no-data" style={{ marginTop: '16px' }}>
                        No train samples or GPT annotations (<code>marked_code_samples.md</code>) found.
                        Please run Option 2 in <code>NIF.py</code>.
                    </p>
                </div>
            </section>
        );
    }

    const goPrev = () => setSampleIndex(((sampleIndex - 1) + maxCount) % maxCount);
    const goNext = () => setSampleIndex((sampleIndex + 1) % maxCount);

    const trainSample = sampleIndex < trainSamples.length ? trainSamples[sampleIndex] : null;
    const gptBlock    = sampleIndex < gptBlocks.length    ? gptBlocks[sampleIndex]    : null;

    return (
        <section className="analysis-section">
            <div className="section-header">
                <h2>Section 2: Related Train Samples</h2>
                <p className="section-desc">
                    与 test sample 最相关的训练样本。<strong>黄色热力图</strong> 为模型 saliency 分数，
                    <strong>紫色标注</strong> 为 GPT 标记的关键 token（<code>&lt;ATTN&gt;</code>）。
                    可切换 "Saliency"、"GPT" 和 "Both" 三种视图进行对比。
                </p>
            </div>

            <div className="train-nav">
                <span className="badge badge-train">TRAIN</span>
                <span>Sample {sampleIndex + 1} / {maxCount}</span>
                <span className="nav-buttons">
                    <button onClick={goPrev}>← prev</button>
                    <button onClick={goNext}>next →</button>
                </span>
                <span className="layer-toggle">
                    <button className={layerMode === 'both'     ? 'active' : ''} onClick={() => setLayerMode('both')}>Both</button>
                    <button className={layerMode === 'saliency' ? 'active' : ''} onClick={() => setLayerMode('saliency')}>Saliency</button>
                    <button className={layerMode === 'gpt'      ? 'active' : ''} onClick={() => setLayerMode('gpt')}>GPT</button>
                </span>
            </div>

            <TrainSampleView
                key={sampleIndex}
                trainSample={trainSample}
                gptBlock={gptBlock}
                layerMode={layerMode}
                hasSaliency={trainSample !== null}
                hasGpt={gptBlock !== null}
            />
        </section>
    );
}

/* ------------------------------------------------------------------ */
/* TrainSampleView                                                    */
/* ------------------------------------------------------------------ */

function TrainSampleView({
    trainSample,
    gptBlock,
    layerMode,
    hasSaliency,
    hasGpt,
}: {
    trainSample: any | null;
    gptBlock: { cleanedText: string; spans: [number, number][] } | null;
    layerMode: LayerMode;
    hasSaliency: boolean;
    hasGpt: boolean;
}) {
    const showSaliency = layerMode === 'both' || layerMode === 'saliency';
    const showGpt      = layerMode === 'both' || layerMode === 'gpt';

    if (hasSaliency && trainSample) {
        const tokens     = convertTokens(trainSample['before_original']['full_tokens']);
        const sal        = convertRawSaliencyToObject(trainSample['before_original']['saliency_list']);
        const startIdx   = trainSample['before_original']['start_index'];
        const gptIndices = (showGpt && hasGpt && gptBlock)
            ? computeGptTokenIndices(tokens, gptBlock)
            : undefined;

        return (
            <div className="train-panel-columns">
                <div className="panel-col">
                    <div className="col-label">
                        {layerMode === 'both'      ? 'Model Saliency + GPT Annotation'
                         : layerMode === 'saliency' ? 'Model Saliency'
                         : 'GPT Annotation Only'}
                    </div>
                    <SwitchTokenCodeBlock
                        key={`train-${layerMode}`}
                        tokens={tokens}
                        salienciesByToken={showSaliency ? sal : {}}
                        answerStartIndex={startIdx}
                        gptAnnotationIndices={gptIndices}
                    />
                </div>
            </div>
        );
    }

    if (hasGpt && gptBlock && showGpt) {
        return (
            <div className="train-panel-columns">
                <div className="panel-col">
                    <div className="col-label">GPT Annotation</div>
                    <GptAnnotatedBlock cleanedText={gptBlock.cleanedText} spans={gptBlock.spans} />
                </div>
            </div>
        );
    }

    if (layerMode === 'saliency' && !hasSaliency) {
        return <div className="no-data">该样本暂无 saliency 数据。请使用 NIF.py 重新生成 JSON。</div>;
    }

    return <div className="no-data">No data for this sample</div>;
}

/* ------------------------------------------------------------------ */
/* computeGptTokenIndices                                             */
/* ------------------------------------------------------------------ */

function computeGptTokenIndices(
    tokens: string[],
    gptBlock: { cleanedText: string; spans: [number, number][] }
): Set<number> {
    const indices  = new Set<number>();
    const fullText = tokens.join('');

    const tokenStarts: number[] = [];
    let offset = 0;
    for (const t of tokens) { tokenStarts.push(offset); offset += t.length; }

    for (const [spanStart, spanEnd] of gptBlock.spans) {
        const spanText = gptBlock.cleanedText.slice(spanStart, spanEnd);
        const matchIdx = fullText.indexOf(spanText);
        if (matchIdx >= 0) {
            const matchEnd = matchIdx + spanText.length;
            for (let ti = 0; ti < tokens.length; ti++) {
                const tStart = tokenStarts[ti];
                const tEnd   = tStart + tokens[ti].length;
                if (tEnd > matchIdx && tStart < matchEnd) indices.add(ti);
            }
        }
    }
    return indices;
}

/* ------------------------------------------------------------------ */
/* GptAnnotatedBlock                                                  */
/* ------------------------------------------------------------------ */

function GptAnnotatedBlock({ cleanedText, spans }: { cleanedText: string; spans: [number, number][] }) {
    const elements: React.ReactNode[] = [];
    let cursor = 0;
    for (let si = 0; si < spans.length; si++) {
        const [start, end] = spans[si];
        if (cursor < start) elements.push(<span key={`u${si}`}>{cleanedText.slice(cursor, start)}</span>);
        elements.push(<span key={`m${si}`} className="attn-highlight">{cleanedText.slice(start, end)}</span>);
        cursor = end;
    }
    if (cursor < cleanedText.length) elements.push(<span key="tail">{cleanedText.slice(cursor)}</span>);
    return (
        <pre className="highlighted-token-code-block">
            <code>{elements}</code>
        </pre>
    );
}

export default App;
