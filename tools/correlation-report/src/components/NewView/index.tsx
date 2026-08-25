import { useCallback, useEffect, useState, type ChangeEvent } from 'react';
import styles from './NewView.module.css';
import {
    ReportPanel,
    normalizeAllTokensReport,
    normalizeImportedReport,
    type AllTokensExperimentMeta,
    type AllTokensReport,
} from './ReportPanel';

export type { AllTokensExperimentMeta };

interface Props {
    metas: AllTokensExperimentMeta[];
}

interface SlotState {
    report: AllTokensReport | null;
    meta: AllTokensExperimentMeta | null;
    status: string | null;
    error: string | null;
}

interface RawEvalFile {
    fileName: string;
    label: string;
    nRows: number;
    folder?: string;
    reportFamily?: 'ce' | 'saliency' | null;
}

interface RawEvalRow {
    line: number;
    task_id: string;
    predict_preview?: string;
    label_preview?: string;
}

function emptySlot(): SlotState {
    return { report: null, meta: null, status: null, error: null };
}

function modelLabelFrom(report: AllTokensReport | null, meta: AllTokensExperimentMeta | null, fallback: string): string {
    return report?.experiment_meta.model_name
        || meta?.label
        || fallback;
}

function defaultEifApiOrigin(): string {
    if (typeof window === 'undefined') return 'http://127.0.0.1:8766';
    return window.location.origin;
}

export function NewView({ metas }: Props) {
    const [slot, setSlot] = useState<SlotState>(() => emptySlot());
    const [dragging, setDragging] = useState(false);
    const [rawFiles, setRawFiles] = useState<RawEvalFile[]>([]);
    const [rawFile, setRawFile] = useState('');
    const [rawRows, setRawRows] = useState<RawEvalRow[]>([]);
    const [rawLine, setRawLine] = useState('');
    const [rawBusy, setRawBusy] = useState(false);
    const [rawError, setRawError] = useState<string | null>(null);

    const activatePayload = useCallback((payload: unknown, sourceName: string) => {
        try {
            const imported = normalizeImportedReport(payload, sourceName);
            const modelName = imported.report.experiment_meta.model_name;
            const meta: AllTokensExperimentMeta = {
                ...imported.meta,
                label: modelName
                    ? `${modelName} · ${imported.meta.label.replace(/ \(uploaded\)$/, '')}`
                    : imported.meta.label,
                fileName: `uploaded:${sourceName}`,
            };
            setSlot({
                report: imported.report,
                meta,
                status: `Loaded ${imported.report.per_token_results.length} token(s) from ${sourceName}`
                    + (modelName ? ` [${modelName}]` : ''),
                error: null,
            });
        } catch (error) {
            const message = error instanceof Error ? error.message : 'Failed to parse JSON.';
            setSlot({
                report: null,
                meta: null,
                status: null,
                error: `Failed to import ${sourceName}: ${message}`,
            });
        }
    }, []);

    const loadMeta = useCallback(async (meta: AllTokensExperimentMeta) => {
        setSlot({
            report: null,
            meta,
            status: `Loading ${meta.fileName}...`,
            error: null,
        });
        try {
            const resp = await fetch(`/data/results/${meta.fileName}`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = normalizeAllTokensReport(await resp.json() as AllTokensReport);
            const modelName = data.experiment_meta.model_name;
            setSlot({
                report: data,
                meta: {
                    ...meta,
                    label: modelName ? `${modelName} · ${meta.taskId}` : meta.label,
                },
                status: `Loaded ${meta.fileName}` + (modelName ? ` [${modelName}]` : ''),
                error: null,
            });
        } catch (error) {
            const message = error instanceof Error ? error.message : 'Failed to load.';
            setSlot({
                report: null,
                meta: null,
                status: null,
                error: `Failed to load ${meta.fileName}: ${message}`,
            });
        }
    }, []);

    const handleFile = useCallback(async (file: File | null | undefined) => {
        if (!file) return;
        try {
            const text = await file.text();
            activatePayload(JSON.parse(text), file.name);
        } catch (error) {
            const message = error instanceof Error ? error.message : 'Failed to parse JSON.';
            setSlot({
                report: null,
                meta: null,
                status: null,
                error: `Failed to import ${file.name}: ${message}`,
            });
        }
    }, [activatePayload]);

    useEffect(() => {
        let cancelled = false;
        void (async () => {
            try {
                const resp = await fetch(`${defaultEifApiOrigin()}/api/raw-eval-files`);
                if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                const data = await resp.json() as { status?: string; files?: RawEvalFile[] };
                if (cancelled || data.status !== 'success') return;
                const files = Array.isArray(data.files) ? data.files : [];
                setRawFiles(files);
                if (files.length > 0) {
                    setRawFile(prev => prev || files[0].fileName);
                }
            } catch (error) {
                if (!cancelled) {
                    setRawError(error instanceof Error ? error.message : 'Failed to list raw eval files');
                }
            }
        })();
        return () => { cancelled = true; };
    }, []);

    useEffect(() => {
        if (!rawFile) {
            setRawRows([]);
            setRawLine('');
            return;
        }
        let cancelled = false;
        setRawBusy(true);
        setRawError(null);
        void (async () => {
            try {
                const url = new URL(`${defaultEifApiOrigin()}/api/raw-eval-rows`);
                url.searchParams.set('file', rawFile);
                const resp = await fetch(url.toString());
                if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                const data = await resp.json() as { status?: string; rows?: RawEvalRow[]; message?: string };
                if (cancelled) return;
                if (data.status !== 'success') {
                    throw new Error(data.message || 'Failed to load raw rows');
                }
                const rows = Array.isArray(data.rows) ? data.rows : [];
                setRawRows(rows);
                setRawLine(rows[0] ? String(rows[0].line) : '');
            } catch (error) {
                if (!cancelled) {
                    setRawRows([]);
                    setRawLine('');
                    setRawError(error instanceof Error ? error.message : 'Failed to load raw rows');
                }
            } finally {
                if (!cancelled) setRawBusy(false);
            }
        })();
        return () => { cancelled = true; };
    }, [rawFile]);

    const loadRawSample = useCallback(async () => {
        if (!rawFile || !rawLine) return;
        setRawBusy(true);
        setRawError(null);
        setSlot({
            report: null,
            meta: null,
            status: `Tokenizing ${rawFile} line ${rawLine}…`,
            error: null,
        });
        try {
            const resp = await fetch(`${defaultEifApiOrigin()}/api/raw-eval-sample`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ fileName: rawFile, line: Number(rawLine) }),
            });
            const rawText = await resp.text();
            let parsed: { status?: string; report?: AllTokensReport; message?: string } = {};
            if (rawText.trim()) {
                parsed = JSON.parse(rawText) as typeof parsed;
            }
            if (!resp.ok || parsed.status !== 'success' || !parsed.report) {
                throw new Error(parsed.message || `Raw sample failed (HTTP ${resp.status})`);
            }
            const data = normalizeAllTokensReport(parsed.report);
            const taskId = data.experiment_meta.task_id || `line_${rawLine}`;
            const family = data.experiment_meta.report_family;
            const familyTag = family === 'ce' ? 'CE' : family === 'saliency' ? 'SA' : 'raw';
            const meta: AllTokensExperimentMeta = {
                taskId,
                label: `[${familyTag}] ${rawFile} · L${rawLine}`,
                fileName: rawFile,
            };
            setSlot({
                report: data,
                meta,
                status: `Raw ${familyTag} · ${taskId} · LoRA=${family || '?'} · live saliency/probs`,
                error: null,
            });
        } catch (error) {
            const message = error instanceof Error ? error.message : 'Failed to load raw sample';
            setRawError(message);
            setSlot({ report: null, meta: null, status: null, error: message });
        } finally {
            setRawBusy(false);
        }
    }, [rawFile, rawLine]);

    // Optional URL query: ?reportUrl=...
    useEffect(() => {
        if (typeof window === 'undefined') return;
        const params = new URLSearchParams(window.location.search);
        const reportUrl = params.get('reportUrl') ?? params.get('report_url') ?? params.get('leftUrl');
        if (!reportUrl) return;

        void (async () => {
            try {
                const url = new URL(reportUrl, window.location.href);
                const resp = await fetch(url.toString(), { cache: 'no-store' });
                if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                activatePayload(await resp.json(), url.pathname.split('/').pop() || url.host);
            } catch (error) {
                const message = error instanceof Error ? error.message : 'Failed to load URL.';
                setSlot({ report: null, meta: null, status: null, error: message });
            }
        })();
    }, [activatePayload]);

    return (
        <div className={styles.root}>
            <div className={styles.slotImportCard}>
                <div className={styles.slotImportHeader}>
                    <div>
                        <div className={styles.slotImportTitle}>Correlation Report</div>
                        <div className={styles.slotImportDesc}>
                            {slot.report
                                ? modelLabelFrom(slot.report, slot.meta, 'loaded')
                                : 'Import JSON、选预处理报告，或打开 correlation_matching_results/raw_ce|raw_sa 下的评测 JSONL'}
                        </div>
                    </div>
                    {slot.report && (
                        <button type="button" className={styles.slotClearBtn} onClick={() => setSlot(emptySlot())}>
                            Clear
                        </button>
                    )}
                </div>

                <div className={styles.rawEvalBar}>
                    <div className={styles.rawEvalTitle}>Raw 评测 JSONL（raw_ce=CE LoRA · raw_sa=Saliency LoRA）</div>
                    <div className={styles.rawEvalControls}>
                        <label className={styles.rawEvalLabel}>
                            文件
                            <select
                                value={rawFile}
                                disabled={rawBusy || rawFiles.length === 0}
                                onChange={e => setRawFile(e.target.value)}
                            >
                                {rawFiles.length === 0 && <option value="">（无 raw_ce|raw_sa/*.jsonl）</option>}
                                {rawFiles.map(f => (
                                    <option key={f.fileName} value={f.fileName}>
                                        {f.label} · {f.nRows} rows
                                    </option>
                                ))}
                            </select>
                        </label>
                        <label className={styles.rawEvalLabel}>
                            样本
                            <select
                                value={rawLine}
                                disabled={rawBusy || rawRows.length === 0}
                                onChange={e => setRawLine(e.target.value)}
                            >
                                {rawRows.length === 0 && <option value="">—</option>}
                                {rawRows.map(r => (
                                    <option key={r.line} value={String(r.line)}>
                                        L{r.line} · {r.task_id}
                                    </option>
                                ))}
                            </select>
                        </label>
                        <button
                            type="button"
                            className={styles.rawEvalOpenBtn}
                            disabled={rawBusy || !rawFile || !rawLine}
                            onClick={() => void loadRawSample()}
                        >
                            {rawBusy ? '打开中…' : '打开样本'}
                        </button>
                    </div>
                    {rawError && <div className={styles.importError}>{rawError}</div>}
                </div>

                <label
                    className={`${styles.importDropZone} ${dragging ? styles.importDropZoneActive : ''}`}
                    onDragOver={event => {
                        event.preventDefault();
                        setDragging(true);
                    }}
                    onDragLeave={() => setDragging(false)}
                    onDrop={event => {
                        event.preventDefault();
                        setDragging(false);
                        void handleFile(event.dataTransfer.files?.[0]);
                    }}
                >
                    <input
                        type="file"
                        accept=".json,application/json"
                        className={styles.importFileInput}
                        onChange={(event: ChangeEvent<HTMLInputElement>) => {
                            void handleFile(event.target.files?.[0]);
                            event.target.value = '';
                        }}
                    />
                    <span className={styles.importDropMain}>Choose JSON</span>
                    <span className={styles.importDropSub}>all-token report</span>
                </label>

                {metas.length > 0 && (
                    <div className={styles.slotMetaList}>
                        {metas.map((m, i) => (
                            <button
                                key={`${m.fileName}-${i}`}
                                type="button"
                                className={`${styles.metaBtn} ${slot.meta?.fileName === m.fileName ? styles.metaBtnActive : ''}`}
                                onClick={() => void loadMeta(m)}
                            >
                                {m.label}
                            </button>
                        ))}
                    </div>
                )}

                {slot.status && <div className={styles.importStatus}>{slot.status}</div>}
                {slot.error && <div className={styles.importError}>{slot.error}</div>}
            </div>

            {!slot.report && (
                <div className={styles.emptyState}>
                    Import or select a correlation report / raw eval sample to begin.
                </div>
            )}

            {slot.report && slot.meta && (
                <div className={styles.singleGrid}>
                    <div className={styles.modelColumn}>
                        <ReportPanel
                            report={slot.report}
                            meta={slot.meta}
                            modelLabel={modelLabelFrom(slot.report, slot.meta, 'Model')}
                        />
                    </div>
                </div>
            )}
        </div>
    );
}
