import { useCallback, useEffect, useRef, useState, type ChangeEvent } from 'react';
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

function friendlyApiError(message: string): string {
    if (/HTTP 500|Failed to list raw eval files|Failed to load raw rows/i.test(message)) {
        return 'Evaluation API is unavailable. Start the backend and reload.';
    }
    return message;
}

function hasReportUrlQuery(): boolean {
    if (typeof window === 'undefined') return false;
    const params = new URLSearchParams(window.location.search);
    return Boolean(params.get('reportUrl') ?? params.get('report_url') ?? params.get('leftUrl'));
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
    const loadGenRef = useRef(0);

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

    const loadRawSampleAt = useCallback(async (fileName: string, line: string) => {
        if (!fileName || !line) return;
        const gen = ++loadGenRef.current;
        setRawBusy(true);
        setRawError(null);
        setSlot({
            report: null,
            meta: null,
            status: `Tokenizing ${fileName} line ${line}…`,
            error: null,
        });
        try {
            const resp = await fetch(`${defaultEifApiOrigin()}/api/raw-eval-sample`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ fileName, line: Number(line) }),
            });
            const rawText = await resp.text();
            let parsed: { status?: string; report?: AllTokensReport; message?: string } = {};
            if (rawText.trim()) {
                parsed = JSON.parse(rawText) as typeof parsed;
            }
            if (gen !== loadGenRef.current) return;
            if (!resp.ok || parsed.status !== 'success' || !parsed.report) {
                throw new Error(parsed.message || `Raw sample failed (HTTP ${resp.status})`);
            }
            const data = normalizeAllTokensReport(parsed.report);
            const taskId = data.experiment_meta.task_id || `line_${line}`;
            const family = data.experiment_meta.report_family;
            const familyTag = family === 'ce' ? 'CE' : family === 'saliency' ? 'SAL' : 'raw';
            const meta: AllTokensExperimentMeta = {
                taskId,
                label: `[${familyTag}] ${fileName} · L${line}`,
                fileName,
            };
            setSlot({
                report: data,
                meta,
                status: `Raw ${familyTag} · ${taskId} · LoRA=${family || '?'} · live saliency/probs`,
                error: null,
            });
        } catch (error) {
            if (gen !== loadGenRef.current) return;
            const message = error instanceof Error ? error.message : 'Failed to load raw sample';
            setRawError(message);
            setSlot({ report: null, meta: null, status: null, error: message });
        } finally {
            if (gen === loadGenRef.current) setRawBusy(false);
        }
    }, []);

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
                const firstLine = rows[0] ? String(rows[0].line) : '';
                setRawLine(firstLine);
                if (cancelled) return;
                if (firstLine && !hasReportUrlQuery()) {
                    await loadRawSampleAt(rawFile, firstLine);
                } else {
                    setRawBusy(false);
                }
            } catch (error) {
                if (!cancelled) {
                    setRawRows([]);
                    setRawLine('');
                    setRawError(error instanceof Error ? error.message : 'Failed to load raw rows');
                    setRawBusy(false);
                }
            }
        })();
        return () => { cancelled = true; };
    }, [rawFile, loadRawSampleAt]);

    useEffect(() => {
        const onKeyDown = (ev: KeyboardEvent) => {
            if (rawBusy) return;
            if (ev.ctrlKey || ev.altKey || ev.metaKey) return;
            const tag = (ev.target as HTMLElement | null)?.tagName;
            if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
            if (ev.key !== 'ArrowDown' && ev.key !== 'ArrowUp') return;
            ev.preventDefault();
            if (!rawFile || rawRows.length === 0) return;
            const idx = rawRows.findIndex(r => String(r.line) === rawLine);
            const nextIdx = ev.key === 'ArrowDown'
                ? (idx < 0 ? 0 : idx + 1)
                : (idx < 0 ? 0 : idx - 1);
            if (nextIdx < 0 || nextIdx >= rawRows.length) return;
            const line = String(rawRows[nextIdx].line);
            setRawLine(line);
            void loadRawSampleAt(rawFile, line);
        };
        window.addEventListener('keydown', onKeyDown);
        return () => window.removeEventListener('keydown', onKeyDown);
    }, [rawBusy, rawFile, rawLine, rawRows, loadRawSampleAt]);

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

    const selectRawLine = (line: string) => {
        setRawLine(line);
        void loadRawSampleAt(rawFile, line);
    };

    const sampleIdx = rawRows.findIndex(r => String(r.line) === rawLine);
    const canPrev = !rawBusy && sampleIdx > 0;
    const canNext = !rawBusy && sampleIdx >= 0 && sampleIdx < rawRows.length - 1;
    const displayError = rawError || slot.error;
    const ceRawFiles = rawFiles.filter(f => f.reportFamily === 'ce' || f.folder === 'raw_ce');
    const salRawFiles = rawFiles.filter(f => f.reportFamily === 'saliency' || f.folder === 'raw_sal' || f.folder === 'raw_sa');
    const otherRawFiles = rawFiles.filter(f => !ceRawFiles.includes(f) && !salRawFiles.includes(f));

    const goDelta = (delta: number) => {
        if (rawBusy || rawRows.length === 0) return;
        const idx = sampleIdx < 0 ? 0 : sampleIdx;
        const nextIdx = idx + delta;
        if (nextIdx < 0 || nextIdx >= rawRows.length) return;
        selectRawLine(String(rawRows[nextIdx].line));
    };

    return (
        <div className={styles.root}>
            <div className={styles.slotImportCard}>
                <div className={styles.toolbarRow}>
                    <label className={styles.field}>
                        <span className={styles.fieldLabel}>语料</span>
                        <select
                            value={rawFile}
                            disabled={rawBusy || rawFiles.length === 0}
                            onChange={e => setRawFile(e.target.value)}
                        >
                    {rawFiles.length === 0 && <option value="">暂无 JSONL</option>}
                            {ceRawFiles.length > 0 && (
                                <optgroup label="raw_ce · CE 测试结果 · 从 CE adapter 续训">
                                    {ceRawFiles.map(f => (
                                        <option key={f.fileName} value={f.fileName}>
                                            {f.label} · {f.nRows}
                                        </option>
                                    ))}
                                </optgroup>
                            )}
                            {salRawFiles.length > 0 && (
                                <optgroup label="raw_sal · SAL 测试结果 · 从 sal adapter 续训">
                                    {salRawFiles.map(f => (
                                        <option key={f.fileName} value={f.fileName}>
                                            {f.label} · {f.nRows}
                                        </option>
                                    ))}
                                </optgroup>
                            )}
                            {otherRawFiles.map(f => (
                                <option key={f.fileName} value={f.fileName}>
                                    {f.label} · {f.nRows}
                                </option>
                            ))}
                        </select>
                    </label>
                    <label className={`${styles.field} ${styles.fieldGrow}`}>
                        <span className={styles.fieldLabel}>样本</span>
                        <select
                            value={rawLine}
                            disabled={rawBusy || rawRows.length === 0}
                            onChange={e => selectRawLine(e.target.value)}
                        >
                            {rawRows.length === 0 && <option value="">—</option>}
                            {rawRows.map(r => (
                                <option key={r.line} value={String(r.line)}>
                                    L{r.line} · {r.task_id}
                                </option>
                            ))}
                        </select>
                    </label>
                    <div className={styles.stepper}>
                        <button
                            type="button"
                            className={styles.stepBtn}
                            disabled={!canPrev}
                            aria-label="Previous sample"
                            onClick={() => goDelta(-1)}
                        >
                            ↑
                        </button>
                        <button
                            type="button"
                            className={styles.stepBtn}
                            disabled={!canNext}
                            aria-label="Next sample"
                            onClick={() => goDelta(1)}
                        >
                            ↓
                        </button>
                    </div>
                    {rawBusy && <span className={styles.loadingNote}>载入中…</span>}
                    {!rawBusy && rawRows.length > 0 && sampleIdx >= 0 && (
                        <span className={styles.counter}>
                            {sampleIdx + 1}
                            <span className={styles.counterSep}>/</span>
                            {rawRows.length}
                        </span>
                    )}
                </div>

                {displayError && (
                    <p className={styles.footnoteError}>{friendlyApiError(displayError)}</p>
                )}

                {/* 暂时隐藏：JSON 文件导入 + 预处理报告列表 + 打开样本按钮 */}
                {false && (
                    <>
                        <button
                            type="button"
                            className={styles.rawEvalOpenBtn}
                            disabled={rawBusy || !rawFile || !rawLine}
                            onClick={() => void loadRawSampleAt(rawFile, rawLine)}
                        >
                            {rawBusy ? '打开中…' : '打开样本'}
                        </button>
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
                    </>
                )}
            </div>

            {!slot.report && (
                <div className={styles.emptyState}>
                    {rawBusy
                        ? '正在载入第一条评测样本…'
                        : '尚未载入样本。选择语料后，API 就绪即可用 ↓ 切换。'}
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
