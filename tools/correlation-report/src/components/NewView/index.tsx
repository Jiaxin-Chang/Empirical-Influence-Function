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

function emptySlot(): SlotState {
    return { report: null, meta: null, status: null, error: null };
}

function modelLabelFrom(report: AllTokensReport | null, meta: AllTokensExperimentMeta | null, fallback: string): string {
    return report?.experiment_meta.model_name
        || meta?.label
        || fallback;
}

export function NewView({ metas }: Props) {
    const [slot, setSlot] = useState<SlotState>(() => emptySlot());
    const [dragging, setDragging] = useState(false);

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
                                : 'Import JSON or pick a bundled experiment'}
                        </div>
                    </div>
                    {slot.report && (
                        <button type="button" className={styles.slotClearBtn} onClick={() => setSlot(emptySlot())}>
                            Clear
                        </button>
                    )}
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
                    Import or select a correlation report to begin.
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
