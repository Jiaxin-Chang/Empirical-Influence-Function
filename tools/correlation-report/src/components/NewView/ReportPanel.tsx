import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import styles from './NewView.module.css';
import { InlineVisualizer } from './InlineVisualizer';
import {
    collectProbeEmphasis,
    loadInlineBundle,
    type InlineBundle,
    type TokenHoverTarget,
} from './inlineBundle';

const TTAV_PREFS_KEY = 'eif:ttav-launch-prefs';
const TTAV_PREPARED_BUNDLES_KEY = 'eif:ttav-prepared-bundles';
const INLINE_PLOT_SIZE_KEY = 'eif:inline-plot-size';
const INLINE_PLOT_POS_KEY = 'eif:inline-plot-pos';
const DEFAULT_TTAV_URL = 'http://1.94.115.154/';
const DEFAULT_TTAV_CONTENT_PATH_TEMPLATE = '/root/project/Dataset/eif_bundles/{sampleId}';
// Empty by default: let the EIF bundle API resolve its own on-server cache
// directory (ttav_bundles/{sampleId} relative to its repo root) instead of a
// hardcoded absolute path that only exists on one developer's machine.
const DEFAULT_EIF_BUNDLE_CACHE_TEMPLATE = '';
const DEFAULT_TTAV_METHOD = 'TimeVis';
const DEFAULT_TTAV_VIS_ID = '1';
function getDefaultEifApiUrl(): string {
    if (typeof window === 'undefined') {
        return 'http://127.0.0.1:8766/api/prepare-ttav-bundle';
    }

    // Same-origin: the /api/* path is reverse-proxied to the EIF bundle API
    // (127.0.0.1:8766) by the page server (vite dev/preview proxy, or nginx).
    // Using window.location.origin avoids port-mismatch and mixed-content issues
    // in VS Code forwarded-localhost and public-nginx setups alike.
    return `${window.location.origin}/api/prepare-ttav-bundle`;
}

const DEFAULT_EIF_API_URL = getDefaultEifApiUrl();

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

interface TrainProbeFocusToken {
    tokenIndex: number;
    token: string;
    tokenDisplay: string;
    pointIndex?: number;
}

interface TrainProbeComparisonPair {
    leftIndex: number;
    leftToken: string;
    leftTokenDisplay: string;
    rightIndex: number;
    rightToken: string;
    rightTokenDisplay: string;
    cosine: number;
}

interface TrainProbeComparisonSummary {
    focusTokens: TrainProbeFocusToken[];
    pairwiseCosine: TrainProbeComparisonPair[];
}

interface UnlearnPairResult {
    status?: string;
    verdict?: string;
    error?: string;
    before?: { ce?: number; logprob?: number; saliency?: number | null };
    after?: { ce?: number; logprob?: number; saliency?: number | null };
    delta?: { ce?: number; logprob?: number; saliency?: number | null };
    update?: { paramSpace?: string; lastNLayers?: number };
    testEdge?: { reportedSaliency?: number | null; saliencyMode?: string };
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

export interface AllTokensReport {
    experiment_meta: {
        test_sample_index: number;
        mode: 'all_tokens';
        tokens_analyzed: number;
        model_name?: string;
        model_path?: string | null;
        base_model_path?: string | null;
        task_id?: string;
    };
    test_sample_baseline: {
        full_tokens: string[];
        correct_full_tokens: string[];
        prompt_len: number;
    };
    per_token_results: PerTokenResult[];
    train_sample_details: Record<string, TrainSampleDetail>;
}

export interface ImportedReport {
    report: AllTokensReport;
    meta: AllTokensExperimentMeta;
    format: 'all_tokens' | 'generic_saliency';
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function asFiniteNumber(value: unknown): number | null {
    if (typeof value !== 'number' || !Number.isFinite(value)) return null;
    return value;
}

function asString(value: unknown): string | null {
    return typeof value === 'string' ? value : null;
}

function asStringArray(value: unknown): string[] | null {
    if (!Array.isArray(value) || !value.every(item => typeof item === 'string')) return null;
    return value;
}

function asNumberArray(value: unknown): number[] | null {
    if (!Array.isArray(value)) return null;
    const values = value.map(asFiniteNumber);
    if (values.some(item => item === null)) return null;
    return values as number[];
}

function firstString(record: Record<string, unknown>, keys: string[]): string | null {
    for (const key of keys) {
        const value = asString(record[key]);
        if (value) return value;
    }
    return null;
}

function firstStringArray(record: Record<string, unknown>, keys: string[]): string[] | null {
    for (const key of keys) {
        const value = asStringArray(record[key]);
        if (value) return value;
    }
    return null;
}

function firstNumber(record: Record<string, unknown>, keys: string[]): number | null {
    for (const key of keys) {
        const value = asFiniteNumber(record[key]);
        if (value !== null) return value;
    }
    return null;
}

function stemFromSource(sourceName: string): string {
    const lastPart = sourceName.split(/[\\/]/).pop() || sourceName;
    return lastPart.replace(/\.json$/i, '') || 'uploaded-report';
}

function buildImportedMeta(report: AllTokensReport, sourceName: string): AllTokensExperimentMeta {
    const stem = stemFromSource(sourceName);
    const testIndex = report.experiment_meta.test_sample_index;
    const taskId = testIndex >= 0 ? `uploaded_test${testIndex}_${stem}` : `uploaded_${stem}`;
    return {
        taskId,
        label: `${stem} (uploaded)`,
        fileName: sourceName,
    };
}

export function isAllTokensReportLike(value: unknown): value is AllTokensReport {
    if (!isRecord(value)) return false;
    const experimentMeta = value.experiment_meta;
    const baseline = value.test_sample_baseline;
    if (!isRecord(experimentMeta) || !isRecord(baseline)) return false;
    return Array.isArray(value.per_token_results)
        && asStringArray(baseline.full_tokens) !== null
        && asFiniteNumber(baseline.prompt_len) !== null;
}

function normalizeTestCorrelation(value: unknown, tokens: string[], fallbackTargetIdx: number): TestCorrelation | null {
    if (!isRecord(value)) return null;
    const sourceIdx = asFiniteNumber(value.source_token_index);
    const targetIdx = asFiniteNumber(value.target_token_index) ?? fallbackTargetIdx;
    const score = asFiniteNumber(value.saliency_score);
    if (sourceIdx === null || targetIdx === null || score === null) return null;
    const sourceIndex = Math.trunc(sourceIdx);
    const targetIndex = Math.trunc(targetIdx);
    return {
        source_token: asString(value.source_token) ?? tokens[sourceIndex] ?? '',
        source_token_index: sourceIndex,
        target_token: asString(value.target_token) ?? tokens[targetIndex] ?? '',
        target_token_index: targetIndex,
        saliency_score: score,
    };
}

export function normalizeAllTokensReport(report: AllTokensReport): AllTokensReport {
    const baseline = report.test_sample_baseline;
    const fullTokens = asStringArray(baseline.full_tokens) ?? [];
    const promptLen = Math.max(0, Math.min(fullTokens.length, Math.trunc(asFiniteNumber(baseline.prompt_len) ?? 0)));
    const correctTokens = asStringArray(baseline.correct_full_tokens) ?? fullTokens;
    const perTokenResults = (Array.isArray(report.per_token_results) ? report.per_token_results : [])
        .map((item): PerTokenResult | null => {
            if (!isRecord(item)) return null;
            const targetIdxRaw = asFiniteNumber(item.target_token_index);
            if (targetIdxRaw === null) return null;
            const targetIdx = Math.trunc(targetIdxRaw);
            return {
                target_token_index: targetIdx,
                target_token: asString(item.target_token) ?? fullTokens[targetIdx] ?? '',
                top_correlations: (Array.isArray(item.top_correlations) ? item.top_correlations : [])
                    .map(c => normalizeTestCorrelation(c, fullTokens, targetIdx))
                    .filter((c): c is TestCorrelation => c !== null),
                correlation_pairs: Array.isArray(item.correlation_pairs)
                    ? (item.correlation_pairs as CorrelationPair[])
                    : [],
            };
        })
        .filter((item): item is PerTokenResult => item !== null);

    return {
        experiment_meta: {
            test_sample_index: Math.trunc(asFiniteNumber(report.experiment_meta.test_sample_index) ?? -1),
            mode: 'all_tokens',
            tokens_analyzed: Math.trunc(asFiniteNumber(report.experiment_meta.tokens_analyzed) ?? perTokenResults.length),
            model_name: asString(report.experiment_meta.model_name) ?? undefined,
            model_path: asString(report.experiment_meta.model_path) ?? undefined,
            base_model_path: asString((report.experiment_meta as { base_model_path?: unknown }).base_model_path) ?? undefined,
            task_id: asString(report.experiment_meta.task_id) ?? undefined,
        },
        test_sample_baseline: {
            full_tokens: fullTokens,
            correct_full_tokens: correctTokens,
            prompt_len: promptLen,
        },
        per_token_results: perTokenResults,
        train_sample_details: isRecord(report.train_sample_details)
            ? report.train_sample_details as Record<string, TrainSampleDetail>
            : {},
    };
}

interface GenericSaliencyItem {
    targetIdx: number;
    targetToken?: string;
    scores: number[];
}

function readGenericSaliencyItems(record: Record<string, unknown>, tokenCount: number): GenericSaliencyItem[] {
    const items: GenericSaliencyItem[] = [];

    const pushItem = (raw: unknown) => {
        if (!isRecord(raw)) return;
        const targetIdx = firstNumber(raw, ['target_token_index', 'target_index', 'index']);
        const scores = firstNumberArray(raw, ['scores', 'saliency', 'saliency_scores', 'source_scores']);
        if (targetIdx === null || !scores) return;
        const idx = Math.trunc(targetIdx);
        if (idx < 0 || idx >= tokenCount) return;
        items.push({
            targetIdx: idx,
            targetToken: firstString(raw, ['target_token', 'token']) ?? undefined,
            scores,
        });
    };

    for (const key of ['saliency', 'saliency_list', 'saliencies', 'targets']) {
        const value = record[key];
        if (Array.isArray(value)) value.forEach(pushItem);
    }

    for (const key of ['saliency_by_target', 'saliencyByTarget']) {
        const value = record[key];
        if (!isRecord(value)) continue;
        for (const [targetIdxRaw, scoresRaw] of Object.entries(value)) {
            const targetIdx = Number(targetIdxRaw);
            const scores = asNumberArray(scoresRaw);
            if (!Number.isFinite(targetIdx) || !scores) continue;
            const idx = Math.trunc(targetIdx);
            if (idx < 0 || idx >= tokenCount) continue;
            items.push({ targetIdx: idx, scores });
        }
    }

    return items;
}

function firstNumberArray(record: Record<string, unknown>, keys: string[]): number[] | null {
    for (const key of keys) {
        const value = asNumberArray(record[key]);
        if (value) return value;
    }
    return null;
}

function topCorrelationsFromScores(tokens: string[], targetIdx: number, targetToken: string, scores: number[]): TestCorrelation[] {
    return scores
        .map((score, idx) => ({ idx, score }))
        .filter(({ idx, score }) => idx !== targetIdx && idx < tokens.length && Number.isFinite(score) && score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, 12)
        .map(({ idx, score }) => ({
            source_token: tokens[idx] ?? '',
            source_token_index: idx,
            target_token: targetToken,
            target_token_index: targetIdx,
            saliency_score: score,
        }));
}

function genericSaliencyToAllTokensReport(payload: unknown): AllTokensReport | null {
    if (!isRecord(payload)) return null;

    const targetTestSample = payload.target_test_sample;
    const legacyBefore = isRecord(targetTestSample) && isRecord(targetTestSample.before)
        ? targetTestSample.before
        : null;
    const source = legacyBefore ?? payload;

    const tokens = firstStringArray(source, ['full_tokens', 'tokens', 'token_list'])
        ?? firstStringArray(payload, ['full_tokens', 'tokens', 'token_list']);
    if (!tokens || tokens.length === 0) return null;

    const promptLenRaw = firstNumber(source, ['prompt_len', 'start_index', 'answer_start_index'])
        ?? firstNumber(payload, ['prompt_len', 'start_index', 'answer_start_index'])
        ?? tokens.length;
    const promptLen = Math.max(0, Math.min(tokens.length, Math.trunc(promptLenRaw)));
    const correctTokens = firstStringArray(source, ['correct_full_tokens', 'correct_tokens'])
        ?? firstStringArray(payload, ['correct_full_tokens', 'correct_tokens'])
        ?? tokens;
    const saliencyItems = readGenericSaliencyItems(source, tokens.length);
    if (saliencyItems.length === 0) return null;

    const experimentMeta = payload.experiment_meta;
    const sampleIndex = firstNumber(payload, ['test_sample_index', 'sample_index'])
        ?? (isRecord(experimentMeta) ? firstNumber(experimentMeta, ['test_sample_index']) : null)
        ?? -1;

    return {
        experiment_meta: {
            test_sample_index: Math.trunc(sampleIndex),
            mode: 'all_tokens',
            tokens_analyzed: saliencyItems.length,
        },
        test_sample_baseline: {
            full_tokens: tokens,
            correct_full_tokens: correctTokens,
            prompt_len: promptLen,
        },
        per_token_results: saliencyItems.map(item => {
            const targetToken = item.targetToken ?? tokens[item.targetIdx] ?? '';
            return {
                target_token_index: item.targetIdx,
                target_token: targetToken,
                top_correlations: topCorrelationsFromScores(tokens, item.targetIdx, targetToken, item.scores),
                correlation_pairs: [],
            };
        }),
        train_sample_details: {},
    };
}

export function normalizeImportedReport(payload: unknown, sourceName: string): ImportedReport {
    if (isAllTokensReportLike(payload)) {
        const report = normalizeAllTokensReport(payload);
        return {
            report,
            meta: buildImportedMeta(report, sourceName),
            format: 'all_tokens',
        };
    }

    const genericReport = genericSaliencyToAllTokensReport(payload);
    if (genericReport) {
        return {
            report: genericReport,
            meta: buildImportedMeta(genericReport, sourceName),
            format: 'generic_saliency',
        };
    }

    throw new Error('Unsupported JSON format. Expected an all-token report, or tokens + prompt_len + saliency_list/saliency_by_target.');
}

// Where the visualizer shows up when Open Visualizer / Open Full Probe is used.
// 'inline' renders the plot in this page; 'window' is the original behaviour —
// open the TTAV web app in a new tab — kept intact so the full tool (neighbor
// lines, refine, time travel) stays one click away.
//
// Prepare sample shows nothing in either mode: it only makes the bundle ready,
// exactly as it did before.
type VisualizerMode = 'inline' | 'window';
const DEFAULT_VISUALIZER_MODE: VisualizerMode = 'inline';

interface TtavLaunchPrefs {
    ttavUrl: string;
    contentPathTemplate: string;
    eifBundleCacheTemplate: string;
    visMethod: string;
    visId: string;
    eifApiUrl: string;
    visualizerMode: VisualizerMode;
}

interface TtavJumpPayload {
    source: 'eif';
    sampleId: string;
    contentPath: string;
    visMethod: string;
    visId: string;
    dataType: 'Text';
    taskType: 'Alignment';
    selectedIndices: number[];
    targetIndex?: number;
    selectedSourceIndex?: number;
    promptLen: number;
    // Probe launches only: which pairs the user ticked here, so the visualizer
    // opens showing just those links. The bundle still carries every pair of the
    // group — narrowing it would make each tick a different bundle to precompute,
    // so the filtering is a display concern on the other side.
    visiblePairIds?: string[];
    // Each matched pair is two *edges* — one inside the train sample, one inside
    // the test sample — plus the gradient similarity between them. cos_sim is the
    // report's own verdict on the match and isn't in the bundle (pair_signature
    // omits it), so it travels with the jump instead of forcing a regenerate.
    probeEdges?: {
        pairId: string;
        cosSim: number;
        trainSourceIndex: number;
        trainTargetIndex: number;
        testSourceIndex: number;
        testTargetIndex: number;
    }[];
}

interface TtavStaticBundlePayload {
    sample_id: string;
    vis_method: string;
    vis_id: string;
    overwrite: boolean;
    bundle: {
        model: string;
        classes: string[];
        sample_index: number;
        prompt_len: number;
        labels: number[];
        text_list: string[];
        text_data: string[];
        token_list: string[];
        index: { train: number[]; test: number[] };
        embeddings: number[][];
        projection: number[][];
    };
}

interface PreparedTtavBundleRecord {
    sampleId: string;
    contentPath: string;
    visMethod: string;
    visId: string;
    preparedAt: number;
}

export interface TtavHighlightUpdateMessage {
    command: 'eifHighlightUpdate';
    data: TtavJumpPayload;
}

interface EifPrepareStatusPayload {
    status: 'success';
    sampleId: string;
    stage: string;
    message: string;
    active: boolean;
    error: boolean;
    updatedAt: number;
}

// Floating-plot geometry the user dragged to. `width` is the panel, `height` is
// the canvas alone (the header, legend and hint sit outside it).
interface InlinePlotSize {
    width: number;
    height: number;
}

interface InlinePlotPos {
    left: number;
    top: number;
}

const INLINE_PLOT_MIN_SIDE = 280;
// Header + hint strip under the canvas; subtracted so the *window* reads square.
const INLINE_PLOT_CHROME = 96;

function loadInlinePlotSize(): InlinePlotSize | null {
    if (typeof window === 'undefined') return null;
    try {
        const raw = window.localStorage.getItem(INLINE_PLOT_SIZE_KEY);
        if (!raw) return null;
        const parsed = JSON.parse(raw) as Partial<InlinePlotSize>;
        if (typeof parsed.width !== 'number' || typeof parsed.height !== 'number') return null;
        // Migrate old wide rectangles to a square side.
        const side = Math.max(INLINE_PLOT_MIN_SIDE, Math.min(parsed.width, parsed.height));
        return { width: side, height: side };
    } catch {
        return null;
    }
}

function saveInlinePlotSize(size: InlinePlotSize) {
    if (typeof window === 'undefined') return;
    try {
        window.localStorage.setItem(INLINE_PLOT_SIZE_KEY, JSON.stringify(size));
    } catch {
        // Private-mode storage failures shouldn't break the plot.
    }
}

function loadInlinePlotPos(): InlinePlotPos | null {
    if (typeof window === 'undefined') return null;
    try {
        const raw = window.localStorage.getItem(INLINE_PLOT_POS_KEY);
        if (!raw) return null;
        const parsed = JSON.parse(raw) as Partial<InlinePlotPos>;
        if (typeof parsed.left !== 'number' || typeof parsed.top !== 'number') return null;
        return { left: parsed.left, top: parsed.top };
    } catch {
        return null;
    }
}

function saveInlinePlotPos(pos: InlinePlotPos | null) {
    if (typeof window === 'undefined') return;
    try {
        if (pos === null) window.localStorage.removeItem(INLINE_PLOT_POS_KEY);
        else window.localStorage.setItem(INLINE_PLOT_POS_KEY, JSON.stringify(pos));
    } catch {
        // ignore
    }
}

function clampInlinePlotPos(
    pos: InlinePlotPos,
    panelWidth: number,
    viewport: { width: number; height: number },
): InlinePlotPos {
    const margin = 8;
    const maxLeft = Math.max(margin, viewport.width - panelWidth - margin);
    // Keep at least the header bar on screen.
    const maxTop = Math.max(margin, viewport.height - 48);
    return {
        left: Math.min(Math.max(margin, pos.left), maxLeft),
        top: Math.min(Math.max(margin, pos.top), maxTop),
    };
}

function loadPreparedTtavBundles(): Record<string, PreparedTtavBundleRecord> {
    if (typeof window === 'undefined') return {};

    try {
        const raw = window.localStorage.getItem(TTAV_PREPARED_BUNDLES_KEY);
        if (!raw) return {};
        const parsed = JSON.parse(raw) as Record<string, PreparedTtavBundleRecord>;
        return parsed && typeof parsed === 'object' ? parsed : {};
    } catch {
        return {};
    }
}

export function savePreparedTtavBundle(record: PreparedTtavBundleRecord) {
    if (typeof window === 'undefined') return;

    const current = loadPreparedTtavBundles();
    current[record.sampleId] = record;
    window.localStorage.setItem(TTAV_PREPARED_BUNDLES_KEY, JSON.stringify(current));
}

export function getPreparedTtavBundle(sampleId: string): PreparedTtavBundleRecord | null {
    const current = loadPreparedTtavBundles();
    return current[sampleId] ?? null;
}

export async function loadPrecomputedRealBundle(sampleId: string, visMethod: string, visId: string): Promise<TtavStaticBundlePayload> {
    const bundleResp = await fetch(`/data/real-bundles/${encodeURIComponent(sampleId)}/bundle_payload.json`, {
        cache: 'no-store',
    });

    if (!bundleResp.ok) {
        throw new Error(`Precomputed real bundle not found for ${sampleId} (HTTP ${bundleResp.status}).`);
    }

    const payload = await bundleResp.json() as TtavStaticBundlePayload;
    return {
        ...payload,
        sample_id: sampleId,
        vis_method: visMethod,
        vis_id: visId,
        overwrite: true,
    };
}

function buildEifApiUrl(apiUrl: string, pathname: string): string {
    const url = new URL(apiUrl.trim());
    url.pathname = pathname;
    url.search = '';
    return url.toString();
}

function buildEifPrepareStatusUrl(apiUrl: string, sampleId: string): string {
    const url = new URL(buildEifApiUrl(apiUrl, '/api/prepare-ttav-bundle-status'));
    url.searchParams.set('sampleId', sampleId);
    return url.toString();
}

export async function fetchEifPrepareStatus(
    apiUrl: string,
    sampleId: string,
    timeoutMs = 3000,
): Promise<EifPrepareStatusPayload | null> {
    if (!apiUrl.trim() || !sampleId) return null;

    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);

    try {
        const resp = await fetch(buildEifPrepareStatusUrl(apiUrl, sampleId), {
            cache: 'no-store',
            signal: controller.signal,
        });
        if (!resp.ok) return null;

        const payload = await resp.json() as Partial<EifPrepareStatusPayload> & { status?: string };
    if (payload.status !== 'success' || typeof payload.message !== 'string' || typeof payload.sampleId !== 'string') {
        return null;
    }

        return {
            status: 'success',
            sampleId: payload.sampleId,
            stage: typeof payload.stage === 'string' ? payload.stage : 'unknown',
            message: payload.message,
            active: payload.active === true,
            error: payload.error === true,
            updatedAt: typeof payload.updatedAt === 'number' ? payload.updatedAt : Date.now(),
        };
    } catch {
        return null;
    } finally {
        window.clearTimeout(timeoutId);
    }
}

function loadTtavLaunchPrefs(): TtavLaunchPrefs {
    const fallbackPrefs: TtavLaunchPrefs = {
        ttavUrl: DEFAULT_TTAV_URL,
        contentPathTemplate: DEFAULT_TTAV_CONTENT_PATH_TEMPLATE,
        eifBundleCacheTemplate: DEFAULT_EIF_BUNDLE_CACHE_TEMPLATE,
        visMethod: DEFAULT_TTAV_METHOD,
        visId: DEFAULT_TTAV_VIS_ID,
        eifApiUrl: DEFAULT_EIF_API_URL,
        visualizerMode: DEFAULT_VISUALIZER_MODE,
    };

    if (typeof window === 'undefined') {
        return fallbackPrefs;
    }

    try {
        const raw = window.localStorage.getItem(TTAV_PREFS_KEY);
        if (!raw) throw new Error('missing prefs');
        const parsed = JSON.parse(raw) as Partial<TtavLaunchPrefs>;

        const legacyUrl = parsed.ttavUrl?.includes('localhost:5174');
        const legacyMethod = parsed.visMethod?.trim().toUpperCase() === 'UMAP';
        const legacyPath = parsed.contentPathTemplate?.includes('/root/project/time-travelling-visualizer/data/eif_bundles/');
        const legacyEifCachePath = parsed.eifBundleCacheTemplate?.includes('/root/project/time-travelling-visualizer/data/eif_bundles/')
            || parsed.eifBundleCacheTemplate?.includes('/root/project/Empirical-Influence-Function/ttav_bundles/');
        const defaultApiHost = new URL(DEFAULT_EIF_API_URL).host;
        const parsedApiHost = parsed.eifApiUrl ? (() => {
            try {
                return new URL(parsed.eifApiUrl).host;
            } catch {
                return '';
            }
        })() : '';
        const legacyApiUrl = parsed.eifApiUrl?.includes('124.70.161.19:8765')
            || parsed.eifApiUrl?.includes('0.0.0.0:8765')
            || parsed.eifApiUrl?.includes('127.0.0.1:8765')
            || parsed.eifApiUrl?.includes('localhost:8765')
            || (parsedApiHost !== '' && parsedApiHost !== defaultApiHost);

        return {
            ttavUrl: legacyUrl ? DEFAULT_TTAV_URL : (parsed.ttavUrl || DEFAULT_TTAV_URL),
            contentPathTemplate: legacyPath
                ? DEFAULT_TTAV_CONTENT_PATH_TEMPLATE
                : (parsed.contentPathTemplate || DEFAULT_TTAV_CONTENT_PATH_TEMPLATE),
            eifBundleCacheTemplate: legacyEifCachePath
                ? DEFAULT_EIF_BUNDLE_CACHE_TEMPLATE
                : (parsed.eifBundleCacheTemplate || DEFAULT_EIF_BUNDLE_CACHE_TEMPLATE),
            visMethod: legacyMethod ? DEFAULT_TTAV_METHOD : (parsed.visMethod || DEFAULT_TTAV_METHOD),
            visId: parsed.visId || DEFAULT_TTAV_VIS_ID,
            eifApiUrl: legacyApiUrl ? DEFAULT_EIF_API_URL : (parsed.eifApiUrl || DEFAULT_EIF_API_URL),
            visualizerMode: parsed.visualizerMode === 'window' ? 'window' : DEFAULT_VISUALIZER_MODE,
        };
    } catch {
        return fallbackPrefs;
    }
}

export function resolveContentPath(template: string, sampleId: string): string {
    return template
        .replaceAll('{sampleId}', sampleId)
        .replaceAll('{taskId}', sampleId);
}

function inferSampleIdFromMeta(meta: AllTokensExperimentMeta, report?: AllTokensReport | null): string {
    // Same regex as infer_sample_id() in src/export_ttav_bundle.py — must stay in
    // sync so the id the frontend asks for matches the directory name the backend
    // actually wrote to ttav_bundles/ and ttav_bundles_real/. Non-greedy on the
    // first group so it captures the full "{model}_{task}" prefix (e.g.
    // "ce_only_codesearchnet_go_test_...") while still handling legacy filenames
    // with a trailing suffix after "_all_tokens" (e.g. "..._all_tokens_new.json").
    const fileStem = meta.fileName.replace(/\.json$/i, '');
    const match = fileStem.match(/^correlation_matching_results_(.+?)_all_tokens(?:_(.+))?$/);
    if (match) {
        const [, prefix, suffix] = match;
        // A trailing "salr5-8" records the attribution parameters, not which
        // sample this is; the generator drops it when naming bundles, so keeping
        // it would point every lookup at a directory that was never written.
        // Other suffixes (e.g. "_new") do identify the sample and are kept.
        if (!suffix || /^salr[\d-]+$/i.test(suffix)) return prefix;
        return `${prefix}_${suffix}`;
    }
    // report.experiment_meta.task_id is model-agnostic (model lives separately in
    // model_name) — only safe to use as a last resort when the filename doesn't
    // follow the expected convention at all, since using it directly would drop
    // the model prefix and collide ce_only/ce_saliency bundles for the same task.
    if (report?.experiment_meta.task_id) {
        const modelName = report.experiment_meta.model_name;
        return modelName ? `${modelName}_${report.experiment_meta.task_id}` : report.experiment_meta.task_id;
    }
    return meta.taskId || meta.label || (report ? `test${report.experiment_meta.test_sample_index}` : fileStem);
}

// ─── Token helpers ────────────────────────────────────────────────────────────

function decodeToken(t: string): string {
    return t.replaceAll('Ċ', '\n').replaceAll('Ġ', ' ').replaceAll('ĉ', '  ');
}

function decodeTokens(tokens: string[]): string[] {
    return tokens.map(decodeToken);
}

function cosSimilarityColor(s: number): { bg: string; fg: string } {
    if (s > 0.6) return { bg: '#dcfce7', fg: '#15803d' };
    if (s > 0.3) return { bg: '#fef9c3', fg: '#854d0e' };
    return { bg: '#fee2e2', fg: '#b91c1c' };
}

export function shouldFallbackToDirectPrepare(message: string): boolean {
    const lower = message.toLowerCase();
    return lower.includes('failed to fetch')
        || lower.includes('networkerror')
        || lower.includes('load failed')
        || lower.includes('not found')
        || lower.includes('http 404')
        || lower.includes('http 500')
        || lower.includes('http 502')
        || lower.includes('http 503')
        || lower.includes('http 504')
        || lower.includes('eif local bundle cache not found')
        || lower.includes('non-json response');
}


// ─── Export helpers ───────────────────────────────────────────────────────────

function formatContextForExport(tokens: string[]): string {
    return tokens.map(tok => {
        const marked = tok.startsWith('→[') && tok.endsWith(']←');
        const raw = decodeToken(marked ? tok.slice(2, -2) : tok);
        return marked ? `[${raw.trim() || '·'}]` : raw;
    }).join('');
}

interface ExportOptions {
    tokenRange: 'all' | 'selected' | 'range';
    selectedTokenIdx?: number;
    rangeFrom?: number;
    rangeTo?: number;
    cosSimThreshold: number;
    hideZero: boolean;
}

export function generateExportMarkdown(report: AllTokensReport, options: ExportOptions): string {
    const { test_sample_baseline: baseline, per_token_results, train_sample_details, experiment_meta } = report;
    const promptLen = baseline.prompt_len;

    let tokensToExport: PerTokenResult[];
    if (options.tokenRange === 'selected' && options.selectedTokenIdx != null) {
        const r = per_token_results.find(r => r.target_token_index === options.selectedTokenIdx);
        tokensToExport = r ? [r] : [];
    } else if (options.tokenRange === 'range' && options.rangeFrom != null && options.rangeTo != null) {
        tokensToExport = per_token_results.filter(r =>
            r.target_token_index >= options.rangeFrom! && r.target_token_index <= options.rangeTo!
        );
    } else {
        tokensToExport = [...per_token_results];
    }

    const keepPair = (p: CorrelationPair) =>
        p.cos_sim >= options.cosSimThreshold && !(options.hideZero && p.cos_sim === 0);

    const L: string[] = [];

    // Header
    L.push(`# Attribution Report: Test Sample #${experiment_meta.test_sample_index}\n`);
    L.push(`- Tokens exported: ${tokensToExport.length} / ${per_token_results.length} analyzed`);
    L.push(`- Prompt length: ${promptLen}`);
    L.push(`- cos_sim filter: ≥${options.cosSimThreshold.toFixed(3)}${options.hideZero ? ', hiding cos=0' : ''}`);
    L.push('');

    // Test Code
    const promptText = decodeTokens(baseline.full_tokens.slice(0, promptLen)).join('');
    const modelResp = decodeTokens(baseline.full_tokens.slice(promptLen)).join('');
    const gtTokens = baseline.correct_full_tokens ?? [];
    const correctResp = decodeTokens(gtTokens.slice(promptLen)).join('');

    L.push('## Test Code\n');
    L.push('### Prompt\n' + '```');
    L.push(promptText.trimEnd());
    L.push('```' + '\n');
    L.push('### Model Output (response)\n' + '```');
    L.push(modelResp.trimEnd());
    L.push('```' + '\n');
    if (correctResp) {
        L.push('### Ground Truth (response)\n' + '```');
        L.push(correctResp.trimEnd());
        L.push('```' + '\n');
    }

    // Per-token analysis
    L.push('## Token Analysis\n');

    for (const tr of tokensToExport) {
        const tgt = decodeToken(tr.target_token).trim() || '·';
        const absIdx = tr.target_token_index;
        const correctTok = gtTokens[absIdx];
        const correctText = correctTok ? (decodeToken(correctTok).trim() || '·') : '?';
        const isCorrect = tr.target_token === correctTok;

        L.push(`### \`${tgt}\` @ pos ${absIdx} | GT: \`${correctText}\` | ${isCorrect ? '✓' : '✗'}\n`);

        // Feature Attribution
        L.push('Feature Attribution:');
        for (const c of tr.top_correlations) {
            const src = decodeToken(c.source_token).trim() || '·';
            L.push(`- \`${src}\`@${c.source_token_index} → \`${tgt}\` (saliency: ${c.saliency_score.toFixed(4)})`);
        }
        L.push('');

        // Data Attribution
        const pairs = tr.correlation_pairs.filter(keepPair);
        if (pairs.length === 0) {
            L.push('Data Attribution: No matches above threshold.\n');
            L.push('---\n');
            continue;
        }

        const byTrain = new Map<number, CorrelationPair[]>();
        for (const p of pairs) {
            if (!byTrain.has(p.train_sample_id)) byTrain.set(p.train_sample_id, []);
            byTrain.get(p.train_sample_id)!.push(p);
        }

        L.push('Data Attribution:');
        for (const [tid, tpairs] of byTrain) {
            const detail = train_sample_details[String(tid)];
            const coarse = detail?.coarse_cos_sim ?? tpairs[0]?.coarse_cos_sim ?? 0;
            L.push(`\n**Train #${tid}** (coarse: ${coarse.toFixed(4)}):`);
            for (const p of tpairs.sort((a, b) => b.cos_sim - a.cos_sim)) {
                const tSrc = decodeToken(p.test_correlation.source_token).trim() || '·';
                const tTgt = decodeToken(p.test_correlation.target_token).trim() || '·';
                const rSrc = decodeToken(p.train_correlation.source_token).trim() || '·';
                const rTgt = decodeToken(p.train_correlation.target_token).trim() || '·';
                const srcCtx = formatContextForExport(p.train_context.source_context);
                const tgtCtx = formatContextForExport(p.train_context.target_context);
                L.push(`- cos=${p.cos_sim.toFixed(4)} | test: \`${tSrc}\`→\`${tTgt}\` ⇔ train: \`${rSrc}\`→\`${rTgt}\``);
                L.push(`  src_ctx: ${srcCtx}`);
                L.push(`  tgt_ctx: ${tgtCtx}`);
            }
        }
        L.push('\n---\n');
    }

    // Training Samples 汇总（每个 train sample 只出现一次）
    const referencedTrainIds = new Set<number>();
    for (const tr of tokensToExport) {
        for (const p of tr.correlation_pairs.filter(keepPair)) {
            referencedTrainIds.add(p.train_sample_id);
        }
    }

    if (referencedTrainIds.size > 0) {
        L.push('## Training Samples\n');
        for (const tid of Array.from(referencedTrainIds).sort((a, b) => a - b)) {
            const detail = train_sample_details[String(tid)];
            if (!detail) continue;
            const tokens = decodeTokens(detail.full_tokens);
            const promptPart = tokens.slice(0, detail.answer_start_index).join('');
            const respPart = tokens.slice(detail.answer_start_index).join('');
            L.push(`### Train #${tid} (coarse: ${detail.coarse_cos_sim.toFixed(4)})\n`);
            L.push('Prompt:\n' + '```');
            L.push(promptPart.trimEnd());
            L.push('```' + '\n');
            L.push('Response:\n' + '```');
            L.push(respPart.trimEnd());
            L.push('```' + '\n');
        }
    }

    return L.join('\n');
}

export function downloadMarkdown(text: string, filename: string) {
    const blob = new Blob([text], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

// ─── Token Renderer ───────────────────────────────────────────────────────────

type TokenState = 'normal' | 'response' | 'response-model' | 'response-gold' | 'selected' | 'source-highlight' | 'analyzed' | 'analyzed-gold';
type ResponseTone = 'default' | 'model' | 'gold';

/** Static map so CSS-modules never drop dynamically accessed classes. */
const TOKEN_STATE_CLASS: Record<TokenState, string> = {
    normal: styles['token-normal'],
    response: styles['token-response'],
    'response-model': styles['token-response-model'],
    'response-gold': styles['token-response-gold'],
    selected: styles['token-selected'],
    'source-highlight': styles['token-source-highlight'],
    analyzed: styles['token-analyzed'],
    'analyzed-gold': styles['token-analyzed-gold'],
};

function TokenSpan({
    token,
    state,
    onClick,
    title,
    hovered = false,
    annotated = false,
    onHoverChange,
}: {
    token: string;
    state: TokenState;
    onClick?: () => void;
    title?: string;
    /** Linked to the scatter plot: true when this token's point is hovered. */
    hovered?: boolean;
    /** GT annotation source for the current saliency target(s). */
    annotated?: boolean;
    onHoverChange?: (hovered: boolean) => void;
}) {
    const display = token === '\n' ? '↵\n' : token === '  ' ? '→' : token;
    const ref = useRef<HTMLSpanElement | null>(null);

    // Code blocks scroll inside a fixed height, so the linked token is usually
    // out of view — highlighting it without scrolling would look like nothing
    // happened.
    //
    // scrollIntoView walks every scrollable ancestor, so this scrolls the <pre>
    // *and* the page. That is wanted here: the plot is a floating panel pinned to
    // the viewport, so the page is free to travel to the token without carrying
    // the plot away. (It was briefly restricted to the <pre>'s own scrollTop,
    // back when the plot sat in the document and page scrolling hid it.)
    useEffect(() => {
        if (hovered) ref.current?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    }, [hovered]);

    return (
        <span
            ref={ref}
            className={`${styles.token} ${TOKEN_STATE_CLASS[state]}`
                + (hovered ? ` ${styles['token-linked']}` : '')
                + (annotated ? ` ${styles['token-annotated-source']}` : '')}
            onClick={onClick}
            title={title}
            onMouseEnter={onHoverChange ? () => onHoverChange(true) : undefined}
            onMouseLeave={onHoverChange ? () => onHoverChange(false) : undefined}
            style={{ cursor: onClick ? 'pointer' : 'default' }}
        >
            {display}
        </span>
    );
}

// ─── Code Panel (tokens display) ─────────────────────────────────────────────

function resolveResponseState(tone: ResponseTone): TokenState {
    if (tone === 'model') return 'response-model';
    if (tone === 'gold') return 'response-gold';
    return 'response';
}

function CodeTokenStream({
    tokens,
    promptLen,
    responseTone = 'default',
    highlightSourceIndices,
    selectedTargetIndex,
    analyzedIndices,
    onTokenClick,
    linkedTokenIndex,
    onTokenHover,
    compact = false,
}: {
    tokens: string[];
    promptLen: number;
    responseTone?: ResponseTone;
    highlightSourceIndices?: Set<number>;
    selectedTargetIndex?: number;
    analyzedIndices?: Set<number>;
    onTokenClick?: (idx: number) => void;
    linkedTokenIndex?: number | null;
    onTokenHover?: (idx: number | null) => void;
    compact?: boolean;
}) {
    const responseState = resolveResponseState(responseTone);
    return (
        <pre className={`${styles.codeBlock}${compact ? ` ${styles.codeBlockCompact}` : ''}`}>
            <code>
                {tokens.map((tok, i) => {
                    const isResponse = i >= promptLen;
                    const isSelected = i === selectedTargetIndex;
                    const isSource = highlightSourceIndices?.has(i) ?? false;
                    const isAnalyzed = analyzedIndices?.has(i) ?? false;

                    let state: TokenState = 'normal';
                    if (isSelected) state = 'selected';
                    else if (isSource) state = 'source-highlight';
                    else if (isAnalyzed && isResponse) {
                        // Keep gold tokens green; model analyzed stays pink.
                        state = responseTone === 'gold' ? 'analyzed-gold' : 'analyzed';
                    }
                    else if (isResponse) state = responseState;

                    // Clickable iff this index has per_token_results. Do NOT gate on
                    // isTrivialToken: intervention now attributes punctuation (}, ), …).
                    const clickable = Boolean(onTokenClick && isAnalyzed);
                    return (
                        <TokenSpan
                            key={i}
                            token={tok}
                            state={state}
                            onClick={clickable && onTokenClick ? () => onTokenClick(i) : undefined}
                            title={clickable ? `Token ${i}: "${decodeToken(tok)}"` : undefined}
                            hovered={linkedTokenIndex === i}
                            onHoverChange={onTokenHover
                                ? (isHovered) => onTokenHover(isHovered ? i : null)
                                : undefined}
                        />
                    );
                })}
            </code>
        </pre>
    );
}

/** Model (red) + Gold answer only (green) in one panel. */
function OutputComparePanel({
    modelTokens,
    goldResponseTokens,
    promptLen,
    highlightSourceIndices,
    selectedTargetIndex,
    analyzedIndices,
    onTokenClick,
    goldSelectedLocalIndex,
    goldHighlightSourceIndices,
    onGoldTokenClick,
    linkedTokenIndex,
    onTokenHover,
    saliencySelected,
    saliencySelectEnabled,
    onToggleSaliencySelect,
}: {
    modelTokens: string[];
    /** Gold answer tokens only — no prompt prefix. */
    goldResponseTokens: string[];
    promptLen: number;
    highlightSourceIndices?: Set<number>;
    selectedTargetIndex?: number;
    analyzedIndices?: Set<number>;
    onTokenClick?: (idx: number) => void;
    goldSelectedLocalIndex?: number | null;
    goldHighlightSourceIndices?: Set<number>;
    onGoldTokenClick?: (localIdx: number) => void;
    linkedTokenIndex?: number | null;
    onTokenHover?: (idx: number | null) => void;
    /** Whether the current test saliency edge is ticked for probe filtering. */
    saliencySelected?: boolean;
    saliencySelectEnabled?: boolean;
    onToggleSaliencySelect?: () => void;
}) {
    const hasGold = goldResponseTokens.length > 0;
    const goldAnalyzed = useMemo(
        () => new Set(goldResponseTokens.map((_, i) => i)),
        [goldResponseTokens],
    );
    return (
        <div className={styles.codePanel}>
            <div className={styles.codePanelHeader}>
                <span className={styles.badge} style={{ background: '#dc2626' }}>MODEL</span>
                <span className={styles.codePanelLabel}>
                    {hasGold ? 'Model Output vs Gold' : 'Model Output'}
                </span>
                {onToggleSaliencySelect && (
                    <button
                        type="button"
                        aria-pressed={saliencySelected === true}
                        disabled={!saliencySelectEnabled}
                        title={
                            !saliencySelectEnabled
                                ? '先点一个 output token，再选一条 source→target saliency 边'
                                : (saliencySelected
                                    ? '取消选择当前 saliency 对（不再单独筛 embedding）'
                                    : '选择当前 saliency 对，embedding 图只显示这对')
                        }
                        onClick={onToggleSaliencySelect}
                        style={{
                            marginLeft: 'auto',
                            border: saliencySelected ? '1px solid #7c3aed' : '1px solid #cbd5e1',
                            background: saliencySelected ? '#f5f3ff' : '#ffffff',
                            color: !saliencySelectEnabled
                                ? '#94a3b8'
                                : (saliencySelected ? '#6d28d9' : '#64748b'),
                            borderRadius: 999,
                            padding: '2px 8px',
                            fontSize: 11,
                            fontWeight: 700,
                            cursor: saliencySelectEnabled ? 'pointer' : 'not-allowed',
                            opacity: saliencySelectEnabled ? 1 : 0.55,
                        }}
                    >
                        {saliencySelected ? '已选' : '选择'}
                    </button>
                )}
            </div>
            <div className={styles.outputCompareBody}>
                <div className={styles.outputSection}>
                    <div className={styles.outputSectionHeader}>
                        <span className={`${styles.outputSectionTitle} ${styles.outputSectionTitleModel}`}>
                            Model
                        </span>
                    </div>
                    <CodeTokenStream
                        tokens={modelTokens}
                        promptLen={promptLen}
                        responseTone="model"
                        highlightSourceIndices={highlightSourceIndices}
                        selectedTargetIndex={selectedTargetIndex}
                        analyzedIndices={analyzedIndices}
                        onTokenClick={onTokenClick}
                        linkedTokenIndex={linkedTokenIndex}
                        onTokenHover={onTokenHover}
                        compact={hasGold}
                    />
                </div>
                {hasGold && (
                    <div className={styles.outputSection}>
                        <div className={styles.outputSectionHeader}>
                            <span className={`${styles.outputSectionTitle} ${styles.outputSectionTitleGold}`}>
                                Gold
                            </span>
                            <span style={{ marginLeft: 8, fontSize: 11, color: '#64748b' }}>
                                点击 → 现场 teacher-force 归因
                            </span>
                        </div>
                        <CodeTokenStream
                            tokens={goldResponseTokens}
                            promptLen={0}
                            responseTone="gold"
                            selectedTargetIndex={goldSelectedLocalIndex ?? undefined}
                            highlightSourceIndices={goldHighlightSourceIndices}
                            analyzedIndices={goldAnalyzed}
                            onTokenClick={onGoldTokenClick}
                            compact
                        />
                    </div>
                )}
            </div>
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

/** GT annotation edges: trainId -> (targetIdx -> sourceIdx[]). */
type TrainGtEdges = Record<string, Record<string, number[]>>;

function annotatedSourcesForPairs(
    edgesByTarget: Record<string, number[]> | undefined,
    pairs: CorrelationPair[],
): Set<number> {
    const out = new Set<number>();
    if (!edgesByTarget) return out;
    for (const pair of pairs) {
        const srcs = edgesByTarget[String(pair.train_correlation.target_token_index)];
        if (!srcs) continue;
        for (const src of srcs) out.add(src);
    }
    return out;
}

function TrainSampleViewer({
    detail,
    highlightPairs,
    linkedTokenIndex,
    onTokenHover,
    annotatedSourceIndices,
}: {
    detail: TrainSampleDetail;
    highlightPairs: CorrelationPair[];
    /** Train-side token currently hovered in the scatter plot, if any. */
    linkedTokenIndex?: number | null;
    onTokenHover?: (idx: number | null) => void;
    /** GT annotation sources for the highlighted pairs' train targets. */
    annotatedSourceIndices?: Set<number>;
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
                        const isAnnotated = annotatedSourceIndices?.has(i) ?? false;
                        let state: TokenState = i >= detail.answer_start_index ? 'response' : 'normal';
                        if (isTgt) state = 'selected';
                        else if (isSrc) state = 'source-highlight';
                        return (
                            <TokenSpan
                                key={i}
                                token={tok}
                                state={state}
                                hovered={linkedTokenIndex === i}
                                annotated={isAnnotated}
                                title={isAnnotated
                                    ? `GT annotation source → target @${[...targetIndices].join(',')}`
                                    : undefined}
                                onHoverChange={onTokenHover
                                    ? (isHovered) => onTokenHover(isHovered ? i : null)
                                    : undefined}
                            />
                        );
                    })}
                </code>
            </pre>
        </div>
    );
}

// ─── Correlation Pair Card ────────────────────────────────────────────────────

function formatSigned(value: number | null | undefined, digits = 4): string {
    if (value == null || Number.isNaN(value)) return '—';
    const sign = value > 0 ? '+' : '';
    return `${sign}${value.toFixed(digits)}`;
}

function unlearnVerdictLabel(verdict: string | undefined): { text: string; color: string } {
    switch (verdict) {
        case 'supports_causal':
            return { text: '像真因果：遗忘后 target 更难', color: '#166534' };
        case 'saliency_only':
            return { text: 'saliency 下降但 CE 几乎不动（test 已近满分）', color: '#a16207' };
        case 'no_effect':
            return { text: '几乎无影响', color: '#64748b' };
        case 'opposite_effect':
            return { text: '反向：遗忘后 target 反而更容易', color: '#9a3412' };
        default:
            return { text: '结果不明确', color: '#6b7280' };
    }
}

function PairCard({
    pair,
    detail,
    selected,
    onToggleSelect,
    annotatedSourceIndices,
    onUnlearn,
    unlearnBusy,
    unlearnResult,
    onOpenAnnotationViewer,
}: {
    pair: CorrelationPair;
    detail?: TrainSampleDetail;
    selected?: boolean;
    onToggleSelect?: () => void;
    annotatedSourceIndices?: Set<number>;
    onUnlearn?: () => void;
    unlearnBusy?: boolean;
    unlearnResult?: UnlearnPairResult | null;
    /** Open annotation-viewer focused on this train sample + target. */
    onOpenAnnotationViewer?: () => void;
}) {
    const [expanded, setExpanded] = useState(false);
    const { bg, fg } = cosSimilarityColor(pair.cos_sim);
    const verdict = unlearnVerdictLabel(unlearnResult?.verdict);

    return (
        <div
            className={styles.pairCard}
            style={selected ? { borderColor: '#7c3aed', boxShadow: '0 0 0 1px rgba(124,58,237,0.18)' } : undefined}
        >
            <div
                className={styles.pairCardHeader}
                onClick={() => setExpanded(e => !e)}
            >
                {onToggleSelect && (
                    <button
                        type="button"
                        onClick={(event) => {
                            event.stopPropagation();
                            onToggleSelect();
                        }}
                        aria-pressed={selected === true}
                        title={selected ? '取消选择这个 correlation pair' : '选择这个 correlation pair：embedding 图只显示已选对'}
                        style={{
                            border: selected ? '1px solid #7c3aed' : '1px solid #cbd5e1',
                            background: selected ? '#f5f3ff' : '#ffffff',
                            color: selected ? '#6d28d9' : '#64748b',
                            borderRadius: 999,
                            padding: '2px 8px',
                            fontSize: 11,
                            fontWeight: 700,
                            cursor: 'pointer',
                        }}
                    >
                        {selected ? '已选' : '选择'}
                    </button>
                )}

                {onUnlearn && (
                    <button
                        type="button"
                        onClick={(event) => {
                            event.stopPropagation();
                            onUnlearn();
                        }}
                        disabled={unlearnBusy}
                        title="对该 train target 在匹配参数空间（默认 last-1 LoRA）做一步 CE ascent，看 test 的 logP / saliency 是否变化"
                        style={{
                            border: '1px solid #fda4af',
                            background: unlearnBusy ? '#ffe4e6' : '#fff1f2',
                            color: '#be123c',
                            borderRadius: 999,
                            padding: '2px 8px',
                            fontSize: 11,
                            fontWeight: 700,
                            cursor: unlearnBusy ? 'wait' : 'pointer',
                        }}
                    >
                        {unlearnBusy ? 'Unlearning…' : 'Unlearn'}
                    </button>
                )}

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

                <span
                    className={styles.trainBadge}
                    role="link"
                    tabIndex={0}
                    title="在 annotation-viewer 中打开该 train 样本并选中本 pair 的 target"
                    onClick={(event) => {
                        event.stopPropagation();
                        onOpenAnnotationViewer?.();
                    }}
                    onKeyDown={(event) => {
                        if (event.key !== 'Enter' && event.key !== ' ') return;
                        event.preventDefault();
                        event.stopPropagation();
                        onOpenAnnotationViewer?.();
                    }}
                    style={onOpenAnnotationViewer ? { cursor: 'pointer', textDecoration: 'underline' } : undefined}
                >
                    TRAIN #{pair.train_sample_id}
                </span>
                <span className={styles.expandIcon}>{expanded ? '▼' : '▶'}</span>
            </div>

            {unlearnResult && (
                <div style={{
                    margin: '0 10px 8px',
                    padding: '8px 10px',
                    borderRadius: 8,
                    background: '#fff7ed',
                    border: '1px solid #fed7aa',
                    fontSize: 11,
                    color: '#9a3412',
                    display: 'grid',
                    gap: 4,
                }}>
                    <div style={{ fontWeight: 700, color: verdict.color }}>{verdict.text}</div>
                    <div>
                        ΔlogP(test target) = <strong>{formatSigned(unlearnResult.delta?.logprob)}</strong>
                        {' · '}
                        Δsaliency = <strong>{formatSigned(unlearnResult.delta?.saliency, 6)}</strong>
                        {' · '}
                        ΔCE = <strong>{formatSigned(unlearnResult.delta?.ce)}</strong>
                    </div>
                    <div style={{ color: '#78716c' }}>
                        before logP {formatSigned(unlearnResult.before?.logprob)} → after {formatSigned(unlearnResult.after?.logprob)}
                        {unlearnResult.before?.saliency != null && (
                            <> · sal {formatSigned(unlearnResult.before.saliency, 6)} → {formatSigned(unlearnResult.after?.saliency, 6)}</>
                        )}
                    </div>
                    {unlearnResult.testEdge?.reportedSaliency != null && (
                        <div style={{ color: '#a8a29e' }}>
                            报告里该边 saliency = {formatSigned(unlearnResult.testEdge.reportedSaliency, 4)}
                            {unlearnResult.before?.saliency != null && (
                                <>
                                    {' · '}重算 before = {formatSigned(unlearnResult.before.saliency, 4)}
                                    {Math.abs(
                                        (unlearnResult.before.saliency ?? 0)
                                        - (unlearnResult.testEdge.reportedSaliency ?? 0),
                                    ) > Math.max(0.5, 0.2 * Math.abs(unlearnResult.testEdge.reportedSaliency ?? 0))
                                        ? ' ⚠ 与报告不一致（多为 checkpoint 不同或 hidden 层索引）'
                                        : ''}
                                </>
                            )}
                        </div>
                    )}
                    <div style={{ color: '#a8a29e' }}>
                        一步 {unlearnResult.update?.paramSpace || 'match-space'} 探针
                        {unlearnResult.update?.lastNLayers != null
                            ? ` · last-${unlearnResult.update.lastNLayers}`
                            : ''}
                        {unlearnResult.testEdge?.saliencyMode
                            ? ` · ${unlearnResult.testEdge.saliencyMode}`
                            : ''}
                        {' · '}已 restore · 不写回 adapter
                    </div>
                    {unlearnResult.error && (
                        <div style={{ color: '#b91c1c' }}>{unlearnResult.error}</div>
                    )}
                </div>
            )}

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
                        <TrainSampleViewer
                            detail={detail}
                            highlightPairs={[pair]}
                            annotatedSourceIndices={annotatedSourceIndices}
                        />
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
    onProbeEmbeddings,
    probeBusy,
    selectedPairIds,
    onTogglePairSelection,
    comparisonSummary,
    linkedTokenIndex,
    onTokenHover,
    gtEdgesByTarget,
    onUnlearnPair,
    unlearningPairId,
    unlearnResults,
    onOpenAnnotationViewer,
}: {
    trainIdx: number;
    pairs: CorrelationPair[];
    detail?: TrainSampleDetail;
    onProbeEmbeddings?: (trainIdx: number, pairs: CorrelationPair[]) => void;
    probeBusy?: boolean;
    selectedPairIds?: string[];
    onTogglePairSelection?: (trainIdx: number, pairId: string) => void;
    comparisonSummary?: TrainProbeComparisonSummary;
    /** Set only for the group the open probe belongs to. */
    linkedTokenIndex?: number | null;
    onTokenHover?: (idx: number | null) => void;
    /** GT annotation edges for this train sample: targetIdx -> sourceIdx[]. */
    gtEdgesByTarget?: Record<string, number[]>;
    onUnlearnPair?: (pair: CorrelationPair) => void;
    unlearningPairId?: string | null;
    unlearnResults?: Record<string, UnlearnPairResult>;
    onOpenAnnotationViewer?: (pair: CorrelationPair) => void;
}) {
    const [collapsed, setCollapsed] = useState(false);
    const annotatedSourceIndices = useMemo(
        () => annotatedSourcesForPairs(gtEdgesByTarget, pairs),
        [gtEdgesByTarget, pairs],
    );
    const bestSim = Math.max(...pairs.map(p => p.cos_sim));
    const { bg, fg } = cosSimilarityColor(bestSim);
    const selectedPairIdSet = useMemo(() => new Set(selectedPairIds ?? []), [selectedPairIds]);
    const selectedPairCount = selectedPairIdSet.size;
    const comparisonTokens = comparisonSummary?.focusTokens ?? [];
    // comparisonSummary.pairwiseCosine is intentionally not displayed: it measures
    // token-to-token similarity *within the train sample*, which says nothing
    // about whether a train↔test match is sound. Showing it next to the match
    // invited reading it as the verdict — cos_sim is that, and it's already in the
    // pair rows above. The field stays in the payload for other uses.

    return (
        <div className={styles.trainGroup}>
            <div className={styles.trainGroupHeader} onClick={() => setCollapsed(c => !c)}>
                <span className={styles.trainGroupId}>TRAIN #{trainIdx}</span>
                <span className={styles.trainGroupCoarse}>coarse {(detail?.coarse_cos_sim ?? pairs[0]?.coarse_cos_sim ?? 0).toFixed(4)}</span>
                <span className={styles.trainGroupCount}>{pairs.length} pairs</span>
                <span className={styles.cosSim} style={{ background: bg, color: fg }}>best {bestSim.toFixed(4)}</span>
                <span style={{ marginLeft: 8, fontSize: 11, color: '#6b7280', fontWeight: 600 }}>
                    已选 {selectedPairCount}
                </span>
                {detail && onProbeEmbeddings && (
                    <button
                        type="button"
                        onClick={event => {
                            event.stopPropagation();
                            onProbeEmbeddings(trainIdx, pairs);
                        }}
                        disabled={probeBusy}
                        style={{
                            marginLeft: 8,
                            padding: '4px 10px',
                            borderRadius: 999,
                            border: '1px solid #c4b5fd',
                            background: probeBusy ? '#ede9fe' : '#faf5ff',
                            color: '#6d28d9',
                            fontSize: 11,
                            fontWeight: 700,
                            cursor: probeBusy ? 'wait' : 'pointer',
                        }}
                    >
                        {probeBusy ? 'Probing…' : 'Open Full Probe'}
                    </button>
                )}
                <span className={styles.expandIcon} style={{ marginLeft: 'auto' }}>{collapsed ? '▶' : '▼'}</span>
            </div>
            {!collapsed && (
                <div className={styles.trainGroupBody}>
                    {detail && (
                        <div className={styles.trainFullView}>
                            <div className={styles.subLabel}>
                                完整训练样本 — 黄底=saliency source，橙底=target；
                                红下划线=该 target 的 GT 标注 source
                                {annotatedSourceIndices.size > 0 ? `（${annotatedSourceIndices.size}）` : ''}
                            </div>
                            <TrainSampleViewer
                                detail={detail}
                                highlightPairs={pairs}
                                linkedTokenIndex={linkedTokenIndex}
                                onTokenHover={onTokenHover}
                                annotatedSourceIndices={annotatedSourceIndices}
                            />
                        </div>
                    )}
                    {selectedPairCount > 0 && comparisonTokens.length > 0 && (
                        <div style={{ padding: '0 0 14px' }}>
                            <div className={styles.subLabel}>已选 token（在 probe 中高亮）</div>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 8 }}>
                                {comparisonTokens.map(token => (
                                    <span
                                        key={`${token.tokenIndex}-${token.pointIndex ?? 'na'}`}
                                        style={{
                                            fontSize: 12,
                                            padding: '4px 8px',
                                            borderRadius: 999,
                                            background: '#f5f3ff',
                                            color: '#6d28d9',
                                            border: '1px solid #ddd6fe',
                                            fontFamily: 'monospace',
                                        }}
                                    >
                                        {token.tokenDisplay} @{token.tokenIndex}
                                    </span>
                                ))}
                            </div>
                        </div>
                    )}
                    <div className={styles.pairList}>
                        {pairs.map(pair => (
                            <PairCard
                                key={pair.id}
                                pair={pair}
                                detail={detail}
                                selected={selectedPairIdSet.has(pair.id)}
                                onToggleSelect={onTogglePairSelection ? () => onTogglePairSelection(trainIdx, pair.id) : undefined}
                                annotatedSourceIndices={annotatedSourcesForPairs(gtEdgesByTarget, [pair])}
                                onUnlearn={onUnlearnPair ? () => onUnlearnPair(pair) : undefined}
                                unlearnBusy={unlearningPairId === pair.id}
                                unlearnResult={unlearnResults?.[pair.id] ?? null}
                                onOpenAnnotationViewer={
                                    onOpenAnnotationViewer
                                        ? () => onOpenAnnotationViewer(pair)
                                        : undefined
                                }
                            />
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

// ─── Main NewView Component ───────────────────────────────────────────────────

export interface AllTokensExperimentMeta {
    taskId: string;
    label: string;
    fileName: string;
}

export interface ReportPanelProps {
    report: AllTokensReport;
    meta: AllTokensExperimentMeta;
    modelLabel?: string;
    /** When provided with onSelectedTokIdxChange, token selection is controlled by parent (for dual-link). */
    selectedTokIdx?: number | null;
    onSelectedTokIdxChange?: (idx: number | null) => void;
    compact?: boolean;
}

export function ReportPanel({
    report,
    meta,
    modelLabel,
    selectedTokIdx: controlledTokIdx,
    onSelectedTokIdxChange,
    compact = false,
}: ReportPanelProps) {
    const importedReportActive = meta.fileName.startsWith('uploaded:') || meta.label.includes('(uploaded)');
    const selectedMeta = meta;

    // Selected output token (by absolute sequence index)
    const [internalTokIdx, setInternalTokIdx] = useState<number | null>(null);
    const selectedTokIdx = controlledTokIdx !== undefined ? controlledTokIdx : internalTokIdx;
    const setSelectedTokIdx = (updater: number | null | ((prev: number | null) => number | null)) => {
        const prev = selectedTokIdx;
        const next = typeof updater === 'function' ? updater(prev) : updater;
        if (onSelectedTokIdxChange) onSelectedTokIdxChange(next);
        else setInternalTokIdx(next);
    };
    // Selected test correlation (source_token_index)
    const [selectedTestCorrIdx, setSelectedTestCorrIdx] = useState<number | null>(null);
    // Gold live attribution (teacher-force API)
    type AttrMode = 'predict' | 'gold';
    const [attrMode, setAttrMode] = useState<AttrMode>('predict');
    const [goldLocalIdx, setGoldLocalIdx] = useState<number | null>(null);
    const [goldTopCorrelations, setGoldTopCorrelations] = useState<TestCorrelation[]>([]);
    const [goldSelectedCorrIdx, setGoldSelectedCorrIdx] = useState<number | null>(null);
    const [goldPairs, setGoldPairs] = useState<CorrelationPair[]>([]);
    const [goldTrainDetails, setGoldTrainDetails] = useState<Record<string, TrainSampleDetail>>({});
    const [goldBusy, setGoldBusy] = useState(false);
    const clearGoldLive = useCallback(() => {
        setGoldLocalIdx(null);
        setGoldTopCorrelations([]);
        setGoldSelectedCorrIdx(null);
        setGoldPairs([]);
        setGoldTrainDetails({});
        setGoldBusy(false);
    }, []);
    // cos_sim filter defaults (UI controls removed with the old header)
    const threshold = 0.0;
    const hideZero = false;

    const [ttavUrl] = useState(() => loadTtavLaunchPrefs().ttavUrl);
    const [ttavContentPathTemplate] = useState(() => loadTtavLaunchPrefs().contentPathTemplate);
    const [eifBundleCacheTemplate] = useState(() => loadTtavLaunchPrefs().eifBundleCacheTemplate);
    const [ttavVisMethod] = useState(() => loadTtavLaunchPrefs().visMethod);
    const [ttavVisId] = useState(() => loadTtavLaunchPrefs().visId);
    const [eifApiUrl] = useState(() => loadTtavLaunchPrefs().eifApiUrl);
    const [ttavLaunchError, setTtavLaunchError] = useState<string | null>(null);
    const [ttavLaunchStatus, setTtavLaunchStatus] = useState<string | null>(null);
    const [probingTrainSampleId, setProbingTrainSampleId] = useState<number | null>(null);
    const [selectedTrainPairIdsByGroup, setSelectedTrainPairIdsByGroup] = useState<Record<number, string[]>>({});
    const [trainProbeComparisons, setTrainProbeComparisons] = useState<Record<number, TrainProbeComparisonSummary>>({});
    /** Model-side tick: current test saliency edge participates in probe filtering. */
    const [modelSaliencySelected, setModelSaliencySelected] = useState(false);
    const [visualizerMode] = useState<VisualizerMode>('inline');
    const [unlearningPairId, setUnlearningPairId] = useState<string | null>(null);
    const [unlearnResultsByPairId, setUnlearnResultsByPairId] = useState<Record<string, UnlearnPairResult>>({});
    // GT annotation edges from smoke_train_data_oversample_llm.jsonl (train 0..4).
    const [trainGtEdges, setTrainGtEdges] = useState<TrainGtEdges | null>(null);
    // The in-page plot appears only after Prepare sample / Open Visualizer /
    // Open Full Probe — nothing is probed eagerly on sample selection.
    const [inlineBundle, setInlineBundle] = useState<InlineBundle | null>(null);
    const [inlineBundleError, setInlineBundleError] = useState<string | null>(null);
    const [inlineBundleLoading, setInlineBundleLoading] = useState(false);
    const [hoverTarget, setHoverTarget] = useState<TokenHoverTarget | null>(null);
    // The dock floats over the page, so it has to be dismissable without
    // throwing the loaded bundle away.
    const [inlinePlotCollapsed, setInlinePlotCollapsed] = useState(false);
    // The floating plot is position:fixed, so it needs the model column's
    // viewport coordinates to sit over that section. Measured rather than
    // guessed: the page width comes from `width: 95vw` plus padding.
    const bottomLeftRef = useRef<HTMLDivElement | null>(null);
    const [dockRect, setDockRect] = useState<{ left: number; width: number } | null>(null);
    // null until the user drags the handle, so the default keeps tracking the
    // column width instead of freezing at whatever it was on first render.
    const [inlinePlotSize, setInlinePlotSize] = useState<InlinePlotSize | null>(() => loadInlinePlotSize());
    // null = docked to the left column's bottom-left; set after the user drags.
    const [inlinePlotPos, setInlinePlotPos] = useState<InlinePlotPos | null>(() => loadInlinePlotPos());
    const [inlinePlotMoving, setInlinePlotMoving] = useState(false);
    const [viewport, setViewport] = useState(() => ({
        width: typeof window === 'undefined' ? 800 : window.innerWidth,
        height: typeof window === 'undefined' ? 900 : window.innerHeight,
    }));
    const resizeOriginRef = useRef<{ x: number; y: number; width: number; height: number } | null>(null);
    const moveOriginRef = useRef<{ x: number; y: number; left: number; top: number } | null>(null);
    const inlineVisualizerRef = useRef<HTMLDivElement | null>(null);

    const displayModelLabel = modelLabel
        || report.experiment_meta.model_name
        || meta.label
        || 'model';

    // Reset secondary selection when selected token changes
    useEffect(() => {
        setSelectedTestCorrIdx(null);
        setModelSaliencySelected(false);
    }, [selectedTokIdx]);

    useEffect(() => {
        setModelSaliencySelected(false);
    }, [selectedTestCorrIdx]);

    useEffect(() => {
        let cancelled = false;
        void (async () => {
            try {
                const resp = await fetch('/data/train-gt-edges.json', { cache: 'no-store' });
                if (!resp.ok) return;
                const data = await resp.json() as TrainGtEdges;
                if (!cancelled) setTrainGtEdges(data);
            } catch {
                // Annotation overlay is optional; missing file just skips the underline.
            }
        })();
        return () => { cancelled = true; };
    }, []);

    useEffect(() => {
        setSelectedTrainPairIdsByGroup({});
        setTrainProbeComparisons({});
        setUnlearnResultsByPairId({});
        setUnlearningPairId(null);
    }, [selectedTokIdx, selectedTestCorrIdx]);

    useEffect(() => {
        if (typeof window === 'undefined') return;
        window.localStorage.setItem(TTAV_PREFS_KEY, JSON.stringify({
            ttavUrl,
            contentPathTemplate: ttavContentPathTemplate,
            eifBundleCacheTemplate,
            visMethod: ttavVisMethod,
            visId: ttavVisId,
            eifApiUrl,
            visualizerMode,
        } satisfies TtavLaunchPrefs));
    }, [ttavUrl, ttavContentPathTemplate, eifBundleCacheTemplate, ttavVisMethod, ttavVisId, eifApiUrl, visualizerMode]);

    const modelTokens   = useMemo(() => decodeTokens(report.test_sample_baseline.full_tokens), [report]);
    const correctTokens = useMemo(() => decodeTokens(report.test_sample_baseline.correct_full_tokens ?? []), [report]);
    const promptLen     = report?.test_sample_baseline.prompt_len ?? 0;
    // Gold panel shows the answer only — same slice used by Markdown export.
    const goldResponseTokens = useMemo(
        () => (correctTokens.length > promptLen ? correctTokens.slice(promptLen) : correctTokens),
        [correctTokens, promptLen],
    );
    const selectedSampleId = inferSampleIdFromMeta(selectedMeta, report);

    // Keep the floating plot aligned with the model/output section.
    // ResizeObserver fires once on observe, so it supplies the initial
    // measurement too.
    //
    // No scroll listener on purpose: the section's horizontal position only
    // moves on resize, and measuring per scroll event would force a layout
    // read on every frame of every scroll for a value that never changed.
    useEffect(() => {
        const node = bottomLeftRef.current;
        if (!node) return;

        const sync = () => {
            const rect = node.getBoundingClientRect();
            setDockRect(prev => (
                prev && Math.abs(prev.left - rect.left) < 0.5 && Math.abs(prev.width - rect.width) < 0.5
                    ? prev
                    : { left: rect.left, width: rect.width }
            ));
            // Tracked so the size clamp re-runs on a vertical-only resize, which
            // leaves the column's rect untouched but can still leave a dragged
            // panel taller than the window.
            setViewport(prev => (
                prev.width === window.innerWidth && prev.height === window.innerHeight
                    ? prev
                    : { width: window.innerWidth, height: window.innerHeight }
            ));
        };

        const observer = new ResizeObserver(sync);
        observer.observe(node);
        window.addEventListener('resize', sync);
        return () => {
            observer.disconnect();
            window.removeEventListener('resize', sync);
        };
    }, []);

    // Drop the plot when the report switches to another sample. Its points map to
    // the previous sample's token indices, so leaving it up would highlight the
    // wrong tokens in the code panels — silently, and convincingly.
    useEffect(() => {
        setInlineBundle(null);
        setInlineBundleError(null);
        setHoverTarget(null);
        setAttrMode('predict');
        clearGoldLive();
    }, [selectedSampleId, clearGoldLive]);

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

    const ttavSelectedIndices = useMemo(() => {
        const selected = new Set<number>();
        if (selectedTokIdx !== null) selected.add(selectedTokIdx);
        sourceHighlightIndices.forEach(idx => selected.add(idx));
        return Array.from(selected).sort((a, b) => a - b);
    }, [selectedTokIdx, sourceHighlightIndices]);

    // Pairs to show in the right panel — only after the user picks one test (s→t) edge.
    const allDisplayPairs = useMemo(() => {
        const keep = (p: CorrelationPair) =>
            p.cos_sim >= threshold && !(hideZero && p.cos_sim === 0);

        if (attrMode === 'gold') {
            if (goldSelectedCorrIdx === null) return [];
            return goldPairs
                .filter(p =>
                    keep(p) && p.test_correlation.source_token_index === goldSelectedCorrIdx
                )
                .sort((a, b) => b.cos_sim - a.cos_sim);
        }

        // Require an explicit Top-Correlation click before showing train matches.
        if (!selectedResult || selectedTestCorrIdx === null) return [];

        return selectedResult.correlation_pairs
            .filter(p =>
                keep(p) && p.test_correlation.source_token_index === selectedTestCorrIdx
            )
            .sort((a, b) => b.cos_sim - a.cos_sim);
    }, [
        attrMode, selectedResult, selectedTestCorrIdx, threshold, hideZero,
        goldPairs, goldSelectedCorrIdx,
    ]);

    // Group pairs by train_sample_id; keep Top-10 trains by best pair cos for this edge.
    const trainGroups = useMemo(() => {
        const map = new Map<number, CorrelationPair[]>();
        allDisplayPairs.forEach(p => {
            if (!map.has(p.train_sample_id)) map.set(p.train_sample_id, []);
            map.get(p.train_sample_id)!.push(p);
        });
        return Array.from(map.entries())
            .map(([id, pairs]) => ({
                id,
                pairs: [...pairs].sort((a, b) => b.cos_sim - a.cos_sim),
                bestSim: Math.max(...pairs.map(p => p.cos_sim)),
            }))
            .sort((a, b) => b.bestSim - a.bestSim)
            .slice(0, 10);
    }, [allDisplayPairs]);

    const trainPanelEmptyHint = useMemo(() => {
        if (attrMode === 'gold') {
            if (goldLocalIdx === null) {
                return '点击右侧 Gold 答案中的任意 token，现场计算 teacher-force saliency。';
            }
            if (goldBusy && goldTopCorrelations.length === 0) {
                return '正在计算 Gold saliency…（首次会加载模型/bank，可能较慢）';
            }
            if (goldSelectedCorrIdx === null) {
                return '选择一条 Gold saliency 边，现场检索 Top-10 train 并跑 Stage3。';
            }
            if (goldBusy) {
                return '正在检索 train + Stage3 matching…';
            }
            return 'No matching pairs for this gold edge. Try another source→target.';
        }
        if (!selectedResult) {
            return 'Click an analyzed output token, then choose one Top Correlation (source→target) to retrieve related training samples.';
        }
        if (selectedTestCorrIdx === null) {
            return 'Select one Top Correlation on the left to show its Top-10 related training samples and matching pairs.';
        }
        if (importedReportActive && allDisplayPairs.length === 0) {
            return 'No training correlation pairs are included for this selected source→target edge.';
        }
        return 'No matching pairs for this source→target edge. Try lowering the cos_sim threshold.';
    }, [
        attrMode, goldLocalIdx, goldBusy, goldTopCorrelations.length, goldSelectedCorrIdx,
        selectedResult, selectedTestCorrIdx, importedReportActive, allDisplayPairs.length,
    ]);

    const goldHighlightSourceIndices = useMemo(() => {
        if (attrMode !== 'gold' || goldTopCorrelations.length === 0) return new Set<number>();
        const abs = goldSelectedCorrIdx !== null
            ? [goldSelectedCorrIdx]
            : goldTopCorrelations.map(c => c.source_token_index);
        // Gold panel uses local indices (answer-only). Prompt sources are skipped.
        return new Set(
            abs
                .filter(i => i >= promptLen)
                .map(i => i - promptLen),
        );
    }, [attrMode, goldTopCorrelations, goldSelectedCorrIdx, promptLen]);

    const handleGoldTokenClick = useCallback((localIdx: number) => {
        if (importedReportActive) {
            setTtavLaunchError('上传的报告不支持 Gold 现场归因（需要服务器上的模型与 train bank）。');
            return;
        }
        const absIdx = promptLen + localIdx;
        setAttrMode('gold');
        setSelectedTokIdx(null);
        setSelectedTestCorrIdx(null);
        setGoldLocalIdx(localIdx);
        setGoldSelectedCorrIdx(null);
        setGoldPairs([]);
        setGoldTrainDetails({});
        setGoldTopCorrelations([]);
        setGoldBusy(true);
        setTtavLaunchError(null);
        setTtavLaunchStatus(`Gold live saliency @ ${absIdx}…`);

        void (async () => {
            try {
                const resp = await fetch(buildEifApiUrl(eifApiUrl, '/api/gold-saliency'), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        reportFileName: selectedMeta.fileName,
                        targetIndex: absIdx,
                    }),
                });
                const raw = await resp.text();
                let parsed: Record<string, unknown> = {};
                try {
                    parsed = raw.trim() ? JSON.parse(raw) as Record<string, unknown> : {};
                } catch {
                    throw new Error(`Gold saliency API non-JSON (HTTP ${resp.status}): ${raw.slice(0, 200)}`);
                }
                if (!resp.ok || parsed.status !== 'success') {
                    throw new Error(
                        typeof parsed.message === 'string'
                            ? parsed.message
                            : `Gold saliency failed (HTTP ${resp.status})`,
                    );
                }
                const top = Array.isArray(parsed.topCorrelations)
                    ? parsed.topCorrelations as TestCorrelation[]
                    : [];
                setGoldTopCorrelations(top);
                setTtavLaunchStatus(
                    `Gold saliency ready · ${top.length} sources @ idx ${absIdx}`,
                );
            } catch (error) {
                const msg = error instanceof Error ? error.message : 'Gold saliency failed';
                setTtavLaunchError(msg);
                setTtavLaunchStatus(null);
                clearGoldLive();
            } finally {
                setGoldBusy(false);
            }
        })();
    }, [
        importedReportActive, promptLen, eifApiUrl, selectedMeta.fileName,
        clearGoldLive, setSelectedTokIdx,
    ]);

    const handleGoldCorrClick = useCallback((sourceAbsIdx: number) => {
        if (goldLocalIdx === null) return;
        const absTarget = promptLen + goldLocalIdx;
        const next = goldSelectedCorrIdx === sourceAbsIdx ? null : sourceAbsIdx;
        setGoldSelectedCorrIdx(next);
        setGoldPairs([]);
        setGoldTrainDetails({});
        if (next === null) return;

        setGoldBusy(true);
        setTtavLaunchError(null);
        setTtavLaunchStatus(
            `Gold retrieve+Stage3 · src ${next} → tgt ${absTarget}…`,
        );
        void (async () => {
            try {
                const resp = await fetch(buildEifApiUrl(eifApiUrl, '/api/gold-retrieve-stage3'), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        reportFileName: selectedMeta.fileName,
                        sourceIndex: next,
                        targetIndex: absTarget,
                    }),
                });
                const raw = await resp.text();
                let parsed: Record<string, unknown> = {};
                try {
                    parsed = raw.trim() ? JSON.parse(raw) as Record<string, unknown> : {};
                } catch {
                    throw new Error(`Gold Stage3 API non-JSON (HTTP ${resp.status}): ${raw.slice(0, 200)}`);
                }
                if (!resp.ok || parsed.status !== 'success') {
                    throw new Error(
                        typeof parsed.message === 'string'
                            ? parsed.message
                            : `Gold Stage3 failed (HTTP ${resp.status})`,
                    );
                }
                const pairs = Array.isArray(parsed.correlationPairs)
                    ? parsed.correlationPairs as CorrelationPair[]
                    : [];
                const details = (
                    typeof parsed.trainSampleDetails === 'object'
                    && parsed.trainSampleDetails !== null
                ) ? parsed.trainSampleDetails as Record<string, TrainSampleDetail> : {};
                setGoldPairs(pairs);
                setGoldTrainDetails(details);
                setTtavLaunchStatus(
                    `Gold Stage3 ready · ${pairs.length} pairs · ${Object.keys(details).length} trains`,
                );
            } catch (error) {
                const msg = error instanceof Error ? error.message : 'Gold Stage3 failed';
                setTtavLaunchError(msg);
                setTtavLaunchStatus(null);
            } finally {
                setGoldBusy(false);
            }
        })();
    }, [goldLocalIdx, goldSelectedCorrIdx, promptLen, eifApiUrl, selectedMeta.fileName]);

    // Load a prepared bundle into the in-page plot. All three entry points route
    // here when the mode is 'inline'; the window path below is left untouched.
    //
    // This only reads the precomputed bundle from disk, so it works under
    // EIF_CACHE_ONLY and does not need the TTAV app or its backend to be running.
    // A miss means the bundle was never prepared, which is what the error says.
    const showInlineBundle = useCallback(async (
        sampleId: string,
    ): Promise<InlineBundle | null> => {
        setInlineBundleLoading(true);
        setInlineBundleError(null);
        setHoverTarget(null);
        try {
            // Always load the full probe; pair/saliency filtering is applied live
            // from the Model / pair 「选择」 ticks below.
            const bundle = await loadInlineBundle(sampleId, []);
            setInlineBundle(bundle);
            return bundle;
        } catch (error) {
            setInlineBundle(null);
            setInlineBundleError(error instanceof Error ? error.message : '加载 bundle 失败。');
            return null;
        } finally {
            setInlineBundleLoading(false);
        }
    }, []);

    const toggleTrainPairSelection = (trainIdx: number, pairId: string) => {
        setSelectedTrainPairIdsByGroup(current => {
            const prev = new Set(current[trainIdx] ?? []);
            if (prev.has(pairId)) {
                prev.delete(pairId);
            } else {
                prev.add(pairId);
            }
            return {
                ...current,
                [trainIdx]: Array.from(prev),
            };
        });
        setTrainProbeComparisons(current => {
            if (!(trainIdx in current)) return current;
            const next = { ...current };
            delete next[trainIdx];
            return next;
        });
    };

    const handleOpenAnnotationViewer = (pair: CorrelationPair) => {
        const base = (
            import.meta.env.VITE_ANNOTATION_VIEWER_URL as string | undefined
        )?.trim() || 'http://127.0.0.1:5174';
        const url = new URL(base);
        url.searchParams.set('sample', String(pair.train_sample_id));
        url.searchParams.set('target', String(pair.train_correlation.target_token_index));
        window.open(url.toString(), '_blank', 'noopener,noreferrer');
    };

    const handleUnlearnPair = (pair: CorrelationPair) => {
        if (!report || !selectedMeta || importedReportActive) return;

        setUnlearningPairId(pair.id);
        setTtavLaunchError(null);
        setTtavLaunchStatus(`Unlearning probe for pair ${pair.id}…`);

        void (async () => {
            try {
                const resp = await fetch(buildEifApiUrl(eifApiUrl, '/api/unlearn-pair-probe'), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        reportFileName: selectedMeta.fileName,
                        sampleId: selectedSampleId,
                        pairId: pair.id,
                        trainSampleId: pair.train_sample_id,
                        testSourceIndex: pair.test_correlation.source_token_index,
                        testTargetIndex: pair.test_correlation.target_token_index,
                        trainSourceIndex: pair.train_correlation.source_token_index,
                        trainTargetIndex: pair.train_correlation.target_token_index,
                        modelPath: report.experiment_meta.model_path ?? null,
                        baseModelPath: report.experiment_meta.base_model_path ?? null,
                        // LoRA last-layer is high-dimensional; unit-normalized η=1
                        // barely moves CE when the test token is already near-perfect.
                        unlearnLr: 20.0,
                        recomputeSaliency: true,
                    }),
                });
                const rawText = await resp.text();
                let parsed: Record<string, unknown> = {};
                if (rawText.trim()) {
                    try {
                        parsed = JSON.parse(rawText) as Record<string, unknown>;
                    } catch {
                        throw new Error(
                            `Unlearn API returned non-JSON (HTTP ${resp.status}): ${rawText.slice(0, 240)}`,
                        );
                    }
                }
                if (!resp.ok || parsed.status !== 'success') {
                    const message = typeof parsed.message === 'string'
                        ? parsed.message
                        : `Unlearn API failed (HTTP ${resp.status})`;
                    throw new Error(message);
                }

                const result = parsed as UnlearnPairResult;
                setUnlearnResultsByPairId(current => ({ ...current, [pair.id]: result }));
                const dLog = typeof result.delta?.logprob === 'number'
                    ? result.delta.logprob.toFixed(4)
                    : '?';
                setTtavLaunchStatus(`Unlearn ${pair.id} 完成 · ΔlogP=${dLog} · ${result.verdict ?? ''}`);
            } catch (error) {
                const msg = error instanceof Error ? error.message : 'Unlearn probe failed';
                setUnlearnResultsByPairId(current => ({
                    ...current,
                    [pair.id]: { status: 'error', error: msg, verdict: 'inconclusive' },
                }));
                setTtavLaunchError(msg);
                setTtavLaunchStatus(null);
            } finally {
                setUnlearningPairId(current => (current === pair.id ? null : current));
            }
        })();
    };

    const handleOpenTrainProbe = (trainIdx: number, pairs: CorrelationPair[]) => {
        if (!report || !selectedMeta) return;

        const trimmedUrl = ttavUrl.trim() || DEFAULT_TTAV_URL;
        const visMethod = ttavVisMethod.trim() || DEFAULT_TTAV_METHOD;
        const visId = ttavVisId.trim() || DEFAULT_TTAV_VIS_ID;

        // Probe is always rendered in-page now; the TTAV jump chrome was removed.
        const selectedPairIdSet = new Set(selectedTrainPairIdsByGroup[trainIdx] ?? []);
        const selectedPairs = selectedPairIdSet.size > 0
            ? pairs.filter(pair => selectedPairIdSet.has(pair.id))
            : [];
        const focusTrainIndices = Array.from(new Set(selectedPairs.flatMap(pair => [
            pair.train_correlation.source_token_index,
            pair.train_correlation.target_token_index,
        ]))).sort((a, b) => a - b);

        setProbingTrainSampleId(trainIdx);
        setTtavLaunchError(null);
        setTtavLaunchStatus(`Preparing full-train embedding probe for TRAIN #${trainIdx}...`);
        void (async () => {
            try {
                const probeResp = await fetch(buildEifApiUrl(eifApiUrl, '/api/prepare-ttav-train-probe'), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        reportFileName: selectedMeta.fileName,
                        sampleId: selectedSampleId,
                        trainSampleId: trainIdx,
                        ttavUploadUrl: new URL('/registerEIFBundle', trimmedUrl).toString(),
                        ttavUrl: trimmedUrl,
                        visMethod,
                        visId,
                        contextRadius: 1,
                        includeFullTrain: true,
                        focusTrainIndices,
                        renderTarget: 'inline',
                        focusPairIds: selectedPairs.map(pair => pair.id),
                        probePairs: pairs.map(pair => ({
                            id: pair.id,
                            trainSourceIndex: pair.train_correlation.source_token_index,
                            trainTargetIndex: pair.train_correlation.target_token_index,
                            testSourceIndex: pair.test_correlation.source_token_index,
                            testTargetIndex: pair.test_correlation.target_token_index,
                        })),
                    }),
                });

                const rawText = await probeResp.text();
                let parsedJson: Record<string, unknown> | null = null;
                if (rawText.trim()) {
                    try {
                        parsedJson = JSON.parse(rawText) as Record<string, unknown>;
                    } catch {
                        throw new Error(
                            `EIF train probe API returned a non-JSON response (HTTP ${probeResp.status}). ` +
                            `${rawText.slice(0, 240)}`
                        );
                    }
                }

                const apiJson = parsedJson ?? {};
                if (!probeResp.ok || apiJson.status !== 'success') {
                    const baseMessage = 'EIF train probe API failed (HTTP ' + probeResp.status + ')';
                    const message = typeof apiJson.message === 'string'
                        ? baseMessage + ': ' + apiJson.message
                        : baseMessage;
                    throw new Error(message);
                }

                const comparisonSummary = (typeof apiJson.comparisonSummary === 'object' && apiJson.comparisonSummary !== null)
                    ? apiJson.comparisonSummary as TrainProbeComparisonSummary
                    : { focusTokens: [], pairwiseCosine: [] };
                setTrainProbeComparisons(current => ({
                    ...current,
                    [trainIdx]: comparisonSummary,
                }));

                const probeSampleId = typeof apiJson.pregeneratedSampleId === 'string'
                    ? apiJson.pregeneratedSampleId
                    : null;
                if (!probeSampleId) {
                    throw new Error('EIF API 未返回 pregeneratedSampleId，无法在本页加载 probe。请更新 EIF bundle API 后重试。');
                }
                const bundle = await showInlineBundle(probeSampleId);
                setTtavLaunchStatus(bundle
                    ? `已在本页显示 TRAIN #${trainIdx} 的 probe（${bundle.projection.length} 个点）。`
                    : null);
            } catch (error) {
                const msg = error instanceof Error ? error.message : 'Failed to open embedding probe';
                setTtavLaunchError(msg);
                setTtavLaunchStatus(null);
            } finally {
                setProbingTrainSampleId(current => (current === trainIdx ? null : current));
            }
        })();
    };

    const inlinePlotGeometry = useMemo(() => {
        if (!dockRect) return { width: 0, height: INLINE_PLOT_MIN_SIDE, canvasHeight: INLINE_PLOT_MIN_SIDE };
        const maxSide = Math.max(
            INLINE_PLOT_MIN_SIDE,
            Math.min(dockRect.width, viewport.height - 160),
        );
        const rawSide = inlinePlotSize?.width ?? maxSide;
        const side = Math.min(Math.max(rawSide, INLINE_PLOT_MIN_SIDE), maxSide);
        return {
            width: side,
            height: side,
            canvasHeight: Math.max(INLINE_PLOT_MIN_SIDE - INLINE_PLOT_CHROME, side - INLINE_PLOT_CHROME),
        };
    }, [dockRect, inlinePlotSize, viewport]);

    // Resize from the top-right corner: keep a square, grow/shrink by the
    // larger of the two deltas so the grip still feels natural.
    const handleResizePointerDown = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
        event.preventDefault();
        event.currentTarget.setPointerCapture(event.pointerId);
        resizeOriginRef.current = {
            x: event.clientX,
            y: event.clientY,
            width: inlinePlotGeometry.width,
            height: inlinePlotGeometry.height,
        };
    }, [inlinePlotGeometry]);

    const handleResizePointerMove = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
        const origin = resizeOriginRef.current;
        if (!origin) return;
        const delta = Math.max(event.clientX - origin.x, origin.y - event.clientY);
        const side = origin.width + delta;
        setInlinePlotSize({ width: side, height: side });
    }, []);

    const handleResizePointerUp = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
        if (!resizeOriginRef.current) return;
        resizeOriginRef.current = null;
        event.currentTarget.releasePointerCapture(event.pointerId);
        // Persist the clamped square, not the raw drag.
        saveInlinePlotSize({ width: inlinePlotGeometry.width, height: inlinePlotGeometry.height });
    }, [inlinePlotGeometry]);

    // Drag the header to reposition. Starts from the panel's current screen
    // rect so the first move doesn't jump, whether it was docked (bottom) or
    // already free-floating (top/left).
    const handleMovePointerDown = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
        const target = event.target as HTMLElement;
        if (target.closest('button') || target.closest(`.${styles.inlineVisualizerResizeGrip}`)) return;
        event.preventDefault();
        event.currentTarget.setPointerCapture(event.pointerId);
        const panel = inlineVisualizerRef.current;
        if (!panel) return;
        const rect = panel.getBoundingClientRect();
        moveOriginRef.current = {
            x: event.clientX,
            y: event.clientY,
            left: rect.left,
            top: rect.top,
        };
        setInlinePlotMoving(true);
        setInlinePlotPos({ left: rect.left, top: rect.top });
    }, []);

    const handleMovePointerMove = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
        const origin = moveOriginRef.current;
        if (!origin) return;
        setInlinePlotPos(clampInlinePlotPos(
            {
                left: origin.left + (event.clientX - origin.x),
                top: origin.top + (event.clientY - origin.y),
            },
            inlinePlotGeometry.width,
            viewport,
        ));
    }, [inlinePlotGeometry.width, viewport]);

    const handleMovePointerUp = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
        if (!moveOriginRef.current) return;
        moveOriginRef.current = null;
        setInlinePlotMoving(false);
        try {
            event.currentTarget.releasePointerCapture(event.pointerId);
        } catch {
            // already released
        }
        setInlinePlotPos(prev => {
            if (!prev) return prev;
            const next = clampInlinePlotPos(prev, inlinePlotGeometry.width, viewport);
            saveInlinePlotPos(next);
            return next;
        });
    }, [inlinePlotGeometry.width, viewport]);

    const resetInlinePlotPos = useCallback(() => {
        setInlinePlotPos(null);
        saveInlinePlotPos(null);
    }, []);

    // Keep a free-floated panel on-screen when the window or plot size changes.
    useEffect(() => {
        setInlinePlotPos(prev => {
            if (!prev) return prev;
            const next = clampInlinePlotPos(prev, inlinePlotGeometry.width, viewport);
            if (next.left === prev.left && next.top === prev.top) return prev;
            saveInlinePlotPos(next);
            return next;
        });
    }, [inlinePlotGeometry.width, viewport]);

    // Which points/links the Model + pair 「选择」 ticks emphasize. null = full
    // default view; otherwise the cloud stays, but everything else washes out.
    const probeEmphasis = useMemo(() => {
        if (!inlineBundle || inlineBundle.kind !== 'probe') return null;

        const trainId = inlineBundle.trainSampleId;
        const selectedPairIds = trainId != null
            ? (selectedTrainPairIdsByGroup[trainId] ?? [])
            : [];
        const idSet = new Set(selectedPairIds);
        const selectedPairs = (trainId != null
            ? (trainGroups.find(g => g.id === trainId)?.pairs ?? [])
            : []
        )
            .filter(pair => idSet.has(pair.id))
            .map(pair => ({
                id: pair.id,
                trainSourceIndex: pair.train_correlation.source_token_index,
                trainTargetIndex: pair.train_correlation.target_token_index,
                testSourceIndex: pair.test_correlation.source_token_index,
                testTargetIndex: pair.test_correlation.target_token_index,
            }));

        return collectProbeEmphasis(inlineBundle, {
            selectedPairs,
            includeTestSaliency: modelSaliencySelected,
            testSourceIndex: selectedTestCorrIdx,
            testTargetIndex: selectedTokIdx,
        });
    }, [
        inlineBundle,
        modelSaliencySelected,
        selectedTrainPairIdsByGroup,
        selectedTestCorrIdx,
        selectedTokIdx,
        trainGroups,
    ]);

    const probeSelectionActive = Boolean(probeEmphasis && probeEmphasis.points.size > 0);

    const inlinePlotSelection = useMemo(() => {
        if (!inlineBundle) return [];
        if (probeEmphasis) return Array.from(probeEmphasis.points).sort((a, b) => a - b);
        if (inlineBundle.kind === 'probe') return inlineBundle.selectedPoints;
        return ttavSelectedIndices.filter(idx => idx < inlineBundle.projection.length);
    }, [inlineBundle, probeEmphasis, ttavSelectedIndices]);

    const linkedTestTokenIndex = hoverTarget?.side === 'test' ? hoverTarget.tokenIndex : null;
    const linkedTrainTokenIndex = hoverTarget?.side === 'train' ? hoverTarget.tokenIndex : null;
    const linkedTrainSampleId = hoverTarget?.side === 'train' ? hoverTarget.trainSampleId : null;

    const handleTestTokenHover = useCallback((idx: number | null) => {
        if (idx === null) {
            setHoverTarget(current => (current?.side === 'test' ? null : current));
            return;
        }
        if (!inlineBundle) return;
        setHoverTarget({
            side: 'test',
            tokenIndex: idx,
            trainSampleId: null,
            role: null,
            token: report.test_sample_baseline.full_tokens[idx] ?? '',
        });
    }, [inlineBundle, report]);

    const makeTrainTokenHoverHandler = useCallback((trainIdx: number, tokens: string[]) =>
        (idx: number | null) => {
            if (idx === null) {
                setHoverTarget(current => (current?.side === 'train' ? null : current));
                return;
            }
            if (!inlineBundle || inlineBundle.trainSampleId !== trainIdx) return;
            setHoverTarget({
                side: 'train',
                tokenIndex: idx,
                trainSampleId: trainIdx,
                role: null,
                token: tokens[idx] ?? '',
            });
        }, [inlineBundle]);

    // ── Render ──

    return (
        <div className={`${styles.root}${compact ? ` ${styles.panelRootCompact}` : ''}`}>
            <div className={styles.modelBanner}>
                <span className={styles.modelBannerLabel}>Model</span>
                <span className={styles.modelBannerName}>{displayModelLabel}</span>
                <span className={styles.modelBannerMeta}>
                    test#{report.experiment_meta.test_sample_index}
                    {report.experiment_meta.task_id ? ` · ${report.experiment_meta.task_id}` : ''}
                    {` · ${report.per_token_results.length} tokens`}
                </span>
            </div>

                    {/* ── Left: Model+Gold · Right: Train matches ── */}
                    <div className={styles.bottomSection}>
                        <div className={styles.bottomLeft} ref={bottomLeftRef}>
                            <OutputComparePanel
                                modelTokens={modelTokens}
                                goldResponseTokens={goldResponseTokens}
                                promptLen={promptLen}
                                highlightSourceIndices={attrMode === 'predict' ? sourceHighlightIndices : undefined}
                                selectedTargetIndex={attrMode === 'predict' ? (selectedTokIdx ?? undefined) : undefined}
                                analyzedIndices={analyzedIndices}
                                onTokenClick={idx => {
                                    setAttrMode('predict');
                                    clearGoldLive();
                                    setSelectedTokIdx(prev => prev === idx ? null : idx);
                                }}
                                goldSelectedLocalIndex={attrMode === 'gold' ? goldLocalIdx : null}
                                goldHighlightSourceIndices={goldHighlightSourceIndices}
                                onGoldTokenClick={handleGoldTokenClick}
                                linkedTokenIndex={linkedTestTokenIndex}
                                onTokenHover={inlineBundle ? handleTestTokenHover : undefined}
                                saliencySelected={modelSaliencySelected}
                                saliencySelectEnabled={selectedTokIdx !== null && selectedTestCorrIdx !== null}
                                onToggleSaliencySelect={() => setModelSaliencySelected(v => !v)}
                            />

                            {attrMode === 'gold' && goldLocalIdx !== null && (
                                <div className={styles.correlationList}>
                                    <div className={styles.correlationListTitle}>
                                        Gold live · Top Correlations for "
                                        {decodeToken(goldResponseTokens[goldLocalIdx] ?? '').trim()}"
                                        {' '}@ idx {promptLen + goldLocalIdx}
                                        {goldBusy ? ' …' : ''}
                                    </div>
                                    <div className={styles.correlationListHint}>
                                        Teacher-force gold path. Click one edge to run bank Top-10 + Stage3.
                                    </div>
                                    <div className={styles.correlationListItems}>
                                        {goldTopCorrelations.map(c => (
                                            <button
                                                key={c.source_token_index}
                                                type="button"
                                                disabled={goldBusy}
                                                className={`${styles.corrBtn} ${c.source_token_index === goldSelectedCorrIdx ? styles.corrBtnActive : ''}`}
                                                onClick={() => handleGoldCorrClick(c.source_token_index)}
                                            >
                                                <div className={styles.corrBtnLeft}>
                                                    <span className={styles.corrLabel}>source → target</span>
                                                    <span className={styles.corrSourceTok}>
                                                        {(decodeToken(c.source_token).trim() || '·')}
                                                        <span className={styles.corrArrow}>→</span>
                                                        {decodeToken(c.target_token).trim() || '·'}
                                                    </span>
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

                            {attrMode === 'predict' && selectedResult && (
                                <div className={styles.correlationList}>
                                    <div className={styles.correlationListTitle}>
                                        Top Correlations for "{decodeToken(selectedResult.target_token).trim()}" @ idx {selectedResult.target_token_index}
                                    </div>
                                    <div className={styles.correlationListHint}>
                                        Click one source→target edge to load its Top-10 training matches on the right.
                                    </div>
                                    <div className={styles.correlationListItems}>
                                        {selectedResult.top_correlations.slice(0, 4).map(c => (
                                            <button
                                                key={c.source_token_index}
                                                type="button"
                                                title={`Select ${decodeToken(c.source_token).trim() || '·'} → ${decodeToken(selectedResult.target_token).trim()} for train retrieval`}
                                                className={`${styles.corrBtn} ${c.source_token_index === selectedTestCorrIdx ? styles.corrBtnActive : ''}`}
                                                onClick={() => setSelectedTestCorrIdx(
                                                    prev => prev === c.source_token_index ? null : c.source_token_index
                                                )}
                                            >
                                                <div className={styles.corrBtnLeft}>
                                                    <span className={styles.corrLabel}>source → target</span>
                                                    <span className={styles.corrSourceTok}>
                                                        {(decodeToken(c.source_token).trim() || '·')}
                                                        <span className={styles.corrArrow}>→</span>
                                                        {decodeToken(selectedResult.target_token).trim() || '·'}
                                                    </span>
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
                                {(ttavLaunchError || ttavLaunchStatus) && (
                                    <div style={{
                                        padding: '8px 12px',
                                        fontSize: 11,
                                        color: ttavLaunchError ? '#b91c1c' : '#6b7280',
                                        borderBottom: '1px solid #e5e7eb',
                                    }}>
                                        {ttavLaunchError ?? ttavLaunchStatus}
                                    </div>
                                )}

                                {trainGroups.length === 0 ? (
                                    <div className={styles.emptyState} style={{ padding: '32px 0' }}>
                                        {trainPanelEmptyHint}
                                    </div>
                                ) : (
                                    <div className={styles.trainGroupList}>
                                        {trainGroups.map(({ id, pairs }) => (
                                            <TrainSampleGroup
                                                key={id}
                                                trainIdx={id}
                                                pairs={pairs}
                                                detail={
                                                    attrMode === 'gold'
                                                        ? (goldTrainDetails[String(id)] ?? report.train_sample_details[String(id)])
                                                        : report.train_sample_details[String(id)]
                                                }
                                                onProbeEmbeddings={importedReportActive ? undefined : handleOpenTrainProbe}
                                                probeBusy={probingTrainSampleId === id}
                                                selectedPairIds={selectedTrainPairIdsByGroup[id] ?? []}
                                                onTogglePairSelection={toggleTrainPairSelection}
                                                comparisonSummary={trainProbeComparisons[id]}
                                                linkedTokenIndex={linkedTrainSampleId === id ? linkedTrainTokenIndex : null}
                                                onTokenHover={inlineBundle?.trainSampleId === id
                                                    ? makeTrainTokenHoverHandler(id, report.train_sample_details[String(id)]?.full_tokens ?? [])
                                                    : undefined}
                                                gtEdgesByTarget={trainGtEdges?.[String(id)]}
                                                onUnlearnPair={importedReportActive ? undefined : handleUnlearnPair}
                                                unlearningPairId={unlearningPairId}
                                                unlearnResults={unlearnResultsByPairId}
                                                onOpenAnnotationViewer={handleOpenAnnotationViewer}
                                            />
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                
            {/* ── Embedding plot: floats over the left column, above everything ── */}
            {(inlineBundle || inlineBundleError || inlineBundleLoading) && (
                <div
                    ref={inlineVisualizerRef}
                    className={`${styles.inlineVisualizer}${inlinePlotCollapsed ? ` ${styles.inlineVisualizerCollapsed}` : ''}${inlinePlotMoving ? ` ${styles.inlineVisualizerMoving}` : ''}`}
                    style={!compact && dockRect
                        ? (inlinePlotPos
                            ? {
                                left: inlinePlotPos.left,
                                top: inlinePlotPos.top,
                                bottom: 'auto',
                                width: inlinePlotGeometry.width,
                            }
                            : { left: dockRect.left, width: inlinePlotGeometry.width })
                        : undefined}
                >
                    {/* Top-right grip. The panel is anchored bottom-left, so this
                        corner is the one that grows it in both axes. */}
                    {!compact && !inlinePlotCollapsed && (
                        <div
                            className={styles.inlineVisualizerResizeGrip}
                            onPointerDown={handleResizePointerDown}
                            onPointerMove={handleResizePointerMove}
                            onPointerUp={handleResizePointerUp}
                            onPointerCancel={handleResizePointerUp}
                            onDoubleClick={() => {
                                // Back to a square that fits the left column.
                                setInlinePlotSize(null);
                                if (typeof window !== 'undefined') {
                                    window.localStorage.removeItem(INLINE_PLOT_SIZE_KEY);
                                }
                            }}
                            title="拖动调整正方形大小；双击恢复默认"
                        />
                    )}
                    <div
                        className={styles.inlineVisualizerHeader}
                        onPointerDown={!compact ? handleMovePointerDown : undefined}
                        onPointerMove={!compact ? handleMovePointerMove : undefined}
                        onPointerUp={!compact ? handleMovePointerUp : undefined}
                        onPointerCancel={!compact ? handleMovePointerUp : undefined}
                        onDoubleClick={!compact ? (event) => {
                            const target = event.target as HTMLElement;
                            if (target.closest('button')) return;
                            resetInlinePlotPos();
                        } : undefined}
                        title={!compact ? '拖动标题栏移动浮窗；双击恢复默认位置' : undefined}
                    >
                        <span className={styles.badge} style={{ background: '#7c3aed' }}>PLOT</span>
                        <span className={styles.codePanelLabel}>
                            {inlineBundle?.kind === 'probe'
                                ? `Full Probe · TRAIN #${inlineBundle.trainSampleId ?? '?'}`
                                : 'Sample Embedding'}
                        </span>
                        {inlineBundle && (
                            <span className={styles.inlineVisualizerMeta}>
                                {probeSelectionActive ? '已强调选中对 · ' : ''}
                                {inlineBundle.projection.length} 点
                                {inlineBundle.links.length > 0
                                    ? ` · ${inlineBundle.links.length} 条边`
                                    : ''}
                                {probeSelectionActive
                                    ? ` · 高亮 ${probeEmphasis!.points.size}`
                                    : ''}
                            </span>
                        )}
                        {/* Hover readout lives in the header while collapsed, so the
                            code↔point link still says something with the plot shut. */}
                        {inlinePlotCollapsed && hoverTarget && (
                            <span className={styles.inlineVisualizerMeta}>
                                {hoverTarget.side === 'train'
                                    ? `TRAIN #${hoverTarget.trainSampleId ?? '?'}`
                                    : 'TEST'} · @{hoverTarget.tokenIndex} · "{decodeToken(hoverTarget.token)}"
                            </span>
                        )}
                        <button
                            type="button"
                            className={styles.inlineVisualizerDockBtn}
                            onClick={() => setInlinePlotCollapsed(v => !v)}
                            title={inlinePlotCollapsed ? '展开投影图' : '收起投影图（保留已加载的 bundle）'}
                        >
                            {inlinePlotCollapsed ? '展开 ▲' : '收起 ▼'}
                        </button>
                        <button
                            type="button"
                            className={styles.inlineVisualizerClose}
                            onClick={() => {
                                setInlineBundle(null);
                                setInlineBundleError(null);
                                setHoverTarget(null);
                            }}
                        >
                            关闭
                        </button>
                    </div>
                    {inlineBundleLoading && (
                        <div className={styles.inlineVisualizerNotice}>正在加载 bundle…</div>
                    )}
                    {inlineBundleError && !inlineBundleLoading && (
                        <div className={`${styles.inlineVisualizerNotice} ${styles.inlineVisualizerError}`}>
                            {inlineBundleError}
                        </div>
                    )}
                    {inlineBundle && !inlineBundleLoading && (
                        <>
                            <InlineVisualizer
                                key={inlineBundle.sampleId}
                                bundle={inlineBundle}
                                hoverTarget={hoverTarget}
                                onHoverTargetChange={setHoverTarget}
                                selectedPoints={inlinePlotSelection}
                                emphasizedPoints={probeEmphasis?.points ?? null}
                                emphasizedLinkKeys={probeEmphasis?.linkKeys ?? null}
                                canvasHeight={inlinePlotGeometry.canvasHeight}
                            />
                            <div className={styles.inlineVisualizerHint}>
                                {hoverTarget
                                    ? `${hoverTarget.side === 'train'
                                        ? `TRAIN #${hoverTarget.trainSampleId ?? '?'}`
                                        : 'TEST'} · @${hoverTarget.tokenIndex}`
                                        + `${hoverTarget.role ? ` · ${hoverTarget.role}` : ''}`
                                        + ` · "${decodeToken(hoverTarget.token)}"`
                                    : (probeSelectionActive
                                        ? '已选对高亮，其余点极淡保留分布；取消全部选择后恢复默认。'
                                        : '鼠标划过任意一个点，即可在上方代码中定位它；反之亦然。')}
                            </div>
                        </>
                    )}
                </div>
            )}
        </div>
    );
}
