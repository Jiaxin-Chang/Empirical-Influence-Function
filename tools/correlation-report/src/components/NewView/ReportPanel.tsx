import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
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
    score?: number;
    struct_score?: number;
    text_score?: number;
    subtype?: string;
    retrieval?: string;
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
    direction?: 'unlearn' | 'learn' | string;
    restored?: boolean;
    before?: { ce?: number; logprob?: number; saliency?: number | null };
    after?: { ce?: number; logprob?: number; saliency?: number | null };
    delta?: { ce?: number; logprob?: number; saliency?: number | null };
    update?: { paramSpace?: string; lastNLayers?: number; direction?: string; steps?: number; unlearnLr?: number };
    testEdge?: { reportedSaliency?: number | null; saliencyMode?: string };
    intervention?: { active?: boolean; direction?: string; pairId?: string | null; steps?: number };
}

/** Default normalized LoRA step size for Learn/Unlearn (||Δθ||₂ = η). */
const DEFAULT_PAIR_INTERVENE_LR = 0.05;

interface NextTokenProbRow {
    token: string;
    tokenId: number;
    prob: number;
    isActual?: boolean;
}

interface NextTokenProbResult {
    status?: string;
    mode?: 'predict' | 'gold' | string;
    targetIndex?: number;
    actualToken?: string;
    actualProb?: number;
    top?: NextTokenProbRow[];
    intervention?: { active?: boolean; direction?: string; pairId?: string | null };
    error?: string;
    viewFamily?: string;
    liveFamily?: string;
    availableViews?: AdapterViewTab[];
    flip?: DegradationFlip;
}

interface AdapterViewTab {
    id: string;
    label: string;
    family?: string;
    path?: string;
}

interface DegradationFlipSide {
    argmaxId?: number;
    argmaxToken?: string;
    argmaxProb?: number;
    logitGained?: number;
    logitLost?: number;
    logitMargin?: number;
    pGained?: number;
    pLost?: number;
    nllLost?: number;
}

interface DegradationFlip {
    viewFamily?: string;
    liveFamily?: string;
    gainedToken?: string;
    gainedTokenId?: number;
    lostToken?: string;
    lostTokenId?: number;
    flipped?: boolean;
    live?: DegradationFlipSide;
    compare?: DegradationFlipSide;
    deltaMargin?: number;
    deltaNllLost?: number;
}

interface DegradeProgress {
    done: number;
    total: number;
    nTrains?: number;
    trainIdx?: number | null;
    src?: number | null;
    dst?: number | null;
    srcTok?: string;
    dstTok?: string;
    stage?: string;
    message?: string;
    cacheHits?: number;
    cacheMisses?: number;
}

interface PerTokenResult {
    target_token_index: number;
    target_token: string;
    top_correlations: TestCorrelation[];
    correlation_pairs: CorrelationPair[];
}

interface TrainSampleDetail {
    full_tokens: string[];
    full_token_ids?: number[];
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
        full_token_ids?: number[];
        correct_full_token_ids?: number[];
        full_tokens_display?: string[];
        correct_full_tokens_display?: string[];
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
    return String(t ?? '').replaceAll('Ċ', '\n').replaceAll('Ġ', ' ').replaceAll('ĉ', '  ');
}

function decodeTokens(tokens: string[]): string[] {
    return tokens.map(decodeToken);
}

/** Prefer server-built display surfaces (byte-fallback merges); else raw tokens. */
function pickDisplayTokens(raw: string[], display?: string[] | null): string[] {
    if (display && display.length === raw.length) return decodeTokens(display);
    return decodeTokens(raw);
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
    const display = token === '\n' ? '↵\n' : token === '  ' ? '→' : token === '' ? '\u200b' : token;
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
    /** Default: only analyzed response tokens. `prompt` = context/prefix only. */
    clickScope = 'analyzed',
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
    clickScope?: 'analyzed' | 'prompt' | 'all';
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

                    const clickable = Boolean(onTokenClick) && (
                        clickScope === 'all'
                        || (clickScope === 'prompt' && !isResponse)
                        || (clickScope === 'analyzed' && isAnalyzed)
                    );
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
    modelClickScope = 'analyzed',
    goldSelectedLocalIndex,
    goldHighlightSourceIndices,
    onGoldTokenClick,
    goldHint,
    linkedTokenIndex,
    onTokenHover,
    saliencySelected,
    saliencySelectEnabled,
    onToggleSaliencySelect,
    headerExtra,
}: {
    modelTokens: string[];
    /** Gold answer tokens only — no prompt prefix. */
    goldResponseTokens: string[];
    promptLen: number;
    highlightSourceIndices?: Set<number>;
    selectedTargetIndex?: number;
    analyzedIndices?: Set<number>;
    onTokenClick?: (idx: number) => void;
    modelClickScope?: 'analyzed' | 'prompt' | 'all';
    goldSelectedLocalIndex?: number | null;
    goldHighlightSourceIndices?: Set<number>;
    onGoldTokenClick?: (localIdx: number) => void;
    goldHint?: string;
    linkedTokenIndex?: number | null;
    onTokenHover?: (idx: number | null) => void;
    /** Whether the current test saliency edge is ticked for probe filtering. */
    saliencySelected?: boolean;
    saliencySelectEnabled?: boolean;
    onToggleSaliencySelect?: () => void;
    headerExtra?: ReactNode;
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
                <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
                {headerExtra}
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
            </div>
            <div className={styles.outputCompareBody}>
                <div className={styles.outputSection}>
                    <div className={styles.outputSectionHeader}>
                        <span className={`${styles.outputSectionTitle} ${styles.outputSectionTitleModel}`}>
                            Model
                        </span>
                        {modelClickScope === 'prompt' && (
                            <span style={{ marginLeft: 8, fontSize: 11, color: '#7c3aed' }}>
                                点击灰色上下文 token → 选 source
                            </span>
                        )}
                    </div>
                    <CodeTokenStream
                        tokens={modelTokens}
                        promptLen={promptLen}
                        responseTone="model"
                        highlightSourceIndices={highlightSourceIndices}
                        selectedTargetIndex={selectedTargetIndex}
                        analyzedIndices={analyzedIndices}
                        onTokenClick={onTokenClick}
                        clickScope={modelClickScope}
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
                                {goldHint ?? '点击 → 现场 teacher-force 归因'}
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

function unlearnVerdictLabel(verdict: string | undefined, direction?: string): { text: string; color: string } {
    const isLearn = direction === 'learn';
    switch (verdict) {
        case 'supports_causal':
            return {
                text: isLearn
                    ? '像真因果：学习后 target 更容易'
                    : '像真因果：遗忘后 target 更难',
                color: '#166534',
            };
        case 'saliency_only':
            return {
                text: isLearn
                    ? 'saliency 上升，但 CE 几乎不动'
                    : 'saliency 下降，但 CE 几乎不动',
                color: '#a16207',
            };
        case 'no_effect':
            return { text: '几乎无影响', color: '#64748b' };
        case 'opposite_effect':
            return {
                text: isLearn
                    ? '反向：学习后 target 反而更难'
                    : '反向：遗忘后 target 反而更容易',
                color: '#9a3412',
            };
        default:
            return { text: '结果不明确', color: '#6b7280' };
    }
}

function formatProbPct(p: number): string {
    if (!Number.isFinite(p)) return '—';
    const pct = p * 100;
    if (pct < 0.001 && pct > 0) return `${pct.toExponential(2)}%`;
    return `${pct.toFixed(3)}%`;
}

function NextTokenProbPanel({
    result,
    busy,
    error,
    interventionActive,
    interventionDirection,
    interventionSteps,
    viewFamily,
    onViewFamilyChange,
    degradePairs,
    degradeBusy,
    degradeError,
    degradeProgress,
    onRetrieveDegrade,
}: {
    result: NextTokenProbResult | null;
    busy?: boolean;
    error?: string | null;
    interventionActive?: boolean;
    interventionDirection?: string | null;
    interventionSteps?: number;
    viewFamily: string;
    onViewFamilyChange?: (id: string) => void;
    degradePairs?: CorrelationPair[];
    degradeBusy?: boolean;
    degradeError?: string | null;
    degradeProgress?: DegradeProgress | null;
    onRetrieveDegrade?: () => void;
}) {
    const rows = result?.top ?? [];
    const maxP = Math.max(...rows.map(r => r.prob), 1e-12);
    const modeLabel = result?.mode === 'gold' ? 'teacher-forced' : 'model predict';
    const steps = Math.max(1, interventionSteps ?? 1);
    const views = result?.availableViews?.length
        ? result.availableViews
        : [
            { id: 'live', label: '当前' },
            { id: 'ce', label: 'CE' },
            { id: 'base', label: 'Base' },
        ];
    const flip = result?.flip;
    const showDegrade = viewFamily !== 'live' && flip != null && (result?.viewFamily === viewFamily);
    const viewTitle = viewFamily === 'live'
        ? '当前'
        : (views.find(v => v.id === viewFamily)?.label ?? viewFamily);

    return (
        <div className={styles.probPanel}>
            <div className={styles.probPanelHeader}>
                <div className={styles.probPanelTitle}>
                    top next-token probabilities
                    {result?.targetIndex != null
                        ? ` · ${modeLabel} @ ${result.targetIndex}`
                        : ' (distribution that produced this token)'}
                    {viewFamily !== 'live' ? ` · ${viewTitle}` : ''}
                </div>
                {onViewFamilyChange && (
                    <div className={styles.probViewTabs} role="tablist" aria-label="adapter probability view">
                        {views.map(v => (
                            <button
                                key={v.id}
                                type="button"
                                role="tab"
                                aria-selected={viewFamily === v.id}
                                className={`${styles.probViewTab}${viewFamily === v.id ? ` ${styles.probViewTabActive}` : ''}`}
                                disabled={busy || interventionActive}
                                title={v.path || v.family || v.label}
                                onClick={() => onViewFamilyChange(v.id)}
                            >
                                {v.label}
                            </button>
                        ))}
                    </div>
                )}
                {result?.actualProb != null && (
                    <span className={styles.probPanelMeta}>
                        P(actual={JSON.stringify(decodeToken(result.actualToken ?? ''))})={formatProbPct(result.actualProb)}
                    </span>
                )}
            </div>
            {viewFamily === 'live' && onViewFamilyChange && (
                <div className={styles.degradeHint}>
                    切 CE / Base 对比同一位置的概率，并打开退化归因
                </div>
            )}
            {interventionActive && (
                <div className={styles.probInterveneBanner}>
                    Active {interventionDirection === 'learn' ? 'Learn' : 'Unlearn'}
                    {steps > 1 ? ` ×${steps}` : ''}
                    — probabilities reflect modified weights. Recover from the correlation card on the right.
                </div>
            )}
            <div className={styles.probPanelBody}>
                {busy && <div className={styles.probEmpty}>Loading next-token distribution…</div>}
                {!busy && error && <div className={styles.probEmpty} style={{ color: '#f38ba8' }}>{error}</div>}
                {!busy && !error && rows.length === 0 && (
                    <div className={styles.probEmpty}>
                        Click a Model or Gold answer token to show the next-token distribution.
                    </div>
                )}
                {!busy && rows.map((row, i) => {
                    const width = `${Math.max(0.5, (row.prob / maxP) * 100)}%`;
                    const display = decodeToken(row.token).replace(/\n/g, '\\n');
                    const isGained = flip != null && row.tokenId === flip.gainedTokenId;
                    const isLost = flip != null && row.tokenId === flip.lostTokenId;
                    return (
                        <div key={`${row.tokenId}-${i}`} className={styles.probRow}>
                            <span className={`${styles.probTok}${row.isActual ? ` ${styles.probTokActual}` : ''}${isGained ? ` ${styles.probTokGained}` : ''}${isLost ? ` ${styles.probTokLost}` : ''}`}>
                                {display || '·'}
                                {isGained ? ' ↑live' : ''}
                                {isLost ? ' ↓cmp' : ''}
                            </span>
                            <span className={styles.probPct}>{formatProbPct(row.prob)}</span>
                            <div className={styles.probBarTrack}>
                                <div
                                    className={`${styles.probBarFill}${row.isActual ? ` ${styles.probBarFillTop}` : ''}`}
                                    style={{ width }}
                                />
                            </div>
                        </div>
                    );
                })}
            </div>
            {showDegrade && flip && (
                <div className={styles.degradeCard}>
                    <div className={styles.degradeTitle}>
                        退化归因 · 当前 vs {viewTitle}
                    </div>
                    <div className={styles.degradeFormula}>
                        Δ = (logit_{decodeToken(flip.gainedToken ?? '')} − logit_{decodeToken(flip.lostToken ?? '')})
                        <sub>当前</sub> − (·)<sub>{viewTitle}</sub>
                    </div>
                    {flip.flipped ? (
                        <div className={styles.degradeFlip}>
                            决策翻转：{viewTitle} argmax={JSON.stringify(decodeToken(flip.compare?.argmaxToken ?? ''))}
                            {' → '}
                            当前 argmax={JSON.stringify(decodeToken(flip.live?.argmaxToken ?? ''))}
                        </div>
                    ) : (
                        <div className={styles.degradeNoFlip}>
                            相对 {viewTitle} 没有翻转（两边 argmax 都是 {JSON.stringify(decodeToken(flip.live?.argmaxToken ?? ''))}）。
                            可切到 Base 再比；仍可按边归因，看哪些 L_sal 把质量推向 {JSON.stringify(decodeToken(flip.gainedToken ?? ''))}。
                        </div>
                    )}
                    <div className={styles.degradeMetrics}>
                        <span>Δ margin {formatSigned(flip.deltaMargin)}</span>
                        <span>Δ NLL({JSON.stringify(decodeToken(flip.lostToken ?? ''))}) {formatSigned(flip.deltaNllLost)}</span>
                        <span>当前 P(lost)={formatProbPct(flip.live?.pLost ?? NaN)}</span>
                        <span>{viewTitle} P(lost)={formatProbPct(flip.compare?.pLost ?? NaN)}</span>
                    </div>
                    {onRetrieveDegrade && (
                        <button
                            type="button"
                            className={styles.degradeBtn}
                            disabled={degradeBusy || busy}
                            onClick={onRetrieveDegrade}
                        >
                            {degradeBusy ? '按边归因中…' : '按边归因 L_sal'}
                        </button>
                    )}
                    {degradeBusy && (
                        <div className={styles.degradeProgress}>
                            <div className={styles.degradeProgressMeta}>
                                {degradeProgress && degradeProgress.total > 0
                                    ? `${degradeProgress.done} / ${degradeProgress.total} 标注边`
                                    : '正在统计标注边…'}
                                {typeof degradeProgress?.trainIdx === 'number'
                                    ? `  · train #${degradeProgress.trainIdx}`
                                    : ''}
                                {typeof degradeProgress?.cacheHits === 'number'
                                    || typeof degradeProgress?.cacheMisses === 'number'
                                    ? `  · hit ${degradeProgress?.cacheHits ?? 0} / miss ${degradeProgress?.cacheMisses ?? 0}`
                                    : ''}
                            </div>
                            <div className={styles.degradeProgressTrack}>
                                <div
                                    className={styles.degradeProgressFill}
                                    style={{
                                        width: degradeProgress && degradeProgress.total > 0
                                            ? `${Math.min(100, (degradeProgress.done / degradeProgress.total) * 100)}%`
                                            : '8%',
                                    }}
                                />
                            </div>
                            {degradeProgress?.message && (
                                <div className={styles.degradeProgressMsg}>{degradeProgress.message}</div>
                            )}
                        </div>
                    )}
                    {degradeError && (
                        <div className={styles.degradeErr}>{degradeError}</div>
                    )}
                    {degradePairs && degradePairs.length > 0 && (
                        <div className={styles.degradeTrains}>
                            <div className={styles.degradeNoFlip}>
                                contrib = −cos(∇f, ∇L_sal)。正值 = 这条边顺着 New←Wrap。完整列表在右侧梯度栏，可 Unlearn 验证。
                            </div>
                            {degradePairs.slice(0, 8).map(p => {
                                const src = decodeToken(p.train_correlation.source_token).trim() || '·';
                                const dst = decodeToken(p.train_correlation.target_token).trim() || '·';
                                const contrib = p.cos_sim;
                                return (
                                    <div key={p.id} className={styles.degradeTrainRow}>
                                        <div className={styles.degradeTrainMeta}>
                                            <span>train #{p.train_sample_id}</span>
                                            <span>{src} → {dst}</span>
                                            <span style={{ color: contrib > 0 ? '#f38ba8' : '#a6adc8' }}>
                                                contrib {contrib >= 0 ? '+' : ''}{contrib.toFixed(4)}
                                            </span>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

function PairCard({
    pair,
    detail,
    selected,
    onToggleSelect,
    annotatedSourceIndices,
    onUnlearn,
    onLearn,
    onRecover,
    unlearnBusy,
    learnBusy,
    recoverBusy,
    unlearnResult,
    interveneActiveForPair,
    interventionSteps,
    interveneLr,
    onOpenAnnotationViewer,
    onAutoAnnotateContinue,
    defaultExpanded = false,
}: {
    pair: CorrelationPair;
    detail?: TrainSampleDetail;
    selected?: boolean;
    onToggleSelect?: () => void;
    annotatedSourceIndices?: Set<number>;
    onUnlearn?: () => void;
    onLearn?: () => void;
    onRecover?: () => void;
    unlearnBusy?: boolean;
    learnBusy?: boolean;
    recoverBusy?: boolean;
    unlearnResult?: UnlearnPairResult | null;
    interveneActiveForPair?: boolean;
    interventionSteps?: number;
    interveneLr?: number;
    /** Open annotation-viewer focused on this train sample + source/target. */
    onOpenAnnotationViewer?: () => void;
    onAutoAnnotateContinue?: () => void;
    defaultExpanded?: boolean;
}) {
    const [expanded, setExpanded] = useState(defaultExpanded);
    const { bg, fg } = cosSimilarityColor(pair.cos_sim);
    const verdict = unlearnVerdictLabel(unlearnResult?.verdict, unlearnResult?.direction);
    const busy = Boolean(unlearnBusy || learnBusy || recoverBusy);
    const steps = Math.max(1, interventionSteps ?? 1);
    const lr = interveneLr ?? DEFAULT_PAIR_INTERVENE_LR;

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
                        disabled={busy}
                        title={`对该 train target 做一步 CE ascent（Unlearn，η=${lr} normalized）。可连点累加，Recover 一次回到最初。`}
                        style={{
                            border: '1px solid #fda4af',
                            background: unlearnBusy ? '#ffe4e6' : '#fff1f2',
                            color: '#be123c',
                            borderRadius: 999,
                            padding: '2px 8px',
                            fontSize: 11,
                            fontWeight: 700,
                            cursor: busy ? 'wait' : 'pointer',
                        }}
                    >
                        {unlearnBusy ? 'Unlearning…' : 'Unlearn'}
                    </button>
                )}

                {onLearn && (
                    <button
                        type="button"
                        onClick={(event) => {
                            event.stopPropagation();
                            onLearn();
                        }}
                        disabled={busy}
                        title={`对该 train target 做一步 CE descent（Learn，η=${lr} normalized）。可连点累加，Recover 一次回到最初。`}
                        style={{
                            border: '1px solid #86efac',
                            background: learnBusy ? '#dcfce7' : '#f0fdf4',
                            color: '#15803d',
                            borderRadius: 999,
                            padding: '2px 8px',
                            fontSize: 11,
                            fontWeight: 700,
                            cursor: busy ? 'wait' : 'pointer',
                        }}
                    >
                        {learnBusy ? 'Learning…' : 'Learn'}
                    </button>
                )}

                {interveneActiveForPair && onRecover && (
                    <button
                        type="button"
                        onClick={(event) => {
                            event.stopPropagation();
                            onRecover();
                        }}
                        disabled={busy}
                        title="恢复到 Learn/Unlearn 之前的原始权重"
                        style={{
                            border: '1px solid #67e8f9',
                            background: recoverBusy ? '#cffafe' : '#ecfeff',
                            color: '#0e7490',
                            borderRadius: 999,
                            padding: '2px 8px',
                            fontSize: 11,
                            fontWeight: 700,
                            cursor: busy ? 'wait' : 'pointer',
                        }}
                    >
                        {recoverBusy ? 'Recovering…' : (steps > 1 ? `Recover ×${steps}` : 'Recover')}
                    </button>
                )}

                <span className={styles.pairId}>{pair.id}</span>

                <span
                    className={styles.cosSim}
                    style={{ background: bg, color: fg }}
                    title={pair.retrieval === 'degrade_sal_edge'
                        ? 'contrib = −cos(∇f, ∇L_sal)'
                        : 'cos_sim'}
                >
                    {pair.retrieval === 'degrade_sal_edge' ? 'contrib ' : ''}
                    {(pair.score ?? pair.cos_sim).toFixed(4)}
                </span>
                {(() => {
                    const st = pair as CorrelationPair & {
                        struct_score?: number;
                        text_score?: number;
                        ast?: Record<string, string>;
                        retrieval?: string;
                    };
                    if (st.retrieval !== 'structural_ast' && st.struct_score == null) return null;
                    const ast = st.ast || {};
                    return (
                        <span style={{ fontSize: 10, color: '#6d28d9', fontWeight: 600 }}>
                            {ast.train_lca
                                ? `lca=${ast.train_lca}/${ast.train_relation || '?'}`
                                : 'AST'}
                            {' · '}struct {(st.struct_score ?? 0).toFixed(3)}
                            {' · '}text {(st.text_score ?? 0).toFixed(3)}
                        </span>
                    );
                })()}

                <span className={styles.corrTag} style={{ background: '#eff6ff', borderColor: '#bfdbfe', color: '#1d4ed8' }}>
                    <span className={styles.corrLabel}>test </span>
                    <strong>{(pair.test_correlation.source_token ?? '').trim() || '·'}</strong>
                    <span className={styles.arrow}> → </span>
                    <strong>{(pair.test_correlation.target_token ?? '').trim() || '·'}</strong>
                </span>

                <span className={styles.corrArrow}>⇔</span>

                <span className={styles.corrTag} style={{ background: '#fffbeb', borderColor: '#fde68a', color: '#92400e' }}>
                    <span className={styles.corrLabel}>train </span>
                    <strong>{(pair.train_correlation.source_token ?? '').trim() || '·'}</strong>
                    <span className={styles.arrow}> → </span>
                    <strong>{(pair.train_correlation.target_token ?? '').trim() || '·'}</strong>
                    <span className={styles.offset}>+{pair.train_correlation.response_token_offset}</span>
                </span>

                <span
                    className={styles.trainBadge}
                    role="link"
                    tabIndex={0}
                    title="在 annotation-viewer 中打开该 train，黄=source / 橙=target（仅可视化）"
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
                {onAutoAnnotateContinue && (
                    <button
                        type="button"
                        onClick={(event) => {
                            event.stopPropagation();
                            onAutoAnnotateContinue();
                        }}
                        title="打开 annotation-viewer 并用 LLM 围绕该 train pair 自动标注 → 写入续训小集（不含旧标注）"
                        style={{
                            border: '1px solid #c4b5fd',
                            background: '#f5f3ff',
                            color: '#6d28d9',
                            borderRadius: 999,
                            padding: '2px 8px',
                            fontSize: 11,
                            fontWeight: 700,
                            cursor: 'pointer',
                        }}
                    >
                        自动标注
                    </button>
                )}
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
                        ΔCE = <strong>{formatSigned(unlearnResult.delta?.ce)}</strong>
                        {' · '}
                        Δsaliency = <strong>{formatSigned(unlearnResult.delta?.saliency, 6)}</strong>
                    </div>
                    <div style={{ color: '#78716c', fontSize: 10 }}>
                        CE {formatSigned(unlearnResult.before?.ce)} → {formatSigned(unlearnResult.after?.ce)}
                        {unlearnResult.before?.saliency != null && (
                            <>
                                {' · '}sal {formatSigned(unlearnResult.before.saliency, 6)} → {formatSigned(unlearnResult.after?.saliency, 6)}
                            </>
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
                        一步 {unlearnResult.direction === 'learn' ? 'Learn' : 'Unlearn'}{' '}
                        {unlearnResult.update?.paramSpace || 'match-space'}
                        {unlearnResult.update?.lastNLayers != null
                            ? ` · last-${unlearnResult.update.lastNLayers}`
                            : ''}
                        {unlearnResult.update?.steps != null && unlearnResult.update.steps > 1
                            ? ` · stacked ×${unlearnResult.update.steps}`
                            : ''}
                        {unlearnResult.testEdge?.saliencyMode
                            ? ` · ${unlearnResult.testEdge.saliencyMode}`
                            : ''}
                        {unlearnResult.restored === false
                            ? ' · 权重已修改（可 Recover）'
                            : ' · 已 restore · 不写回 adapter'}
                    </div>
                    {unlearnResult.error && (
                        <div style={{ color: '#b91c1c' }}>{unlearnResult.error}</div>
                    )}
                </div>
            )}

            {expanded && (
                <div className={styles.pairCardBody}>
                    {pair.retrieval !== 'structural_ast' && (
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
                    )}
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
    onLearnPair,
    onRecoverIntervention,
    interveningPairId,
    interveningDirection,
    recoverBusy,
    activeInterventionPairId,
    interventionSteps,
    interveneLr,
    unlearnResults,
    onOpenAnnotationViewer,
    onAutoAnnotateContinue,
    pairDefaultExpanded = false,
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
    onLearnPair?: (pair: CorrelationPair) => void;
    onRecoverIntervention?: () => void;
    interveningPairId?: string | null;
    interveningDirection?: 'unlearn' | 'learn' | null;
    recoverBusy?: boolean;
    activeInterventionPairId?: string | null;
    interventionSteps?: number;
    interveneLr?: number;
    unlearnResults?: Record<string, UnlearnPairResult>;
    onOpenAnnotationViewer?: (pair: CorrelationPair) => void;
    onAutoAnnotateContinue?: (pair: CorrelationPair) => void;
    pairDefaultExpanded?: boolean;
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
                <span className={styles.trainGroupCoarse}>
                    {pairs[0]?.retrieval === 'degrade_sal_edge' ? 'contrib' : 'coarse'}{' '}
                    {(detail?.coarse_cos_sim ?? pairs[0]?.coarse_cos_sim ?? 0).toFixed(4)}
                </span>
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
                                onLearn={onLearnPair ? () => onLearnPair(pair) : undefined}
                                onRecover={onRecoverIntervention}
                                unlearnBusy={interveningPairId === pair.id && interveningDirection === 'unlearn'}
                                learnBusy={interveningPairId === pair.id && interveningDirection === 'learn'}
                                recoverBusy={recoverBusy}
                                unlearnResult={unlearnResults?.[pair.id] ?? null}
                                interveneActiveForPair={activeInterventionPairId === pair.id}
                                interventionSteps={
                                    activeInterventionPairId === pair.id ? interventionSteps : undefined
                                }
                                interveneLr={interveneLr}
                                onOpenAnnotationViewer={
                                    onOpenAnnotationViewer
                                        ? () => onOpenAnnotationViewer(pair)
                                        : undefined
                                }
                                onAutoAnnotateContinue={
                                    onAutoAnnotateContinue
                                        ? () => onAutoAnnotateContinue(pair)
                                        : undefined
                                }
                                defaultExpanded={pairDefaultExpanded}
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
    // Gold live attribution (teacher-force API) | manual = user-picked source→gold-target
    type AttrMode = 'predict' | 'gold' | 'manual';
    const [attrMode, setAttrMode] = useState<AttrMode>('predict');
    const [goldLocalIdx, setGoldLocalIdx] = useState<number | null>(null);
    const [goldTopCorrelations, setGoldTopCorrelations] = useState<TestCorrelation[]>([]);
    const [goldSelectedCorrIdx, setGoldSelectedCorrIdx] = useState<number | null>(null);
    const [goldPairs, setGoldPairs] = useState<CorrelationPair[]>([]);
    const [goldTrainDetails, setGoldTrainDetails] = useState<Record<string, TrainSampleDetail>>({});
    const [goldBusy, setGoldBusy] = useState(false);
    /** Manual pair pick: source = context (prompt), target = gold completion (absolute idx). */
    const [manualSourceIdx, setManualSourceIdx] = useState<number | null>(null);
    const [manualTargetAbsIdx, setManualTargetAbsIdx] = useState<number | null>(null);
    /** Gradient Stage3 pairs for 指定 pair (same API as gold edge retrieve). */
    const [manualGradPairs, setManualGradPairs] = useState<CorrelationPair[]>([]);
    const [manualTrainDetails, setManualTrainDetails] = useState<Record<string, TrainSampleDetail>>({});
    const [manualGradBusy, setManualGradBusy] = useState(false);
    const clearGoldLive = useCallback(() => {
        setGoldLocalIdx(null);
        setGoldTopCorrelations([]);
        setGoldSelectedCorrIdx(null);
        setGoldPairs([]);
        setGoldTrainDetails({});
        setGoldBusy(false);
    }, []);
    const clearManualPair = useCallback(() => {
        setManualSourceIdx(null);
        setManualTargetAbsIdx(null);
        setManualGradPairs([]);
        setManualTrainDetails({});
        setManualGradBusy(false);
        setManualEdgeSaliency(null);
        setManualEdgeSaliencyErr(null);
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
    const [interveningPairId, setInterveningPairId] = useState<string | null>(null);
    const [interveningDirection, setInterveningDirection] = useState<'unlearn' | 'learn' | null>(null);
    const [recoverBusy, setRecoverBusy] = useState(false);
    const [activeInterventionPairId, setActiveInterventionPairId] = useState<string | null>(null);
    const [activeInterventionDirection, setActiveInterventionDirection] = useState<string | null>(null);
    const [interventionSteps, setInterventionSteps] = useState(0);
    const [pairInterveneLr, setPairInterveneLr] = useState(DEFAULT_PAIR_INTERVENE_LR);
    const [pairInterveneLrInput, setPairInterveneLrInput] = useState(String(DEFAULT_PAIR_INTERVENE_LR));
    const [continueStepsInput, setContinueStepsInput] = useState('20');
    const [continueLrInput, setContinueLrInput] = useState('2e-5');
    const [continueStepsDefault, setContinueStepsDefault] = useState(20);
    const [continueLrDefault, setContinueLrDefault] = useState('2e-5');
    const [continueBusy, setContinueBusy] = useState(false);
    const [continueJobId, setContinueJobId] = useState<string | null>(null);
    const [continueResultSummary, setContinueResultSummary] = useState<string | null>(null);
    const [continueAdapterActive, setContinueAdapterActive] = useState(false);
    const [continueRecoverBusy, setContinueRecoverBusy] = useState(false);
    /** After continue-train: live predict top-k overlay (null = use report JSON). */
    const [predictLiveTop, setPredictLiveTop] = useState<TestCorrelation[] | null>(null);
    const [predictLiveBusy, setPredictLiveBusy] = useState(false);
    /** 指定 pair: live saliency of the selected source→target edge. */
    const [manualEdgeSaliency, setManualEdgeSaliency] = useState<number | null>(null);
    const [manualEdgeSaliencyBusy, setManualEdgeSaliencyBusy] = useState(false);
    const [manualEdgeSaliencyErr, setManualEdgeSaliencyErr] = useState<string | null>(null);
    const [structuralAttributionEnabled, setStructuralAttributionEnabled] = useState(false);
    const [structuralPairs, setStructuralPairs] = useState<CorrelationPair[]>([]);
    const [structuralTrainDetails, setStructuralTrainDetails] = useState<Record<string, TrainSampleDetail>>({});
    const [structuralBusy, setStructuralBusy] = useState(false);
    const [structuralError, setStructuralError] = useState<string | null>(null);
    const [structuralMeta, setStructuralMeta] = useState<string | null>(null);
    const enterManualPairMode = useCallback(() => {
        setAttrMode('manual');
        setSelectedTokIdx(null);
        setSelectedTestCorrIdx(null);
        clearGoldLive();
        clearManualPair();
        setStructuralAttributionEnabled(true);
        setModelSaliencySelected(false);
        setTtavLaunchError(null);
        setDegradePairs([]);
        setDegradeTrainDetails({});
        setDegradeError(null);
        setDegradeProgress(null);
        setTtavLaunchStatus('指定 pair：先点 Model 上下文选 source，再点 Gold 选 target');
    }, [clearGoldLive, clearManualPair, setSelectedTokIdx]);
    const exitManualPairMode = useCallback(() => {
        clearManualPair();
        setAttrMode('predict');
        setTtavLaunchStatus(null);
    }, [clearManualPair]);
    const [unlearnResultsByPairId, setUnlearnResultsByPairId] = useState<Record<string, UnlearnPairResult>>({});
    const [tokenProbResult, setTokenProbResult] = useState<NextTokenProbResult | null>(null);
    const [tokenProbBusy, setTokenProbBusy] = useState(false);
    const [tokenProbError, setTokenProbError] = useState<string | null>(null);
    const [tokenProbFocus, setTokenProbFocus] = useState<{ mode: 'predict' | 'gold'; index: number } | null>(null);
    const tokenProbFocusRef = useRef<{ mode: 'predict' | 'gold'; index: number } | null>(null);
    const [probViewFamily, setProbViewFamily] = useState('live');
    const [degradePairs, setDegradePairs] = useState<CorrelationPair[]>([]);
    const [degradeTrainDetails, setDegradeTrainDetails] = useState<Record<string, TrainSampleDetail>>({});
    const [degradeBusy, setDegradeBusy] = useState(false);
    const [degradeError, setDegradeError] = useState<string | null>(null);
    const [degradeProgress, setDegradeProgress] = useState<DegradeProgress | null>(null);
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

    const familyPrefix = meta.fileName.startsWith('ce/')
        ? '[ce] '
        : meta.fileName.startsWith('saliency/')
            ? '[saliency] '
            : '';
    const rawLabel = modelLabel
        || report.experiment_meta.model_name
        || meta.label
        || 'model';
    const displayModelLabel = (
        familyPrefix && !String(rawLabel).startsWith('[ce]') && !String(rawLabel).startsWith('[saliency]')
            ? `${familyPrefix}${rawLabel}`
            : rawLabel
    );

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
        setInterveningPairId(null);
        setInterveningDirection(null);
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

    // Sync Continue train steps/lr defaults from eif_api.env (EIF_CONTINUE_MAX_STEPS).
    useEffect(() => {
        let cancelled = false;
        void (async () => {
            try {
                const resp = await fetch(`${eifApiUrl.replace(/\/$/, '')}/api/continue-train-eval-defaults`);
                if (!resp.ok || cancelled) return;
                const data = await resp.json() as {
                    status?: string;
                    defaults?: { max_steps?: number; learning_rate?: number };
                };
                if (data.status !== 'success' || !data.defaults || cancelled) return;
                const steps = Number(data.defaults.max_steps);
                if (Number.isFinite(steps) && steps >= 1) {
                    const s = String(Math.floor(steps));
                    setContinueStepsDefault(Math.floor(steps));
                    setContinueStepsInput(prev => (prev === '50' || prev === '20' ? s : prev));
                }
                const lr = Number(data.defaults.learning_rate);
                if (Number.isFinite(lr) && lr > 0) {
                    const nice = lr === 2e-5
                        ? '2e-5'
                        : lr.toExponential().replace(/\.0+e/, 'e').replace(/e\+/, 'e');
                    setContinueLrDefault(nice);
                    setContinueLrInput(prev => (prev === '2e-5' ? nice : prev));
                }
            } catch {
                // keep local defaults
            }
        })();
        return () => { cancelled = true; };
    }, [eifApiUrl]);

    const [fullTokensDisplay, setFullTokensDisplay] = useState<string[] | null>(
        () => report.test_sample_baseline.full_tokens_display ?? null,
    );
    const [correctTokensDisplay, setCorrectTokensDisplay] = useState<string[] | null>(
        () => report.test_sample_baseline.correct_full_tokens_display ?? null,
    );

    useEffect(() => {
        setFullTokensDisplay(report.test_sample_baseline.full_tokens_display ?? null);
        setCorrectTokensDisplay(report.test_sample_baseline.correct_full_tokens_display ?? null);
    }, [report]);

    useEffect(() => {
        if (importedReportActive || !selectedMeta) return;
        const baseline = report.test_sample_baseline;
        const haveFull = (baseline.full_tokens_display?.length ?? 0) === baseline.full_tokens.length;
        const goldRaw = baseline.correct_full_tokens ?? [];
        const haveGold = goldRaw.length === 0
            || (baseline.correct_full_tokens_display?.length ?? 0) === goldRaw.length;
        if (haveFull && haveGold) return;
        if (!baseline.full_token_ids?.length) return;

        let cancelled = false;
        void (async () => {
            try {
                const resp = await fetch(buildEifApiUrl(eifApiUrl, '/api/token-display-surfaces'), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ reportFileName: selectedMeta.fileName }),
                });
                const raw = await resp.text();
                const parsed = raw.trim() ? JSON.parse(raw) as Record<string, unknown> : {};
                if (!resp.ok || parsed.status !== 'success' || cancelled) return;
                const full = parsed.fullTokensDisplay;
                const gold = parsed.correctFullTokensDisplay;
                if (Array.isArray(full) && full.every(x => typeof x === 'string')) {
                    setFullTokensDisplay(full as string[]);
                }
                if (Array.isArray(gold) && gold.every(x => typeof x === 'string')) {
                    setCorrectTokensDisplay(gold as string[]);
                }
            } catch {
                // Keep raw surfaces; attribution still works via ids on the server.
            }
        })();
        return () => { cancelled = true; };
    }, [report, selectedMeta, importedReportActive, eifApiUrl]);

    const modelTokens = useMemo(
        () => pickDisplayTokens(report.test_sample_baseline.full_tokens, fullTokensDisplay),
        [report, fullTokensDisplay],
    );
    const correctTokens = useMemo(
        () => pickDisplayTokens(
            report.test_sample_baseline.correct_full_tokens ?? [],
            correctTokensDisplay,
        ),
        [report, correctTokensDisplay],
    );
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
        clearManualPair();
    }, [selectedSampleId, clearGoldLive, clearManualPair]);

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
        if (attrMode === 'predict') {
            const top = predictLiveTop ?? selectedResult?.top_correlations;
            if (!top || top.length === 0) return new Set<number>();
            if (selectedTestCorrIdx !== null) return new Set([selectedTestCorrIdx]);
            return new Set(top.map(c => c.source_token_index));
        }
        if (!selectedResult) return new Set<number>();
        if (selectedTestCorrIdx !== null) return new Set([selectedTestCorrIdx]);
        return new Set(selectedResult.top_correlations.map(c => c.source_token_index));
    }, [attrMode, predictLiveTop, selectedResult, selectedTestCorrIdx]);

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

        if (degradePairs.length > 0) {
            return [...degradePairs].sort((a, b) => b.cos_sim - a.cos_sim);
        }

        // Manual pair: gradient Stage3 on the user-picked gold edge.
        if (attrMode === 'manual') {
            return manualGradPairs
                .filter(keep)
                .sort((a, b) => b.cos_sim - a.cos_sim);
        }

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
        goldPairs, goldSelectedCorrIdx, manualGradPairs, degradePairs,
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

    const resolveTrainDetail = useCallback((id: number): TrainSampleDetail | undefined => {
        const key = String(id);
        if (degradeTrainDetails[key]) return degradeTrainDetails[key];
        if (attrMode === 'manual') {
            return manualTrainDetails[key] ?? report.train_sample_details[key];
        }
        if (attrMode === 'gold') {
            return goldTrainDetails[key] ?? report.train_sample_details[key];
        }
        return report.train_sample_details[key];
    }, [
        attrMode, manualTrainDetails, goldTrainDetails, degradeTrainDetails,
        report.train_sample_details,
    ]);

    const trainPanelEmptyHint = useMemo(() => {
        if (degradeBusy) {
            if (degradeProgress && degradeProgress.total > 0) {
                return `按边归因：${degradeProgress.done} / ${degradeProgress.total} 标注边`
                    + (typeof degradeProgress.trainIdx === 'number' ? ` · train #${degradeProgress.trainIdx}` : '');
            }
            return '按边归因：正在统计标注边并算 ∇f…';
        }
        if (degradeError) {
            return `按边归因失败：${degradeError}`;
        }
        if (degradePairs.length > 0) {
            return 'contrib = −cos(∇f, ∇L_sal)。正值 = 这条边顺着当前相对对照的翻转；可在卡片上 Unlearn 验证。';
        }
        if (attrMode === 'manual') {
            if (manualSourceIdx === null && manualTargetAbsIdx === null) {
                return '指定 pair：点 Model 灰色上下文选 source，再点 Gold 选 target；上下栏分别跑梯度 Stage3 与结构检索。';
            }
            if (manualSourceIdx === null) {
                return '已选 target — 请再点 Model 上下文 token 作为 source。';
            }
            if (manualTargetAbsIdx === null) {
                return '已选 source — 请再点 Gold complete token 作为 target。';
            }
            if (manualGradBusy) {
                return '指定 pair：正在梯度检索 train + Stage3…';
            }
            if (manualGradPairs.length === 0) {
                return '指定 pair 梯度检索无结果。可换一条边，或看下方结构归因。';
            }
            return '指定 pair 梯度归因结果（对该 gold edge 做 bank 匹配）。';
        }
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
        degradeBusy, degradeError, degradePairs.length,
        attrMode, manualSourceIdx, manualTargetAbsIdx, manualGradBusy, manualGradPairs.length,
        goldLocalIdx, goldBusy, goldTopCorrelations.length, goldSelectedCorrIdx,
        selectedResult, selectedTestCorrIdx, importedReportActive, allDisplayPairs.length,
        degradeProgress,
    ]);

    // Gold sources in the shared prompt → yellow on Model stream (same indices).
    // Do not paint Model answer: those indices would be predict text, not gold.
    const goldModelHighlightSourceIndices = useMemo(() => {
        if (attrMode !== 'gold' || goldTopCorrelations.length === 0) return new Set<number>();
        const abs = goldSelectedCorrIdx !== null
            ? [goldSelectedCorrIdx]
            : goldTopCorrelations.map(c => c.source_token_index);
        return new Set(
            abs.filter(i => i >= 0 && i < promptLen && i < modelTokens.length),
        );
    }, [attrMode, goldTopCorrelations, goldSelectedCorrIdx, promptLen, modelTokens.length]);

    // Gold answer panel: yellow-highlight answer-local saliency sources (like Model).
    const goldHighlightSourceIndices = useMemo(() => {
        if (attrMode !== 'gold' || goldTopCorrelations.length === 0) return new Set<number>();
        const abs = goldSelectedCorrIdx !== null
            ? [goldSelectedCorrIdx]
            : goldTopCorrelations.map(c => c.source_token_index);
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

        // Manual pair pick: gold click only sets target (no saliency API).
        if (attrMode === 'manual') {
            setManualTargetAbsIdx(prev => (prev === absIdx ? null : absIdx));
            setTtavLaunchError(null);
            setTtavLaunchStatus(
                `指定 pair target = "${decodeToken(goldResponseTokens[localIdx] ?? '').trim() || '·'}" @ ${absIdx}`,
            );
            return;
        }

        setAttrMode('gold');
        setSelectedTokIdx(null);
        setSelectedTestCorrIdx(null);
        clearManualPair();
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
        clearGoldLive, clearManualPair, setSelectedTokIdx, attrMode, goldResponseTokens,
    ]);

    const handleManualSourceClick = useCallback((idx: number) => {
        if (idx < 0 || idx >= promptLen) {
            setTtavLaunchError('指定 pair 的 source 必须是上下文（prompt）token，不能点 model 输出。');
            return;
        }
        setManualSourceIdx(prev => (prev === idx ? null : idx));
        setTtavLaunchError(null);
        const surf = decodeToken(modelTokens[idx] ?? correctTokens[idx] ?? '').trim() || '·';
        setTtavLaunchStatus(`指定 pair source = "${surf}" @ ${idx}`);
    }, [promptLen, modelTokens, correctTokens]);

    const handleGoldCorrClick = useCallback((sourceAbsIdx: number) => {
        if (goldLocalIdx === null) return;
        const absTarget = promptLen + goldLocalIdx;
        const next = goldSelectedCorrIdx === sourceAbsIdx ? null : sourceAbsIdx;
        setGoldSelectedCorrIdx(next);
        setGoldPairs([]);
        setGoldTrainDetails({});
        setDegradePairs([]);
        setDegradeTrainDetails({});
        setDegradeProgress(null);
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

    // 指定 pair: once source+target are set, run the same gold Stage3 gradient retrieve.
    useEffect(() => {
        if (importedReportActive) return;
        if (attrMode !== 'manual') return;
        if (manualSourceIdx === null || manualTargetAbsIdx === null) {
            setManualGradPairs([]);
            setManualTrainDetails({});
            setManualGradBusy(false);
            return;
        }
        let cancelled = false;
        setManualGradBusy(true);
        setManualGradPairs([]);
        setManualTrainDetails({});
        setTtavLaunchError(null);
        setTtavLaunchStatus(
            `指定 pair 梯度 Stage3 · src ${manualSourceIdx} → tgt ${manualTargetAbsIdx}…`,
        );
        void (async () => {
            try {
                const resp = await fetch(buildEifApiUrl(eifApiUrl, '/api/gold-retrieve-stage3'), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        reportFileName: selectedMeta.fileName,
                        sourceIndex: manualSourceIdx,
                        targetIndex: manualTargetAbsIdx,
                    }),
                });
                const raw = await resp.text();
                let parsed: Record<string, unknown> = {};
                try {
                    parsed = raw.trim() ? JSON.parse(raw) as Record<string, unknown> : {};
                } catch {
                    throw new Error(`指定 pair Stage3 non-JSON (HTTP ${resp.status}): ${raw.slice(0, 200)}`);
                }
                if (!resp.ok || parsed.status !== 'success') {
                    throw new Error(
                        typeof parsed.message === 'string'
                            ? parsed.message
                            : `指定 pair Stage3 failed (HTTP ${resp.status})`,
                    );
                }
                if (cancelled) return;
                const pairs = Array.isArray(parsed.correlationPairs)
                    ? parsed.correlationPairs as CorrelationPair[]
                    : [];
                const details = (
                    typeof parsed.trainSampleDetails === 'object'
                    && parsed.trainSampleDetails !== null
                ) ? parsed.trainSampleDetails as Record<string, TrainSampleDetail> : {};
                setManualGradPairs(pairs);
                setManualTrainDetails(details);
                setTtavLaunchStatus(
                    `指定 pair 梯度 ready · ${pairs.length} pairs · ${Object.keys(details).length} trains`,
                );
            } catch (error) {
                if (cancelled) return;
                const msg = error instanceof Error ? error.message : '指定 pair Stage3 failed';
                setTtavLaunchError(msg);
                setTtavLaunchStatus(null);
                setManualGradPairs([]);
                setManualTrainDetails({});
            } finally {
                if (!cancelled) setManualGradBusy(false);
            }
        })();
        return () => { cancelled = true; };
    }, [
        attrMode, manualSourceIdx, manualTargetAbsIdx,
        importedReportActive, eifApiUrl, selectedMeta.fileName,
    ]);

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

    const handleOpenAnnotationViewer = (pair: CorrelationPair, opts?: { autoAnnotate?: boolean }) => {
        const base = (
            import.meta.env.VITE_ANNOTATION_VIEWER_URL as string | undefined
        )?.trim() || 'http://127.0.0.1:5174';
        void (async () => {
            const url = new URL(base);
            url.searchParams.set('sample', String(pair.train_sample_id));
            url.searchParams.set('target', String(pair.train_correlation.target_token_index));
            url.searchParams.set('source', String(pair.train_correlation.source_token_index));
            const mode =
                attrMode === 'gold' ? 'gold' : attrMode === 'manual' ? 'manual' : 'predict';
            // predict → model tokens (MID = model output); gold/manual → correct tokens.
            const probeTokens = mode === 'predict' ? modelTokens : correctTokens;
            const probeSrcIdx = pair.test_correlation.source_token_index;
            const probeDstIdx = pair.test_correlation.target_token_index;
            const probeSrcTok =
                decodeToken(pair.test_correlation.source_token).trim()
                || decodeToken(probeTokens[probeSrcIdx] ?? '').trim()
                || '·';
            const probeDstTok =
                decodeToken(pair.test_correlation.target_token).trim()
                || decodeToken(probeTokens[probeDstIdx] ?? '').trim()
                || '·';
            url.searchParams.set('probeSrc', probeSrcTok);
            url.searchParams.set('probeDst', probeDstTok);
            url.searchParams.set('queryMode', mode);
            const payload = {
                probe_tokens: probeTokens,
                probe_answer_start: promptLen,
                probe_focus_src: probeSrcIdx,
                probe_focus_dst: probeDstIdx,
                probe_src_token: probeSrcTok,
                probe_dst_token: probeDstTok,
                probe_mid_text: null as string | null,
                query_mode: mode,
            };
            try {
                const resp = await fetch(`${base.replace(/\/$/, '')}/api/probe-focus-cache`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                if (resp.ok) {
                    const data = await resp.json() as { probe_id?: string };
                    if (data.probe_id) url.searchParams.set('probeId', data.probe_id);
                }
            } catch {
                // Viewer can still run with surfaces-only if cache POST fails.
            }
            if (opts?.autoAnnotate) {
                url.searchParams.set('autoAnnotate', '1');
            }
            window.open(url.toString(), '_blank', 'noopener,noreferrer');
        })();
    };

    const fetchTokenProbs = useCallback((
        mode: 'predict' | 'gold',
        targetIndex: number,
        opts?: { keepDegrade?: boolean; viewFamily?: string },
    ) => {
        if (!report || !selectedMeta || importedReportActive) return;
        if (!(targetIndex > 0)) {
            setTokenProbError('Cannot score next-token probs at index 0.');
            setTokenProbResult(null);
            return;
        }
        const prev = tokenProbFocusRef.current;
        const focusChanged = !prev || prev.mode !== mode || prev.index !== targetIndex;
        tokenProbFocusRef.current = { mode, index: targetIndex };
        setTokenProbFocus({ mode, index: targetIndex });
        setTokenProbBusy(true);
        setTokenProbError(null);
        if (focusChanged && !opts?.keepDegrade) {
            setDegradePairs([]);
            setDegradeTrainDetails({});
            setDegradeError(null);
            setDegradeProgress(null);
        }
        const viewFamily = opts?.viewFamily ?? probViewFamily;
        void (async () => {
            try {
                const resp = await fetch(buildEifApiUrl(eifApiUrl, '/api/next-token-probs'), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        reportFileName: selectedMeta.fileName,
                        mode,
                        targetIndex,
                        topK: 10,
                        viewFamily,
                        // Prefer eif_api.env (same as gold live); do not force report checkpoint.
                        modelPath: null,
                        baseModelPath: null,
                    }),
                });
                const rawText = await resp.text();
                let parsed: Record<string, unknown> = {};
                if (rawText.trim()) {
                    try {
                        parsed = JSON.parse(rawText) as Record<string, unknown>;
                    } catch {
                        throw new Error(
                            `Probs API returned non-JSON (HTTP ${resp.status}): ${rawText.slice(0, 240)}`,
                        );
                    }
                }
                if (!resp.ok || parsed.status !== 'success') {
                    const message = typeof parsed.message === 'string'
                        ? parsed.message
                        : `Probs API failed (HTTP ${resp.status})`;
                    throw new Error(message);
                }
                const result = parsed as NextTokenProbResult;
                setTokenProbResult(result);
            } catch (error) {
                const msg = error instanceof Error ? error.message : 'Failed to load next-token probs';
                setTokenProbError(msg);
                setTokenProbResult(null);
            } finally {
                setTokenProbBusy(false);
            }
        })();
    }, [report, selectedMeta, importedReportActive, eifApiUrl, probViewFamily]);

    useEffect(() => {
        if (importedReportActive) return;
        if (attrMode === 'predict' && selectedTokIdx != null && selectedTokIdx > 0) {
            fetchTokenProbs('predict', selectedTokIdx);
        }
    }, [attrMode, selectedTokIdx, importedReportActive, fetchTokenProbs]);

    useEffect(() => {
        if (importedReportActive) return;
        if (attrMode === 'gold' && goldLocalIdx != null) {
            const abs = promptLen + goldLocalIdx;
            if (abs > 0) fetchTokenProbs('gold', abs);
        }
    }, [attrMode, goldLocalIdx, promptLen, importedReportActive, fetchTokenProbs]);

    useEffect(() => {
        if (importedReportActive) return;
        if (attrMode === 'manual' && manualTargetAbsIdx != null && manualTargetAbsIdx > 0) {
            fetchTokenProbs('gold', manualTargetAbsIdx);
        }
    }, [attrMode, manualTargetAbsIdx, importedReportActive, fetchTokenProbs]);

    const refreshTokenProbs = useCallback((viewFamily?: string) => {
        if (tokenProbFocus) {
            fetchTokenProbs(tokenProbFocus.mode, tokenProbFocus.index, {
                keepDegrade: true,
                viewFamily,
            });
        }
    }, [tokenProbFocus, fetchTokenProbs]);

    const fetchDegradeRetrieve = useCallback(() => {
        if (!report || !selectedMeta || importedReportActive) return;
        if (!tokenProbFocus || !(tokenProbFocus.index > 0)) return;
        if (probViewFamily === 'live') return;
        setDegradeBusy(true);
        setDegradeError(null);
        setDegradeProgress({ done: 0, total: 0, message: '启动…' });
        void (async () => {
            const pollOnce = () => {
                void (async () => {
                    try {
                        const stResp = await fetch(
                            buildEifApiUrl(eifApiUrl, '/api/degradation-retrieve-progress'),
                        );
                        const st = await stResp.json() as Record<string, unknown>;
                        const done = typeof st.done === 'number' ? st.done : 0;
                        const total = typeof st.total === 'number' ? st.total : 0;
                        setDegradeProgress({
                            done,
                            total,
                            nTrains: typeof st.nTrains === 'number' ? st.nTrains : undefined,
                            trainIdx: typeof st.trainIdx === 'number' ? st.trainIdx : null,
                            src: typeof st.src === 'number' ? st.src : null,
                            dst: typeof st.dst === 'number' ? st.dst : null,
                            srcTok: typeof st.srcTok === 'string' ? st.srcTok : '',
                            dstTok: typeof st.dstTok === 'string' ? st.dstTok : '',
                            stage: typeof st.stage === 'string' ? st.stage : undefined,
                            message: typeof st.message === 'string' ? st.message : undefined,
                            cacheHits: typeof st.cacheHits === 'number' ? st.cacheHits : undefined,
                            cacheMisses: typeof st.cacheMisses === 'number' ? st.cacheMisses : undefined,
                        });
                    } catch {
                        /* keep last progress */
                    }
                })();
            };
            pollOnce();
            const poll = window.setInterval(pollOnce, 400);
            try {
                const resp = await fetch(buildEifApiUrl(eifApiUrl, '/api/degradation-retrieve'), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        reportFileName: selectedMeta.fileName,
                        mode: tokenProbFocus.mode,
                        targetIndex: tokenProbFocus.index,
                        compareFamily: probViewFamily,
                        gainedTokenId: tokenProbResult?.flip?.gainedTokenId,
                        lostTokenId: tokenProbResult?.flip?.lostTokenId,
                    }),
                });
                const rawText = await resp.text();
                let parsed: Record<string, unknown> = {};
                if (rawText.trim()) {
                    try {
                        parsed = JSON.parse(rawText) as Record<string, unknown>;
                    } catch {
                        throw new Error(
                            `Degrade API returned non-JSON (HTTP ${resp.status}): ${rawText.slice(0, 240)}`,
                        );
                    }
                }
                if (!resp.ok || parsed.status !== 'success') {
                    const message = typeof parsed.message === 'string'
                        ? parsed.message
                        : `Degrade retrieve failed (HTTP ${resp.status})`;
                    throw new Error(message);
                }
                const pairs = Array.isArray(parsed.correlationPairs)
                    ? parsed.correlationPairs as CorrelationPair[]
                    : [];
                const details = (
                    parsed.trainSampleDetails
                    && typeof parsed.trainSampleDetails === 'object'
                ) ? parsed.trainSampleDetails as Record<string, TrainSampleDetail> : {};
                setDegradePairs(pairs);
                setDegradeTrainDetails(details);
                if (parsed.flip && typeof parsed.flip === 'object') {
                    setTokenProbResult(prev => (
                        prev ? { ...prev, flip: parsed.flip as DegradationFlip } : prev
                    ));
                }
            } catch (error) {
                const msg = error instanceof Error ? error.message : 'Degradation retrieve failed';
                setDegradeError(msg);
                setDegradePairs([]);
                setDegradeTrainDetails({});
            } finally {
                window.clearInterval(poll);
                setDegradeBusy(false);
            }
        })();
    }, [
        report, selectedMeta, importedReportActive, eifApiUrl,
        tokenProbFocus, probViewFamily, tokenProbResult?.flip?.gainedTokenId,
        tokenProbResult?.flip?.lostTokenId,
    ]);

    useEffect(() => {
        setProbViewFamily('live');
        setDegradePairs([]);
        setDegradeTrainDetails({});
        setDegradeError(null);
        setDegradeProgress(null);
    }, [selectedMeta?.fileName]);

    const saliencyFocusRef = useRef({
        attrMode,
        selectedTokIdx,
        goldLocalIdx,
        manualSourceIdx,
        manualTargetAbsIdx,
        continueAdapterActive,
        promptLen,
    });
    saliencyFocusRef.current = {
        attrMode,
        selectedTokIdx,
        goldLocalIdx,
        manualSourceIdx,
        manualTargetAbsIdx,
        continueAdapterActive,
        promptLen,
    };

    const fetchLiveSaliency = useCallback(async (opts: {
        mode: 'predict' | 'gold';
        targetIndex: number;
        sourceIndex?: number | null;
        topK?: number;
    }): Promise<Record<string, unknown>> => {
        if (!selectedMeta) throw new Error('No report selected');
        const resp = await fetch(buildEifApiUrl(eifApiUrl, '/api/gold-saliency'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                reportFileName: selectedMeta.fileName,
                mode: opts.mode,
                targetIndex: opts.targetIndex,
                topK: opts.topK ?? 4,
                ...(opts.sourceIndex != null ? { sourceIndex: opts.sourceIndex } : {}),
            }),
        });
        const raw = await resp.text();
        let parsed: Record<string, unknown> = {};
        try {
            parsed = raw.trim() ? JSON.parse(raw) as Record<string, unknown> : {};
        } catch {
            throw new Error(`Live saliency non-JSON (HTTP ${resp.status}): ${raw.slice(0, 200)}`);
        }
        if (!resp.ok || parsed.status !== 'success') {
            throw new Error(
                typeof parsed.message === 'string'
                    ? parsed.message
                    : `Live saliency failed (HTTP ${resp.status})`,
            );
        }
        return parsed;
    }, [eifApiUrl, selectedMeta]);

    /** Refresh predict/gold top-k and 指定-pair edge saliency after continue / recover. */
    const refreshSaliencyPanels = useCallback(async (opts?: {
        /** After recover: drop live predict overlay and show report JSON again. */
        restorePredictReport?: boolean;
    }) => {
        if (importedReportActive || !selectedMeta) return;
        const focus = saliencyFocusRef.current;
        const restorePredict = Boolean(opts?.restorePredictReport);

        if (restorePredict) {
            setPredictLiveTop(null);
            setPredictLiveBusy(false);
        }

        // Predict top-k: live only while continued adapter is active (or refreshing into it).
        if (focus.attrMode === 'predict' && focus.selectedTokIdx != null && focus.selectedTokIdx > 0) {
            if (restorePredict) {
                // already cleared
            } else {
                setPredictLiveBusy(true);
                try {
                    const parsed = await fetchLiveSaliency({
                        mode: 'predict',
                        targetIndex: focus.selectedTokIdx,
                        topK: 4,
                    });
                    const top = Array.isArray(parsed.topCorrelations)
                        ? parsed.topCorrelations as TestCorrelation[]
                        : [];
                    setPredictLiveTop(top);
                } catch (error) {
                    const msg = error instanceof Error ? error.message : 'Predict live saliency failed';
                    setTtavLaunchError(msg);
                } finally {
                    setPredictLiveBusy(false);
                }
            }
        }

        // Gold top-k: always re-fetch live (env or continued adapter after cache evict).
        if (focus.attrMode === 'gold' && focus.goldLocalIdx != null) {
            const abs = focus.promptLen + focus.goldLocalIdx;
            setGoldBusy(true);
            try {
                const parsed = await fetchLiveSaliency({
                    mode: 'gold',
                    targetIndex: abs,
                    topK: 4,
                });
                const top = Array.isArray(parsed.topCorrelations)
                    ? parsed.topCorrelations as TestCorrelation[]
                    : [];
                setGoldTopCorrelations(top);
            } catch (error) {
                const msg = error instanceof Error ? error.message : 'Gold live saliency failed';
                setTtavLaunchError(msg);
            } finally {
                setGoldBusy(false);
            }
        }

        // 指定 pair edge saliency
        if (
            focus.attrMode === 'manual'
            && focus.manualSourceIdx != null
            && focus.manualTargetAbsIdx != null
        ) {
            setManualEdgeSaliencyBusy(true);
            setManualEdgeSaliencyErr(null);
            try {
                const parsed = await fetchLiveSaliency({
                    mode: 'gold',
                    targetIndex: focus.manualTargetAbsIdx,
                    sourceIndex: focus.manualSourceIdx,
                    topK: 4,
                });
                const score = typeof parsed.edgeSaliency === 'number' ? parsed.edgeSaliency : null;
                setManualEdgeSaliency(score);
            } catch (error) {
                const msg = error instanceof Error ? error.message : 'Pair edge saliency failed';
                setManualEdgeSaliencyErr(msg);
                setManualEdgeSaliency(null);
            } finally {
                setManualEdgeSaliencyBusy(false);
            }
        }
    }, [importedReportActive, selectedMeta, fetchLiveSaliency]);

    // While continued adapter is active, keep predict panel on live scores when target changes.
    useEffect(() => {
        if (importedReportActive) return;
        if (attrMode !== 'predict' || selectedTokIdx == null || selectedTokIdx <= 0) return;
        if (!continueAdapterActive) {
            setPredictLiveTop(null);
            return;
        }
        let cancelled = false;
        setPredictLiveBusy(true);
        void (async () => {
            try {
                const parsed = await fetchLiveSaliency({
                    mode: 'predict',
                    targetIndex: selectedTokIdx,
                    topK: 4,
                });
                if (cancelled) return;
                const top = Array.isArray(parsed.topCorrelations)
                    ? parsed.topCorrelations as TestCorrelation[]
                    : [];
                setPredictLiveTop(top);
            } catch (error) {
                if (cancelled) return;
                const msg = error instanceof Error ? error.message : 'Predict live saliency failed';
                setTtavLaunchError(msg);
            } finally {
                if (!cancelled) setPredictLiveBusy(false);
            }
        })();
        return () => { cancelled = true; };
    }, [
        attrMode, selectedTokIdx, continueAdapterActive,
        importedReportActive, fetchLiveSaliency,
    ]);

    // 指定 pair: fetch edge saliency whenever source+target are set.
    useEffect(() => {
        if (importedReportActive) return;
        if (attrMode !== 'manual') return;
        if (manualSourceIdx == null || manualTargetAbsIdx == null) {
            setManualEdgeSaliency(null);
            setManualEdgeSaliencyErr(null);
            return;
        }
        let cancelled = false;
        setManualEdgeSaliencyBusy(true);
        setManualEdgeSaliencyErr(null);
        void (async () => {
            try {
                const parsed = await fetchLiveSaliency({
                    mode: 'gold',
                    targetIndex: manualTargetAbsIdx,
                    sourceIndex: manualSourceIdx,
                    topK: 4,
                });
                if (cancelled) return;
                const score = typeof parsed.edgeSaliency === 'number' ? parsed.edgeSaliency : null;
                setManualEdgeSaliency(score);
            } catch (error) {
                if (cancelled) return;
                const msg = error instanceof Error ? error.message : 'Pair edge saliency failed';
                setManualEdgeSaliencyErr(msg);
                setManualEdgeSaliency(null);
            } finally {
                if (!cancelled) setManualEdgeSaliencyBusy(false);
            }
        })();
        return () => { cancelled = true; };
    }, [
        attrMode, manualSourceIdx, manualTargetAbsIdx,
        importedReportActive, fetchLiveSaliency, continueAdapterActive,
    ]);

    const activeQueryEdge = useMemo(() => {
        if (attrMode === 'manual') {
            if (manualSourceIdx === null || manualTargetAbsIdx === null) return null;
            if (
                manualSourceIdx < 0
                || manualSourceIdx >= promptLen
                || manualTargetAbsIdx < promptLen
                || manualTargetAbsIdx >= correctTokens.length
            ) {
                return null;
            }
            return {
                sourceIndex: manualSourceIdx,
                targetIndex: manualTargetAbsIdx,
                sourceToken: decodeToken(correctTokens[manualSourceIdx] ?? modelTokens[manualSourceIdx] ?? ''),
                targetToken: decodeToken(correctTokens[manualTargetAbsIdx] ?? ''),
                tokens: correctTokens,
                promptLen,
            };
        }
        if (attrMode === 'gold') {
            if (goldLocalIdx === null || goldSelectedCorrIdx === null) return null;
            const absTgt = promptLen + goldLocalIdx;
            const corr = goldTopCorrelations.find(c => c.source_token_index === goldSelectedCorrIdx);
            const srcTok = corr?.source_token
                ?? correctTokens[goldSelectedCorrIdx]
                ?? '';
            const dstTok = corr?.target_token
                ?? correctTokens[absTgt]
                ?? goldResponseTokens[goldLocalIdx]
                ?? '';
            return {
                sourceIndex: goldSelectedCorrIdx,
                targetIndex: absTgt,
                sourceToken: decodeToken(srcTok),
                targetToken: decodeToken(dstTok),
                tokens: correctTokens,
                promptLen,
            };
        }
        if (!selectedResult || selectedTestCorrIdx === null) return null;
        const corr = selectedResult.top_correlations.find(
            c => c.source_token_index === selectedTestCorrIdx,
        );
        return {
            sourceIndex: selectedTestCorrIdx,
            targetIndex: selectedResult.target_token_index,
            sourceToken: decodeToken(corr?.source_token ?? modelTokens[selectedTestCorrIdx] ?? ''),
            targetToken: decodeToken(corr?.target_token ?? selectedResult.target_token),
            tokens: modelTokens,
            promptLen,
        };
    }, [
        attrMode, manualSourceIdx, manualTargetAbsIdx,
        goldLocalIdx, goldSelectedCorrIdx, goldTopCorrelations,
        promptLen, correctTokens, goldResponseTokens, modelTokens,
        selectedResult, selectedTestCorrIdx,
    ]);

    const fetchStructuralPairs = useCallback((edge: {
        sourceIndex: number;
        targetIndex: number;
        sourceToken: string;
        targetToken: string;
        tokens: string[];
        promptLen: number;
    }) => {
        if (importedReportActive) return;
        setStructuralBusy(true);
        setStructuralError(null);
        setStructuralMeta(null);
        void (async () => {
            try {
                const resp = await fetch(buildEifApiUrl(eifApiUrl, '/api/structural-pair-retrieve'), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        sourceToken: edge.sourceToken,
                        targetToken: edge.targetToken,
                        sourceIndex: edge.sourceIndex,
                        targetIndex: edge.targetIndex,
                        tokens: edge.tokens,
                        promptLen: edge.promptLen,
                        topK: 40,
                        structWeight: 0.8,
                        textWeight: 0.2,
                    }),
                });
                const raw = await resp.text();
                let parsed: Record<string, unknown> = {};
                if (raw.trim()) {
                    try {
                        parsed = JSON.parse(raw) as Record<string, unknown>;
                    } catch {
                        throw new Error(`Structural API non-JSON (HTTP ${resp.status}): ${raw.slice(0, 200)}`);
                    }
                }
                if (!resp.ok || parsed.status !== 'success') {
                    throw new Error(
                        typeof parsed.message === 'string'
                            ? parsed.message
                            : `Structural retrieve failed (HTTP ${resp.status})`,
                    );
                }
                const pairs = Array.isArray(parsed.pairs)
                    ? parsed.pairs as CorrelationPair[]
                    : [];
                setStructuralPairs(pairs);
                const details = (
                    typeof parsed.trainSampleDetails === 'object'
                    && parsed.trainSampleDetails !== null
                ) ? parsed.trainSampleDetails as Record<string, TrainSampleDetail> : {};
                setStructuralTrainDetails(details);
                const q = (parsed.query || {}) as Record<string, unknown>;
                const ast = (q.ast || {}) as Record<string, unknown>;
                setStructuralMeta(
                    `按pair · ${pairs.length} top`
                    + (typeof parsed.nScoredPairs === 'number' ? ` / scored=${parsed.nScoredPairs}` : '')
                    + (ast.lca ? ` · query lca=${String(ast.lca)}/${String(ast.relation || '')}` : '')
                    + ` · w=0.8/0.2`,
                );
            } catch (error) {
                setStructuralPairs([]);
                setStructuralTrainDetails({});
                setStructuralError(error instanceof Error ? error.message : 'Structural retrieve failed');
            } finally {
                setStructuralBusy(false);
            }
        })();
    }, [importedReportActive, eifApiUrl]);

    useEffect(() => {
        if (!structuralAttributionEnabled) {
            setStructuralPairs([]);
            setStructuralTrainDetails({});
            setStructuralError(null);
            setStructuralMeta(null);
            return;
        }
        if (!activeQueryEdge) {
            setStructuralPairs([]);
            setStructuralTrainDetails({});
            setStructuralMeta(null);
            return;
        }
        fetchStructuralPairs(activeQueryEdge);
    }, [structuralAttributionEnabled, activeQueryEdge, fetchStructuralPairs]);

    const handlePairIntervene = useCallback((pair: CorrelationPair, direction: 'unlearn' | 'learn') => {
        if (!report || !selectedMeta || importedReportActive) return;

        setInterveningPairId(pair.id);
        setInterveningDirection(direction);
        setTtavLaunchError(null);
        const verb = direction === 'learn' ? 'Learning' : 'Unlearning';
        setTtavLaunchStatus(`${verb} pair ${pair.id}…`);

        const trainKey = String(pair.train_sample_id);
        const trainDetail =
            degradeTrainDetails[trainKey]
            ?? (attrMode === 'manual'
                ? (manualTrainDetails[trainKey] ?? report.train_sample_details[trainKey])
                : attrMode === 'gold'
                    ? (goldTrainDetails[trainKey] ?? report.train_sample_details[trainKey])
                    : report.train_sample_details[trainKey]);

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
                        modelPath: null,
                        baseModelPath: null,
                        unlearnLr: pairInterveneLr,
                        recomputeSaliency: true,
                        direction,
                        persist: true,
                        completionMode: attrMode === 'predict' ? 'predict' : 'gold',
                        trainSampleDetail: trainDetail ?? null,
                    }),
                });
                const rawText = await resp.text();
                let parsed: Record<string, unknown> = {};
                if (rawText.trim()) {
                    try {
                        parsed = JSON.parse(rawText) as Record<string, unknown>;
                    } catch {
                        throw new Error(
                            `${verb} API returned non-JSON (HTTP ${resp.status}): ${rawText.slice(0, 240)}`,
                        );
                    }
                }
                if (!resp.ok || parsed.status !== 'success') {
                    const message = typeof parsed.message === 'string'
                        ? parsed.message
                        : `${verb} API failed (HTTP ${resp.status})`;
                    throw new Error(message);
                }

                const result = parsed as UnlearnPairResult;
                setUnlearnResultsByPairId(current => ({ ...current, [pair.id]: result }));
                if (result.restored === false) {
                    setActiveInterventionPairId(pair.id);
                    setActiveInterventionDirection(direction);
                    const steps =
                        result.intervention?.steps
                        ?? result.update?.steps
                        ?? 1;
                    setInterventionSteps(Math.max(1, Number(steps) || 1));
                } else {
                    setActiveInterventionPairId(null);
                    setActiveInterventionDirection(null);
                    setInterventionSteps(0);
                }
                const dCe = typeof result.delta?.ce === 'number'
                    ? result.delta.ce.toFixed(4)
                    : '?';
                const dSal = typeof result.delta?.saliency === 'number'
                    ? result.delta.saliency.toFixed(4)
                    : '?';
                const stepN = result.update?.steps ?? result.intervention?.steps ?? 1;
                setTtavLaunchStatus(
                    `${direction === 'learn' ? 'Learn' : 'Unlearn'} ${pair.id} 完成`
                    + ` · step ${stepN} · ΔCE=${dCe} · Δsal=${dSal} · ${result.verdict ?? ''}`,
                );
                setProbViewFamily('live');
                refreshTokenProbs('live');
            } catch (error) {
                const msg = error instanceof Error ? error.message : `${verb} failed`;
                setUnlearnResultsByPairId(current => ({
                    ...current,
                    [pair.id]: { status: 'error', error: msg, verdict: 'inconclusive', direction },
                }));
                setTtavLaunchError(msg);
                setTtavLaunchStatus(null);
            } finally {
                setInterveningPairId(null);
                setInterveningDirection(null);
            }
        })();
    }, [
        report, selectedMeta, importedReportActive, eifApiUrl, selectedSampleId,
        attrMode, goldTrainDetails, manualTrainDetails, degradeTrainDetails,
        refreshTokenProbs, pairInterveneLr,
    ]);

    const handleRecoverIntervention = useCallback(() => {
        setRecoverBusy(true);
        setTtavLaunchError(null);
        setTtavLaunchStatus('Recovering model weights…');
        void (async () => {
            try {
                const resp = await fetch(buildEifApiUrl(eifApiUrl, '/api/pair-intervene-recover'), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: '{}',
                });
                const rawText = await resp.text();
                let parsed: Record<string, unknown> = {};
                if (rawText.trim()) {
                    try {
                        parsed = JSON.parse(rawText) as Record<string, unknown>;
                    } catch {
                        throw new Error(`Recover API returned non-JSON (HTTP ${resp.status})`);
                    }
                }
                if (!resp.ok || parsed.status !== 'success') {
                    throw new Error(
                        typeof parsed.message === 'string'
                            ? parsed.message
                            : `Recover failed (HTTP ${resp.status})`,
                    );
                }
                setActiveInterventionPairId(null);
                setActiveInterventionDirection(null);
                setInterventionSteps(0);
                setTtavLaunchStatus(
                    parsed.recovered ? 'Recovered to original weights.' : 'No active intervention to recover.',
                );
                refreshTokenProbs();
            } catch (error) {
                const msg = error instanceof Error ? error.message : 'Recover failed';
                setTtavLaunchError(msg);
                setTtavLaunchStatus(null);
            } finally {
                setRecoverBusy(false);
            }
        })();
    }, [eifApiUrl, refreshTokenProbs]);

    const handleContinueTrainEval = useCallback(() => {
        if (importedReportActive) return;
        const maxSteps = Math.max(
            1,
            Math.floor(Number(continueStepsInput)) || continueStepsDefault || 20,
        );
        const learningRate = Number(continueLrInput);
        if (!Number.isFinite(learningRate) || learningRate <= 0) {
            setTtavLaunchError('续训 lr 必须是 > 0 的数字（如 2e-5）');
            return;
        }
        // Train only on ANNOTATION_CONTINUE_TRAIN_DATA (small annotated subset).
        // Do NOT pass related-train ids / full EIF_TRAIN_DATA — that is not continue-train.
        setContinueBusy(true);
        setContinueResultSummary(null);
        setTtavLaunchError(null);
        setTtavLaunchStatus('续训：小集 → 评测 line_hit（eval_before 优先读 EIF_CONTINUE_EVAL_BEFORE_CACHE）…');
        void (async () => {
            try {
                const resp = await fetch(buildEifApiUrl(eifApiUrl, '/api/continue-train-eval'), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        maxSteps,
                        learningRate,
                        lossMode: 'ce_saliency',
                        evalBefore: true,
                    }),
                });
                const raw = await resp.text();
                let parsed: Record<string, unknown> = {};
                if (raw.trim()) {
                    try {
                        parsed = JSON.parse(raw) as Record<string, unknown>;
                    } catch {
                        throw new Error(`Continue-train API non-JSON (HTTP ${resp.status}): ${raw.slice(0, 200)}`);
                    }
                }
                if (!resp.ok || parsed.status !== 'success') {
                    throw new Error(
                        typeof parsed.message === 'string'
                            ? parsed.message
                            : `Continue-train failed (HTTP ${resp.status})`,
                    );
                }
                const jobId = String(parsed.jobId || '');
                if (!jobId) throw new Error('Continue-train response missing jobId');
                setContinueJobId(jobId);
                setTtavLaunchStatus(`Continue-train job ${jobId} running…`);

                for (;;) {
                    await new Promise(r => setTimeout(r, 2000));
                    const stResp = await fetch(
                        `${buildEifApiUrl(eifApiUrl, '/api/continue-train-eval-status')}?jobId=${encodeURIComponent(jobId)}`,
                    );
                    const stRaw = await stResp.text();
                    let st: Record<string, unknown> = {};
                    if (stRaw.trim()) {
                        try {
                            st = JSON.parse(stRaw) as Record<string, unknown>;
                        } catch {
                            continue;
                        }
                    }
                    const stage = String(st.stage || '');
                    const message = typeof st.message === 'string' ? st.message : stage;
                    setTtavLaunchStatus(`[${jobId}] ${message}`);
                    if (stage === 'completed') {
                        const result = (st.result || {}) as Record<string, unknown>;
                        const before = (result.before || {}) as Record<string, number | null>;
                        const after = (result.after || {}) as Record<string, number | null>;
                        const delta = (result.delta || {}) as Record<string, number | null>;
                        const per = (result.perSampleDeltas || {}) as Record<string, unknown>;
                        const preBlock = (per.line_hit_pre || {}) as Record<string, unknown>;
                        const recBlock = (per.line_hit_rec || {}) as Record<string, unknown>;
                        const fmt = (v: number | null | undefined) =>
                            typeof v === 'number' && Number.isFinite(v) ? v.toFixed(2) : '—';
                        const fmtIdx = (v: unknown, n: unknown) => {
                            const arr = Array.isArray(v) ? v.map(String) : [];
                            const count = typeof n === 'number' ? n : arr.length;
                            const head = arr.slice(0, 12).join(',');
                            return `${count}[${head}${arr.length > 12 ? '…' : ''}]`;
                        };
                        const summary =
                            `line_hit_pre ${fmt(before.line_hit_pre)} → ${fmt(after.line_hit_pre)}`
                            + ` (Δ ${fmt(delta.line_hit_pre)})`
                            + ` · line_hit_rec ${fmt(before.line_hit_rec)} → ${fmt(after.line_hit_rec)}`
                            + ` (Δ ${fmt(delta.line_hit_rec)})`
                            + ` · test↑pre ${fmtIdx(preBlock.increased, preBlock.nIncreased)}`
                            + ` ↓pre ${fmtIdx(preBlock.decreased, preBlock.nDecreased)}`
                            + ` · test↑rec ${fmtIdx(recBlock.increased, recBlock.nIncreased)}`
                            + ` ↓rec ${fmtIdx(recBlock.decreased, recBlock.nDecreased)}`
                            + ` · out ${String(result.outputDir || '')}`;
                        setContinueResultSummary(summary);
                        setContinueAdapterActive(true);
                        // Metrics live only under Continue train; don't repeat in status strip.
                        setTtavLaunchStatus(null);
                        // Live probes now use continued adapter — refresh probs + saliency panels.
                        refreshTokenProbs();
                        void refreshSaliencyPanels();
                        break;
                    }
                    if (stage === 'error' || st.error === true) {
                        throw new Error(message || 'Continue-train failed');
                    }
                }
            } catch (error) {
                const msg = error instanceof Error ? error.message : 'Continue-train failed';
                setTtavLaunchError(msg);
                setTtavLaunchStatus(null);
            } finally {
                setContinueBusy(false);
            }
        })();
    }, [importedReportActive, continueStepsInput, continueLrInput, continueStepsDefault, eifApiUrl, refreshTokenProbs, refreshSaliencyPanels]);

    const handleContinueAdapterRecover = useCallback(() => {
        if (importedReportActive) return;
        setContinueRecoverBusy(true);
        setTtavLaunchError(null);
        setTtavLaunchStatus('Recovering live adapter to eif_api.env…');
        void (async () => {
            try {
                const resp = await fetch(buildEifApiUrl(eifApiUrl, '/api/continue-adapter-recover'), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: '{}',
                });
                const raw = await resp.text();
                let parsed: Record<string, unknown> = {};
                if (raw.trim()) {
                    try {
                        parsed = JSON.parse(raw) as Record<string, unknown>;
                    } catch {
                        throw new Error(`Recover API non-JSON (HTTP ${resp.status}): ${raw.slice(0, 200)}`);
                    }
                }
                if (!resp.ok || parsed.status !== 'success') {
                    throw new Error(
                        typeof parsed.message === 'string'
                            ? parsed.message
                            : `Continue-adapter recover failed (HTTP ${resp.status})`,
                    );
                }
                setContinueAdapterActive(false);
                setActiveInterventionPairId(null);
                setActiveInterventionDirection(null);
                setInterventionSteps(0);
                setTtavLaunchStatus(
                    typeof parsed.message === 'string'
                        ? parsed.message
                        : 'Restored env adapter.',
                );
                // Probs + saliency: reload with env adapter; predict panel returns to report JSON.
                refreshTokenProbs();
                void refreshSaliencyPanels({ restorePredictReport: true });
            } catch (error) {
                const msg = error instanceof Error ? error.message : 'Continue-adapter recover failed';
                setTtavLaunchError(msg);
                setTtavLaunchStatus(null);
            } finally {
                setContinueRecoverBusy(false);
            }
        })();
    }, [importedReportActive, eifApiUrl, refreshTokenProbs, refreshSaliencyPanels]);

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
                                highlightSourceIndices={
                                    attrMode === 'manual'
                                        ? (manualSourceIdx != null ? new Set([manualSourceIdx]) : undefined)
                                        : attrMode === 'predict'
                                            ? sourceHighlightIndices
                                            : attrMode === 'gold'
                                                ? goldModelHighlightSourceIndices
                                                : undefined
                                }
                                selectedTargetIndex={attrMode === 'predict' ? (selectedTokIdx ?? undefined) : undefined}
                                analyzedIndices={analyzedIndices}
                                modelClickScope={attrMode === 'manual' ? 'prompt' : 'analyzed'}
                                onTokenClick={idx => {
                                    if (attrMode === 'manual') {
                                        handleManualSourceClick(idx);
                                        return;
                                    }
                                    setAttrMode('predict');
                                    clearGoldLive();
                                    clearManualPair();
                                    setSelectedTokIdx(prev => prev === idx ? null : idx);
                                }}
                                goldSelectedLocalIndex={
                                    attrMode === 'manual'
                                        ? (manualTargetAbsIdx != null && manualTargetAbsIdx >= promptLen
                                            ? manualTargetAbsIdx - promptLen
                                            : null)
                                        : attrMode === 'gold' ? goldLocalIdx : null
                                }
                                goldHighlightSourceIndices={
                                    attrMode === 'manual' ? undefined : goldHighlightSourceIndices
                                }
                                onGoldTokenClick={handleGoldTokenClick}
                                goldHint={
                                    attrMode === 'manual'
                                        ? '点击 → 选指定 pair 的 target（仅 Gold complete）'
                                        : undefined
                                }
                                linkedTokenIndex={linkedTestTokenIndex}
                                onTokenHover={inlineBundle ? handleTestTokenHover : undefined}
                                saliencySelected={modelSaliencySelected}
                                saliencySelectEnabled={
                                    attrMode !== 'manual'
                                    && selectedTokIdx !== null
                                    && selectedTestCorrIdx !== null
                                }
                                onToggleSaliencySelect={() => setModelSaliencySelected(v => !v)}
                                headerExtra={(
                                    <button
                                        type="button"
                                        aria-pressed={attrMode === 'manual'}
                                        disabled={importedReportActive || goldResponseTokens.length === 0}
                                        title={
                                            importedReportActive
                                                ? '上传报告不支持指定 pair'
                                                : goldResponseTokens.length === 0
                                                    ? '需要 Gold complete tokens'
                                                    : (attrMode === 'manual'
                                                        ? '退出指定 pair 模式'
                                                        : '手动选上下文 source + Gold target，结构归因检索后 Learn 提升该 token 概率')
                                        }
                                        onClick={() => {
                                            if (attrMode === 'manual') exitManualPairMode();
                                            else enterManualPairMode();
                                        }}
                                        style={{
                                            border: attrMode === 'manual' ? '1px solid #7c3aed' : '1px solid #cbd5e1',
                                            background: attrMode === 'manual' ? '#f5f3ff' : '#ffffff',
                                            color: attrMode === 'manual' ? '#6d28d9' : '#64748b',
                                            borderRadius: 999,
                                            padding: '2px 10px',
                                            fontSize: 11,
                                            fontWeight: 700,
                                            cursor: importedReportActive || goldResponseTokens.length === 0
                                                ? 'not-allowed'
                                                : 'pointer',
                                            opacity: importedReportActive || goldResponseTokens.length === 0 ? 0.55 : 1,
                                        }}
                                    >
                                        {attrMode === 'manual' ? '指定 pair · 开' : '指定 pair'}
                                    </button>
                                )}
                            />

                            {attrMode === 'manual' && (
                                <div
                                    className={styles.correlationList}
                                    style={{
                                        borderColor: '#c4b5fd',
                                        background: '#faf5ff',
                                    }}
                                >
                                    <div className={styles.correlationListTitle}>
                                        指定 pair · 结构归因 query
                                    </div>
                                    <div style={{
                                        padding: '8px 12px',
                                        fontSize: 12,
                                        color: '#4c1d95',
                                        display: 'flex',
                                        flexWrap: 'wrap',
                                        gap: 8,
                                        alignItems: 'center',
                                    }}>
                                        <span>
                                            source:{' '}
                                            <strong>
                                                {manualSourceIdx == null
                                                    ? '（点 Model 上下文）'
                                                    : `"${decodeToken(modelTokens[manualSourceIdx] ?? correctTokens[manualSourceIdx] ?? '').trim() || '·'}" @ ${manualSourceIdx}`}
                                            </strong>
                                        </span>
                                        <span style={{ color: '#a78bfa' }}>→</span>
                                        <span>
                                            target:{' '}
                                            <strong>
                                                {manualTargetAbsIdx == null
                                                    ? '（点 Gold）'
                                                    : `"${decodeToken(correctTokens[manualTargetAbsIdx] ?? '').trim() || '·'}" @ ${manualTargetAbsIdx}`}
                                            </strong>
                                        </span>
                                        <button
                                            type="button"
                                            onClick={clearManualPair}
                                            style={{
                                                marginLeft: 'auto',
                                                border: '1px solid #c4b5fd',
                                                background: '#fff',
                                                borderRadius: 6,
                                                padding: '2px 8px',
                                                fontSize: 11,
                                                cursor: 'pointer',
                                                color: '#6d28d9',
                                            }}
                                        >
                                            清除选中
                                        </button>
                                    </div>
                                    <div style={{ padding: '0 12px 8px', fontSize: 11, color: '#6b7280' }}>
                                        {manualSourceIdx != null && manualTargetAbsIdx != null
                                            ? (structuralAttributionEnabled
                                                ? '选齐后：上栏梯度 Stage3 + 下栏结构检索会并行跑；Learn 用 gold completion。'
                                                : '已触发梯度 Stage3。勾选「结构归因」可同时看 AST pair。')
                                            : '先选 source（上下文），再选 target（Gold complete）。'}
                                    </div>
                                    {manualSourceIdx != null && manualTargetAbsIdx != null && (
                                        <div style={{
                                            margin: '0 12px 10px',
                                            padding: '8px 10px',
                                            borderRadius: 8,
                                            border: '1px solid #ddd6fe',
                                            background: '#ffffff',
                                            fontSize: 12,
                                            color: '#4c1d95',
                                            display: 'flex',
                                            flexWrap: 'wrap',
                                            gap: 8,
                                            alignItems: 'center',
                                        }}>
                                            <span style={{ fontWeight: 700, color: '#5b21b6', fontSize: 13 }}>
                                                edge saliency
                                            </span>
                                            {manualEdgeSaliencyBusy ? (
                                                <span style={{ color: '#7c3aed', fontWeight: 700 }}>计算中…</span>
                                            ) : manualEdgeSaliencyErr ? (
                                                <span style={{ color: '#b91c1c' }}>{manualEdgeSaliencyErr}</span>
                                            ) : manualEdgeSaliency != null ? (
                                                <span
                                                    style={{
                                                        color: '#6d28d9',
                                                        fontWeight: 800,
                                                        fontSize: 22,
                                                        lineHeight: 1.1,
                                                        fontVariantNumeric: 'tabular-nums',
                                                        letterSpacing: '-0.02em',
                                                    }}
                                                >
                                                    {manualEdgeSaliency.toFixed(4)}
                                                </span>
                                            ) : (
                                                <span style={{ color: '#9ca3af', fontSize: 18, fontWeight: 700 }}>—</span>
                                            )}
                                            <span style={{ color: '#9ca3af', fontSize: 11 }}>
                                                {continueAdapterActive
                                                    ? '· live（续训 adapter）'
                                                    : '· live（env adapter）'}
                                                {' · recover 后会重算回到原 adapter'}
                                            </span>
                                        </div>
                                    )}
                                </div>
                            )}

                            {attrMode === 'gold' && goldLocalIdx !== null && (
                                <div className={styles.correlationList}>
                                    <div className={styles.correlationListTitle}>
                                        Gold live · Top Correlations for "
                                        {decodeToken(goldResponseTokens[goldLocalIdx] ?? '').trim()}"
                                        {' '}@ idx {promptLen + goldLocalIdx}
                                        {goldBusy ? ' …' : ''}
                                        {continueAdapterActive ? ' · 续训 adapter' : ''}
                                    </div>
                                    <div className={styles.correlationListHint}>
                                        Teacher-force gold path. Click one edge to run bank Top-10 + Stage3.
                                        {' '}续训结束后会自动重算；Recover 后回到 env adapter。
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
                                        {predictLiveBusy ? ' …' : ''}
                                        {predictLiveTop
                                            ? (continueAdapterActive ? ' · live(续训)' : ' · live')
                                            : ' · report'}
                                    </div>
                                    <div className={styles.correlationListHint}>
                                        Click one source→target edge to load its Top-10 training matches on the right.
                                        {' '}续训后切到 live 分数；Recover 后回到报告原版。
                                    </div>
                                    <div className={styles.correlationListItems}>
                                        {(predictLiveTop ?? selectedResult.top_correlations).slice(0, 4).map(c => (
                                            <button
                                                key={c.source_token_index}
                                                type="button"
                                                title={`Select ${decodeToken(c.source_token).trim() || '·'} → ${decodeToken(c.target_token || selectedResult.target_token).trim()} for train retrieval`}
                                                className={`${styles.corrBtn} ${c.source_token_index === selectedTestCorrIdx ? styles.corrBtnActive : ''}`}
                                                onClick={() => {
                                                    setDegradePairs([]);
                                                    setDegradeTrainDetails({});
                                                    setSelectedTestCorrIdx(
                                                        prev => prev === c.source_token_index ? null : c.source_token_index,
                                                    );
                                                }}
                                            >
                                                <div className={styles.corrBtnLeft}>
                                                    <span className={styles.corrLabel}>source → target</span>
                                                    <span className={styles.corrSourceTok}>
                                                        {(decodeToken(c.source_token).trim() || '·')}
                                                        <span className={styles.corrArrow}>→</span>
                                                        {decodeToken(c.target_token || selectedResult.target_token).trim() || '·'}
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

                            <NextTokenProbPanel
                                result={tokenProbResult}
                                busy={tokenProbBusy}
                                error={tokenProbError}
                                interventionActive={Boolean(activeInterventionPairId)}
                                interventionDirection={activeInterventionDirection}
                                interventionSteps={interventionSteps}
                                viewFamily={probViewFamily}
                                onViewFamilyChange={setProbViewFamily}
                                degradePairs={degradePairs}
                                degradeBusy={degradeBusy}
                                degradeError={degradeError}
                                degradeProgress={degradeProgress}
                                onRetrieveDegrade={fetchDegradeRetrieve}
                            />

                        </div>

                        {/* ── Right Column: Training pairs ── */}
                        <div className={styles.bottomRight}>
                            <div className={styles.bottomPanel}>
                                {!importedReportActive && (
                                    <div style={{
                                        padding: '6px 12px',
                                        fontSize: 11,
                                        color: '#475569',
                                        borderBottom: '1px solid #e5e7eb',
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: 8,
                                        flexWrap: 'wrap',
                                    }}>
                                        <label
                                            style={{
                                                display: 'flex',
                                                alignItems: 'center',
                                                gap: 6,
                                                fontWeight: 700,
                                                color: structuralAttributionEnabled ? '#6d28d9' : '#475569',
                                                cursor: 'pointer',
                                            }}
                                            title="勾选后右侧分上下栏：上=梯度归因；下=对 train 全量枚举 context×completion，用 tree-sitter AST pair 相似 + 文本相似（不依赖 attention_edges 标签）"
                                        >
                                            <input
                                                type="checkbox"
                                                checked={structuralAttributionEnabled}
                                                onChange={(e) => setStructuralAttributionEnabled(e.target.checked)}
                                            />
                                            结构归因
                                        </label>
                                        {structuralAttributionEnabled && (
                                            <span style={{ color: '#7c3aed', fontSize: 10 }}>
                                                AST全量枚举 + 磁盘缓存 · 结构:文本=8:2
                                            </span>
                                        )}
                                    </div>
                                )}
                                {!importedReportActive && (
                                    <div style={{
                                        padding: '6px 12px',
                                        fontSize: 11,
                                        color: '#475569',
                                        borderBottom: '1px solid #e5e7eb',
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: 8,
                                        flexWrap: 'wrap',
                                    }}>
                                        <span style={{ fontWeight: 700 }}>Learn/Unlearn η</span>
                                        <input
                                            type="number"
                                            min={0}
                                            step="any"
                                            value={pairInterveneLrInput}
                                            disabled={Boolean(interveningPairId) || recoverBusy}
                                            onChange={(e) => setPairInterveneLrInput(e.target.value)}
                                            title="归一化 LoRA 步长 ||Δθ||₂ = η。改完后点「确认」生效。"
                                            style={{
                                                width: 72,
                                                padding: '2px 6px',
                                                borderRadius: 6,
                                                border: '1px solid #cbd5e1',
                                                fontSize: 11,
                                                fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
                                            }}
                                        />
                                        <button
                                            type="button"
                                            disabled={Boolean(interveningPairId) || recoverBusy}
                                            onClick={() => {
                                                const n = Number(pairInterveneLrInput);
                                                if (!Number.isFinite(n) || n < 0) {
                                                    setPairInterveneLrInput(String(pairInterveneLr));
                                                    setTtavLaunchError('η 必须是 ≥ 0 的数字');
                                                    return;
                                                }
                                                setPairInterveneLr(n);
                                                setPairInterveneLrInput(String(n));
                                                setTtavLaunchError(null);
                                                setTtavLaunchStatus(`Learn/Unlearn η 已设为 ${n}`);
                                            }}
                                            style={{
                                                padding: '2px 10px',
                                                borderRadius: 6,
                                                border: '1px solid #94a3b8',
                                                background: '#f8fafc',
                                                color: '#334155',
                                                fontSize: 11,
                                                fontWeight: 700,
                                                cursor: interveningPairId || recoverBusy ? 'not-allowed' : 'pointer',
                                            }}
                                        >
                                            确认
                                        </button>
                                        <span style={{ color: '#94a3b8' }}>
                                            当前 {pairInterveneLr} · default {DEFAULT_PAIR_INTERVENE_LR}
                                        </span>
                                    </div>
                                )}
                                {!importedReportActive && (
                                    <div style={{
                                        padding: '6px 12px',
                                        fontSize: 11,
                                        color: '#475569',
                                        borderBottom: '1px solid #e5e7eb',
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: 8,
                                        flexWrap: 'wrap',
                                    }}>
                                        <span style={{ fontWeight: 700 }}>Continue train</span>
                                        <label style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                                            steps
                                            <input
                                                type="number"
                                                min={1}
                                                value={continueStepsInput}
                                                disabled={continueBusy}
                                                onChange={(e) => setContinueStepsInput(e.target.value)}
                                                title={`Default from EIF_CONTINUE_MAX_STEPS (eif_api.env): ${continueStepsDefault}`}
                                                style={{
                                                    width: 56,
                                                    padding: '2px 6px',
                                                    borderRadius: 6,
                                                    border: '1px solid #cbd5e1',
                                                    fontSize: 11,
                                                    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
                                                }}
                                            />
                                        </label>
                                        <label style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                                            lr
                                            <input
                                                type="text"
                                                value={continueLrInput}
                                                disabled={continueBusy}
                                                onChange={(e) => setContinueLrInput(e.target.value)}
                                                title={`AdamW lr on LoRA. Default ${continueLrDefault}. Not the Learn/Unlearn η.`}
                                                style={{
                                                    width: 64,
                                                    padding: '2px 6px',
                                                    borderRadius: 6,
                                                    border: '1px solid #cbd5e1',
                                                    fontSize: 11,
                                                    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
                                                }}
                                            />
                                        </label>
                                        <span style={{ color: '#94a3b8', fontSize: 11 }}>
                                            当前 steps {continueStepsInput} · default {continueStepsDefault}
                                        </span>
                                        <button
                                            type="button"
                                            disabled={continueBusy || Boolean(interveningPairId) || recoverBusy || continueRecoverBusy}
                                            onClick={handleContinueTrainEval}
                                            title="只在 ANNOTATION_CONTINUE_TRAIN_DATA（新标注小集）上从当前 saliency adapter 续训 CE+saliency；评测只看 line_hit_pre / line_hit_rec（EIF_TEST_DATA）"
                                            style={{
                                                padding: '2px 10px',
                                                borderRadius: 6,
                                                border: '1px solid #86efac',
                                                background: continueBusy ? '#dcfce7' : '#f0fdf4',
                                                color: '#15803d',
                                                fontSize: 11,
                                                fontWeight: 700,
                                                cursor: continueBusy ? 'wait' : 'pointer',
                                            }}
                                        >
                                            {continueBusy ? 'Training…' : '续训(小集) + line_hit'}
                                        </button>
                                        <button
                                            type="button"
                                            disabled={
                                                continueBusy
                                                || continueRecoverBusy
                                                || Boolean(interveningPairId)
                                                || recoverBusy
                                                || !continueAdapterActive
                                            }
                                            onClick={handleContinueAdapterRecover}
                                            title="清除续训后的 live adapter 覆盖，恢复为 eif_api.env 中的 EIF_ADAPTER_PATH_*，并刷新 token 概率"
                                            style={{
                                                padding: '2px 10px',
                                                borderRadius: 6,
                                                border: '1px solid #a5f3fc',
                                                background: continueRecoverBusy ? '#cffafe' : '#ecfeff',
                                                color: '#0e7490',
                                                fontSize: 11,
                                                fontWeight: 700,
                                                cursor:
                                                    continueBusy || continueRecoverBusy || !continueAdapterActive
                                                        ? 'not-allowed'
                                                        : 'pointer',
                                                opacity: continueAdapterActive ? 1 : 0.45,
                                            }}
                                        >
                                            {continueRecoverBusy ? 'Recovering…' : 'Recover 原 adapter'}
                                        </button>
                                        {continueAdapterActive && (
                                            <span style={{ color: '#0e7490', fontSize: 11 }}>
                                                live=续训 adapter
                                            </span>
                                        )}
                                        {continueJobId && (
                                            <span style={{ color: '#94a3b8' }}>job {continueJobId}</span>
                                        )}
                                        {continueResultSummary && (
                                            <span style={{ color: '#166534', width: '100%' }}>
                                                {continueResultSummary}
                                            </span>
                                        )}
                                    </div>
                                )}
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

                                {structuralAttributionEnabled ? (
                                    <div style={{
                                        display: 'flex',
                                        flexDirection: 'column',
                                        height: '100%',
                                        minHeight: 360,
                                        overflow: 'hidden',
                                    }}>
                                        <div style={{
                                            flex: '1 1 50%',
                                            minHeight: 160,
                                            overflow: 'auto',
                                            borderBottom: '1px solid #e5e7eb',
                                        }}>
                                            <div style={{
                                                position: 'sticky',
                                                top: 0,
                                                zIndex: 1,
                                                padding: '6px 12px',
                                                fontSize: 11,
                                                fontWeight: 700,
                                                background: '#f8fafc',
                                                borderBottom: '1px solid #e5e7eb',
                                                color: '#1d4ed8',
                                            }}>
                                                上 · 梯度归因
                                                {degradePairs.length > 0
                                                    ? ' · L_sal 按边'
                                                    : (attrMode === 'manual' ? ' · 指定 pair' : '')}
                                            </div>
                                            {trainGroups.length === 0 ? (
                                                <div className={styles.emptyState} style={{ padding: '24px 0' }}>
                                                    {trainPanelEmptyHint}
                                                </div>
                                            ) : (
                                                <div className={styles.trainGroupList}>
                                                    {trainGroups.map(({ id, pairs }) => (
                                                        <TrainSampleGroup
                                                            key={`grad-${id}`}
                                                            trainIdx={id}
                                                            pairs={pairs}
                                                            detail={resolveTrainDetail(id)}
                                                            onProbeEmbeddings={importedReportActive ? undefined : handleOpenTrainProbe}
                                                            probeBusy={probingTrainSampleId === id}
                                                            selectedPairIds={selectedTrainPairIdsByGroup[id] ?? []}
                                                            onTogglePairSelection={toggleTrainPairSelection}
                                                            comparisonSummary={trainProbeComparisons[id]}
                                                            linkedTokenIndex={linkedTrainSampleId === id ? linkedTrainTokenIndex : null}
                                                            onTokenHover={inlineBundle?.trainSampleId === id
                                                                ? makeTrainTokenHoverHandler(id, resolveTrainDetail(id)?.full_tokens ?? [])
                                                                : undefined}
                                                            gtEdgesByTarget={trainGtEdges?.[String(id)]}
                                                            onUnlearnPair={importedReportActive ? undefined : (pair) => handlePairIntervene(pair, 'unlearn')}
                                                            onLearnPair={importedReportActive ? undefined : (pair) => handlePairIntervene(pair, 'learn')}
                                                            onRecoverIntervention={importedReportActive ? undefined : handleRecoverIntervention}
                                                            interveningPairId={interveningPairId}
                                                            interveningDirection={interveningDirection}
                                                            recoverBusy={recoverBusy}
                                                            activeInterventionPairId={activeInterventionPairId}
                                                            interventionSteps={interventionSteps}
                                                            interveneLr={pairInterveneLr}
                                                            unlearnResults={unlearnResultsByPairId}
                                                            onOpenAnnotationViewer={handleOpenAnnotationViewer}
                                                            onAutoAnnotateContinue={(pair) =>
                                                                handleOpenAnnotationViewer(pair, { autoAnnotate: true })
                                                            }
                                                        />
                                                    ))}
                                                </div>
                                            )}
                                        </div>
                                        <div style={{
                                            flex: '1 1 50%',
                                            minHeight: 160,
                                            overflow: 'auto',
                                            background: '#faf5ff',
                                        }}>
                                            <div style={{
                                                position: 'sticky',
                                                top: 0,
                                                zIndex: 1,
                                                padding: '6px 12px',
                                                fontSize: 11,
                                                fontWeight: 700,
                                                background: '#f3e8ff',
                                                borderBottom: '1px solid #e9d5ff',
                                                color: '#6d28d9',
                                            }}>
                                                下 · 结构归因（AST · PRE∪SUF×MID
                                                {attrMode === 'manual' ? ' · 指定 pair' : ''}）
                                                {structuralBusy ? ' …首次会建缓存，可能较慢' : ''}
                                                {structuralMeta ? ` · ${structuralMeta}` : ''}
                                            </div>
                                            {structuralError && (
                                                <div style={{ padding: 12, color: '#b91c1c', fontSize: 11 }}>
                                                    {structuralError}
                                                </div>
                                            )}
                                            {!structuralError && !structuralBusy && structuralPairs.length === 0 && (
                                                <div className={styles.emptyState} style={{ padding: '24px 0' }}>
                                                    {activeQueryEdge
                                                        ? 'No AST/text pairs above threshold (check tree-sitter / language).'
                                                        : (attrMode === 'manual'
                                                            ? '指定 pair：选齐上下文 source 与 Gold target 后自动检索。'
                                                            : '先选中一条 source→target 边。')}
                                                </div>
                                            )}
                                            <div style={{
                                                padding: '6px 12px',
                                                fontSize: 10,
                                                color: '#6b7280',
                                                borderBottom: '1px solid #e9d5ff',
                                            }}>
                                                按 pair 展示（非按 train 聚合）。黄=source / 橙=target；
                                                分数=0.8·结构+0.2·文本。点 TRAIN# 打开标注页也会黄/橙高亮。
                                            </div>
                                            <div className={styles.pairList} style={{ padding: '8px 10px 16px' }}>
                                                {structuralPairs.map((pair) => (
                                                    <PairCard
                                                        key={pair.id}
                                                        pair={pair}
                                                        detail={
                                                            structuralTrainDetails[String(pair.train_sample_id)]
                                                            ?? resolveTrainDetail(pair.train_sample_id)
                                                        }
                                                        annotatedSourceIndices={annotatedSourcesForPairs(
                                                            trainGtEdges?.[String(pair.train_sample_id)],
                                                            [pair],
                                                        )}
                                                        onUnlearn={importedReportActive ? undefined : () => handlePairIntervene(pair, 'unlearn')}
                                                        onLearn={importedReportActive ? undefined : () => handlePairIntervene(pair, 'learn')}
                                                        onRecover={importedReportActive ? undefined : handleRecoverIntervention}
                                                        unlearnBusy={interveningPairId === pair.id && interveningDirection === 'unlearn'}
                                                        learnBusy={interveningPairId === pair.id && interveningDirection === 'learn'}
                                                        recoverBusy={recoverBusy}
                                                        unlearnResult={unlearnResultsByPairId[pair.id] ?? null}
                                                        interveneActiveForPair={activeInterventionPairId === pair.id}
                                                        interventionSteps={
                                                            activeInterventionPairId === pair.id ? interventionSteps : undefined
                                                        }
                                                        interveneLr={pairInterveneLr}
                                                        onOpenAnnotationViewer={() => handleOpenAnnotationViewer(pair)}
                                                        onAutoAnnotateContinue={() =>
                                                            handleOpenAnnotationViewer(pair, { autoAnnotate: true })
                                                        }
                                                        defaultExpanded
                                                    />
                                                ))}
                                            </div>
                                        </div>
                                    </div>
                                ) : trainGroups.length === 0 ? (
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
                                                detail={resolveTrainDetail(id)}
                                                onProbeEmbeddings={importedReportActive ? undefined : handleOpenTrainProbe}
                                                probeBusy={probingTrainSampleId === id}
                                                selectedPairIds={selectedTrainPairIdsByGroup[id] ?? []}
                                                onTogglePairSelection={toggleTrainPairSelection}
                                                comparisonSummary={trainProbeComparisons[id]}
                                                linkedTokenIndex={linkedTrainSampleId === id ? linkedTrainTokenIndex : null}
                                                onTokenHover={inlineBundle?.trainSampleId === id
                                                    ? makeTrainTokenHoverHandler(id, resolveTrainDetail(id)?.full_tokens ?? [])
                                                    : undefined}
                                                gtEdgesByTarget={trainGtEdges?.[String(id)]}
                                                onUnlearnPair={importedReportActive ? undefined : (pair) => handlePairIntervene(pair, 'unlearn')}
                                                onLearnPair={importedReportActive ? undefined : (pair) => handlePairIntervene(pair, 'learn')}
                                                onRecoverIntervention={importedReportActive ? undefined : handleRecoverIntervention}
                                                interveningPairId={interveningPairId}
                                                interveningDirection={interveningDirection}
                                                recoverBusy={recoverBusy}
                                                activeInterventionPairId={activeInterventionPairId}
                                                interventionSteps={interventionSteps}
                                                interveneLr={pairInterveneLr}
                                                unlearnResults={unlearnResultsByPairId}
                                                onOpenAnnotationViewer={handleOpenAnnotationViewer}
                                                onAutoAnnotateContinue={(pair) =>
                                                    handleOpenAnnotationViewer(pair, { autoAnnotate: true })
                                                }
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
