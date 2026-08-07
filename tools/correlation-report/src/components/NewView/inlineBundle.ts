// Reading a prepared TTAV bundle, and mapping its points back to source tokens.
//
// The point→token mapping is a lookup, not a computation: for a plain bundle the
// point index *is* the token index, and a probe bundle carries `point_records`
// stating each point's side and token index. Both were checked against the
// report's own token lists (155 test + 97 train tokens, zero mismatches), which
// is what makes it safe to drive the code highlight off a hovered point.
//
// Kept apart from InlineVisualizer.tsx so the rendering and the data mapping can
// be read — and changed — independently.

import type { ProbeLink } from './plotOverlay';

// matplotlib's tab10, which is what the TTAV backend assigns to classes
// (tool/server/server_utils.py: get_coloring_list → plt.get_cmap('tab10')).
// Hardcoded rather than fetched because it is a fixed palette and this view has
// to work without the TTAV backend running.
export const TAB10 = [
    'rgba(31, 119, 180, 1)',
    'rgba(255, 127, 14, 1)',
    'rgba(44, 160, 44, 1)',
    'rgba(214, 39, 40, 1)',
    'rgba(148, 103, 189, 1)',
    'rgba(140, 86, 75, 1)',
    'rgba(227, 119, 194, 1)',
    'rgba(127, 127, 127, 1)',
    'rgba(188, 189, 34, 1)',
    'rgba(23, 190, 207, 1)',
];
export const FALLBACK_COLOR = 'rgba(116, 116, 116, 1)';

// Which token in which sample a point stands for. The report uses this to move
// the highlight into the right code panel, and hands it back when the user
// hovers a token instead of a point.
export interface TokenHoverTarget {
    side: 'test' | 'train';
    tokenIndex: number;
    /** Set for probe bundles, so the report knows which TRAIN group to light up. */
    trainSampleId: number | null;
    /** train_context / test_target / … for probes, 'prompt' / 'output' otherwise. */
    role: string | null;
    token: string;
}

interface PointRecord {
    point_index: number;
    side: 'train' | 'test';
    role: string;
    token_index: number;
    token: string;
    token_display: string;
    is_focus?: boolean;
}

export interface InlineBundle {
    sampleId: string;
    kind: 'sample' | 'probe';
    classes: string[];
    labels: number[];
    tokens: string[];
    /** Display names per point, e.g. "P131: \n" or "XS60:  error". */
    textList: string[];
    projection: number[][];
    promptLen: number;
    /** Probe bundles only. */
    pointRecords: PointRecord[] | null;
    trainSampleId: number | null;
    /** Probe bundles only: the pairs, already resolved to point indices. */
    links: ProbeLink[];
    /** Every pair id in the bundle, for the "showing N of M" note. */
    allPairIds: string[];
    /**
     * Points the report put in focus. Probe bundles carry their own set (the
     * tokens of the matched pairs); for a plain bundle it stays empty and the
     * caller supplies the live selection from the report UI instead.
     */
    selectedPoints: number[];
}

interface RawBundle {
    selected_indices?: unknown;
    bundle?: {
        classes?: unknown;
        labels?: unknown;
        token_list?: unknown;
        text_list?: unknown;
        projection?: unknown;
        prompt_len?: unknown;
        probe_metadata?: {
            train_sample_id?: unknown;
            point_records?: unknown;
            pair_signature?: unknown;
        };
    };
}

function asNumberArray(value: unknown): number[] {
    return Array.isArray(value) ? value.filter((v): v is number => typeof v === 'number') : [];
}

function asStringArray(value: unknown): string[] {
    return Array.isArray(value) ? value.filter((v): v is string => typeof v === 'string') : [];
}

/**
 * Fetch a prepared bundle and reduce it to what the plot needs.
 *
 * Reads the projection-only endpoint: the full payload is 9-10 MB per sample
 * because of the 3584-dim embeddings, and none of them are used here.
 *
 * `visiblePairIds` mirrors the pairs ticked in the report. Empty means show all,
 * matching TTAV — a probe is opened precisely to look at its pairs, so defaulting
 * to none would be backwards.
 */
export async function loadInlineBundle(
    sampleId: string,
    visiblePairIds: string[] = [],
): Promise<InlineBundle> {
    const resp = await fetch(`/data/real-bundles/${encodeURIComponent(sampleId)}/projection.json`, {
        cache: 'no-store',
    });
    if (!resp.ok) {
        throw new Error(`未找到 ${sampleId} 的预计算 bundle (HTTP ${resp.status})。`);
    }

    const raw = await resp.json() as RawBundle;
    const bundle = raw.bundle;
    if (!bundle) throw new Error(`${sampleId} 的 bundle 结构异常：缺少 bundle 字段。`);

    const projection = Array.isArray(bundle.projection)
        ? (bundle.projection as unknown[]).filter((p): p is number[] =>
            Array.isArray(p) && p.length >= 2 && typeof p[0] === 'number' && typeof p[1] === 'number')
        : [];
    if (projection.length === 0) throw new Error(`${sampleId} 的 bundle 里没有可用的 projection。`);

    const meta = bundle.probe_metadata;
    const pointRecords = Array.isArray(meta?.point_records)
        ? (meta!.point_records as PointRecord[]).filter(r =>
            r && typeof r.point_index === 'number' && typeof r.token_index === 'number')
        : null;

    // Point index → token index, per side. A probe merges two samples into one
    // matrix, so a pair's endpoints only become drawable once both resolve.
    const trainPointByToken = new Map<number, number>();
    const testPointByToken = new Map<number, number>();
    pointRecords?.forEach(rec => {
        const target = rec.side === 'train' ? trainPointByToken
            : rec.side === 'test' ? testPointByToken
            : null;
        target?.set(rec.token_index, rec.point_index);
    });

    const signature = Array.isArray(meta?.pair_signature) ? meta!.pair_signature as Array<{
        id?: unknown; cos_sim?: unknown;
        trainSourceIndex?: unknown; trainTargetIndex?: unknown;
        testSourceIndex?: unknown; testTargetIndex?: unknown;
    }> : [];

    const visible = visiblePairIds.length > 0 ? new Set(visiblePairIds) : null;
    const links: ProbeLink[] = [];
    const allPairIds: string[] = [];
    // A probe is anchored on one test edge, so every pair shares it. Drawing it
    // per pair would stack N identical segments and let an arbitrary one decide
    // the colour — so it is drawn once, uncoloured, as the reference the train
    // edges are compared against.
    const seenTestEdges = new Set<string>();

    for (const entry of signature) {
        const pairId = typeof entry.id === 'string' ? entry.id : null;
        if (!pairId) continue;
        allPairIds.push(pairId);
        if (visible && !visible.has(pairId)) continue;

        const cosSim = typeof entry.cos_sim === 'number' ? entry.cos_sim : null;
        const resolve = (map: Map<number, number>, idx: unknown) =>
            typeof idx === 'number' ? map.get(idx) ?? null : null;

        const trainFrom = resolve(trainPointByToken, entry.trainSourceIndex);
        const trainTo = resolve(trainPointByToken, entry.trainTargetIndex);
        if (trainFrom !== null && trainTo !== null) {
            links.push({ fromPoint: trainFrom, toPoint: trainTo, cosSim, side: 'train', pairId });
        }

        const testFrom = resolve(testPointByToken, entry.testSourceIndex);
        const testTo = resolve(testPointByToken, entry.testTargetIndex);
        if (testFrom !== null && testTo !== null) {
            const edgeKey = `${testFrom}->${testTo}`;
            if (!seenTestEdges.has(edgeKey)) {
                seenTestEdges.add(edgeKey);
                links.push({ fromPoint: testFrom, toPoint: testTo, cosSim: null, side: 'test', pairId });
            }
        }
    }

    const trainSampleId = typeof meta?.train_sample_id === 'number' ? meta.train_sample_id : null;

    // The focus points a probe ships with — the same list TTAV reads. Falling
    // back to the point_records' own is_focus flag keeps older bundles working,
    // since the two are written from the same set.
    const selectedPoints = asNumberArray(raw.selected_indices);
    if (selectedPoints.length === 0 && pointRecords) {
        pointRecords.forEach(rec => {
            if (rec.is_focus) selectedPoints.push(rec.point_index);
        });
    }

    return {
        sampleId,
        kind: pointRecords ? 'probe' : 'sample',
        classes: asStringArray(bundle.classes),
        labels: asNumberArray(bundle.labels),
        tokens: asStringArray(bundle.token_list),
        textList: asStringArray(bundle.text_list),
        projection,
        promptLen: typeof bundle.prompt_len === 'number' ? bundle.prompt_len : 0,
        pointRecords,
        trainSampleId,
        links,
        allPairIds,
        selectedPoints: selectedPoints.sort((a, b) => a - b),
    };
}

/** Point index → the token it stands for, or null if the bundle can't say. */
export function resolvePointToToken(bundle: InlineBundle, pointIndex: number): TokenHoverTarget | null {
    if (bundle.kind === 'probe') {
        const rec = bundle.pointRecords?.find(r => r.point_index === pointIndex);
        if (!rec) return null;
        return {
            side: rec.side,
            tokenIndex: rec.token_index,
            trainSampleId: rec.side === 'train' ? bundle.trainSampleId : null,
            role: rec.role,
            token: rec.token,
        };
    }
    // A plain bundle is one sample's tokens in order, so the point index is the
    // token index. Verified against the report's test_sample_baseline.full_tokens.
    if (pointIndex < 0 || pointIndex >= bundle.projection.length) return null;
    return {
        side: 'test',
        tokenIndex: pointIndex,
        trainSampleId: null,
        role: pointIndex >= bundle.promptLen ? 'output' : 'prompt',
        token: bundle.tokens[pointIndex] ?? '',
    };
}

/** The inverse, for highlighting a point when the user hovers a token. */
export function resolveTokenToPoint(bundle: InlineBundle, target: TokenHoverTarget): number | null {
    if (bundle.kind === 'probe') {
        const rec = bundle.pointRecords?.find(r =>
            r.side === target.side && r.token_index === target.tokenIndex);
        return rec ? rec.point_index : null;
    }
    if (target.side !== 'test') return null;
    return target.tokenIndex >= 0 && target.tokenIndex < bundle.projection.length
        ? target.tokenIndex
        : null;
}

/** One correlation pair's four endpoints, used to light them up in the probe. */
export interface ProbeEmphasisPair {
    id: string;
    trainSourceIndex: number;
    trainTargetIndex: number;
    testSourceIndex: number;
    testTargetIndex: number;
}

export interface ProbeEmphasis {
    /** Points that stay full-colour / ringed; everything else is washed out. */
    points: Set<number>;
    /** Links drawn at full strength; others are almost invisible. */
    linkKeys: Set<string>;
}

function linkKey(link: ProbeLink): string {
    return `${link.side}:${link.fromPoint}->${link.toPoint}:${link.pairId}`;
}

function findProbePoint(
    bundle: InlineBundle,
    side: 'train' | 'test',
    tokenIndex: number,
): number | null {
    const rec = bundle.pointRecords?.find(r => r.side === side && r.token_index === tokenIndex);
    return rec ? rec.point_index : null;
}

/**
 * Which probe points/links the Model / pair 「选择」 ticks should emphasize.
 * Returns null when nothing is ticked — the plot stays in its default full view.
 *
 * Resolves endpoints via token indices (not only link.pairId), so Model-only or
 * train-only selection still finds the right points even when the shared test
 * edge was stored under a different pair's id.
 */
export function collectProbeEmphasis(
    bundle: InlineBundle,
    opts: {
        selectedPairs: ProbeEmphasisPair[];
        includeTestSaliency: boolean;
        testSourceIndex: number | null;
        testTargetIndex: number | null;
    },
): ProbeEmphasis | null {
    if (bundle.kind !== 'probe') return null;

    const hasPairs = opts.selectedPairs.length > 0;
    const includeTest = opts.includeTestSaliency
        && opts.testSourceIndex != null
        && opts.testTargetIndex != null;
    if (!hasPairs && !includeTest) return null;

    const points = new Set<number>();
    const linkKeys = new Set<string>();
    const pairIdSet = new Set(opts.selectedPairs.map(p => p.id));

    const addEndpoint = (side: 'train' | 'test', tokenIndex: number) => {
        const idx = findProbePoint(bundle, side, tokenIndex);
        if (idx !== null) points.add(idx);
    };

    for (const pair of opts.selectedPairs) {
        addEndpoint('train', pair.trainSourceIndex);
        addEndpoint('train', pair.trainTargetIndex);
        addEndpoint('test', pair.testSourceIndex);
        addEndpoint('test', pair.testTargetIndex);
    }

    if (includeTest) {
        addEndpoint('test', opts.testSourceIndex!);
        addEndpoint('test', opts.testTargetIndex!);
    }

    for (const link of bundle.links) {
        const byPairId = hasPairs && pairIdSet.has(link.pairId);
        const bothEndsHot = points.has(link.fromPoint) && points.has(link.toPoint);
        const modelTestEdge = includeTest
            && link.side === 'test'
            && bothEndsHot;
        // Train-only: pairId may not match the single stored test-edge row, so
        // also accept any link whose endpoints we already resolved from tokens.
        if (byPairId || modelTestEdge || (hasPairs && bothEndsHot)) {
            linkKeys.add(linkKey(link));
            points.add(link.fromPoint);
            points.add(link.toPoint);
        }
    }

    if (points.size === 0) return null;
    return { points, linkKeys };
}

export { linkKey as probeLinkKey };

