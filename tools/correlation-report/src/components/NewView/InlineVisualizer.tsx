// In-page scatter plot of a prepared TTAV bundle.
//
// This renders the same picture as the TTAV web app rather than an approximation
// of it: same component (embedding-atlas's EmbeddingView, pinned to the version
// TTAV depends on), same 2-D coordinates (the bundle ships a precomputed
// `projection`; TTAV's backend just writes it to disk unchanged), the same tab10
// palette, and the same point size. Pan and zoom come from EmbeddingView.
//
// What it adds over the standalone app is the link back to the source: hovering
// a point reports which token it stands for, and the report highlights that
// token in the code panels. See inlineBundle.ts for the mapping itself.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { EmbeddingView, type DataPoint, type ViewportState } from 'embedding-atlas/react';
import { PlotOverlay } from './plotOverlay';
import {
    TAB10,
    FALLBACK_COLOR,
    resolvePointToToken,
    resolveTokenToPoint,
    type InlineBundle,
    type TokenHoverTarget,
} from './inlineBundle';
import styles from './NewView.module.css';

/** Restate an `rgba(r, g, b, a)` colour at a different alpha. */
function withAlpha(color: string, alpha: number): string {
    const match = color.match(/^rgba?\(([^)]+)\)$/);
    if (!match) return color;
    const [r, g, b] = match[1].split(',').map(part => part.trim());
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function describeToken(token: string): string {
    if (token === '\n') return '↵';
    if (token.trim() === '') return `"${token}"`;
    return token;
}

const POINT_SIZE = 2;
/** Unselected points when a Model/pair tick is active — barely visible cloud. */
const SELECTION_DIM_ALPHA = 0.045;

export function InlineVisualizer({
    bundle,
    hoverTarget,
    onHoverTargetChange,
    selectedPoints,
    emphasizedPoints,
    emphasizedLinkKeys,
    canvasHeight,
}: {
    bundle: InlineBundle;
    /** Set by the floating panel's resize handle. */
    canvasHeight: number;
    /** Hover driven from the code panels; keeps both directions in one state. */
    hoverTarget: TokenHoverTarget | null;
    onHoverTargetChange: (target: TokenHoverTarget | null) => void;
    /**
     * Points to ring and always label. A probe brings its own set; a plain bundle
     * gets the report's live selection (target token + attribution sources), so
     * clicking a different token in the code updates the plot.
     */
    selectedPoints: number[];
    /**
     * When non-null, points outside this set are washed out so the selection
     * reads against the full embedding cloud instead of replacing it.
     */
    emphasizedPoints?: Set<number> | null;
    /** Matching link keys for the overlay; same selection as emphasizedPoints. */
    emphasizedLinkKeys?: Set<string> | null;
}) {
    const containerRef = useRef<HTMLDivElement | null>(null);
    const [size, setSize] = useState({ width: 0, height: 0 });
    const [viewportState, setViewportState] = useState<ViewportState | null>(null);
    // Labels default on, matching TTAV (state.unified.ts: showLabel/showIndex both
    // true). Density is handled in the overlay — a gap, a budget, and a priority
    // order — rather than by switching them off past some size.
    const [showLabels, setShowLabels] = useState(true);

    // EmbeddingView needs explicit pixel dimensions.
    useEffect(() => {
        const node = containerRef.current;
        if (!node) return;
        const observer = new ResizeObserver(entries => {
            const rect = entries[0]?.contentRect;
            if (rect) setSize({ width: Math.floor(rect.width), height: Math.floor(rect.height) });
        });
        observer.observe(node);
        return () => observer.disconnect();
    }, []);

    // Note: the caller keys this component by sampleId, so a different bundle
    // remounts it and the viewport starts fresh. Carrying the previous sample's
    // zoom over would apply it to coordinates it has nothing to do with.

    const prepared = useMemo(() => {
        const n = bundle.projection.length;
        const x = new Float32Array(n);
        const y = new Float32Array(n);
        const category = new Uint8Array(n);
        const categoryColors: string[] = [];
        const slotByKey = new Map<string, number>();
        const dataPoints: DataPoint[] = [];
        const selectionActive = Boolean(emphasizedPoints && emphasizedPoints.size > 0);

        for (let i = 0; i < n; i++) {
            x[i] = bundle.projection[i][0];
            y[i] = bundle.projection[i][1];

            const label = bundle.labels[i] ?? 0;
            const isContext = (bundle.classes[label] ?? '').endsWith('_context');
            const isHot = !selectionActive || emphasizedPoints!.has(i);
            // Per-(label, hot/dim) slots so selection can wash out points without
            // hiding them — EmbeddingView colours by category, not per point.
            const slotKey = `${label}:${isHot ? 'hot' : 'dim'}`;
            let slot = slotByKey.get(slotKey);
            if (slot === undefined) {
                slot = categoryColors.length;
                slotByKey.set(slotKey, slot);
                const base = TAB10[label] ?? FALLBACK_COLOR;
                const alpha = !isHot
                    ? SELECTION_DIM_ALPHA
                    : (isContext ? 0.28 : 1);
                categoryColors.push(withAlpha(base, alpha));
            }
            category[i] = slot;

            const resolved = resolvePointToToken(bundle, i);
            const text = resolved
                ? `@${resolved.tokenIndex}  ${describeToken(resolved.token)}\n${resolved.side} · ${resolved.role ?? ''}`
                : `Index: ${i}`;
            dataPoints.push({ x: x[i], y: y[i], category: label, text, identifier: i, fields: {} });
        }

        // The class legend follows label order, not slot order, so it reads in
        // the order the bundle declares its classes.
        const legend = bundle.classes.map((name, label) => ({
            name,
            color: TAB10[label] ?? FALLBACK_COLOR,
            count: bundle.labels.filter(l => l === label).length,
        })).filter(entry => entry.count > 0);

        // Who gets a label first when space runs out. Emphasized / non-context
        // points come first so a selection isn't drowned by the cloud.
        const labelOrder = Array.from({ length: n }, (_, i) => i).sort((a, b) => {
            const hotA = selectionActive && emphasizedPoints!.has(a) ? 0 : 1;
            const hotB = selectionActive && emphasizedPoints!.has(b) ? 0 : 1;
            if (hotA !== hotB) return hotA - hotB;
            const ctxA = (bundle.classes[bundle.labels[a] ?? 0] ?? '').endsWith('_context') ? 1 : 0;
            const ctxB = (bundle.classes[bundle.labels[b] ?? 0] ?? '').endsWith('_context') ? 1 : 0;
            return ctxA - ctxB || a - b;
        });

        return { data: { x, y, category }, categoryColors, dataPoints, legend, labelOrder };
    }, [bundle, emphasizedPoints]);

    // One hover state, two directions: a point hover reports the token upward,
    // and a token hover from the report comes back down as a highlighted point.
    const hoveredPoint = useMemo(
        () => (hoverTarget ? resolveTokenToPoint(bundle, hoverTarget) : null),
        [bundle, hoverTarget],
    );

    const tooltip = hoveredPoint !== null ? prepared.dataPoints[hoveredPoint] ?? null : null;

    const handleTooltip = useCallback((point: DataPoint | null) => {
        if (!point) {
            onHoverTargetChange(null);
            return;
        }
        onHoverTargetChange(resolvePointToToken(bundle, point.identifier as number));
    }, [bundle, onHoverTargetChange]);

    // Nearest-point lookup for hover/selection, in data space.
    const querySelection = useCallback(async (x: number, y: number, unitDistance: number) => {
        const { data } = prepared;
        let best = -1;
        let bestD2 = Infinity;
        for (let i = 0; i < data.x.length; i++) {
            const dx = data.x[i] - x;
            const dy = data.y[i] - y;
            const d2 = dx * dx + dy * dy;
            if (d2 < bestD2) { bestD2 = d2; best = i; }
        }
        if (best < 0 || Math.sqrt(bestD2) > unitDistance * 10) return null;
        return prepared.dataPoints[best];
    }, [prepared]);

    const overlay = useMemo(() => ({
        class: PlotOverlay as never,
        props: {
            links: bundle.links,
            pointX: prepared.data.x,
            pointY: prepared.data.y,
            hoveredPoint,
            selectedPoints,
            emphasizedLinkKeys: emphasizedLinkKeys ?? null,
            textList: bundle.textList,
            showLabel: showLabels,
            showIndex: showLabels,
            labelOrder: prepared.labelOrder,
            pointSize: POINT_SIZE,
        },
    }), [bundle.links, bundle.textList, prepared, hoveredPoint, selectedPoints, emphasizedLinkKeys, showLabels]);

    return (
        <div className={styles.inlineVisualizerBody}>
            <div
                className={styles.inlineVisualizerCanvas}
                ref={containerRef}
                style={{ height: canvasHeight }}
            >
                {size.width > 0 && size.height > 0 && (
                    <EmbeddingView
                        data={prepared.data}
                        categoryColors={prepared.categoryColors}
                        width={size.width}
                        height={size.height}
                        // Matches TTAV's defaults (state.unified.ts: pointSize 2,
                        // chart.tsx: mode 'points', colorScheme 'light') so the
                        // same bundle looks the same in both places.
                        config={{ mode: 'points', colorScheme: 'light', pointSize: POINT_SIZE }}
                        // Empty (but non-null) turns off embedding-atlas's own
                        // label generation. Left unset it density-clusters the
                        // points in a worker to invent cluster names — labels we
                        // neither want (plotOverlay draws per-token ones) nor can
                        // rely on: the worker URL does not survive Vite's dep
                        // pre-bundling, so the await never settles and the view
                        // sits on "Generating labels..." forever.
                        labels={[]}
                        tooltip={tooltip}
                        onTooltip={handleTooltip}
                        viewportState={viewportState}
                        onViewportState={setViewportState}
                        querySelection={querySelection}
                        customOverlay={overlay}
                    />
                )}
            </div>
            <div className={styles.inlineVisualizerLegend}>
                {prepared.legend.map(entry => (
                    <span key={entry.name} className={styles.inlineLegendItem}>
                        <span className={styles.inlineLegendSwatch} style={{ background: entry.color }} />
                        {entry.name}
                        <span className={styles.inlineLegendCount}>{entry.count}</span>
                    </span>
                ))}
                <label className={styles.inlineLegendToggle}>
                    <input
                        type="checkbox"
                        checked={showLabels}
                        onChange={e => setShowLabels(e.target.checked)}
                    />
                    token 名
                </label>
            </div>
        </div>
    );
}
