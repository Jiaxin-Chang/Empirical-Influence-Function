// Overlay for the in-page scatter plot: pair links, selection rings, point labels.
//
// Ported from TTAV's NeighborOverlay
// (time-travelling-visualizer/web/src/component/chart.tsx, class NeighborOverlay).
// The two draw the same picture on purpose — the visual encoding here is the
// report's published reading of a probe, so if it changes here it must change
// there too. What is deliberately left out is the projection-diagnosis half:
// neighbor rings, refine trails, box select, click-to-select. Those answer "is
// this projection trustworthy", which is not what this view is for.
//
// It plugs into EmbeddingView's `customOverlay` slot, which hands us a `proxy`
// that converts data coordinates to screen coordinates. Going through the proxy
// on every render is what keeps everything glued to the points while the user
// pans and zooms.

const SVG_NS = 'http://www.w3.org/2000/svg';

// Gap enforced between label boxes, in screen px. Density falls out of this and
// the zoom level together: zooming spreads the points, so more labels fit.
const LABEL_GAP = 12;

// Ceiling on automatically-placed labels. Collision culling alone still admits
// several hundred on a large probe, which reads as texture rather than as text.
// Focus labels are placed before this and are never counted against it.
const MAX_AUTO_LABELS = 70;

export interface ProbeLink {
    fromPoint: number;
    toPoint: number;
    /** 'train' draws solid, 'test' draws dashed. */
    side: 'train' | 'test';
    /** Signed gradient similarity; null for the neutral reference edge. */
    cosSim: number | null;
    pairId: string;
}

interface OverlayProxy {
    width: number;
    height: number;
    location(x: number, y: number): { x: number; y: number };
}

export interface PlotOverlayProps {
    proxy: OverlayProxy;
    links: ProbeLink[];
    /** Data-space coordinates indexed by point id. */
    pointX: Float32Array;
    pointY: Float32Array;
    /** Point id currently hovered, so its links can be lifted above the rest. */
    hoveredPoint: number | null;
    /** Points the report considers in focus — ringed, and always labelled. */
    selectedPoints: number[];
    /**
     * When set, links whose key is missing are drawn almost invisible so the
     * selected pairs stay readable against the full cloud.
     */
    emphasizedLinkKeys?: Set<string> | null;
    /** The bundle's text_list, e.g. "P131: \n" or "XS60:  error". */
    textList: string[];
    showLabel: boolean;
    /** Point ids in the order they should compete for label space. */
    labelOrder?: number[];
    showIndex: boolean;
    pointSize: number;
}

/**
 * TTAV's formatPointLabel (chart.tsx), kept in sync with it.
 *
 * The prefix in text_list already carries the token index and, for probes, the
 * side and role — `O136` is output token 136, `XT136` is the test target at 136.
 * So a label reads "O136:  fmt" / "XT136:  fmt" and needs no separate point id.
 *
 * The pattern covers every bundle kind: `P`/`O` for a plain bundle, `TC`/`TS`/`TT`
 * and `XC`/`XS`/`XT` for a probe, `AP`/`AA`/`BP`/`BA` for an arbitrary pair.
 * A prefix missing from this list falls into the generic branch, where it comes
 * out as "3588. BP1059:  …" with the point index bolted on, duplicating the token
 * index the prefix already carries — which is how both the probe and pair
 * prefixes were first noticed as missing.
 */
function formatPointLabel(id: number, rawLabel: string, showLabel: boolean, showIndex: boolean): string {
    // \u00a0 = non-breaking space, \u2420 = SYMBOL FOR SPACE. Both stand in for a
    // real space in token displays and have to collapse back to one. Written as
    // escapes rather than the literal characters TTAV uses, so the intent
    // survives copy-paste and no linter trips over invisible input.
    const normalized = rawLabel.replace(/\u00a0/g, ' ').replace(/\s+/g, ' ').trim();
    const tokenMatch = normalized.match(/^([PO]\d+|[TX][CST]\d+|[AB][PA]\d+):\s*(.*)$/);

    if (tokenMatch) {
        const prefix = tokenMatch[1];
        const tail = tokenMatch[2].replace(/[\u2420\u0020]+/g, ' ').trim() || '·';
        if (showLabel) return `${prefix}: ${tail}`;
        if (showIndex) return String(id);
        return '';
    }

    if (showLabel && showIndex) return normalized ? `${id}. ${normalized}` : String(id);
    if (showLabel) return normalized;
    if (showIndex) return String(id);
    return '';
}

export class PlotOverlay {
    private el: HTMLDivElement | null;
    private svg: SVGSVGElement | null = null;
    private props: PlotOverlayProps;

    constructor(target: HTMLDivElement, props: PlotOverlayProps) {
        this.el = target;
        this.props = props;
        this.svg = document.createElementNS(SVG_NS, 'svg');
        this.svg.style.display = 'block';
        this.svg.style.pointerEvents = 'none';
        this.el.appendChild(this.svg);
        this.render();
    }

    update(nextProps: Partial<PlotOverlayProps>) {
        this.props = { ...this.props, ...nextProps };
        this.render();
    }

    destroy() {
        if (this.svg && this.el) this.el.removeChild(this.svg);
        this.svg = null;
        this.el = null;
    }

    private locate(pointId: number): { x: number; y: number } | null {
        const { pointX, pointY, proxy } = this.props;
        if (pointId < 0 || pointId >= pointX.length) return null;
        return proxy.location(pointX[pointId], pointY[pointId]);
    }

    // A small filled triangle at the segment's MIDPOINT rather than an SVG
    // end-marker, so the arrowhead never sits on top of the destination point.
    private drawMidpointArrow(
        parent: SVGGElement,
        x1: number, y1: number, x2: number, y2: number,
        color: string, size = 7,
    ) {
        const mx = (x1 + x2) / 2;
        const my = (y1 + y2) / 2;
        const angleDeg = Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI;
        const half = size * 0.6;
        const arrow = document.createElementNS(SVG_NS, 'path');
        arrow.setAttribute('d', `M ${-half},${-half} L ${size},0 L ${-half},${half} Z`);
        arrow.setAttribute('fill', color);
        arrow.setAttribute('transform', `translate(${mx},${my}) rotate(${angleDeg})`);
        parent.appendChild(arrow);
    }

    private render() {
        const { proxy } = this.props;
        if (!this.svg) return;

        this.svg.setAttribute('width', String(proxy.width));
        this.svg.setAttribute('height', String(proxy.height));
        while (this.svg.firstChild) this.svg.removeChild(this.svg.firstChild);

        // Same z-order as TTAV: links under rings under labels.
        this.renderLinks();
        this.renderSelectionRings();
        this.renderLabels();
    }

    private renderLinks() {
        const { links, hoveredPoint, emphasizedLinkKeys } = this.props;
        if (!this.svg || links.length === 0) return;

        const group = document.createElementNS(SVG_NS, 'g');

        // Draw the hovered point's own links last so they end up on top.
        const touchesHover = (link: ProbeLink) =>
            hoveredPoint !== null && (link.fromPoint === hoveredPoint || link.toPoint === hoveredPoint);
        const linkKeyOf = (link: ProbeLink) =>
            `${link.side}:${link.fromPoint}->${link.toPoint}:${link.pairId}`;
        const ordered = [...links].sort((a, b) => Number(touchesHover(a)) - Number(touchesHover(b)));

        for (const link of ordered) {
            const from = this.locate(link.fromPoint);
            const to = this.locate(link.toPoint);
            if (!from || !to) continue;

            // cos_sim is signed and spans roughly ±0.35 in practice, with ~39% of
            // pairs negative — so the sign picks the colour (warm = the two
            // dependencies agree, cool = they oppose) and only the magnitude
            // drives width. Grading width on the signed value would render a
            // strong negative match as a hairline and hide it.
            //
            // Test edges carry no cos_sim by design (the number compares a train
            // edge against the test edge, so it says nothing about the test edge
            // alone) and land in the neutral grey branch.
            const cos = typeof link.cosSim === 'number' ? link.cosSim : null;
            const strength = cos === null ? 0.35 : Math.min(1, Math.abs(cos) / 0.35);
            const colour = cos === null ? '#7F8C8D' : (cos >= 0 ? '#D35400' : '#2471A3');
            const selectionDimmed = Boolean(
                emphasizedLinkKeys && emphasizedLinkKeys.size > 0 && !emphasizedLinkKeys.has(linkKeyOf(link)),
            );
            const dimmed = selectionDimmed || (hoveredPoint !== null && !touchesHover(link));
            const opacityScale = selectionDimmed ? 0.06 : (dimmed ? 0.25 : 1);

            const line = document.createElementNS(SVG_NS, 'line');
            line.setAttribute('x1', String(from.x));
            line.setAttribute('y1', String(from.y));
            line.setAttribute('x2', String(to.x));
            line.setAttribute('y2', String(to.y));
            line.setAttribute('stroke', colour);
            line.setAttribute('stroke-width', String(selectionDimmed ? 1 : (1.2 + strength * 3.3)));
            line.setAttribute('stroke-opacity', String((0.5 + strength * 0.45) * opacityScale));
            line.setAttribute('stroke-linecap', 'round');
            // Dashed marks the test-sample edge, so a pair's two edges stay
            // tellable apart when they share a colour and width.
            if (link.side === 'test') line.setAttribute('stroke-dasharray', '7 4');
            group.appendChild(line);

            // Edges are directed (source → target); without an arrow the reader
            // can't tell which token influences which.
            if (!dimmed && !selectionDimmed) this.drawMidpointArrow(group, from.x, from.y, to.x, to.y, colour, 7);
        }

        this.svg.appendChild(group);
    }

    // Gold rings on the points the report put in focus — for a probe, the tokens
    // of the matched pairs; for a plain bundle, the target token and its
    // attribution sources. The hovered point gets the dark ring instead, the
    // same treatment TTAV gives its focus centre.
    private renderSelectionRings() {
        const { selectedPoints, hoveredPoint, pointSize } = this.props;
        if (!this.svg) return;

        const ringed = new Set(selectedPoints);
        if (hoveredPoint !== null) ringed.add(hoveredPoint);
        if (ringed.size === 0) return;

        const group = document.createElementNS(SVG_NS, 'g');
        ringed.forEach(pointId => {
            const loc = this.locate(pointId);
            if (!loc) return;
            const isHovered = pointId === hoveredPoint;
            const ring = document.createElementNS(SVG_NS, 'circle');
            ring.setAttribute('cx', String(loc.x));
            ring.setAttribute('cy', String(loc.y));
            ring.setAttribute('r', String(pointSize + 3));
            ring.setAttribute('fill', 'none');
            ring.setAttribute('stroke', isHovered ? '#111827' : '#f59e0b');
            ring.setAttribute('stroke-width', isHovered ? '2.5' : '2');
            group.appendChild(ring);
        });
        this.svg.appendChild(group);
    }

    // Point names. Every point is offered a label, but one that would overlap an
    // already-placed label is dropped — which is why only a readable subset shows
    // at any zoom, and why zooming in reveals more. Selected points are placed
    // first and are allowed to overlap, so a point in focus is never unlabelled.
    private renderLabels() {
        const { selectedPoints, hoveredPoint, textList, showLabel, showIndex, pointSize, pointX } = this.props;
        if (!this.svg) return;

        const forced = new Set(selectedPoints);
        if (hoveredPoint !== null) forced.add(hoveredPoint);
        if (!showLabel && !showIndex && forced.size === 0) return;

        const group = document.createElementNS(SVG_NS, 'g');
        const occupied: { x: number; y: number; width: number; height: number }[] = [];
        // Real separation, not just non-overlap. At the old 2px two labels could sit
        // flush against each other, which on a 5000-point probe produced a wall of
        // text rather than a labelled plot.
        const padding = LABEL_GAP;
        let autoLabels = 0;

        const renderLabel = (loc: { x: number; y: number }, content: string, forceVisible: boolean) => {
            const fontSize = forceVisible ? 13 : 10;
            const charWidth = forceVisible ? 7.5 : 6;
            const charHeight = forceVisible ? 13 : 10;
            const boxWidth = content.length * charWidth;
            const baseOffset = pointSize + 2;
            const candidates = forceVisible
                ? [
                    { dx: baseOffset, dy: -baseOffset },
                    { dx: baseOffset, dy: charHeight + 4 },
                    { dx: -(boxWidth + baseOffset), dy: -baseOffset },
                    { dx: -(boxWidth + baseOffset), dy: charHeight + 4 },
                    { dx: -(boxWidth / 2), dy: -(pointSize + 10) },
                    { dx: -(boxWidth / 2), dy: charHeight + pointSize + 6 },
                ]
                : [{ dx: baseOffset, dy: -baseOffset }];

            let chosen: { labelX: number; labelY: number; boxX: number; boxY: number } | null = null;
            for (const candidate of candidates) {
                const labelX = loc.x + candidate.dx;
                const labelY = loc.y + candidate.dy;
                const boxY = labelY - charHeight;
                const collides = occupied.some(box =>
                    labelX < box.x + box.width + padding
                    && labelX + boxWidth + padding > box.x
                    && boxY < box.y + box.height + padding
                    && boxY + charHeight + padding > box.y);
                if (!collides) { chosen = { labelX, labelY, boxX: labelX, boxY }; break; }
            }
            if (!chosen) {
                if (!forceVisible) return;
                const labelX = loc.x + baseOffset;
                const labelY = loc.y - baseOffset;
                chosen = { labelX, labelY, boxX: labelX, boxY: labelY - charHeight };
            }

            const textEl = document.createElementNS(SVG_NS, 'text');
            textEl.setAttribute('x', String(chosen.labelX));
            textEl.setAttribute('y', String(chosen.labelY));
            textEl.setAttribute('fill', forceVisible ? '#111827' : '#000');
            textEl.setAttribute('font-size', String(fontSize));
            textEl.setAttribute('font-family', 'Console, monospace');
            if (forceVisible) {
                // White outline behind bold text so a focus label stays readable
                // where points are dense.
                textEl.setAttribute('font-weight', '700');
                textEl.setAttribute('paint-order', 'stroke');
                textEl.setAttribute('stroke', '#ffffff');
                textEl.setAttribute('stroke-width', '3');
                textEl.setAttribute('stroke-linejoin', 'round');
            }
            textEl.textContent = content;
            group.appendChild(textEl);
            occupied.push({ x: chosen.boxX, y: chosen.boxY, width: boxWidth, height: charHeight });
        };

        forced.forEach(pointId => {
            const loc = this.locate(pointId);
            if (!loc) return;
            const content = formatPointLabel(pointId, textList[pointId] ?? '', true, true);
            if (content) renderLabel(loc, content, true);
        });

        // Order decides who wins the space, and index order gave it to whichever
        // tokens happened to come first in the train sample. labelOrder puts the
        // meaningful points (sources, targets) ahead of context.
        const order = this.props.labelOrder ?? null;
        const total = pointX.length;
        for (let n = 0; n < total; n++) {
            if (autoLabels >= MAX_AUTO_LABELS) break;
            const i = order ? order[n] : n;
            if (i === undefined || forced.has(i)) continue;
            const loc = this.locate(i);
            // Off-screen points are skipped rather than laid out invisibly: they
            // would otherwise consume the label budget, and skipping them is what
            // makes zooming in reveal more labels instead of the same few.
            if (!loc || loc.x < -40 || loc.y < -20
                || loc.x > this.props.proxy.width + 40
                || loc.y > this.props.proxy.height + 20) continue;
            const content = formatPointLabel(i, textList[i] ?? '', showLabel, showIndex);
            if (!content) continue;
            const before = occupied.length;
            renderLabel(loc, content, false);
            if (occupied.length > before) autoLabels += 1;
        }

        this.svg.appendChild(group);
    }
}
