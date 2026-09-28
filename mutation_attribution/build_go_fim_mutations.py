#!/usr/bin/env python3
"""Build a ground-truth attribution set by rewriting train FIM samples.

Each query keeps the hole's semantic mechanism (role / pattern / operation /
relation) and changes the surface the three baselines read:

* BM25 and token-histogram text similarity: identifiers and literals
* AST structure: node-type and parent-edge histograms
* embedding: same surface rewrite, so bag-of-tokens geometry moves with them

Ground truth is the parent row in ``csn_go_train_fim_10k.jsonl``.
``ground_truth_line`` is the 0-based index in that file.

The ``MUTATIONS`` specs below still store line numbers from the full
``csn_go_train_fim.jsonl``. They are the source of the hand-written
rewrites; ``expand_to_100.py`` is what writes the 10k attribution set.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "csn_go_train_fim_10k.jsonl"
OUT = Path(__file__).resolve().parent / "go_fim_mutations.jsonl"

STOP = {
    "func", "return", "if", "else", "for", "range", "var", "const", "switch",
    "case", "default", "struct", "map", "interface", "type", "package", "import",
    "go", "defer", "select", "chan", "break", "continue", "fallthrough", "goto",
    "new", "make", "append", "copy", "len", "cap", "close", "panic", "recover",
    "print", "println", "nil", "true", "false", "iota", "complex", "real", "imag",
    "byte", "rune", "string", "int", "int8", "int16", "int32", "int64",
    "uint", "uint8", "uint16", "uint32", "uint64", "float32", "float64",
    "bool", "error", "any", "comparable",
}

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _sem(role, pattern, operations, relations):
    return {
        "role": role,
        "pattern": pattern,
        "operations": operations,
        "relations": relations,
    }


def _rel(src, dst, typ):
    return {"source": src, "target": dst, "type": typ}


# prefix / middle / suffix are the mutated FIM split.
# `line` is the 0-based row in the full csn_go_train_fim.jsonl.
# Only specs whose task also sits in csn_go_train_fim_10k.jsonl are kept
# in the attribution file; expand_to_100.py remaps those to 10k indices.
MUTATIONS = [
    {
        "line": 7377,
        "note": "fold is a range loop over a slice literal; the parent is two ifs and one shift expression",
        "prefix": """func gatherMark(raw []byte) (int32, bool) {
	if len(raw) < 6 {
		return -1, false
	}
	tiles := []int{tile(raw[2]), tile(raw[3]), tile(raw[4]), tile(raw[5])}
	acc := 0
	for _, d := range tiles {
		if d < 0 {
			return -1, false
		}

""",
        "middle": """		acc = acc*16 + d
""",
        "suffix": """	}
	return int32(acc), true
}
""",
        "semantic": _sem(
            "assemble one scalar from four validated base-16 digits",
            ["place-value fold of four nibbles"],
            ["fold four digits into one integer and report success"],
            [
                _rel("four validated base-16 digits", "single scalar", "transform"),
                _rel("prior invalid-digit rejection", "successful fold", "semantic_dependency"),
            ],
        ),
    },
    {
        "line": 10664,
        "note": "removed span is released, then the kept parts are concatenated by one call",
        "prefix": """func (pool *slotBin) dropSpan(lo, width int, ring []slot) []slot {
	pool.release(ring[lo : lo+width])
	if lo == 0 && width*2 < len(ring) {
		return ring[width:]
	}

""",
        "middle": """	return joinGap(ring, lo, width)
""",
        "suffix": """}
""",
        "semantic": _sem(
            "close a removed span by sliding the tail forward",
            ["in-place tail shift after a recycled prefix"],
            ["copy each later element down by the removed width"],
            [
                _rel("tail past the removed span", "gap at the removal point", "dataflow"),
                _rel("shifted buffer", "shortened length returned afterward", "semantic_dependency"),
            ],
        ),
    },
    {
        "line": 10727,
        "note": "separator normalization is one rune switch, not successive Replace calls",
        "prefix": """func scrubLabel(label string) *tally {
	out := make([]byte, 0, len(label))
	for _, ch := range label {
		switch ch {
		case '/':
			out = append(out, '-')
""",
        "middle": """		case ':':
			out = append(out, '-')
""",
        "suffix": """		case '.':
			out = append(out, '-')
		default:
			out = append(out, byte(ch))
		}
	}
	cleaned := string(out)
	t := new(tally)
	for _, tail := range []string{"tried", "broke", "ok", "bad", "drop", "open", "slow", "spareOk", "spareBad", "elapsed", "ran"} {
		t.add(cleaned, tail)
	}
	return t
}
""",
        "semantic": _sem(
            "map the colon separator onto the dash used for the other separators",
            ["single-character separator normalization"],
            ["replace a colon with a dash inside the label"],
            [
                _rel("colon in the label", "dash", "transform"),
                _rel("normalized label", "later metric suffixes", "semantic_dependency"),
            ],
        ),
    },
    {
        "line": 10123,
        "note": "cleartext scheme is chosen from a bool map; the secure scheme still refuses first",
        "prefix": """func openDesk(gate string) *desk {
	secure := "htt" + "ps://"
	plain := "htt" + "p://"
	if hasHead(gate, secure) {
		panic("secure scheme refused")
	}
	gate = map[bool]string{
		true:  plain + gate,
		false: gate,
	}[
""",
        "middle": """		!hasHead(gate, plain)
""",
        "suffix": """	]
	return &desk{gate: gate}
}
""",
        "semantic": _sem(
            "attach the cleartext web scheme when the address has none",
            ["default scheme prefix after rejecting the secure scheme"],
            ["prepend the cleartext scheme to the bare address"],
            [
                _rel("bare address", "cleartext scheme prefix", "transform"),
                _rel("secure scheme", "refusal before the prefix", "control"),
            ],
        ),
    },
    {
        "line": 7390,
        "note": "box-kind aliases are a map, not fallthrough cases; both axes scale inside one loop",
        "prefix": """func (sheet *canvas) pinFrame(kind string, rect frame) {
	alias := map[string]string{
		"edge": "Edge", "edgeframe": "Edge",
		"cut": "Cut", "cutframe": "Cut",
		"margin": "Margin", "marginframe": "Margin",
		"picture": "Picture", "pictureframe": "Picture",
	}
	canon, known := alias[lower(kind)]
	if !known {
		sheet.fault = boom("unknown frame kind")
		return
	}

""",
        "middle": """	axes := []*float64{&rect.ax, &rect.ay}
	for _, axis := range axes {
		*axis = *axis * sheet.unit
	}
""",
        "suffix": """	rect.aw = rect.aw*sheet.unit + rect.ax
	rect.ah = rect.ah*sheet.unit + rect.ay
	if sheet.index > 0 {
		sheet.frames[sheet.index][canon] = rect
	}
	sheet.baseFrames[canon] = rect
}
""",
        "semantic": _sem(
            "scale the frame origin by the unit factor",
            ["scale position before size is scaled and shifted"],
            ["multiply both origin axes by the unit factor"],
            [
                _rel("origin axes", "unit factor", "transform"),
                _rel("scaled origin", "size shift applied afterward", "semantic_dependency"),
            ],
        ),
    },
    {
        "line": 7360,
        "note": "window is one composite value; the per-plane reslice is hidden in a call so the parent for-loop disappears",
        "prefix": """func (clip wave) window(lo, hi int) wave {
	if lo > hi {
		panic(note(lo, hi))
	}
	return wave{

""",
        "middle": """		layout: clip.layout,
		count:  hi - lo,
		planes: cut(clip.planes, lo, hi, clip.layout.bytesEach()),
""",
        "suffix": """	}
}
""",
        "semantic": _sem(
            "duplicate the frame header and set the window length",
            ["clone then reslice"],
            ["copy plane headers into a new slice and store the window length"],
            [
                _rel("source plane headers", "independent header slice", "dataflow"),
                _rel("window length", "later per-plane reslice", "semantic_dependency"),
            ],
        ),
    },
    {
        "line": 8100,
        "note": "length dispatch is an if-else chain; the widths are sums so the literals 32/36/38/41/45 disappear",
        "prefix": """func (id *token) readText(raw []byte) error {
	n := len(raw)
	packed, dashed, wrapped := 30+2, 30+6, 30+8
	namedA, namedB := 40+1, 40+5
	if n == packed {

""",
        "middle": """		return id.readPacked(raw)
""",
        "suffix": """	} else if n == dashed {
		return id.readDashed(raw)
	} else if n == wrapped {
		return id.readWrapped(raw)
	} else if n == namedA || n == namedB {
		return id.readNamed(raw)
	}
	return fail("bad token width")
}
""",
        "semantic": _sem(
            "send the compact 32-character form to the packed decoder",
            ["dispatch a decoder by text length"],
            ["call the packed decoder for the shortest form"],
            [
                _rel("text length of the compact form", "packed decoder", "control"),
                _rel("other lengths", "the other decoders after this arm", "semantic_dependency"),
            ],
        ),
    },
    {
        "line": 9929,
        "note": "refusals are rows in a table scanned by one loop; the parent is a chain of ifs",
        "prefix": """func (pkt *hello) check() byte {
	legacy := pkt.banner == ("MQ" + "Isdp") && pkt.rev != (2+1)
	modern := pkt.banner == ("MQ" + "TT") && pkt.rev != (2+2)
	rules := []struct {
		tripped bool
		code    byte
	}{
		{pkt.secretSet && !pkt.nameSet, refuseAuth},

""",
        "middle": """		{pkt.unusedBit != 0, refuseFraming},
		{legacy || modern, refuseRev},
""",
        "suffix": """		{pkt.banner != ("MQ"+"Isdp") && pkt.banner != ("MQ"+"TT"), refuseFraming},
		{len(pkt.whoami) > (1<<16-1) || len(pkt.who) > (1<<16-1) || len(pkt.secret) > (1<<16-1), refuseFraming},
		{len(pkt.whoami) == 0 && !pkt.fresh, refuseName},
	}
	for _, rule := range rules {
		if rule.tripped {
			return rule.code
		}
	}
	return admit
}
""",
        "semantic": _sem(
            "reject a set reserved bit as a framing error, then start the version check",
            ["flag rejection before a protocol-version gate"],
            ["return the framing refusal", "open the mismatched-version test"],
            [
                _rel("set reserved bit", "framing refusal", "control"),
                _rel("banner and revision", "version refusal that follows", "semantic_dependency"),
            ],
        ),
    },
    {
        "line": 10065,
        "note": "watched calls are recorded under different names; the wildcard-address test is code, not the original regex",
        "prefix": """func freshOpenEveryIface(tag string, book opts) (policy, []kind) {
	watched := newCallSet()

""",
        "middle": """	watched.note("socket", "Open")
""",
        "suffix": """	watched.note("secure", "Open")
	policy := &openEveryIface{}
	policy.setCalls(watched)
	policy.setHost(func(addr string) bool {
		wild := "0" + ".0.0.0"
		return addr == wild || (len(addr) > 0 && addr[0] == ':')
	})
	policy.setBlurb("accepts on each iface")
	policy.setTag(tag)
	return policy, []kind{callNode}
}
""",
        "semantic": _sem(
            "record the plain listen entrypoint as a watched call",
            ["register a call site before a sibling secure listen"],
            ["add the plain listen API to the watched set"],
            [
                _rel("plain listen entrypoint", "watched call set", "api"),
                _rel("watched set", "later secure listen registration and wildcard-address test", "semantic_dependency"),
            ],
        ),
    },
    {
        "line": 6494,
        "note": "the parent loops; this folds with a function literal and joins path parts",
        "prefix": """func (group *check) noteFaults(extra, extraLayout string, faults []broken) {
	group.list = fold(faults, func(item broken) broken {

""",
        "middle": """		cast := item.(broken)
		cast.path = joinParts(group.path, extra, cast.path)
		cast.layoutPath = joinParts(group.layout, extraLayout, cast.layoutPath)
		return cast
""",
        "suffix": """	})
}
""",
        "semantic": _sem(
            "cast each failure and prefix both of its namespace strings",
            ["relative namespace prepended onto an existing path"],
            ["cast the failure", "prefix the value path", "prefix the layout path"],
            [
                _rel("relative namespace", "existing failure path", "transform"),
                _rel("prefixed failure", "accumulated failure list", "dataflow"),
            ],
        ),
    },
    {
        "line": 7297,
        "note": "the parent is a for/if; this walks with a callback. Identifiers do not reuse file/src/pkgName",
        "prefix": """func (tool *scan) loadBundle(units []namedSrc) ([]finding, error) {
	bundle := &bundle{units: map[string]*unit{}}
	var saved string
	fault := walk(units, func(path string, body []byte) error {
		if generated(body) {
			return nil
		}
		unit, fault := parseUnit(path, body)
		if fault != nil {
			return fault
		}
		switch saved {
		case "":

""",
        "middle": """			saved = unit.pkgLabel()
""",
        "suffix": """		default:
			if unit.pkgLabel() != saved {
				return mismatch(path, unit.pkgLabel(), saved)
			}
		}
		bundle.units[path] = &unit{owner: bundle, unit: unit, body: body, path: path}
		return nil
	})
	if fault != nil {
		return nil, fault
	}
	if len(bundle.units) == 0 {
		return nil, nil
	}
	return bundle.report(), nil
}
""",
        "semantic": _sem(
            "capture the package name from the first parsed file",
            ["first value becomes the expected identity for later checks"],
            ["store the parsed package name when none is saved yet"],
            [
                _rel("package name of the first file", "saved expected name", "dataflow"),
                _rel("saved name", "mismatch check on later files", "semantic_dependency"),
            ],
        ),
    },
    {
        "line": 9880,
        "note": "span search is a callback, not the parent's for loop; start-cell failure still returns its own sentinel",
        "prefix": """func (spans *merges) spanOf(label string) (int, int, error) {
	const badStart = 0 - 1
	const badEnd = 0 - 2
	if spans == nil {
		return 0, 0, nil
	}
	return findSpan(spans.blocks, func(block area) (int, int, error, bool) {
		if !hasHead(block.mark, label+spanSep) {
			return 0, 0, nil, false
		}
		chunks := splitOnce(block.mark, spanSep)
		x0, y0, fault := coords(chunks[0])
		if fault != nil {

""",
        "middle": """			return badStart, badStart, fault, true
""",
        "suffix": """		}
		x1, y1, fault := coords(chunks[1])
		if fault != nil {
			return badEnd, badEnd, fault, true
		}
		return x1 - x0, y1 - y0, nil, true
	})
}
""",
        "semantic": _sem(
            "surface a start-cell coordinate failure with its own sentinel pair",
            ["distinct numeric sentinel per failing endpoint"],
            ["return the start-cell sentinel twice plus the parse error"],
            [
                _rel("start-cell parse error", "start sentinel pair", "error"),
                _rel("end-cell failure", "a different sentinel later", "semantic_dependency"),
            ],
        ),
    },
    {
        "line": 10755,
        "note": "the parent switches on keys; this inserts a plain glyph then dispatches the rest through a map of functions",
        "prefix": """func onKey(pane *edit, code keyCode, glyph int32, mods keyCode) {
	if glyph != 0 && mods == 0 {

""",
        "middle": """		pane.insert(glyph)
		return
""",
        "suffix": """	}
	actions := map[keyCode]func(){
		keyBlank:  func() { pane.insert(' ') },
		keyRub:    func() { pane.rub(true) },
		keyRub2:   func() { pane.rub(true) },
		keyCut:    func() { pane.rub(false) },
		keyToggle: func() { pane.over = !pane.over },
		keyLine:   func() { pane.breakLine() },
		keyDown:   func() { pane.step(0, 1) },
		keyUp:     func() { pane.step(0, -1) },
		keyLeft:   func() { pane.step(-1, 0) },
		keyRight:  func() { pane.step(1, 0) },
	}
	if act, hit := actions[code]; hit {
		act()
	}
}
""",
        "semantic": _sem(
            "insert a plain character into the editor",
            ["printable input writes itself"],
            ["write the character into the pane"],
            [
                _rel("plain character with no modifier", "inserted text", "dataflow"),
                _rel("other keys", "the edit actions in later arms", "semantic_dependency"),
            ],
        ),
    },
    {
        "line": 10796,
        "note": "character classes are ifs into a byte buffer; lowercase still passes through",
        "prefix": """func cleanIdent(label string) string {
	out := make([]byte, 0, len(label))
	for _, ch := range label {
		if ch >= '0' && ch <= '9' {
			out = append(out, byte(ch))
		} else if ch >= 'a' && ch <= 'z' {

""",
        "middle": """			out = append(out, byte(ch))
""",
        "suffix": """		} else if ch >= 'A' && ch <= 'Z' {
			out = append(out, byte(ch))
		} else {
			out = append(out, '_')
		}
	}
	return string(out)
}
""",
        "semantic": _sem(
            "keep a lowercase letter in the sanitized identifier",
            ["pass an allowed character class through unchanged"],
            ["append the lowercase letter as itself"],
            [
                _rel("lowercase letter", "sanitized identifier bytes", "dataflow"),
                _rel("other letters and digits", "the same pass-through in sibling arms", "semantic_dependency"),
            ],
        ),
    },
    {
        "line": 6853,
        "note": "controller list is built by append; the hole narrows to the bus controller with a two-step assert",
        "prefix": """func busKinds() deviceList {
	items := deviceList{}
	items = append(items, &logicBoard{})
	items = append(items, &legacyBoard{})
	items = append(items, &virtBoard{})
	items = append(items, &sasBoard{})
	return items.keep(func(dev baseDevice) bool {

""",
        "middle": """		narrow, _ := dev.(busController)
		board := narrow.board()
""",
        "suffix": """		board.shared = shareNone
		board.index = -1
		return true
	})
}
""",
        "semantic": _sem(
            "narrow the device to its bus-controller object",
            ["type narrowing before bus defaults"],
            ["assert the bus controller and read the concrete board"],
            [
                _rel("generic device", "concrete bus board", "transform"),
                _rel("concrete board", "sharing mode and bus index set afterward", "semantic_dependency"),
            ],
        ),
    },
    {
        "line": 9928,
        "note": "reads are a slice of function steps plus one runner loop; the parent is a long if-err chain",
        "prefix": """func (pkt *hello) read(in stream) error {
	var fault error
	var flags byte
	steps := []func(){
		func() { pkt.banner, fault = takeText(in) },
		func() { pkt.rev, fault = takeU8(in) },
		func() {
			flags, fault = takeU8(in)
			pkt.unused = flags & 1
			pkt.fresh = flags&(1<<1) != 0
			pkt.willOn = flags&(1<<2) != 0
			pkt.willRank = (flags >> 3) & 3
			pkt.willKeep = flags&(1<<5) != 0
			pkt.secretOn = flags&(1<<6) != 0
		},
		func() {

""",
        "middle": """			pkt.nameOn = flags&128 != 0
			pkt.ticks, fault = takeU16(in)
""",
        "suffix": """		},
		func() { pkt.whoami, fault = takeText(in) },
		func() {
			if pkt.willOn {
				pkt.willPath, fault = takeText(in)
			}
		},
		func() {
			if pkt.willOn && fault == nil {
				pkt.willBody, fault = takeBlob(in)
			}
		},
		func() {
			if pkt.nameOn {
				pkt.who, fault = takeText(in)
			}
		},
		func() {
			if pkt.secretOn {
				pkt.secret, fault = takeBlob(in)
			}
		},
	}
	for _, step := range steps {
		step()
		if fault != nil {
			return fault
		}
	}
	return nil
}
""",
        "semantic": _sem(
            "read the name-present flag from the high bit, then the keepalive",
            ["bitfield flag followed by the next integer field"],
            ["test bit 7 as the name-present flag", "read the following 16-bit keepalive"],
            [
                _rel("high bit of the flag byte", "name-present flag", "transform"),
                _rel("name-present flag", "later conditional read of the name", "semantic_dependency"),
                _rel("following two bytes", "keepalive", "dataflow"),
            ],
        ),
    },
    {
        "line": 10702,
        "note": "state machine is if/else; the reply bit is OR-assigned with a plain expression",
        "prefix": """func (flow *lane) headerBits() uint16 {
	flow.mu.hold()
	defer flow.mu.drop()
	var bits uint16
	if flow.phase == phaseNew {
		bits = bits | bitOpen
		flow.phase = phaseOpenSent
	} else if flow.phase == phaseOpenSeen {

""",
        "middle": """		bits = bits | bitReply
""",
        "suffix": """		flow.phase = phaseReady
	}
	return bits
}
""",
        "semantic": _sem(
            "add the reply bit once the open has been seen",
            ["flag set on a half-open state transition"],
            ["OR the reply bit into the header flags"],
            [
                _rel("open-seen phase", "reply bit", "control"),
                _rel("reply bit", "move to the ready phase", "semantic_dependency"),
            ],
        ),
    },
    {
        "line": 11128,
        "note": "illegal characters are dropped by a rune walk, not a character-class regex",
        "prefix": """func plainName(raw string) string {
	label := lower(raw)
	label = basePart(cleanPath(label))
	label = trimSpace(label)
	var tmp []byte
	for _, ch := range label {
		switch ch {
		case ' ', '&', '_', '=', '+', ':':
			tmp = append(tmp, '-')
		default:
			tmp = append(tmp, byte(ch))
		}
	}
	label = string(tmp)
	kept := make([]byte, 0, len(label))
	for _, ch := range label {
		okLetter := (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9')
		okMark := ch == '-' || ch == '.'

""",
        "middle": """		if okLetter || okMark {
			kept = append(kept, byte(ch))
		}
	}
	label = string(kept)
	for hasSub(label, "--") {

""",
        "suffix": """		label = swap(label, "--", "-")
	}
	return label
}
""",
        "semantic": _sem(
            "drop characters outside the allowed filename set, then start collapsing doubled dashes",
            ["whitelist filter before repeated-separator collapse"],
            ["keep letters, digits, dash, and dot", "begin the doubled-dash scan"],
            [
                _rel("disallowed character", "dropped from the name", "transform"),
                _rel("filtered name", "later collapse of doubled dashes", "semantic_dependency"),
            ],
        ),
    },
    {
        "line": 7347,
        "note": "decode stays outside the hole; the certificate is parsed inside a composite install, not an if-err assignment",
        "prefix": """func (c *peer) loadAnchor(armored string) error {
	raw, fault := decodeArmor(armored)
	if fault != nil {
		return fault
	}
	c.cfg = cfg{roots: newTrustSet().add(

""",
        "middle": """		parseSigned(raw),
""",
        "suffix": """	)}
	return fault
}
""",
        "semantic": _sem(
            "turn decoded certificate bytes into a certificate object",
            ["parse a decoded certificate before installing it as a trust root"],
            ["parse the decoded bytes into a certificate"],
            [
                _rel("decoded certificate bytes", "certificate object", "transform"),
                _rel("certificate object", "trust-root set updated afterward", "semantic_dependency"),
            ],
        ),
    },
    {
        "line": 7538,
        "note": "the search is one call; the hole still loads the candidate span for the containment test",
        "prefix": """func locateSpan(
	edge func(i int) addr,
	whole func(i int) span,
	n int,
	probe addr,
) (hit span, found bool) {
	lo := lowerBound(edge, n, probe)
	switch {
	case lo == n:
		return
	default:

""",
        "middle": """		hit = whole(lo)
""",
        "suffix": """		found = !after(hit.lo, probe) && !after(probe, hit.hi)
		return
	}
}
""",
        "semantic": _sem(
            "materialize the candidate span at the located index",
            ["lookup result bound for a following containment test"],
            ["load the full span at the search index"],
            [
                _rel("search index", "candidate span", "dataflow"),
                _rel("candidate span", "containment test against the probe", "semantic_dependency"),
            ],
        ),
    },
    {
        "line": 8508,
        "note": "newline normalization is a byte walk, not a replacer of \\\\n to \\\\r\\\\n",
        "prefix": """func (p *bridge) emitText(which int, text string) (int, error) {
	sink := p.out
	if which == errStream {
		sink = p.errOut
	}

""",
        "middle": """	raw := []byte(text)
	buf := make([]byte, 0, len(raw)+8)
	for i := 0; i < len(raw); i++ {
		if raw[i] == '\\n' && (i == 0 || raw[i-1] != '\\r') {
			buf = append(buf, '\\r')
		}
		buf = append(buf, raw[i])
	}
	n, fault := sink.push(buf)
""",
        "suffix": """	if fault != nil {
		return n, fault
	}
	return n, nil
}
""",
        "semantic": _sem(
            "normalize newlines to CRLF and write them to the selected stream",
            ["expand bare line feeds before the write"],
            ["insert a carriage return before a bare line feed", "write the normalized bytes"],
            [
                _rel("bare line feed", "carriage-return line-feed pair", "transform"),
                _rel("normalized bytes", "selected output stream", "dataflow"),
            ],
        ),
    },
    {
        "line": 7298,
        "note": "banner scan splits lines itself; the hole still binds the current line for the marker test",
        "prefix": """func marked(body []byte) bool {
	for _, line := range splitLines(body) {

""",
        "middle": """		b := line
""",
        "suffix": """		head, tail := genHead, genTail
		if hasHead(b, head) && hasTail(b, tail) && len(b) >= len(head)+len(tail) {
			return true
		}
	}
	return false
}
""",
        "semantic": _sem(
            "bind the current line for the generated-file marker test",
            ["current line captured before a prefix-and-suffix banner check"],
            ["assign the current line to the buffer the marker test reads"],
            [
                _rel("current line", "banner prefix and suffix test", "dataflow"),
            ],
        ),
    },
    {
        "line": 6864,
        "note": "controller dispatch is a map of functions; alias strings are concatenated",
        "prefix": """func (list deviceList) pick(label string) (baseBoard, error) {
	parallel, serial, fast := "i"+"de", "s"+"csi", "n"+"vme"
	table := map[string]func() (baseBoard, error){
		parallel: func() (baseBoard, error) {

""",
        "middle": """			return list.pickParallel("")
""",
        "suffix": """		},
		serial: func() (baseBoard, error) { return list.pickSerial("") },
		"":     func() (baseBoard, error) { return list.pickSerial("") },
		fast:   func() (baseBoard, error) { return list.pickFast("") },
	}
	if act, hit := table[label]; hit {
		return act()
	}
	if board, hit := list.lookup(label).(baseBoard); hit {
		return board, nil
	}
	return nil, boom("unknown board")
}
""",
        "semantic": _sem(
            "select the parallel-bus finder when the name is that alias",
            ["name dispatch to a controller finder"],
            ["call the parallel-bus finder with an empty extra name"],
            [
                _rel("parallel-bus alias", "parallel-bus finder", "control"),
                _rel("other aliases", "the serial and fast finders in later arms", "semantic_dependency"),
            ],
        ),
    },
    {
        "line": 11866,
        "note": "the last-key arm uses a differently named cursor method; the key is decoded by shifts, not BigEndian",
        "prefix": """func (log *ledger) edgeKey(wantHead bool) (uint64, error) {
	batch, fault := log.db.start(false)
	if fault != nil {
		return 0, fault
	}
	var (
		raw []byte
		ord uint64
	)
	cur := batch.bin(logBucket).walker()
	if wantHead {
		raw, _ = cur.head()
	} else {

""",
        "middle": """		raw, _ = cur.final()
	}
	if raw != nil {

""",
        "suffix": """		ord = uint64(raw[0])<<56 | uint64(raw[1])<<48 | uint64(raw[2])<<40 | uint64(raw[3])<<32 |
			uint64(raw[4])<<24 | uint64(raw[5])<<16 | uint64(raw[6])<<8 | uint64(raw[7])
	}
	batch.abort()
	return ord, nil
}
""",
        "semantic": _sem(
            "take the last key when the caller did not ask for the first",
            ["cursor edge selection"],
            ["read the last cursor key", "open the present-key decode"],
            [
                _rel("request for the tail edge", "last cursor key", "control"),
                _rel("present key", "integer decoded afterward", "semantic_dependency"),
            ],
        ),
    },
    {
        "line": 11588,
        "note": "protocol selection is if/else; tcp/udp/ip literals are split",
        "prefix": """func (t *peer) wireName() string {
	ver := 4
	if t.addr.four() == nil {
		ver = 6
	}
	var label string
	if t.kind == protoEcho4 {
		label = "i" + "p4:" + "echo"
	} else if t.kind == protoEcho6 {
		label = "i" + "p6:" + "echo6"
	} else if t.kind == protoStream {

""",
        "middle": """		label = ("t" + "cp") + digit(ver)
""",
        "suffix": """	} else if t.kind == protoPacket {
		label = ("u" + "dp") + digit(ver)
	} else {
		return "(none)"
	}
	return label
}
""",
        "semantic": _sem(
            "name the stream protocol with the address family version",
            ["protocol kind formatted with the IP version"],
            ["build the stream-protocol name plus the version digit"],
            [
                _rel("stream protocol and IP version", "versioned protocol name", "transform"),
                _rel("packet protocol", "the sibling versioned name", "semantic_dependency"),
            ],
        ),
    },
    {
        "line": 10331,
        "note": "four-channel conversion is a direct call; the empty-color fast path compares fields, not a struct literal",
        "prefix": """func blankCanvas(w, h int, fill shade) *raster {
	if w <= 0 || h <= 0 {
		return &raster{}
	}

""",
        "middle": """	px := toFour(fill)
""",
        "suffix": """	if px.r == 0 && px.g == 0 && px.b == 0 && px.a == 0 {
		return freshRaster(w, h)
	}
	dots := repeatFour(px.r, px.g, px.b, px.a, w*h)
	return &raster{dots: dots, step: 4 * w, w: w, h: h}
}
""",
        "semantic": _sem(
            "convert the fill color into four channels",
            ["color normalization before an empty-color fast path"],
            ["convert the fill color to four channel bytes"],
            [
                _rel("fill color", "four channel bytes", "transform"),
                _rel("four channel bytes", "empty-color fast path and later pixel fill", "semantic_dependency"),
            ],
        ),
    },
    {
        "line": 11643,
        "note": "word summing is recursive ifs, not the parent's for-loops; the hole is still the ones-complement",
        "prefix": """func foldSum(b []byte) uint16 {
	var walk func(i int, acc uint32) uint32
	walk = func(i int, acc uint32) uint32 {
		switch {
		case i >= len(b) && acc > 65535:
			return walk(len(b), acc%65536+acc/65536)
		case i >= len(b):
			return acc
		case i+1 == len(b):
			return walk(i+1, acc+uint32(b[i])*256)
		default:
			return walk(i+2, acc+uint32(b[i])*256+uint32(b[i+1]))
		}
	}
	acc := walk(0, 0)

""",
        "middle": """	low := uint16(acc)
	return low ^ 65535
""",
        "suffix": """}
""",
        "semantic": _sem(
            "return the ones-complement of the folded 16-bit sum",
            ["ones-complement checksum finalization"],
            ["truncate the folded sum and invert its 16 bits"],
            [
                _rel("folded 16-bit sum", "ones-complement result", "transform"),
            ],
        ),
    },
    {
        "line": 7474,
        "note": "parent assigns six matrix fields; this builds one composite after the unit-circle pair",
        "prefix": """func (sheet *canvas) turn(deg, x, y float64) {
	y = (sheet.h - y) * sheet.unit
	x = x * sheet.unit
	rad := deg * (3.141592653589793 / 180)

""",
        "middle": """	cx, sy := circle(rad)
""",
        "suffix": """	sheet.apply(matrix{cx, sy, -sy, cx, x + sy*y - cx*x, y - cx*y - sy*x})
}
""",
        "semantic": _sem(
            "store the unit-circle pair for the rotation angle",
            ["rotation matrix from an angle"],
            ["bind the adjacent and opposite unit-circle components"],
            [
                _rel("angle in radians", "unit-circle pair", "transform"),
                _rel("unit-circle pair", "opposite signs and translation filled afterward", "semantic_dependency"),
            ],
        ),
    },
    {
        "line": 12436,
        "note": "metacharacter escaping appends byte 92, and the scan is an if chain rather than a switch",
        "prefix": """func shieldGlob(glob string) string {
	needs := false
	for _, ch := range glob {
		if ch == '*' || ch == '?' || ch == '[' || ch == '\\\\' {
			needs = true
			break
		}
	}
	if !needs {
		return glob
	}
	out := make([]byte, 0, len(glob)*2)
	for _, ch := range glob {
		if ch == '*' || ch == '?' || ch == '[' || ch == '\\\\' {

""",
        "middle": """			out = append(out, 92)
""",
        "suffix": """		}
		out = append(out, byte(ch))
	}
	return string(out)
}
""",
        "semantic": _sem(
            "prefix a backslash before a glob metacharacter",
            ["escape a metacharacter before copying it"],
            ["append a backslash byte"],
            [
                _rel("glob metacharacter", "leading backslash", "transform"),
                _rel("escaped character", "the character copied immediately after", "semantic_dependency"),
            ],
        ),
    },
    {
        "line": 16404,
        "note": "parent is an if chain; this switches. Hex digits are written by hand, not printf",
        "prefix": """func quoteBlob(sink *buf, raw []byte) {
	sink.put('"')
	hexd := "0123456789abcdef"
	for _, b := range raw {
		switch {
		case b == '\\n':
			sink.put('\\\\')
			sink.put('n')
		case b == '\\\\':
			sink.put('\\\\')
			sink.put('\\\\')
		case b == '"':
			sink.put('\\\\')
			sink.put('"')
		case (b >= 32 && b <= 126) || b == '\\t':
			sink.put(b)
		default:

""",
        "middle": """			sink.put('\\\\')
			sink.put('x')
			sink.put(hexd[b>>4])
			sink.put(hexd[b&15])
""",
        "suffix": """		}
	}
	sink.put('"')
}
""",
        "semantic": _sem(
            "write a non-printable byte as a hex escape",
            ["hex escape for a byte outside the plain range"],
            ["write a backslash, x, and two hex digits"],
            [
                _rel("non-printable byte", "two hex digits", "transform"),
                _rel("hex escape", "quoted byte string", "dataflow"),
            ],
        ),
    },
]


def content_tokens(text: str) -> list[str]:
    out = []
    for raw in _IDENT.findall(text):
        tok = raw.casefold()
        if tok in STOP or len(tok) <= 1:
            continue
        out.append(tok)
    return out


def cosine(a: dict[str, int], b: dict[str, int]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    for key, av in a.items():
        bv = b.get(key)
        if bv:
            dot += float(av) * float(bv)
    na = sum(float(v) * float(v) for v in a.values()) ** 0.5
    nb = sum(float(v) * float(v) for v in b.values()) ** 0.5
    if na <= 0 or nb <= 0:
        return 0.0
    return float(dot / (na * nb))


def hist(tokens: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
    return counts


def _parser():
    from tree_sitter import Language, Parser
    import tree_sitter_go as tsgo

    lang = Language(tsgo.language())
    try:
        return Parser(lang)
    except TypeError:
        parser = Parser()
        parser.set_language(lang)
        return parser


def ast_hists(parser, code: str) -> tuple[dict[str, int], dict[str, int], bool]:
    tree = parser.parse(code.encode("utf-8"))
    types: dict[str, int] = {}
    edges: dict[str, int] = {}
    bad = False

    def walk(node, parent: str | None = None) -> None:
        nonlocal bad
        name = node.type or ""
        if name == "ERROR" or node.is_missing:
            bad = True
        if name:
            types[name] = types.get(name, 0) + 1
        if parent and name:
            key = f"{parent}->{name}"
            edges[key] = edges.get(key, 0) + 1
        for child in node.children:
            walk(child, name)

    walk(tree.root_node)
    return types, edges, bad or tree.root_node.has_error


def rank_of(parent_score: float, others: list[float]) -> int:
    """1 means the parent is the nearest (worst for this experiment)."""
    return 1 + sum(1 for score in others if score > parent_score)


def make_prompt(prefix: str, suffix: str) -> str:
    return (
        "This is a go programming task based on some code contexts.\n"
        "### Given Task:\n"
        "The task is to fill in the missing part of a function according to the provided code context. "
        "And the missing part is marked with <MID>.\n\n"
        "And here is the function you are asked to complete:\n"
        "```go\n"
        f"<PRE> {prefix}<SUF> {suffix}<MID>\n"
        "```\n\n"
        "Ensure that only missing codes marked as <MID> are returned.\n"
        "### Response:"
    )


def full_code(row: dict) -> str:
    return f"{row.get('prefix') or ''}{row.get('middle') or ''}{row.get('suffix') or ''}"


def main() -> None:
    raise SystemExit(
        "These hand-written line numbers index csn_go_train_fim.jsonl. "
        "Rebuild the 10k attribution set with mutation_attribution/expand_to_100.py."
    )


if __name__ == "__main__":
    main()
