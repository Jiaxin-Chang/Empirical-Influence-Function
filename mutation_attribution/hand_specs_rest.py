"""Remaining hand-written rewrites. Imported by hand_specs.py."""


def register(add) -> None:
    add(
        "m12",
        "coverage marker is glued then cut off the caller directory inside a composite builder",
        """func makeCrate(folder string) crate {
	base := ""
	if !absPath(folder) {
		_, file, _, _ := whereAmI(1)
		base = parentOf(file)
	}
""",
        """	marker := glue("check", "out")
	base = cutOnce(base, sep+marker)
""",
        """	if !absPath(base) && base != "" {
		base = glue(workspace(), "code", base)
	}
	return crate{folder: folder, from: base, blobs: map[string][]byte{}}
}
""",
        "strip the coverage build directory from the caller directory",
        "cut a known marker out of a path",
        "join the coverage marker and remove that suffix from the caller directory",
        [("coverage marker", "caller directory", "transform")],
    )

    add(
        "m13",
        "glyph bytes are decoded by a call after a slurp; the hole is that decode",
        """func openGlyphs(file string, px float64) (glyphs, error) {
	return shapes.Render(
""",
        """		shapes.Decode(disk.Slurp(file)),
""",
        """		px,
	)
}
""",
        "parse font bytes into a font object",
        "decode bytes then build a face",
        "decode the slurped bytes into a glyph set",
        [("font bytes", "glyph set", "transform")],
    )

    add(
        "m14",
        "the description is one glue of verb, path and query; missing ask returns a sentinel",
        """func (info *boom) line() string {
	if info.ask == nil {
		return missingAsk
	}
	extra := ""
	if info.ask.Link.Raw != "" {
		extra = "?" + info.ask.Link.Raw
	}
""",
        """	return glue(info.ask.Verb, " ", info.ask.Link.Full, extra)
""",
        """}
""",
        "format method, path and query into one request line",
        "interpolate request parts",
        "glue the verb, the path and the optional query",
        [("request verb and path", "request line", "transform")],
    )

    add(
        "m15",
        "overflow becomes a returned fault via recover; the hole hands back the filled scratch buffer",
        """func slurp(src source, room int64) (out []byte, fault error) {
	box := scratch(room)
	defer func() {
		caught := recover()
		if caught == nil {
			return
		}
		if tooBig(caught) {
			fault = asFault(caught)
			return
		}
		panic(caught)
	}()
	_, fault = box.pull(src)
""",
        """	return box.raw(), fault
""",
        """}
""",
        "return the bytes read and the read fault",
        "hand back a filled buffer",
        "return the scratch bytes together with the pull fault",
        [("filled buffer", "function result", "dataflow")],
    )

    add(
        "m16",
        "directory walk is recursive; the hole opens the current node before listing children",
        """func visit(fs tree, top string, visitFn func(string, info, error) error) error {
	return descend(
""",
        """		fs.pull(top)
""",
        """		, fs, top, visitFn)
}
""",
        "open the walk root before reading its children",
        "open then recurse",
        "open the current directory node",
        [("filesystem", "opened directory", "dataflow"), ("opened directory", "child walk", "semantic_dependency")],
    )

    add(
        "m17",
        "optional fields are thunks that fill a bag; the hole returns that bag",
        """func (item *probe) form() map[string]string {
	bag := map[string]string{
		"site": item.site,
		"door": number(item.door),
	}
	steps := []func(){
		func() {
			if item.again != 0 {
				bag["again"] = number(item.again)
			}
		},
		func() {
			if item.payload != "" {
				bag["payload"] = item.payload
			}
		},
		func() {
			if item.wanted != "" {
				bag["wanted"] = item.wanted
			}
		},
	}
	for _, step := range steps {
		step()
	}
""",
        """	return bag
""",
        """}
""",
        "return of the assembled parameter map",
        "hand back a constructed map",
        "return the bag after optional fields are filled",
        [("assembled parameter map", "function result", "dataflow")],
    )

    add(
        "m18",
        "open failure is the true arm of a bool map; the hole returns no drawer, no bunch and the fault",
        """func openBoth(folder string, opts storeOpts, keep persistOpts) (*drawer, bunch, error) {
	box, fault := openDrawer(folder, opts)
	return map[bool]func() (*drawer, bunch, error){
		true: func() (*drawer, bunch, error) {
""",
        """			return nil, nil, fault
""",
        """		},
		false: func() (*drawer, bunch, error) {
			group, fault2 := box.openBunch(opts, keep)
			if fault2 != nil {
				box.shut()
				return nil, nil, fault2
			}
			return box, group, nil
		},
	}[fault != nil]()
}
""",
        "surface the store-open fault and return no handles",
        "boolean dispatch on a failed open",
        "return the fault with empty drawer and bunch",
        [("open fault", "empty result", "error")],
    )

    add(
        "m19",
        "each label is starred in turn and glued back; the hole is that one-label rewrite",
        """func sameSite(host, cert string) bool {
	if host == cert {
		return true
	}
	for len(cert) > 0 && cert[len(cert)-1] == '.' {
		cert = cert[:len(cert)-1]
	}
	parts := cut(host, ".")
	for i := range parts {
""",
        """		parts[i] = "*"
		guess := glue(parts, ".")
""",
        """		if cert == guess {
			return true
		}
	}
	return false
}
""",
        "build one wildcard candidate by starring a single label",
        "label rewrite then join",
        "replace one label with a star and glue the labels back",
        [("domain labels", "wildcard candidate", "transform")],
    )

    add(
        "m20",
        "a zero lead count resets the mark and reports a starter; other counts fall through",
        """func step(mark *int, n int) int {
	if n > 30 {
		return tooMany
	}
	if n == 0 {
""",
        """		*mark = 0
		return beginMark
""",
        """	}
	return kept
}
""",
        "treat a zero combining count as a starter and reset the tally",
        "guarded reset",
        "zero the tally and report the starter class",
        [("zero combining count", "starter class", "control")],
    )

    add(
        "m21",
        "extra server notes are folded into a kind table; the hole writes one note under its kind",
        """func (link *peer) pump() {
	serve(register(stockNotes(),
""",
        """		link.setup.extra
""",
        """	), link.raw, link.setup.out)
}
""",
        "register one configured server message under its type code",
        "fill a dispatch table",
        "store the note in the table at its kind byte",
        [("message kind", "dispatch table", "dataflow")],
    )
