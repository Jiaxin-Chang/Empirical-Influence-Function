"""Shorter skeletons for long parents that still matched the AST histogram."""


def register(add) -> None:
    add(
        "m02",
        "refusals are flat picker arguments; the hole is the unused-bit refusal and the revision test",
        """func (pkt *hello) check() byte {
	return pick(
		when(pkt.secretSet && !pkt.nameSet, refuseAuth),
""",
        """		when(pkt.unusedBit != 0, refuseFraming),
		when(badRev(pkt), refuseRev),
""",
        """		when(badBanner(pkt), refuseFraming),
		when(tooWide(pkt) || blankWho(pkt), refuseFraming),
		when(true, admit),
	)
}
""",
        "reject a set reserved bit as a framing error, then start the version check",
        "ordered refusal rules",
        "emit the framing refusal for a nonzero unused bit and open the revision test",
        [("unused bit", "framing refusal", "control"), ("revision mismatch", "revision refusal", "control")],
    )
    add(
        "m10",
        "options are applied by one fold call",
        """func makeClient(choices ...chooser) *actor {
	built := actor{budget: defaultWait, retries: defaultTries, again: freshNoRetry()}
""",
        """	fold(choices, &built)
""",
        """	if built.raw == nil {
		built.raw = &web.Agent{Limit: built.budget}
	}
	return &built
}
""",
        "apply the configuration callbacks to the client under construction",
        "fold options into a value",
        "apply every callback to the client",
        [("option callbacks", "client under construction", "dataflow")],
    )
    add(
        "m12",
        "the coverage marker is cut off the caller directory by one call",
        """func makeCrate(folder string) crate {
	base := callerDir(folder)
""",
        """	base = cutOnce(base, glue(sep, "check", "out"))
""",
        """	return crate{folder: folder, from: absOrWorkspace(base), blobs: map[string][]byte{}}
}
""",
        "strip the coverage build directory from the caller directory",
        "cut a known marker out of a path",
        "remove the coverage marker from the caller directory",
        [("coverage marker", "caller directory", "transform")],
    )
    add(
        "m19",
        "wildcard candidates are produced by one helper over the labels",
        """func sameSite(host, cert string) bool {
	if host == cert {
		return true
	}
""",
        """	return anyStar(cut(host, "."), trimDot(cert))
""",
        """}
""",
        "match a certificate domain by starring one label at a time",
        "label rewrite then compare",
        "test starred label candidates against the certificate domain",
        [("domain labels", "wildcard candidate", "transform")],
    )
    add(
        "m20",
        "a zero lead count resets the mark and reports a starter",
        """func step(mark *int, n int) int {
	return chooseCount(n, func() int {
""",
        """		*mark = 0
		return beginMark
""",
        """	}, tooMany, kept)
}
""",
        "treat a zero combining count as a starter and reset the tally",
        "guarded reset",
        "zero the tally and report the starter class",
        [("zero combining count", "starter class", "control")],
    )
    add(
        "m26",
        "the plain pool is built and then wrapped",
        """func makeGreedy(hosts []string, fade span, calc calculator) pool {
""",
        """	plain := makePlain(hosts)
""",
        """	return wrapGreedy(plain, fadeOr(fade), calc)
}
""",
        "build the standard host pool that the greedy pool embeds",
        "construct a base then wrap it",
        "allocate the plain host pool from the host list",
        [("host list", "plain host pool", "init")],
    )
    add(
        "m27",
        "the reload socket path is one binding in a pair list",
        """func makeHub(title string, port uint16) *hub {
	return mount(title, port, []struct{ route string; fn func(*hub) }{
		{"/livereload.js", scriptHandler},
""",
        """		{"/livereload", socketHandler},
""",
        """	})
}
""",
        "route the reload socket path to its handler",
        "table of route bindings",
        "bind the reload path to the socket handler",
        [("reload path", "socket handler", "config")],
    )
    add(
        "m32",
        "the database link is opened, then the checkpoint object is finished around it",
        """func makeMark(app, table, dsn string, opts ...opt) (*mark, error) {
	if app == "" || table == "" {
		return nil, faultMissing("app or table")
	}
""",
        """	link, fault := openPG(dsn)
""",
        """	return finishMark(link, fault, app, table, opts)
}
""",
        "open the database from the connection string",
        "open a driver then build the object",
        "open the database link from the dsn",
        [("connection string", "database link", "init")],
    )
    add(
        "m34",
        "the bytes are summed, then packed by version",
        """func (p header) digest(data []byte) (cid, error) {
""",
        """	sum, fault := mh.Sum(data, p.mhKind, lengthOf(p))
""",
        """	return packSum(p, sum, fault)
}
""",
        "hash the bytes with the configured multihash type and length",
        "sum then wrap by version",
        "compute the multihash of the bytes",
        [("payload bytes", "multihash", "transform")],
    )
    add(
        "m36",
        "a vendor type is qualified only when the package is not the datatype package",
        """func goName(t, pkg string) string {
	return map[bool]func() string{
		true: func() string {
""",
        """			return "datatypes." + trimHead(t)
""",
        """		},
		false: func() string { return trimHead(t) },
	}[needsQualify(t, pkg)]()
}
""",
        "qualify a stripped type name with the datatypes package",
        "prefix a type name",
        "return the datatypes-qualified name",
        [("stripped type name", "qualified type name", "transform")],
    )
    add(
        "m38",
        "the function file is announced, then built",
        """func (b *buildCmd) run() error {
	fn, fault := findFuncHere()
	if fault != nil {
		return fault
	}
""",
        """	announce(b, fn)
""",
        """	return buildAnnounced(b, fn)
}
""",
        "log that a function file is being built",
        "announce before the build call",
        "write the building line for the function file",
        [("function file", "build log", "dataflow")],
    )
    add(
        "m39",
        "a blank service id refuses the update",
        """func (c *agent) updateGCS(in *gcsIn) (*gcs, error) {
	return map[bool]func() (*gcs, error){
		true: func() (*gcs, error) {
""",
        """			return nil, faultNoSvc
""",
        """		},
		false: func() (*gcs, error) { return c.putChecked(in) },
	}[in.svc == ""]()
}
""",
        "refuse the update when the service id is missing",
        "missing-field guard",
        "return the missing-service fault and no object",
        [("blank service id", "missing-service fault", "error")],
    )
    add(
        "m43",
        "the hello server name selects the host record",
        """func (a *armor) configFor(hello *tlsHello) (*tlsCfg, error) {
""",
        """	host := a.hosts[hello.serverName]
""",
        """	return cfgFor(a, hello, host)
}
""",
        "look up the host config by the TLS hello server name",
        "map lookup then maybe build",
        "index the host table with the server name",
        [("server name", "host record", "dataflow")],
    )
    add(
        "m50",
        "an empty request URI becomes the root path",
        """func rawRoute(r *httpReq) string {
""",
        """	return slashIfBlank(r.requestURI, r.URL.Path)
""",
        """}
""",
        "treat an empty request URI as the root path",
        "empty-string default",
        "map a blank URI to slash",
        [("empty request URI", "root path", "transform")],
    )
    add(
        "m51",
        "the host is one write into the connection text",
        """func connText(user, secret, host, db string) string {
	return glueParts(user, secret,
""",
        """		host,
""",
        """		db)
}
""",
        "write the host into the connection string",
        "append a component to a buffer",
        "place the host between credentials and the database name",
        [("host", "connection text", "dataflow")],
    )
    add(
        "m55",
        "an internal node continues at the next child's link",
        """func (t *tree) get(k uint64) (layer, bool) {
	return descend(t.root, func(q node, i int) node {
		if inner, isInner := q.(*inner); isInner {
""",
        """			return inner.kids[i+1].ch
""",
        """		}
		return nil
	}, k)
}
""",
        "follow the next child pointer of an internal node",
        "descend one tree level",
        "move the cursor to the following child's link",
        [("search key", "child link", "control")],
    )
    add(
        "m60",
        "unfinished jobs are put back when the worker stops",
        """func (w *worker) onStop(pid int) {
	noteStop(w, pid)
""",
        """	w.requeue()
""",
        """	w.quit()
}
""",
        "put unfinished jobs back on the queue when stopping",
        "requeue on shutdown",
        "requeue the unfinished jobs",
        [("stop", "job queue", "control")],
    )
    add(
        "m63",
        "headers, status and body are forwarded by one call",
        """func proxy(w writer, resp *httpResp) error {
""",
        """	return copyResp(w, resp)
""",
        """}
""",
        "copy response headers and status, then the body",
        "forward a response",
        "copy headers, write the status, copy the body",
        [("upstream response", "client writer", "dataflow")],
    )
    add(
        "m64",
        "a missing or unnamed key file is generated after a stat",
        """func (k *keys) ensure(nc netConn) error {
""",
        """	return genIfMissing(k, nc, statFile(k.file))
""",
        """}
""",
        "stat the key file and generate one when it is missing or unnamed",
        "stat then branch on absence",
        "stat the file and enter generation when it is absent",
        [("key file name", "stat result", "dataflow"), ("missing file", "generation", "control")],
    )
    add(
        "m66",
        "a byte outside the hex ranges contributes -1",
        """func nibble(c byte) int {
	return map[bool]int{
		true: hexVal(c),
		false:
""",
        """ -1,
""",
        """	}[isHex(c)]
}
""",
        "an out-of-range hex digit contributes -1",
        "default of a digit decode",
        "return -1 for a non-hex byte",
        [("non-hex byte", "invalid nibble", "transform")],
    )
    add(
        "m72",
        "the terminator is sliced off the line",
        """func trimTerm(line []byte, termLen int) ([]byte, error) {
""",
        """	return cutTail(line, termLen)
""",
        """}
""",
        "drop the line terminator from the buffer",
        "reslice off a suffix",
        "shorten the line by the terminator length",
        [("terminated line", "bare line", "transform")],
    )
    add(
        "m76",
        "the attribute is closed by writing a fixed suffix",
        """func (w *htmlW) closeAttr(class string) error {
""",
        """	return writeSuffix(w, class, `">`)
""",
        """}
""",
        "write the closing quote and angle bracket of a tag",
        "write a fixed suffix",
        "write quote-angle-bracket after the class",
        [("open tag", "closed attribute", "dataflow")],
    )
    add(
        "m86",
        "the path-parameter arm reads the token from that source",
        """func (mw *gate) tokenFrom(c *ctx, key string) (string, error) {
	return map[string]func() (string, error){
		"query":  func() (string, error) { return mw.fromQuery(c, key) },
		"cookie": func() (string, error) { return mw.fromCookie(c, key) },
		"param": func() (string, error) {
""",
        """			return mw.fromParam(c, key)
""",
        """		},
	}[mw.from]()
}
""",
        "take the token from the path parameter",
        "switch arm for a source",
        "read the token from the path parameter",
        [("path parameter", "token", "dataflow")],
    )
    add(
        "m91",
        "a JSON decoder is started on the response body",
        """func fetch(url string) (interface{}, error) {
	res, fault := httpGet(url)
	if fault != nil {
		return nil, fault
	}
	defer closeBody(res)
""",
        """	dec := newJSON(bodyOf(res))
""",
        """	return decodeAll(dec)
}
""",
        "start a JSON decoder on the response body",
        "decode a body",
        "construct a decoder over the body",
        [("response body", "JSON decoder", "init")],
    )
