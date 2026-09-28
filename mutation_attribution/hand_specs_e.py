"""Hand-written rewrites m51-m100."""


def register(add) -> None:
    add("m51", "the host is written into the connection-text buffer",
        """func connText(user, secret, host, db string) string {
	buf := new(bytes.Buffer)
	if user != "" {
		buf.WriteString(user)
		if secret != "" {
			buf.WriteString(":" + secret + "@")
		}
	}
	if host != "" {
""",
        """		buf.WriteString(host)
""",
        """	}
	buf.WriteString("/" + db)
	return buf.String()
}
""",
        "write the host into the connection string",
        "append a component to a buffer",
        "write the host text into the buffer",
        [("host", "connection text", "dataflow")])

    add("m52", "a missing sender is replaced by the default factory",
        """func makePipe(factories []factory, o opts) pipe {
	if o.sender == nil {
""",
        """		o.sender = defaultHTTP()
""",
        """	}
	return pipe{factories: factories, sender: o.sender}
}
""",
        "install the default HTTP sender when none was given",
        "fill a missing option",
        "assign the default sender",
        [("missing sender", "default HTTP sender", "init")])

    add("m53", "the small estimate is returned as an unsigned integer",
        """func (h *sketch) count() uint64 {
	est := linear(h.m, h.v)
	if est < two32/30 {
""",
        """		return uint64(est)
""",
        """	}
	return uint64(est * bias(h))
}
""",
        "return the small estimate as a uint64",
        "cast and return an estimate",
        "return the estimate widened to uint64",
        [("cardinality estimate", "unsigned count", "transform")])

    add("m54", "mismatched footer color counts abort",
        """func (t *grid) setFooterColors(colors ...color) {
	if len(t.footers) != len(colors) {
""",
        """		panic("footer color count must match footer count")
""",
        """	}
	t.footerColors = colors
}
""",
        "panic when footer color count does not match the footer count",
        "length guard then panic",
        "panic on a mismatched color count",
        [("footer color count", "panic", "error")])

    add("m55", "an internal node continues at the next child's character",
        """func (t *tree) get(k uint64) (layer, bool) {
	q := t.root
	for q != nil {
		i, found := t.find(q, k)
		if !found {
			return layer{}, false
		}
		switch node := q.(type) {
		case *inner:
""",
        """			q = node.kids[i+1].ch
""",
        """			continue
		case *leaf:
			return node.vals[i], true
		}
	}
	return layer{}, false
}
""",
        "follow the next child pointer of an internal node",
        "descend one tree level",
        "move the cursor to the following child's link",
        [("search key", "child link", "control")])

    add("m56", "with no signatures the flat encoding is parsed",
        """func parseAny(encoded string, u ...user) (value, error) {
	if g, fault := sniff(encoded); fault == nil && g.sigs == nil {
""",
        """		return g.parseFlat(u...)
""",
        """	}
	return parseCompact(encoded, u...)
}
""",
        "parse the flat form when there is no signature section",
        "branch on missing signatures",
        "parse the flat encoding",
        [("unsigned encoding", "flat parser", "control")])

    add("m57", "ForUser is forwarded to the controller with the gathered arguments",
        """func (m *model) forUser(arg0 interface{}, arg1 ...interface{}) reply {
	varargs := []interface{}{arg0}
	for _, a := range arg1 {
		varargs = append(varargs, a)
	}
""",
        """	ret := m.ctrl.call(m, "ForUser", varargs...)
""",
        """	return ret
}
""",
        "forward ForUser to the controller with the gathered arguments",
        "delegate a call",
        "call the controller method ForUser",
        [("gathered arguments", "controller call", "api")])

    add("m58", "a negative receive timeout is returned as an error duration",
        """func (soc *socket) rcvTimeout() (time.Duration, error) {
	v, fault := soc.getInt(zmqRcv)
	if v < 0 {
""",
        """		return time.Duration(-1), fault
""",
        """	}
	return time.Duration(v) * time.Millisecond, nil
}
""",
        "a negative receive-timeout is an error duration",
        "negative value becomes an error result",
        "return a negative duration together with the fault",
        [("negative socket value", "error duration", "error")])

    add("m59", "each listed container is inspected by id",
        """func (p *pod) containers(ctx context) ([]box, error) {
	listed, fault := p.client.list(ctx)
	if fault != nil {
		return nil, fault
	}
	var out []box
	for _, item := range listed {
""",
        """		box, fault := p.client.inspect(ctx, item.id)
""",
        """		if fault != nil {
			return nil, fault
		}
		out = append(out, box)
	}
	return out, nil
}
""",
        "inspect one listed container by id",
        "loop of inspect calls",
        "inspect the container id from the list",
        [("container id", "inspected container", "api")])

    add("m60", "unfinished jobs are put back when the worker stops",
        """func (w *worker) onStop(pid int) {
	log.Printf("stop timeout=%s pid=%d", w.limit, pid)
""",
        """	w.requeue()
""",
        """	w.quit()
}
""",
        "put unfinished jobs back on the queue when stopping",
        "requeue on shutdown",
        "requeue the unfinished jobs",
        [("stop", "job queue", "control")])

    add("m61", "the column is the 1-based offset from the line start",
        """func lineCol(s string, cur int) (ln, col int) {
	ln = 1
	lineStart := 0
	for pos := 0; pos < cur; pos++ {
		if s[pos] == '\\n' {
			ln++
			lineStart = pos + 1
		}
	}
""",
        """	col = cur - lineStart + 1
""",
        """	return ln, col
}
""",
        "column is the 1-based offset from the start of the line",
        "subtract the line start",
        "compute the column from the cursor and the line start",
        [("cursor and line start", "column", "transform")])

    add("m62", "the last two bicubic coefficients are finished from b and c",
        """func bicubic(b, c float64) *cubic {
	f := &cubic{}
	f.a = 1 - b/3
	f.b = -3 + 2*b + c
	f.c = 2 - 3*b/2 - c
""",
        """	f.d = 4*b/3 + 4*c
	f.e = -2*b - 8*c
""",
        """	return f
}
""",
        "finish the bicubic coefficients d and e from b and c",
        "complete a polynomial",
        "assign the d and e coefficients",
        [("cubic parameters", "remaining coefficients", "transform")])

    add("m63", "response headers and status are copied, then the body",
        """func proxy(w writer, resp *httpResp) error {
	if resp == nil {
		return faultGateway
	}
""",
        """	copyHdr(w.header(), resp.header)
	w.writeCode(resp.code)
	_, fault := io.Copy(w, resp.body)
""",
        """	return fault
}
""",
        "copy response headers and status, then the body",
        "forward a response",
        "copy headers, write the status, copy the body",
        [("upstream response", "client writer", "dataflow")])

    add("m64", "the key file is stated, then a missing or unnamed file is created",
        """func (k *keys) ensure(nc netConn) error {
	var fault error
	if len(k.file) > 0 {
""",
        """		_, fault = os.Stat(k.file)
	}
	if os.IsNotExist(fault) || len(k.file) == 0 {
""",
        """		return k.generate(nc)
	}
	return fault
}
""",
        "stat the key file, and if it is missing or unnamed enter creation",
        "stat then branch on absence",
        "stat the named file and open the missing-file branch",
        [("key file name", "stat result", "dataflow"), ("missing file", "generation", "control")])

    add("m65", "the cancel URL is built from the API root and the receipt",
        """func (p *push) cancel(receipt string) (*resp, error) {
""",
        """	endpoint := fmt.Sprintf("%s/receipts/%s/cancel.json", apiRoot, receipt)
""",
        """	return p.post(endpoint)
}
""",
        "build the cancel URL from the API root and the receipt",
        "format a URL",
        "interpolate the receipt into the cancel path",
        [("receipt", "cancel URL", "transform")])

    add("m66", "a hex digit outside 0-9 and a-f contributes -1",
        """func nibble(c byte) int {
	switch {
	case c >= '0' && c <= '9':
		return int(c - '0')
	case c >= 'a' && c <= 'f':
		return int(c - 'a' + 10)
	case c >= 'A' && c <= 'F':
		return int(c - 'A' + 10)
	default:
""",
        """		return -1
""",
        """	}
}
""",
        "an out-of-range hex digit contributes -1",
        "default of a digit decode",
        "return -1 for a non-hex byte",
        [("non-hex byte", "invalid nibble", "transform")])

    add("m67", "the connection is marked shut before pending calls end",
        """func (c *conn) readLoop() {
	c.sending.Lock()
	c.mu.Lock()
""",
        """	c.down = true
""",
        """	c.mu.Unlock()
	c.sending.Unlock()
	c.failPending()
}
""",
        "mark the connection shut down before ending pending calls",
        "set a shutdown flag under the lock",
        "set the shutdown flag",
        [("read loop", "shutdown flag", "control")])

    add("m68", "the originating card is the last ancestor",
        """func (c card) origin(args args) (card, error) {
	elders, fault := c.ancestors(args)
	if fault != nil {
		return c, fault
	}
	if len(elders) > 0 {
""",
        """		return elders[len(elders)-1], nil
""",
        """	}
	return c, nil
}
""",
        "the originating card is the last ancestor",
        "take the tail of a list",
        "return the last ancestor",
        [("ancestor list", "originating card", "dataflow")])

    add("m69", "the command is wrapped in CDATA so XML metacharacters survive",
        """func execReq(command string) string {
""",
        """	command = "<![CDATA[" + command + "]]>"
""",
        """	return "<shell>" + command + "</shell>"
}
""",
        "wrap the command in a CDATA section",
        "escape by wrapping",
        "surround the command with a CDATA marker",
        [("raw command", "CDATA section", "transform")])

    add("m70", "a failed sequence update is logged",
        """func (db *store) saveSeq(last int) {
	_, fault := db.exec("UPDATE cluster SET last_sequence=?", last)
	if fault != nil {
""",
        """		log.Errorf("sequence update failed: %v", fault)
""",
        """	}
}
""",
        "log the failed update of the last sequence",
        "log an error result",
        "log the update fault",
        [("update fault", "error log", "error")])

    add("m71", "an OpError's inner error is the close signal",
        """func closed(fault error) bool {
	if fault == nil {
		return false
	}
	if op, isOp := fault.(*netOp); isOp {
""",
        """		fault = op.inner
""",
        """	}
	return fault == io.EOF || fault == io.ErrClosedPipe
}
""",
        "unwrap an operation error to its inner fault",
        "unwrap then test",
        "replace the fault with the operation's inner fault",
        [("operation error", "inner fault", "transform")])

    add("m72", "the line terminator is cut off the buffer",
        """func trimTerm(line []byte, termLen int) ([]byte, error) {
	if len(line) >= termLen && line[len(line)-termLen] != '\\r' {
		return nil, faultBad
	}
""",
        """	line = line[:len(line)-termLen]
""",
        """	return line, nil
}
""",
        "drop the line terminator from the buffer",
        "reslice off a suffix",
        "shorten the line by the terminator length",
        [("terminated line", "bare line", "transform")])

    add("m73", "a results channel is allocated for the query",
        """func (d *store) query(q query) (results, error) {
""",
        """	results := make(chan row)
""",
        """	go d.run(q, results)
	return results, nil
}
""",
        "allocate the results channel for a datastore query",
        "make a channel then start work",
        "make the results channel",
        [("query", "results channel", "init")])

    add("m74", "logical and physical core counts are recorded",
        """func (cpu *chip) detect() {
	cpu.features = readFeatures()
	cpu.sgx = cpu.features&sgxBit != 0
	cpu.threads = threadsPerCore()
""",
        """	cpu.logical = logicalCores()
	cpu.physical = physicalCores()
""",
        """}
""",
        "record logical and physical core counts",
        "fill hardware counters",
        "store logical and physical core counts",
        [("cpu", "core counts", "dataflow")])

    add("m75", "resources are listed for the group with an empty version",
        """func resourceFor(mapper mapper, info metricInfo) (gvr, error) {
""",
        """	full, fault := mapper.resources(info.group.withVersion(""))
""",
        """	if fault != nil {
		return gvr{}, fault
	}
	return full, nil
}
""",
        "list resources for the group with an empty version",
        "map a group at a blank version",
        "ask the mapper for resources of the unversioned group",
        [("group", "resource list", "api")])

    add("m76", "the tag is closed with a quote and an angle bracket",
        """func (w *htmlW) closeAttr(class string) error {
	if _, fault := io.WriteString(w, class); fault != nil {
		return fault
	}
""",
        """	_, fault = w.Write([]byte(`">`))
""",
        """	return fault
}
""",
        "write the closing quote and angle bracket of a tag",
        "write a fixed suffix",
        "write quote-angle-bracket",
        [("open tag", "closed attribute", "dataflow")])

    add("m77", "a failed port delete is wrapped as an error",
        """func (c *vrs) destroyPort(id string) error {
	cond := c.match(id)
	if fault := c.table.delete(c.client, cond); fault != nil {
""",
        """		return fmt.Errorf("cannot drop port from vrs %v", fault)
""",
        """	}
	return nil
}
""",
        "wrap a port-delete failure",
        "wrap and return an error",
        "return the delete fault wrapped with context",
        [("delete fault", "wrapped error", "error")])

    add("m78", "the term end is the inclusive end date",
        """func (term *span) scan(end inclusiveDate) {
	end = end.addDays(-1)
""",
        """	term.end = end
""",
        """}
""",
        "set the term end to the inclusive end date",
        "store a computed date",
        "assign the adjusted end date",
        [("exclusive end date", "inclusive term end", "transform")])

    add("m79", "the redirect status and a short body are written",
        """func (ctx *webCtx) redirect(status int, loc string) {
	ctx.header().Set("Location", loc)
""",
        """	ctx.writeCode(status)
	ctx.write([]byte("going to: " + loc))
""",
        """}
""",
        "write the redirect status and a short body",
        "status then body",
        "write the status code and a redirect notice",
        [("redirect target", "response body", "dataflow")])

    add("m80", "a challenge buffer is allocated and a server security context is started",
        """func newServerCtx(cred *creds, negotiate []byte) (*srvCtx, []byte, error) {
""",
        """	challenge := make([]byte, maxToken)
	c := newSrv(cred, ascConn)
""",
        """	done, n, fault := updateCtx(c, challenge, negotiate)
	if fault != nil || done || n == 0 {
		c.release()
		return nil, nil, fault
	}
	return c, challenge[:n], nil
}
""",
        "allocate the challenge buffer and start a server security context",
        "allocate then construct",
        "make the challenge buffer and the server context",
        [("credentials", "server security context", "init")])

    add("m81", "the new child is attached under the current entry",
        """func (b *builder) add(name string, size int64, flags uint32) {
	child := &entry{name: name, size: size, flags: flags, parent: b.cur}
""",
        """	b.cur.kids = append(b.cur.kids, child)
""",
        """}
""",
        "attach the new child entry under the current entry",
        "append a child",
        "append the child to the current entry",
        [("child entry", "current entry", "dataflow")])

    add("m82", "the builder takes its plugin from the plugger",
        """func (b *builder) withKind(kind string) {
	if plug, isPlug := interface{}(b).(plugger); isPlug {
""",
        """		b.plugin = plug.plugin()
""",
        """	}
}
""",
        "set the builder's plugin from the plugger",
        "copy a plugin handle",
        "assign the plugin from the plugger",
        [("plugger", "builder plugin", "dataflow")])

    add("m83", "an error channel is created for the listener",
        """func (s *queue) listen(fn func(msg)) error {
""",
        """	faults := make(chan error)
""",
        """	go s.pull(fn, faults)
	return <-faults
}
""",
        "make the error channel the listener will report on",
        "allocate a channel",
        "make the fault channel",
        [("listener", "error channel", "init")])

    add("m84", "a missing context becomes the background context",
        """func withSecret(title string) option {
	return func(o *options) {
		if o.ctx == nil {
""",
        """			o.ctx = context.Background()
""",
        """		}
		o.secret = title
	}
}
""",
        "use a background context when the options have none",
        "fill a nil context",
        "assign the background context",
        [("missing context", "background context", "init")])

    add("m85", "TOML bytes are parsed into a table",
        """func unmarshalTOML(data []byte, v interface{}) error {
""",
        """	table, fault := toml.Parse(data)
""",
        """	if fault != nil {
		return fault
	}
	return fill(v, table)
}
""",
        "parse TOML bytes into a table",
        "parse then fill",
        "parse the bytes as a TOML table",
        [("TOML bytes", "table", "transform")])

    add("m86", "the token is taken from the path parameter",
        """func (mw *gate) tokenFrom(c *ctx, key string) (string, error) {
	switch mw.from {
	case "query":
		return mw.fromQuery(c, key)
	case "cookie":
		return mw.fromCookie(c, key)
	case "param":
""",
        """		return mw.fromParam(c, key)
""",
        """	default:
		return "", faultWhere
	}
}
""",
        "take the token from the path parameter",
        "switch arm for a source",
        "read the token from the path parameter",
        [("path parameter", "token", "dataflow")])

    add("m87", "YAML bytes are unmarshaled into a generic value",
        """func newYAML(body []byte) (*doc, error) {
	var val interface{}
""",
        """	fault := yaml.Unmarshal(body, &val)
""",
        """	if fault != nil {
		return nil, fault
	}
	return &doc{val: val}, nil
}
""",
        "unmarshal YAML bytes into a generic value",
        "unmarshal into an empty interface",
        "unmarshal the body into val",
        [("YAML bytes", "generic value", "transform")])

    add("m88", "one text range is allocated per edge",
        """func edgesToRanges(path []edge) []textRange {
""",
        """	ranges := make([]textRange, len(path))
""",
        """	for i, e := range path {
		ranges[i] = textRange{start: e.from, end: e.to}
	}
	return ranges
}
""",
        "allocate one text range per edge",
        "make a slice sized to the path",
        "allocate the range slice",
        [("edge path", "text ranges", "init")])

    add("m89", "a count other than one is an unexpected-count error",
        """func hostGroupByID(id string) (group *group, fault error) {
	groups, fault := findGroups(id)
	if fault != nil {
		return nil, fault
	}
	if len(groups) == 1 {
		return &groups[0], nil
	}
""",
        """	e := expectedOne(len(groups))
	fault = &e
""",
        """	return nil, fault
}
""",
        "more or fewer than one group is an unexpected-count error",
        "build an error from a count",
        "store the unexpected-count error",
        [("group count", "unexpected-count error", "error")])

    add("m90", "the address is marked as coming from the real-ip header",
        """func remoteOf(r *httpReq) string {
	addr := r.remoteAddr
	if real := r.header.Get("X-Real-IP"); real != "" {
		addr = real
""",
        """		addr += " (X-Real-IP)"
""",
        """	}
	return addr
}
""",
        "mark the address as coming from the real-ip header",
        "annotate a chosen address",
        "append the real-ip marker",
        [("real-ip header", "annotated address", "transform")])

    add("m91", "a JSON decoder is started on the response body",
        """func fetch(url string) (interface{}, error) {
	res, fault := http.Get(url)
	if fault != nil {
		return nil, fault
	}
	defer res.Body.Close()
""",
        """	dec := json.NewDecoder(res.Body)
""",
        """	var out interface{}
	return out, dec.Decode(&out)
}
""",
        "start a JSON decoder on the response body",
        "decode a body",
        "construct a decoder over the body",
        [("response body", "JSON decoder", "init")])

    add("m92", "a non-interactive question is rendered as left unanswered",
        """func (c *ui) question(query string) (string, error) {
	if !c.interactive {
		c.render(catQuestion, query, c.out, colorNone, true)
""",
        """		c.render(catQuestion, "leaving the question unanswered", c.out, colorNone, true)
""",
        """		return "", nil
	}
	return c.prompt(query)
}
""",
        "render a question that will be left unanswered",
        "render a fallback notice",
        "render the unanswered notice",
        [("non-interactive question", "unanswered notice", "dataflow")])

    add("m93", "the second overlap test compares y's start with x's end",
        """func anyOverlap(x, y []byte) bool {
	return len(x) > 0 && len(y) > 0 &&
		ptr(&x[0]) <= ptr(&y[len(y)-1]) &&
""",
        """		ptr(&y[0]) <= ptr(&x[len(x)-1])
""",
        """}
""",
        "test that y starts at or before x ends",
        "pointer-range overlap",
        "compare the pointer of y's first byte with the pointer of x's last byte",
        [("slice y", "slice x", "control")])

    add("m94", "a matching sibling is passed to the callback",
        """func findNodes(input *node, match func(kind, atom) bool, cb func(*node)) {
	for c := input.first; c != nil; c = c.next {
		if match(c.kind, c.atom) {
""",
        """			cb(c)
""",
        """		}
	}
}
""",
        "invoke the callback on the matching sibling",
        "callback on a match",
        "call the callback with the sibling",
        [("matching node", "callback", "control")])

    add("m95", "IPv4 addresses use a 32-bit length",
        """func hostBits(ip netIP) int {
	var n int
	if ip.to4() != nil {
""",
        """		n = 32
""",
        """	} else {
		n = 128
	}
	return n
}
""",
        "IPv4 addresses use a 32-bit length",
        "branch on address family",
        "set the bit length to 32",
        [("IPv4 address", "32-bit length", "transform")])

    add("m96", "an empty value list makes the relation null",
        """func (c col) inStrings(values []string) rel {
	if len(values) == 0 {
""",
        """		return c.isNull()
""",
        """	}
	return c.inList(values)
}
""",
        "an empty value list makes the relation null",
        "empty-input shortcut",
        "return the null relation",
        [("empty value list", "null relation", "control")])

    add("m97", "a short read failure is wrapped",
        """func readSome(file *osFile) ([]byte, error) {
	data := make([]byte, 1000)
	_, fault := file.Read(data)
	if fault != nil {
""",
        """		return nil, fmt.Errorf("read failed: %s", fault)
""",
        """	}
	return data, nil
}
""",
        "wrap a short file read failure",
        "wrap a read fault",
        "return the read fault wrapped with context",
        [("read fault", "wrapped error", "error")])

    add("m98", "a kept symlink is logged at verbosity 5",
        """func (h *copier) onLink(source, dest string) {
	if h.keepLinks() {
""",
        """		glog.V(5).Infof("L %s -> %s", source, dest)
""",
        """	}
}
""",
        "log a symlink that is being kept",
        "verbose log of a link",
        "log the source and destination of the link",
        [("symlink", "verbose log", "dataflow")])

    add("m99", "the ring is locked before a key is removed",
        """func (p *ring) remove(id string) error {
""",
        """	p.Lock()
""",
        """	defer p.Unlock()
	delete(p.keys, id)
	return nil
}
""",
        "lock the ring before removal",
        "lock then mutate",
        "acquire the ring lock",
        [("public ring", "lock", "control")])

    add("m100", "the compare-and-swap flag and the state fault are returned after the callback loop",
        """func (b *bucket) update(k string, exp int, cb func([]byte) ([]byte, error)) (uint64, error) {
	var state casState
	for b.casNext(k, exp, &state) {
		var fault error
		if state.val, fault = cb(state.val); fault != nil {
			return 0, fault
		}
	}
""",
        """	return state.cas, state.fault
""",
        """}
""",
        "return the compare-and-swap token and the state fault",
        "return the loop's final state",
        "return the cas token and the state fault",
        [("cas state", "function result", "dataflow")])
