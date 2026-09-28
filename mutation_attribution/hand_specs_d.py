"""Hand-written rewrites m31-m100."""


def register(add) -> None:
    add("m31", "id-offset map and tombstones are cleared together while closing",
        """func (idx *seriesIdx) shut() (fault error) {
	if idx.raw != nil {
		fault = unmap(idx.raw)
	}
	idx.keyBlob, idx.offBlob, idx.keyMap = nil, nil, nil
""",
        """	idx.offMap, idx.dead = nil, nil
""",
        """	return fault
}
""",
        "drop the id-offset index and the tombstone set during close",
        "nil out closing fields",
        "clear the offset map and the tombstone set",
        [("offset map", "closed index", "dataflow")])

    add("m32", "the postgres link is opened from the connection text before the checkpoint exists",
        """func makeMark(app, table, dsn string, opts ...opt) (*mark, error) {
	if app == "" || table == "" {
		return nil, faultMissing("app or table")
	}
""",
        """	link, fault := sql.Open("postgres", dsn)
""",
        """	if fault != nil {
		return nil, fault
	}
	made := &mark{link: link, app: app, table: table}
	for _, opt := range opts {
		opt(made)
	}
	go made.loop()
	return made, nil
}
""",
        "open the database from the connection string",
        "open a driver then build the object",
        "open a postgres connection from the dsn",
        [("connection string", "database link", "init")])

    add("m33", "after the completed list is handed to the collector it is dropped",
        """func (s *server) sweep() {
	var done []string
	for {
		select {
		case name := <-s.finished:
			done = append(done, name)
		case <-time.After(s.every):
			s.run <- done
""",
        """			done = nil
""",
        """		}
	}
}
""",
        "clear the completed-handler list after handing it off",
        "transfer then reset",
        "drop the completed list once it has been sent",
        [("completed handlers", "empty list", "dataflow")])

    add("m34", "bytes are summed with the prefix type and length",
        """func (p header) digest(data []byte) (cid, error) {
	n := p.mhLen
	if p.mhKind == idKind {
		n = -1
	}
""",
        """	sum, fault := mh.Sum(data, p.mhKind, n)
""",
        """	if fault != nil {
		return undef, fault
	}
	return pack(p.ver, p.codec, sum)
}
""",
        "hash the bytes with the configured multihash type and length",
        "sum then wrap by version",
        "compute the multihash of the bytes",
        [("payload bytes", "multihash", "transform")])

    add("m35", "the port label is rewritten by the spec before the chain is wrapped",
        """func withPort(label string, spec func(string) string, gen chain) chain {
""",
        """	label = spec(label)
""",
        """	if label == "" {
		return gen
	}
	return func(rows ...string) {
		gen(rows...)
		for i := range rows {
			rows[i] = "_" + label + "." + rows[i]
		}
		gen(rows...)
	}
}
""",
        "rewrite the port name through the spec function",
        "transform a name before use",
        "apply the spec to the port label",
        [("raw port label", "rewritten port label", "transform")])

    add("m36", "a softlayer type outside the datatype package is qualified",
        """func goName(t, pkg string) string {
	switch t {
	case "unsignedLong", "unsignedInt":
		return "uint"
	case "boolean":
		return "bool"
	default:
		if hasHead(t, "Soft"+"Layer_") && pkg != "datatypes" {
""",
        """			return "datatypes." + trimHead(t)
""",
        """		}
		return trimHead(t)
	}
}
""",
        "qualify a stripped type name with the datatypes package",
        "prefix a type name",
        "return the datatypes-qualified name",
        [("stripped type name", "qualified type name", "transform")])

    add("m37", "unreading with an empty size stack is refused",
        """func (lx *scan) unread() error {
	if len(lx.widths) == 0 {
""",
        """		return faultUnread
""",
        """	}
	lx.pos -= lx.widths[len(lx.widths)-1]
	lx.widths = lx.widths[:len(lx.widths)-1]
	return nil
}
""",
        "refuse to unread when no rune was consumed",
        "guard on an empty stack",
        "return the invalid-unread fault",
        [("empty rune stack", "unread refusal", "error")])

    add("m38", "the function file is announced before it is built",
        """func (b *buildCmd) run(c *cliCtx) error {
	out := verbOut(b.verbose)
	dir, fault := cwd()
	if fault != nil {
		return fault
	}
	fn, fault := findFunc(dir)
	if fault != nil {
		return fault
	}
""",
        """	fmt.Fprintln(out, "building", fn)
""",
        """	built, fault := buildFunc(out, fn)
	if fault != nil {
		return fault
	}
	fmt.Printf("built %s\\n", built.title())
	return nil
}
""",
        "log that a function file is being built",
        "announce before the build call",
        "write the building line for the function file",
        [("function file", "build log", "dataflow")])

    add("m39", "a blank service id refuses the update before other checks",
        """func (c *agent) updateGCS(in *gcsIn) (*gcs, error) {
	if in.svc == "" {
""",
        """		return nil, faultNoSvc
""",
        """	}
	if in.rev == 0 {
		return nil, faultNoRev
	}
	return c.putForm(in)
}
""",
        "refuse the update when the service id is missing",
        "missing-field guard",
        "return the missing-service fault and no object",
        [("blank service id", "missing-service fault", "error")])

    add("m40", "wrapped function source is parsed into a program",
        """func parseFn(params, body string) (*fnLit, error) {
	src := "(function(" + params + ") {\\n" + body + "\\n})"
""",
        """	p := newParser("", src, 1, nil)
	prog, fault := p.parse()
""",
        """	if fault != nil {
		return nil, fault
	}
	return prog.firstFn(), nil
}
""",
        "parse the wrapped function source into a program",
        "parse then unwrap the function literal",
        "build a parser and parse the wrapped source",
        [("function source", "parsed program", "transform")])

    add("m41", "the read lock is taken before the names are copied",
        """func (ns *stops) names() []string {
	if ns == nil {
		return nil
	}
""",
        """	ns.RLock()
""",
        """	defer ns.RUnlock()
	out := []string{}
	for k := range ns.items {
		out = append(out, k)
	}
	return out
}
""",
        "take the read lock before listing names",
        "lock then copy keys",
        "acquire the read lock",
        [("stopwatch map", "read lock", "control")])

    add("m42", "a recovered error is logged as fatal after a pause",
        """func (a app) crash() {
	caught := recover()
	switch caught.(type) {
	case error:
""",
        """		time.Sleep(5 * time.Second)
		a.log.Fatal("crash", caught.(error))
""",
        """	case nil:
		return
	default:
		time.Sleep(5 * time.Second)
		a.log.Fatal("crash", nil)
	}
}
""",
        "pause, then log the recovered error as fatal",
        "delay then fatal-log",
        "sleep and fatal-log the recovered error",
        [("recovered error", "fatal log", "error")])

    add("m43", "the hello server name selects the host record",
        """func (a *armor) configFor(hello *tlsHello) (*tlsCfg, error) {
""",
        """	host := a.hosts[hello.serverName]
""",
        """	if host == nil || len(host.clients) == 0 {
		return nil, nil
	}
	if host.cfg != nil {
		return host.cfg, nil
	}
	host.cfg = a.buildCfg(hello, host)
	return host.cfg, nil
}
""",
        "look up the host config by the TLS hello server name",
        "map lookup then maybe build",
        "index the host table with the server name",
        [("server name", "host record", "dataflow")])

    add("m44", "the custom group is appended and the builder returned",
        """func (s *groups) custom(name string) *groups {
	g := group{name: name, open: s.open, sep: s.sep}
""",
        """	*s = append(*s, g)
	return s
""",
        """}
""",
        "append a custom option group and return the builder",
        "append then hand back",
        "append the group and return the builder",
        [("custom group", "builder", "dataflow")])

    add("m45", "remote field name and user type are stored on the map definition",
        """func mapFrom(field, userType string) *mapDef {
	md := newMapDef()
""",
        """	md.remoteField, md.remoteType = field, userType
""",
        """	return md
}
""",
        "record the remote field name and its user type",
        "assign a pair of attributes",
        "store the remote field and the remote type",
        [("field name", "map definition", "dataflow")])

    add("m46", "a value already known to be template context is returned as-is",
        """func pongoCtx(data interface{}) ctx {
	if ready, isCtx := data.(ctx); isCtx {
""",
        """		return ready
""",
        """	}
	return ctx{"data": data}
}
""",
        "return the value that is already a template context",
        "pass through an existing context",
        "return the context value",
        [("template context", "function result", "dataflow")])

    add("m47", "a fresh empty set is stored for a new token",
        """func (a *bag) addToken(token string) error {
	if _, fault := a.find(token); fault == nil {
		return faultExists
	}
""",
        """	a[token] = emptySet()
""",
        """	return nil
}
""",
        "create an empty set for a newly stored token",
        "init a map entry",
        "store a new empty set under the token",
        [("token", "empty set", "init")])

    add("m48", "a unix datagram socket is opened at the address",
        """func listenUnix(ctx context, netw string, addr *unixAddr) (*unixConn, error) {
""",
        """	conn, fault := net.ListenUnixgram(netw, addr)
""",
        """	if fault != nil {
		return nil, fault
	}
	return &unixConn{conn: conn, ctx: ctx}, nil
}
""",
        "listen for a unix datagram at the given address",
        "listen then wrap",
        "open a unix datagram listener",
        [("network and address", "datagram listener", "init")])

    add("m49", "the matching slave is cut out of the slice",
        """func (br *bridge) dropSlave(ifc nic) {
	for index, item := range br.slaves {
		if item.title == ifc.title && sameAddr(item.hw, ifc.hw) {
""",
        """			br.slaves = append(br.slaves[:index], br.slaves[index+1:]...)
""",
        """			return
		}
	}
}
""",
        "delete the slave interface at the matched index",
        "slice delete",
        "cut the indexed slave out of the slice",
        [("slave index", "slave list", "transform")])

    add("m50", "an empty request URI is treated as the root path",
        """func rawRoute(r *httpReq) string {
	if r.URL.Path == "" {
		return "/"
	}
	route := r.requestURI
	if route == "" {
""",
        """		route = "/"
""",
        """	}
	return route
}
""",
        "treat an empty request URI as the root path",
        "empty-string default",
        "set the route to slash",
        [("empty request URI", "root path", "transform")])
