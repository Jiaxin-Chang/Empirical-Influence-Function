"""Hand-written rewrites m22-m100. Each hole keeps the parent mechanism."""


def register(add) -> None:
    add(
        "m22",
        "stdout and stderr ends are sibling arguments; the hole opens the sink end and the alarm end",
        """func buildEnds(tweaks ...tune) (bundle, error) {
	cfg := baseEnds()
	for _, tweak := range tweaks {
		tweak(cfg)
	}
	return pack(openEnd(cfg.feed),
""",
        """		openEnd(cfg.sink), openEnd(cfg.alarm),
""",
        """	)
}
""",
        "keep the standard-output pipe, then the standard-error pipe",
        "open each requested end",
        "open the sink end and the alarm end",
        [("requested ends", "pipe bundle", "init")],
    )
    add(
        "m23",
        "a failed membership ask is explained inside the fault arm",
        """func (gs *groups) memberOf(groupID, who, token string) (person, bool, error) {
	return onFault(ask(gs.setup, "GET", groupID, who, token), func(fault error) (person, bool, error) {
""",
        """		return person{}, false, explain(fault)
""",
        """	}, func(resp reply) (person, bool, error) {
		if resp.missing() {
			return person{}, false, nil
		}
		return readPerson(resp.raw)
	})
}
""",
        "turn a failed membership request into an empty member and a translated fault",
        "fault arm before a later decode",
        "return no person, not a member, and the explained fault",
        [("request fault", "explained failure", "error")],
    )
    add(
        "m24",
        "each addon is scanned inside a callback; the hole aborts when that scan fails",
        """func (mux *gate) ready() {
	each(mux.addons, func(plug addon) {
		if fault := plug.scan(mux.routes); fault != nil {
""",
        """			panic(fault)
""",
        """		}
		if plug.idle() {
			panic(idlePlug(plug))
		}
	})
}
""",
        "abort preparation when a plugin scan fails",
        "panic on a failed scan",
        "panic with the scan fault",
        [("scan fault", "preparation abort", "error")],
    )
    add(
        "m25",
        "the locator is parsed before file and head checks",
        """func reachable(raw string, limit span, log book) (bool, error) {
""",
        """	loc, fault := parseLoc(raw)
""",
        """	return afterLoc(loc, fault, func() (bool, error) {
		if loc.kind == "file" {
			return fileThere(loc.full)
		}
		log.Debugf("probe %s", raw)
		return headOk(raw, limit)
	})
}
""",
        "parse the address before testing whether it exists",
        "parse then branch on scheme",
        "parse the raw address into a locator",
        [("raw address", "locator", "transform")],
    )
    add(
        "m26",
        "the plain pool is built first and then embedded",
        """func makeGreedy(hosts []string, fade span, calc calculator) pool {
	if fade <= 0 {
		fade = usualFade
	}
""",
        """	plain := makePlain(hosts).(*plainPool)
""",
        """	return &greedyPool{
		plainPool: *plain,
		epsilon:   float32(startEpsilon),
		fade:      fade,
		calc:      calc,
		timer:     &wallTimer{},
		quit:      make(chan bool),
	}
}
""",
        "build the standard host pool that the epsilon-greedy pool embeds",
        "construct a base then wrap it",
        "allocate the plain host pool from the host list",
        [("host list", "plain host pool", "init")],
    )
    add(
        "m27",
        "routes are registered as pairs; the hole binds the reload socket path",
        """func makeHub(title string, port uint16) *hub {
	routes := web.NewMux()
	built := &hub{title: title, routes: routes, liveCSS: true}
	built.setPort(port)
	pairs := []struct{ route string; fn func(*hub) }{
		{"/livereload.js", scriptHandler},
""",
        """		{"/livereload", socketHandler},
""",
        """	}
	for _, pair := range pairs {
		routes.HandleFunc(pair.route, pair.fn(built))
	}
	return built
}
""",
        "route the reload socket path to its handler",
        "table of route bindings",
        "bind the reload path to the socket handler",
        [("reload path", "socket handler", "config")],
    )
    add(
        "m28",
        "the walking integer is turned into an address inside the loop",
        """func (alloc *spanAlloc) subtract(span *span) {
	cur := big.NewInt(0).SetBytes(span.lo)
	end := big.NewInt(0).SetBytes(span.hi)
	for ; cur.Cmp(end) < 1; cur = cur.Add(big.NewInt(1), cur) {
""",
        """		addr := alloc.intToAddr(cur)
""",
        """		if alloc.spanHolds(addr) {
			alloc.hold(addr)
		}
	}
}
""",
        "turn the current integer into an address while walking a range",
        "convert then maybe reserve",
        "convert the cursor integer into an address",
        [("cursor integer", "address", "transform")],
    )
    add(
        "m29",
        "verb bindings are a slice; the hole adds the log read and the patch",
        """func makeUserGate() *userGate {
	h := &userGate{Router: makeRouter()}
	bind := []struct{ verb, route string; fn func() }{
		{"POST", usersRoute, h.postUser},
		{"GET", usersRoute, h.getUsers},
		{"GET", userRoute, h.getUser},
""",
        """		{"GET", userLogRoute, h.getUserLog},
		{"PATCH", userRoute, h.patchUser},
""",
        """		{"DELETE", userRoute, h.deleteUser},
	}
	for _, item := range bind {
		h.Handle(item.verb, item.route, item.fn)
	}
	return h
}
""",
        "register the user-log read and the user patch",
        "table of verb bindings",
        "bind GET user-log and PATCH user",
        [("user-log route", "log handler", "config"), ("user route", "patch handler", "config")],
    )
    add(
        "m30",
        "the prepared request is sent and decoded in one call",
        """func (c *agent) patch(uri string, data interface{}, decoded interface{}) (*reply, error) {
	req, fault := c.jsonReq("PATCH", uri, data)
	if fault != nil {
		return nil, fault
	}
""",
        """	resp, fault := c.doRest(req, decoded)
""",
        """	if fault != nil {
		return nil, fault
	}
	if resp.code != accepted {
		return nil, faultCode(resp.code)
	}
	return resp, fault
}
""",
        "send the prepared request and decode its body",
        "perform a request then check status",
        "execute the REST call into the decode target",
        [("prepared request", "decoded reply", "api")],
    )
