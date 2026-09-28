"""Hand-written FIM rewrites. Keys are mutation ids."""

from __future__ import annotations

SPECS: dict[str, dict] = {}


def add(mid, note, prefix, middle, suffix, role, pattern, operations, rels):
    SPECS[mid] = {
        "note": note,
        "prefix": prefix if prefix.endswith("\n") or prefix.endswith(" ") else prefix,
        "middle": middle,
        "suffix": suffix,
        "semantic": {
            "role": role,
            "pattern": [pattern] if isinstance(pattern, str) else list(pattern),
            "operations": [operations] if isinstance(operations, str) else list(operations),
            "relations": [
                {"source": src, "target": dst, "type": typ} for src, dst, typ in rels
            ],
        },
    }


add(
    "m02",
    "refusals are picker arguments; the hole is the unused-bit framing refusal followed by the revision test",
    """func (pkt *hello) check() byte {
	legacy := pkt.banner == ("MQ" + "Isdp") && pkt.rev != (2 + 1)
	modern := pkt.banner == ("MQ" + "TT") && pkt.rev != (2 + 2)
	return pick(
		when(pkt.secretSet && !pkt.nameSet, refuseAuth),
""",
    """		when(pkt.unusedBit != 0, refuseFraming),
		when(legacy || modern, refuseRev),
""",
    """		when(pkt.banner != ("MQ"+"Isdp") && pkt.banner != ("MQ"+"TT"), refuseFraming),
		when(len(pkt.whoami) > (1<<16-1) || len(pkt.who) > (1<<16-1) || len(pkt.secret) > (1<<16-1), refuseFraming),
		when(len(pkt.whoami) == 0 && !pkt.fresh, refuseName),
		when(true, admit),
	)
}
""",
    "reject a set reserved bit as a framing error, then start the version check",
    "ordered refusal rules",
    "emit the framing refusal for a nonzero unused bit and open the revision test",
    [
        ("unused bit", "framing refusal", "control"),
        ("revision mismatch", "revision refusal", "control"),
    ],
)

add(
    "m03",
    "the packet is one composite; the hole sets the name-present flag from the high bit and reads the tick count",
    """func (pkt *hello) read(in stream) error {
	flags := takeU8(in)
	*pkt = hello{
		banner:   takeText(in),
		rev:      takeU8(in),
		unused:   flags&1 != 0,
		fresh:    flags&(1<<1) != 0,
		willOn:   flags&(1<<2) != 0,
		willRank: (flags >> 3) & 3,
		willKeep: flags&(1<<5) != 0,
		secretOn: flags&(1<<6) != 0,
""",
    """		nameOn: flags&128 != 0,
		ticks:  takeU16(in),
""",
    """		whoami: takeText(in),
	}
	return attachOptional(pkt, in, flags)
}
""",
    "read the name-present flag from the high bit, then the keepalive",
    "bitfield unpack into a composite",
    "store whether the high flag bit is set and read the two-byte tick count",
    [
        ("high flag bit", "name-present flag", "transform"),
        ("tick count", "packet fields", "dataflow"),
    ],
)

add(
    "m04",
    "phase arms are functions in a map; the hole ORs the reply bit and moves to ready",
    """func (flow *lane) headerBits() uint16 {
	flow.mu.hold()
	defer flow.mu.drop()
	arm := map[int]func(uint16) uint16{
		phaseNew: func(bits uint16) uint16 {
			flow.phase = phaseOpenSent
			return bits | bitOpen
		},
		phaseOpenSeen: func(bits uint16) uint16 {
""",
    """			flow.phase = phaseReady
			return bits | bitReply
""",
    """		},
	}[flow.phase]
	if arm == nil {
		return 0
	}
	return arm(0)
}
""",
    "add the reply bit once the open has been seen",
    "map of phase transitions",
    "OR the reply bit and advance the lane to ready",
    [
        ("open-seen phase", "reply bit", "control"),
        ("reply bit", "ready phase", "dataflow"),
    ],
)

add(
    "m06",
    "the catalog is grabbed by a call, then a bool map chooses whether to expand a single argument list",
    """func runNamed(actor runner, text string, values ...any) (outcome, error) {
""",
    """	catalog := grabCatalog(actor)
""",
    """	text, values = map[bool]func() (string, []any){
		true:  func() (string, []any) { return expandNames(catalog, text, values) },
		false: func() (string, []any) { return text, values },
	}[len(values) == 1]()
	return perform(actor, text, values...)
}
""",
    "bind the executor's name catalog before a possible named-placeholder expansion",
    "capture a dependency then branch on arity",
    "read the catalog off the executor",
    [
        ("executor", "name catalog", "dataflow"),
        ("single argument list", "placeholder expansion", "control"),
    ],
)

add(
    "m07",
    "closed and missing sockets are map arms; the hole is the shut-socket refusal",
    """func wrapCodec(link *socket, kind string) (*codedLink, error) {
	if link == nil {
		return nil, faultAbsent("bare socket")
	}
	return map[bool]func() (*codedLink, error){
		true: func() (*codedLink, error) {
""",
    """			return nil, faultShut
""",
    """		},
		false: func() (*codedLink, error) {
			codec := codecFor(kind)
			if codec == nil {
				return nil, faultUnknown(kind)
			}
			return &codedLink{socket: link, codec: codec}, nil
		},
	}[link.gone()]()
}
""",
    "refuse a closed connection before an encoded wrapper is built",
    "boolean dispatch on liveness",
    "return the shut-socket failure and no wrapper",
    [
        ("closed socket", "shut failure", "error"),
        ("live socket", "codec wrapper", "semantic_dependency"),
    ],
)

add(
    "m08",
    "setup steps are a slice of thunks; the hole clears the private probe flag",
    """func installProbe(vm *machine, cb probe, bits byte, every int) {
	steps := []func(){
		func() {
			if cb == nil || bits == 0 {
				cb, bits = nil, 0
			}
		},
		func() {
			if frame := vm.frame; frame.scripted() {
				vm.prevPC = frame.saved
			}
		},
		func() { vm.callback, vm.budget = cb, every },
		func() { vm.rewindBudget() },
		func() { vm.bits = bits },
		func() {
""",
    """			vm.private = false
""",
    """		},
	}
	for _, step := range steps {
		step()
	}
}
""",
    "clear the private hook flag after the probe is installed",
    "sequenced thunks",
    "set the private flag off",
    [("installed probe", "private flag", "dataflow")],
)

add(
    "m09",
    "flag registrations are a slice of calls; the hole binds the follow switch",
    """func boot() {
	top.Attach(order)
	f := order.Switches()
	bind := []func(){
		func() { f.SpanVarP(&window, "sin"+"ce", "s", 5*span.Unit, "how far back") },
		func() { f.TextVarP(&needle, "fil"+"ter", "F", "", "keep only matches") },
		func() {
""",
    """			f.YesVarP(&tail, "fol"+"low", "f", false, "stay attached")
""",
    """		},
	}
	for _, step := range bind {
		step()
	}
}
""",
    "register the follow-logs boolean on the command",
    "table of flag binders",
    "bind the follow switch, defaulting it off",
    [("follow switch", "command flags", "config")],
)

add(
    "m10",
    "options are applied by ranging a copied slice; the hole is the single apply call",
    """func makeClient(choices ...chooser) *actor {
	built := actor{
		budget:  defaultWait,
		retries: defaultTries,
		again:   freshNoRetry(),
	}
	for _, choice := range choices {
""",
    """		choice(&built)
""",
    """	}
	if built.raw == nil {
		built.raw = &web.Agent{Limit: built.budget}
	}
	return &built
}
""",
    "apply one configuration callback to the client under construction",
    "option loop",
    "invoke the callback on the client",
    [("option callback", "client under construction", "dataflow")],
)

add(
    "m11",
    "the subcommand is taken from the tail of the argument vector via a map, not an if",
    """func (svc *worker) drive() (string, error) {
	hint := "verbs: add | drop | go | halt | peek"
	argv := cli.Vector
	verb := map[bool]string{
		true:  "",
		false: argv[1],
	}[len(argv) <= 1]
	if verb == "" {
		return hint, nil
	}
""",
    """	chosen := verb
""",
    """	return map[string]func() (string, error){
		"add":  svc.Add,
		"drop": svc.Drop,
		"go":   svc.Go,
		"halt": svc.Halt,
		"peek": svc.Peek,
	}[chosen]()
}
""",
    "capture the first CLI argument as the service verb",
    "index a verb map",
    "bind the verb taken from the argument vector",
    [("argument vector", "service verb", "dataflow")],
)


def _load_rest() -> None:
    from hand_specs_rest import register
    from hand_specs_c import register as register_c
    from hand_specs_d import register as register_d
    from hand_specs_e import register as register_e
    from hand_specs_f import register as register_f

    register(add)
    register_c(add)
    register_d(add)
    register_e(add)
    register_f(add)


_load_rest()
