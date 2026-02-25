```go
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
This is a go programming task on some code contents. Given task: The task is to fill in the missing part of a go function according to the provided code   content. Below is the package path:github.com/influxdata/influxdb/pkg/tracing Below is the code repository: github.com/influxdata/influxdb/ Below is the imported package path "github.com/xlab/treeprint" The receiver struct definitions of the function is not exist The parameter struct definition or not exist of the function is not exist The return value struct definitions or not exist of the function is *treeVisitor

// Type treeVisitor definition:
type treeVisitor struct {
	root  <ATTN>treeprint.Tree</ATTN>
	trees []<ATTN>treeprint.Tree</ATTN>
}
// Methods:
- func (v *treeVisitor) Visit(n *TreeNode) Visitor
 The code snippets before the function is func (t *TreeNode) String() string {
	if t == nil {
		return ""
	}
	tv := <ATTN>newTreeVisitor()</ATTN>
	Walk(tv, t)
	return tv.root.String()
}

func Walk(v Visitor, node *TreeNode) {
	if v = v.Visit(node); v == nil {
		return
	}

	for _, c := range node.Children {
		Walk(v, c)
	}
} And here is the function you are asked to complete func newTreeVisitor() *treeVisitor
func newTreeVisitor() *treeVisitor { <MID> } Ensure that only missing codes marked as <MID> are returned<|im_end|>
<|im_start|>assistant
	t := treeprint.New()
	return &treeVisitor{root: t, trees: []treeprint.Tree{t}}<|im_end|>
```

```go
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
This is a go programming task on some code contents. Given task: The task is to fill in the missing part of a go function according to the provided code   content. Below is the package path:github.com/etcd-io/etcd/client/v3/concurrency Below is the code repository: github.com/etcd-io/etcd/ Below is the imported package path "context"; "errors"; "fmt"; "strings"; "sync"; "go.etcd.io/etcd/api/v3/etcdserverpb"; "go.etcd.io/etcd/client/v3" The receiver struct definitions of the function is type Mutex struct {
	s *Session

	pfx   string
	myKey string
	myRev int64
	hdr   *pb.ResponseHeader
}

// Methods:
- func (m *Mutex) TryLock(ctx context.Context) error
- func (m *Mutex) Lock(ctx context.Context) error
- func (m *Mutex) tryAcquire(ctx context.Context) ((*v3.TxnResponse, error))
- func (m *Mutex) Unlock(ctx context.Context) error
- func (m *Mutex) IsOwner() v3.Cmp
- func (m *Mutex) Key() string
- func (m *Mutex) Header() *pb.ResponseHeader
 The parameter struct definition or not exist of the function is ctx context.Context The return value struct definitions or not exist of the function is error The code snippets before the function is func NewMutex(s *Session, pfx string) *Mutex {
	return &Mutex{s, pfx + "/", "", -1, nil}
}

func (m *Mutex) TryLock(ctx context.Context) error {
	resp, err := m.tryAcquire(ctx)
	if err != nil {
		return err
	}
	// if no key on prefix / the minimum rev is key, already hold the lock
	ownerKey := resp.Responses[1].GetResponseRange().Kvs
	if len(ownerKey) == 0 || ownerKey[0].CreateRevision == m.myRev {
		m.hdr = resp.Header
		return nil
	}
	client := m.s.Client()
	// Cannot lock, so delete the key
	if _, err := client.Delete(ctx, m.myKey); err != nil {
		return err
	}
	m.myKey = "\x00"
	m.myRev = -1
	return ErrLocked
} And here is the function you are asked to complete func (m *Mutex) Lock(ctx context.Context) error
func (m *Mutex) Lock(ctx context.Context) error {
	resp, err := m.tryAcquire(ctx)
	if err != nil {
		return err
	}
	// if no key on prefix / the minimum rev is key, already hold the lock
	ownerKey := resp.Responses[1].GetResponseRange().Kvs
	if len(ownerKey) == 0 || ownerKey[0].CreateRevision == m.myRev {
		m.hdr = resp.Header
		return nil
	}
	client := m.s.Client()
	// wait for deletion revisions prior to myKey
	// TODO: early termination if the session key is deleted before other session keys with smaller revisions.
	werr := waitDeletes(ctx, client, m.pfx, m.myRev-1)
	// release lock key if wait failed
	if werr != nil {
		m.Unlock(client.Ctx())
		return werr
	}

	// make sure the session is not expired, and the owner key still exists.
	gresp, werr := client.Get(ctx, m.myKey)
	if werr != nil { <MID> 	}

	if len(gresp.Kvs) == 0 { // is the session key lost?
		return ErrSessionExpired
	}
	m.hdr = gresp.Header

	return nil
} Ensure that only missing codes marked as <MID> are returned<|im_end|>
<|im_start|>assistant
		<ATTN>m.Unlock(client.Ctx())</ATTN>
		<ATTN>return werr</ATTN><|im_end|>
```

```go
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
This is a go programming task on some code contents. Given task: The task is to fill in the missing part of a go function according to the provided code   content. Below is the package path:github.com/urfave/cli Below is the code repository: github.com/urfave/cli/ Below is the imported package path "fmt"; "time" The receiver struct definitions of the function is *Command The parameter struct definition or not exist of the function is name string The return value struct definitions or not exist of the function is any The code snippets before the function is func (a *ArgumentBase[T, C, VC]) Get() any {
	if a.value != nil {
		return *a.value
	}
	return a.Value
}

func (a *ArgumentsBase[T, C, VC]) HasName(s string) bool {
	return s == a.Name
}

func (a *ArgumentsBase[T, C, VC]) Usage() string {
	if a.UsageText != "" {
		return a.UsageText
	}

	usageFormat := ""
	if a.Min == 0 {
		if a.Max == 1 {
			usageFormat = "[%[1]s]"
		} else {
			usageFormat = "[%[1]s ...]"
		}
	} else {
		usageFormat = "%[1]s [%[1]s ...]"
	}
	return fmt.Sprintf(usageFormat, a.Name)
}

func (a *ArgumentsBase[T, C, VC]) Parse(s []string) ([]string, error) {
	tracef("calling arg%[1] parse with args %[2]", &a.Name, s)
	if a.Max == 0 {
		return s, fmt.Errorf("args %s has max 0, not parsing argument", a.Name)
	}
	if a.Max != -1 && a.Min > a.Max {
		return s, fmt.Errorf("args %s has min[%d] > max[%d], not parsing argument", a.Name, a.Min, a.Max)
	}

	count := 0
	var vc VC
	var t T
	value := vc.Create(a.Value, &t, a.Config)
	a.values = []T{}

	tracef("attempting arg%[1] parse", &a.Name)
	for _, arg := range s {
		if err := value.Set(arg); err != nil {
			return s, fmt.Errorf("invalid value %q for argument %s: %v", arg, a.Name, err)
		}
		tracef("set arg%[1] one value", &a.Name, value.Get().(T))
		a.values = append(a.values, value.Get().(T))
		count++
		if count >= a.Max && a.Max > -1 {
			break
		}
	}
	if count < a.Min {
		return s, fmt.Errorf("sufficient count of arg %s not provided, given %d expected %d", a.Name, count, a.Min)
	}

	if a.Destination != nil {
		tracef("appending destination")
		*a.Destination = a.values // append(*a.Destination, a.values...)
	}

	return s[count:], nil
}

func (a *ArgumentsBase[T, C, VC]) Get() any {
	if a.values != nil {
		return a.values
	}
	return []T{}
} And here is the function you are asked to complete func (c *Command) getArgValue(name string) any
func (c *Command) getArgValue(name string) any {
	tracef("command %s looking for args %s", c.Name, name)
	for _, arg := range <ATTN>c.Arguments</ATTN> { <MID> 	}
	tracef("command %s did not find args %s", c.Name, name)
	return nil
} Ensure that only missing codes marked as <MID> are returned<|im_end|>
<|im_start|>assistant
		if arg.<ATTN>HasName</ATTN>(name) {
			tracef("command %s found args %s", c.Name, name)
			return arg.<ATTN>Get</ATTN>()
		}<|im_end|>
```

```go
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
This is a go programming task on some code contents. Given task: The task is to fill in the missing part of a go function according to the provided code   content. Below is the package path:github.com/kubernetes/Kubernetes/staging/src/k8s.io/apiserver/pkg/server Below is the code repository: github.com/kubernetes/Kubernetes/ Below is the imported package path "context"; "crypto/tls"; "fmt"; "io"; "log"; "net"; "net/http"; "os"; "strings"; "time"; "golang.org/x/net/http2"; "k8s.io/component-base/cli/flag"; "k8s.io/klog/v2"; "k8s.io/apimachinery/pkg/util/runtime"; "k8s.io/apiserver/pkg/endpoints/metrics"; "k8s.io/apiserver/pkg/server/dynamiccertificates" The receiver struct definitions of the function is type tlsHandshakeErrorWriter struct {
	out io.Writer
}

// Methods:
- func (w *tlsHandshakeErrorWriter) Write(p *ast.ArrayType) ((int, error))
 The parameter struct definition or not exist of the function is p *ast.ArrayType The return value struct definitions or not exist of the function is (int, error) The code snippets before the function is func (s *SecureServingInfo) tlsConfig(stopCh <-chan struct{}) (*tls.Config, error) {
	tlsConfig := &tls.Config{
		// Can't use SSLv3 because of POODLE and BEAST
		// Can't use TLSv1.0 because of POODLE and BEAST using CBC cipher
		// Can't use TLSv1.1 because of RC4 cipher usage
		MinVersion: tls.VersionTLS12,
		// enable HTTP2 for go's 1.7 HTTP Server
		NextProtos: []string{"h2", "http/1.1"},
	}

	// these are static aspects of the tls.Config
	if s.DisableHTTP2 {
		klog.Info("Forcing use of http/1.1 only")
		tlsConfig.NextProtos = []string{"http/1.1"}
	}
	if s.MinTLSVersion > 0 {
		tlsConfig.MinVersion = s.MinTLSVersion
	}
	if len(s.CipherSuites) > 0 {
		tlsConfig.CipherSuites = s.CipherSuites
		insecureCiphers := flag.InsecureTLSCiphers()
		for i := 0; i < len(s.CipherSuites); i++ {
			for cipherName, cipherID := range insecureCiphers {
				if s.CipherSuites[i] == cipherID {
					klog.Warningf("Use of insecure cipher '%s' detected.", cipherName)
				}
			}
		}
	}

	if s.ClientCA != nil {
		// Populate PeerCertificates in requests, but don't reject connections without certificates
		// This allows certificates to be validated by authenticators, while still allowing other auth types
		tlsConfig.ClientAuth = tls.RequestClientCert
	}

	if s.ClientCA != nil || s.Cert != nil || len(s.SNICerts) > 0 {
		dynamicCertificateController := dynamiccertificates.NewDynamicServingCertificateController(
			tlsConfig,
			s.ClientCA,
			s.Cert,
			s.SNICerts,
			nil, // TODO see how to plumb an event recorder down in here. For now this results in simply klog messages.
		)

		if s.ClientCA != nil {
			s.ClientCA.AddListener(dynamicCertificateController)
		}
		if s.Cert != nil {
			s.Cert.AddListener(dynamicCertificateController)
		}
		// generate a context from stopCh. This is to avoid modifying files which are relying on apiserver
		// TODO: See if we can pass ctx to the current method
		ctx, cancel := context.WithCancel(context.Background())
		go func() {
			select {
			case <-stopCh:
				cancel() // stopCh closed, so cancel our context
			case <-ctx.Done():
			}
		}()
		// start controllers if possible
		if controller, ok := s.ClientCA.(dynamiccertificates.ControllerRunner); ok {
			// runonce to try to prime data.  If this fails, it's ok because we fail closed.
			// Files are required to be populated already, so this is for convenience.
			if err := controller.RunOnce(ctx); err != nil {
				klog.Warningf("Initial population of client CA failed: %v", err)
			}

			go controller.Run(ctx, 1)
		}
		if controller, ok := s.Cert.(dynamiccertificates.ControllerRunner); ok {
			// runonce to try to prime data.  If this fails, it's ok because we fail closed.
			// Files are required to be populated already, so this is for convenience.
			if err := controller.RunOnce(ctx); err != nil {
				klog.Warningf("Initial population of default serving certificate failed: %v", err)
			}

			go controller.Run(ctx, 1)
		}
		for _, sniCert := range s.SNICerts {
			sniCert.AddListener(dynamicCertificateController)
			if controller, ok := sniCert.(dynamiccertificates.ControllerRunner); ok {
				// runonce to try to prime data.  If this fails, it's ok because we fail closed.
				// Files are required to be populated already, so this is for convenience.
				if err := controller.RunOnce(ctx); err != nil {
					klog.Warningf("Initial population of SNI serving certificate failed: %v", err)
				}

				go controller.Run(ctx, 1)
			}
		}

		// runonce to try to prime data.  If this fails, it's ok because we fail closed.
		// Files are required to be populated already, so this is for convenience.
		if err := dynamicCertificateController.RunOnce(); err != nil {
			klog.Warningf("Initial population of dynamic certificates failed: %v", err)
		}
		go dynamicCertificateController.Run(1, stopCh)

		tlsConfig.GetConfigForClient = dynamicCertificateController.GetConfigForClient
	}

	return tlsConfig, nil
}

func (s *SecureServingInfo) Serve(handler http.Handler, shutdownTimeout time.Duration, stopCh <-chan struct{}) (<-chan struct{}, <-chan struct{}, error) {
	if s.Listener == nil {
		return nil, nil, fmt.Errorf("listener must not be nil")
	}

	tlsConfig, err := s.tlsConfig(stopCh)
	if err != nil {
		return nil, nil, err
	}

	secureServer := &http.Server{
		Addr:           s.Listener.Addr().String(),
		Handler:        handler,
		MaxHeaderBytes: 1 << 20,
		TLSConfig:      tlsConfig,

		IdleTimeout:       90 * time.Second, // matches http.DefaultTransport keep-alive timeout
		ReadHeaderTimeout: 32 * time.Second, // just shy of requestTimeoutUpperBound
	}

	if !s.DisableHTTP2 {
		// At least 99% of serialized resources in surveyed clusters were smaller than 256kb.
		// This should be big enough to accommodate most API POST requests in a single frame,
		// and small enough to allow a per connection buffer of this size multiplied by `MaxConcurrentStreams`.
		const resourceBody99Percentile = 256 * 1024

		http2Options := &http2.Server{
			IdleTimeout: 90 * time.Second, // matches http.DefaultTransport keep-alive timeout
			// shrink the per-stream buffer and max framesize from the 1MB default while still accommodating most API POST requests in a single frame
			MaxUploadBufferPerStream: resourceBody99Percentile,
			MaxReadFrameSize:         resourceBody99Percentile,
		}

		// use the overridden concurrent streams setting or make the default of 250 explicit so we can size MaxUploadBufferPerConnection appropriately
		if s.HTTP2MaxStreamsPerConnection > 0 {
			http2Options.MaxConcurrentStreams = uint32(s.HTTP2MaxStreamsPerConnection)
		} else {
			// match http2.initialMaxConcurrentStreams used by clients
			// this makes it so that a malicious client can only open 400 streams before we forcibly close the connection
			// https://github.com/golang/net/commit/b225e7ca6dde1ef5a5ae5ce922861bda011cfabd
			http2Options.MaxConcurrentStreams = 100
		}

		// increase the connection buffer size from the 1MB default to handle the specified number of concurrent streams
		http2Options.MaxUploadBufferPerConnection = http2Options.MaxUploadBufferPerStream * int32(http2Options.MaxConcurrentStreams)
		// apply settings to the server
		if err := http2.ConfigureServer(secureServer, http2Options); err != nil {
			return nil, nil, fmt.Errorf("error configuring http2: %v", err)
		}
	}

	// use tlsHandshakeErrorWriter to handle messages of tls handshake error
	tlsErrorWriter := &tlsHandshakeErrorWriter{os.Stderr}
	tlsErrorLogger := log.New(tlsErrorWriter, "", 0)
	secureServer.ErrorLog = tlsErrorLogger

	klog.Infof("Serving securely on %s", secureServer.Addr)
	return RunServer(secureServer, s.Listener, shutdownTimeout, stopCh)
}

) (<-chan struct{}, <-chan struct{}, error) {
	if ln == nil {
		return nil, nil, fmt.Errorf("listener must not be nil")
	}

	// Shutdown server gracefully.
	serverShutdownCh, listenerStoppedCh := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(serverShutdownCh)
		<-stopCh
		ctx, cancel := context.WithTimeout(context.Background(), shutDownTimeout)
		defer cancel()
		err := server.Shutdown(ctx)
		if err != nil {
			klog.Errorf("Failed to shutdown server: %v", err)
		}
	}()

	go func() {
		defer utilruntime.HandleCrash()
		defer close(listenerStoppedCh)

		var listener net.Listener
		listener = tcpKeepAliveListener{ln}
		if server.TLSConfig != nil {
			listener = tls.NewListener(listener, server.TLSConfig)
		}

		err := server.Serve(listener)

		msg := fmt.Sprintf("Stopped listening on %s", ln.Addr().String())
		select {
		case <-stopCh:
			klog.Info(msg)
		default:
			panic(fmt.Sprintf("%s due to error: %v", msg, err))
		}
	}()

	return serverShutdownCh, listenerStoppedCh, nil
}

func (ln tcpKeepAliveListener) Accept() (net.Conn, error) {
	c, err := ln.Listener.Accept()
	if err != nil {
		return nil, err
	}
	if tc, ok := c.(*net.TCPConn); ok {
		tc.SetKeepAlive(true)
		tc.SetKeepAlivePeriod(defaultKeepAlivePeriod)
	}
	return c, nil
} And here is the function you are asked to complete func (w *tlsHandshakeErrorWriter) Write(p *ast.ArrayType) ((int, error))
func (w *tlsHandshakeErrorWriter) Write(p []byte) (int, error) {
	if strings.Contains(string(p), tlsHandshakeErrorPrefix) { <MID> 	}

	// for non tls handshake error, log it as usual
	return w.out.Write(p)
} Ensure that only missing codes marked as <MID> are returned<|im_end|>
<|im_start|>assistant
		<ATTN>klog.V(5).Info</ATTN>(string(p))
		<ATTN>metrics.TLSHandshakeErrors.Inc()</ATTN>
		<ATTN>return len(p), nil</ATTN><|im_end|>
```

```go
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
This is a go programming task on some code contents. Given task: The task is to fill in the missing part of a go function according to the provided code   content. Below is the package path:github.com/cockroachdb/cockroach/pkg/cli Below is the code repository: github.com/cockroachdb/cockroach/ Below is the imported package path "context"; "flag"; "os"; "path/filepath"; "strconv"; "strings"; "github.com/cockroachdb/cockroach/pkg/base"; "github.com/cockroachdb/cockroach/pkg/cli/cliflags"; "github.com/cockroachdb/cockroach/pkg/storage/fs"; "github.com/cockroachdb/cockroach/pkg/util/envutil"; "github.com/cockroachdb/cockroach/pkg/util/humanizeutil"; "github.com/cockroachdb/cockroach/pkg/util/log"; "github.com/cockroachdb/cockroach/pkg/util/log/channel"; "github.com/cockroachdb/cockroach/pkg/util/log/logconfig"; "github.com/cockroachdb/cockroach/pkg/util/log/logcrash"; "github.com/cockroachdb/cockroach/pkg/util/log/severity"; "github.com/cockroachdb/errors"; "github.com/spf13/cobra" The receiver struct definitions of the function is type stringValue struct {
	*<ATTN>settableString</ATTN>
}

// Methods:
- func (l *stringValue) Set(s string) error
- func (l stringValue) Type() string
- func (l stringValue) String() string
 The parameter struct definition or not exist of the function is <ATTN>s string</ATTN> The return value struct definitions or not exist of the function is <ATTN>error</ATTN> The code snippets before the function is func getDefaultLogDirFromStores() (dir *string, ambiguousLogDirs bool) {
	// Default the log directory to the "logs" subdirectory of the first
	// non-memory store.
	for _, spec := range serverCfg.Stores.Specs {
		if spec.InMemory {
			continue
		}
		if dir != nil {
			ambiguousLogDirs = true
			break
		}
		s := filepath.Join(spec.Path, "logs")
		dir = &s
	}

	return
}

func newLogConfigOverrides() *logConfigFlags {
	l := &logConfigFlags{}
	l.fileMaxSizeVal = humanizeutil.NewBytesValue(&l.fileMaxSize)
	l.maxGroupSizeVal = humanizeutil.NewBytesValue(&l.maxGroupSize)
	l.reset()
	return l
}

func (l *logConfigFlags) reset() {
	d := logconfig.DefaultConfig()

	l.logDir = settableString{}
	l.sqlAuditLogDir = settableString{}
	*l.fileMaxSizeVal = *humanizeutil.NewBytesValue(&l.fileMaxSize)
	*l.maxGroupSizeVal = *humanizeutil.NewBytesValue(&l.maxGroupSize)
	l.fileMaxSize = int64(*d.FileDefaults.MaxFileSize)
	l.maxGroupSize = int64(*d.FileDefaults.MaxGroupSize)
	l.fileThreshold = severity.UNKNOWN
	l.stderrThreshold = severity.UNKNOWN
	l.stderrNoColor = settableBool{}
	l.redactableLogs = settableBool{}
}

func (l *logConfigFlags) anySet() bool {
	return l.logDir.isSet ||
		l.sqlAuditLogDir.isSet ||
		l.fileMaxSizeVal.IsSet() ||
		l.maxGroupSizeVal.IsSet() ||
		l.fileThreshold.IsSet() ||
		l.stderrThreshold.IsSet() ||
		l.stderrNoColor.isSet ||
		l.redactableLogs.isSet
}

) {
	if l.logDir.isSet && l.logDir.s != "" {
		// Note: the case where .s == "" is handled in setupLogging()
		// above.
		c.FileDefaults.Dir = &l.logDir.s
	}
	if l.fileMaxSizeVal.IsSet() {
		s := logconfig.ByteSize(l.fileMaxSize)
		c.FileDefaults.MaxFileSize = &s
	}
	if l.maxGroupSizeVal.IsSet() {
		s := logconfig.ByteSize(l.maxGroupSize)
		c.FileDefaults.MaxGroupSize = &s
	}
	if l.fileThreshold.IsSet() {
		c.FileDefaults.Filter = l.fileThreshold
	}
	if l.stderrThreshold.IsSet() {
		if l.stderrThreshold == severity.DEFAULT {
			c.Sinks.Stderr.Filter = commandSpecificDefaultLegacyStderrOverride
		} else {
			c.Sinks.Stderr.Filter = l.stderrThreshold
		}
	}
	if l.stderrNoColor.isSet {
		c.Sinks.Stderr.NoColor = l.stderrNoColor.val
	}
	if l.redactableLogs.isSet {
		c.FileDefaults.Redactable = &l.redactableLogs.val
		c.Sinks.Stderr.Redactable = &l.redactableLogs.val
	}
} And here is the function you are asked to complete func (l *stringValue) Set(s string) error
func (l *stringValue) Set(s string) error { <MID> } Ensure that only missing codes marked as <MID> are returned<|im_end|>
<|im_start|>assistant
	l.<ATTN>s</ATTN> = s
	l.<ATTN>isSet</ATTN> = true
	return <ATTN>nil</ATTN><|im_end|>
```

```go
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
This is a go programming task on some code contents. Given task: The task is to fill in the missing part of a go function according to the provided code   content. Below is the package path:github.com/cockroachdb/cockroach/pkg/util/admission Below is the code repository: github.com/cockroachdb/cockroach/ Below is the imported package path "time"; "github.com/cockroachdb/cockroach/pkg/roachpb"; "github.com/cockroachdb/cockroach/pkg/util/admission/admissionpb"; "github.com/cockroachdb/cockroach/pkg/util/metric"; "github.com/cockroachdb/cockroach/pkg/util/syncutil"; "github.com/cockroachdb/cockroach/pkg/util/timeutil"; "github.com/cockroachdb/errors"; "github.com/cockroachdb/pebble" The receiver struct definitions of the function is type kvStoreTokenGranter struct {
	knobs             *TestingKnobs
	regularRequester  requester
	elasticRequester  requester
	snapshotRequester requester

	mu struct {
		syncutil.Mutex
		// There is no rate limiting in granting these tokens. That is, they are
		// all burst tokens.

		// The "IO" tokens represent flush/compaction capacity into/out of L0. All
		// work deducts from both availableIOTokens and availableElasticIOTokens.
		// Regular work blocks if availableIOTokens is <= 0 and elastic work
		// blocks if availableElasticIOTokens <= 0.
		availableIOTokens            [admissionpb.NumWorkClasses]int64
		elasticIOTokensUsedByElastic int64
		// TODO(aaditya): add support for read/IOPS tokens.
		// Disk bandwidth tokens.
		diskTokensAvailable diskTokens
		diskTokensError     struct {
			// prevObserved{Writes,Reads} is the observed disk metrics in the last
			// call to adjustDiskTokenErrorLocked. These are used to compute the
			// delta.
			prevObservedWrites             uint64
			prevObservedReads              uint64
			diskWriteTokensAlreadyDeducted int64
			diskReadTokensAlreadyDeducted  int64
		}
		diskTokensUsed [admissionpb.NumStoreWorkTypes]diskTokens
		// exhaustedStart is the time when the corresponding availableIOTokens
		// became <= 0. Ignored when the corresponding availableIOTokens is > 0.
		exhaustedStart [admissionpb.NumWorkClasses]time.Time
		// startingIOTokens is the number of tokens set by setAvailableTokens for
		// regular work. It is used to compute the tokens used, by computing
		// startingIOTokens-availableIOTokens[RegularWorkClass].
		startingIOTokens int64

		// Estimation models.
		l0WriteLM, l0IngestLM, ingestLM, writeAmpLM tokensLinearModel
	}

	ioTokensExhaustedDurationMetric [admissionpb.NumWorkClasses]*metric.Counter
	availableTokensMetric           [admissionpb.NumWorkClasses]*metric.Gauge
	tokensReturnedMetric            *metric.Counter
	tokensTakenMetric               *metric.Counter
}

// Methods:
- func (sg *kvStoreTokenGranter) tryGet((workType admissionpb.StoreWorkType, count int64)) bool
- func (sg *kvStoreTokenGranter) tryGetLocked((count int64, wt admissionpb.StoreWorkType)) bool
- func (sg *kvStoreTokenGranter) returnGrant((workType admissionpb.StoreWorkType, count int64))
- func (sg *kvStoreTokenGranter) returnGrantLocked((count int64, wt admissionpb.StoreWorkType))
- func (sg *kvStoreTokenGranter) subtractTokensForStoreWorkTypeLocked((wt admissionpb.StoreWorkType, count int64))
- func (sg *kvStoreTokenGranter) adjustDiskTokenError(m StoreMetrics)
- func (sg *kvStoreTokenGranter) adjustDiskTokenErrorLocked((readBytes uint64, writeBytes uint64))
- func (sg *kvStoreTokenGranter) tookWithoutPermission((workType admissionpb.StoreWorkType, count int64))
- func (sg *kvStoreTokenGranter) tookWithoutPermissionLocked((count int64, wt admissionpb.StoreWorkType))
- func (sg *kvStoreTokenGranter) subtractIOTokensLocked((count int64, elasticCount int64, settingAvailableTokens bool))
- func (sg *kvStoreTokenGranter) subtractTokensLockedForWorkClass((wc admissionpb.WorkClass, count int64, settingAvailableTokens bool))
- func (sg *kvStoreTokenGranter) tryGrant()
- func (sg *kvStoreTokenGranter) tryGrantLocked()
- func (sg *kvStoreTokenGranter) tryGrantLockedOne() bool
- func (sg *kvStoreTokenGranter) setAvailableTokens((ioTokens int64, elasticIOTokens int64, diskWriteTokens int64, diskReadTokens int64, ioTokenCapacity int64, elasticIOTokenCapacity int64, diskWriteTokensCapacity int64, lastTick bool)) ((ioTokensUsed int64, ioTokensUsedByElasticWork int64))
- func (sg *kvStoreTokenGranter) getDiskTokensUsedAndReset() usedTokens *ast.ArrayType
- func (sg *kvStoreTokenGranter) setLinearModels((l0WriteLM tokensLinearModel, l0IngestLM tokensLinearModel, ingestLM tokensLinearModel, writeAmpLM tokensLinearModel))
- func (sg *kvStoreTokenGranter) storeReplicatedWorkAdmitted((wt admissionpb.StoreWorkType, originalTokens int64, admittedInfo storeReplicatedWorkAdmittedInfo, canGrantAnother bool)) additionalTokens int64
- func (sg *kvStoreTokenGranter) storeReplicatedWorkAdmittedLocked((wt admissionpb.StoreWorkType, originalTokens int64, admittedInfo storeReplicatedWorkAdmittedInfo, canGrantAnother bool)) additionalTokens int64
 The parameter struct definition or not exist of the function is (count int64, wt admissionpb.StoreWorkType) The return value struct definitions or not exist of the function is bool The code snippets before the function is func (cg *kvStoreTokenChildGranter) tookWithoutPermission(count int64) {
	cg.parent.tookWithoutPermission(cg.workType, count)
}

func (cg *kvStoreTokenChildGranter) continueGrantChain(grantChainID grantChainID) {
	// Ignore since grant chains are not used for store tokens.
}

) (additionalTokens int64) {
	// NB: the token/metric adjustments we want to make here are the same as we
	// want to make through the storeReplicatedWorkAdmittedLocked, so we (ab)use
	// it. The one difference is that post token adjustments, if we observe the
	// granter was previously exhausted but is no longer so, we're allowed to
	// admit other waiting requests.
	return cg.parent.storeReplicatedWorkAdmitted(
		cg.workType, originalTokens, storeReplicatedWorkAdmittedInfo(doneInfo), true /* canGrantAnother */)
}

) (additionalTokens int64) {
	return cg.parent.storeReplicatedWorkAdmittedLocked(cg.workType, originalTokens, admittedInfo, false /* canGrantAnother */)
}

func (sg *kvStoreTokenGranter) tryGet(workType admissionpb.StoreWorkType, count int64) bool {
	sg.mu.Lock()
	defer sg.mu.Unlock()
	return sg.tryGetLocked(count, workType)
} And here is the function you are asked to complete func (sg *kvStoreTokenGranter) tryGetLocked((count int64, wt admissionpb.StoreWorkType)) bool
func (sg *kvStoreTokenGranter) tryGetLocked(count int64, wt admissionpb.StoreWorkType) bool {
	// NB: ideally if regularRequester.hasWaitingRequests() returns true and
	// wc==elasticWorkClass we should reject this request, since it means that
	// more important regular work is waiting. However, we rely on the
	// assumption that elasticWorkClass, once throttled, will have a non-empty
	// queue, and since the only case where tryGetLocked is called for
	// elasticWorkClass is when the queue is empty, this case should be rare
	// (and not cause a performance isolation failure).

	// NB: For disk write tokens, we apply `writeAmpLM` to account for write
	// amplification. This model is used again later in
	// `storeReplicatedWorkAdmittedLocked()`. There is an obvious gap in
	// accounting here, since the model could change between the two calls for the
	// same request. For example:
	// (1) We have a request that requests 50 tokens, and write-amp LM is
	//     currently 10.0x + 1. We will deduct 501 tokens for disk writes.
	// (2) Before we adjust the actual write bytes used by the write request, the
	//     write-amp model is updated to 5.0x + 1.
	// (3) In `storeReplicatedWorkAdmittedLocked()`, we learn that the request
	//     used 200 actual bytes. Now we will apply the new LM to get 251 tokens
	//     initially deducted, and apply the LM for 1001 actual bytes used. We
	//     will now deduct 750 additional tokens from the bucket. Now we have
	//     deducted 1251 tokens rather than 1001.
	// This can also go the other way, where we deduct fewer tokens than actually
	// needed. We are generally okay with this since the model changes
	// infrequently (every 15s), and the disk bandwidth limiter is designed to
	// generally under admit and only pace elastic work.
	diskWriteTokens := count
	if wt != admissionpb.SnapshotIngestStoreWorkType {
		// Snapshot ingests do not incur the write amplification described above, so
		// we skip applying the model for those writes.
		diskWriteTokens = sg.mu.writeAmpLM.applyLinearModel(count)
	}
	switch wt {
	case admissionpb.RegularStoreWorkType:
		if <ATTN></ATTN>sg.mu.availableIOTokens[admissionpb.RegularWorkClass]<ATTN></ATTN> > 0 { <MID> 		}
	case admissionpb.ElasticStoreWorkType:
		if sg.mu.diskTokensAvailable.writeByteTokens > 0 &&
			<ATTN></ATTN>sg.mu.availableIOTokens[admissionpb.RegularWorkClass]<ATTN></ATTN> > 0 &&
			<ATTN></ATTN>sg.mu.availableIOTokens[admissionpb.ElasticWorkClass]<ATTN></ATTN> > 0 {
			<ATTN></ATTN>sg.subtractIOTokensLocked(count, count, false)<ATTN></ATTN>
			sg.mu.elasticIOTokensUsedByElastic += count
			sg.mu.diskTokensAvailable.writeByteTokens -= diskWriteTokens
			sg.mu.diskTokensError.diskWriteTokensAlreadyDeducted += diskWriteTokens
			sg.mu.diskTokensUsed[wt].writeByteTokens += diskWriteTokens
			return true
		}
	case admissionpb.SnapshotIngestStoreWorkType:
		// Snapshot ingests do not go into L0, so we only subject them to
		// writeByteTokens.
		if sg.mu.diskTokensAvailable.writeByteTokens > 0 {
			sg.mu.diskTokensAvailable.writeByteTokens -= diskWriteTokens
			sg.mu.diskTokensError.diskWriteTokensAlreadyDeducted += diskWriteTokens
			sg.mu.diskTokensUsed[wt].writeByteTokens += diskWriteTokens
			return true
		}
	}
	return false
} Ensure that only missing codes marked as <MID> are returned<|im_end|>
<|im_start|>assistant
			sg.subtractIOTokensLocked(count, count, false)
			sg.mu.diskTokensAvailable.writeByteTokens -= diskWriteTokens
			sg.mu.diskTokensError.diskWriteTokensAlreadyDeducted += diskWriteTokens
			sg.mu.diskTokensUsed[wt].writeByteTokens += diskWriteTokens
			return true<|im_end|>
```

```go
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
This is a go programming task on some code contents. Given task: The task is to fill in the missing part of a go function according to the provided code   content. Below is the package path:github.com/kubernetes/Kubernetes/vendor/google.golang.org/grpc Below is the code repository: github.com/kubernetes/Kubernetes/ Below is the imported package path "context"; "errors"; "fmt"; "io"; "math"; "net"; "net/http"; "reflect"; "runtime"; "strings"; "sync"; "sync/atomic"; "time"; "google.golang.org/grpc/codes"; "google.golang.org/grpc/credentials"; "google.golang.org/grpc/encoding"; "google.golang.org/grpc/encoding/proto"; "google.golang.org/grpc/experimental/stats"; "google.golang.org/grpc/grpclog"; "google.golang.org/grpc/internal"; "google.golang.org/grpc/internal/binarylog"; "google.golang.org/grpc/internal/channelz"; "google.golang.org/grpc/internal/grpcsync"; "google.golang.org/grpc/internal/grpcutil"; "google.golang.org/grpc/internal/stats"; "google.golang.org/grpc/internal/transport"; "google.golang.org/grpc/keepalive"; "google.golang.org/grpc/mem"; "google.golang.org/grpc/metadata"; "google.golang.org/grpc/peer"; "google.golang.org/grpc/stats"; "google.golang.org/grpc/status"; "google.golang.org/grpc/tap" The receiver struct definitions of the function is type listenSocket struct {
	net.Listener
	channelz *channelz.Socket
}

// Methods:
- func (l *listenSocket) <ATTN>Close</ATTN>() error
 The parameter struct definition or not exist of the function is not exist The return value struct definitions or not exist of the function is error The code snippets before the function is func (s *Server) printf(format string, a ...any) {
	if s.events != nil {
		s.events.Printf(format, a...)
	}
}

func (s *Server) errorf(format string, a ...any) {
	if s.events != nil {
		s.events.Errorf(format, a...)
	}
}

func (s *Server) RegisterService(sd *ServiceDesc, ss any) {
	if ss != nil {
		ht := reflect.TypeOf(sd.HandlerType).Elem()
		st := reflect.TypeOf(ss)
		if !st.Implements(ht) {
			logger.Fatalf("grpc: Server.RegisterService found the handler of type %v that does not satisfy %v", st, ht)
		}
	}
	s.register(sd, ss)
}

func (s *Server) register(sd *ServiceDesc, ss any) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.printf("RegisterService(%q)", sd.ServiceName)
	if s.serve {
		logger.Fatalf("grpc: Server.RegisterService after Server.Serve for %q", sd.ServiceName)
	}
	if _, ok := s.services[sd.ServiceName]; ok {
		logger.Fatalf("grpc: Server.RegisterService found duplicate service registration for %q", sd.ServiceName)
	}
	info := &serviceInfo{
		serviceImpl: ss,
		methods:     make(map[string]*MethodDesc),
		streams:     make(map[string]*StreamDesc),
		mdata:       sd.Metadata,
	}
	for i := range sd.Methods {
		d := &sd.Methods[i]
		info.methods[d.MethodName] = d
	}
	for i := range sd.Streams {
		d := &sd.Streams[i]
		info.streams[d.StreamName] = d
	}
	s.services[sd.ServiceName] = info
}

func (s *Server) GetServiceInfo() map[string]ServiceInfo {
	ret := make(map[string]ServiceInfo)
	for n, srv := range s.services {
		methods := make([]MethodInfo, 0, len(srv.methods)+len(srv.streams))
		for m := range srv.methods {
			methods = append(methods, MethodInfo{
				Name:           m,
				IsClientStream: false,
				IsServerStream: false,
			})
		}
		for m, d := range srv.streams {
			methods = append(methods, MethodInfo{
				Name:           m,
				IsClientStream: d.ClientStreams,
				IsServerStream: d.ServerStreams,
			})
		}

		ret[n] = ServiceInfo{
			Methods:  methods,
			Metadata: srv.mdata,
		}
	}
	return ret
} And here is the function you are asked to complete func (l *listenSocket) Close() error
func (l *listenSocket) Close() error { <MID> } Ensure that only missing codes marked as <MID> are returned<|im_end|>
<|im_start|>assistant
	err := l.<ATTN>Listener</ATTN>.<ATTN>Close</ATTN>()
	<ATTN>channelz.RemoveEntry</ATTN>(l.<ATTN>channelz.ID</ATTN>)
	<ATTN>channelz.Info</ATTN>(logger, l.channelz, "ListenSocket deleted")
	return err<|im_end|>
```

```go
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
This is a go programming task on some code contents. Given task: The task is to fill in the missing part of a go function according to the provided code   content. Below is the package path:github.com/kubernetes/Kubernetes/cmd/cloud-controller-manager Below is the code repository: github.com/kubernetes/Kubernetes/ Below is the imported package path "context"; "errors"; "fmt"; "net"; "strings"; "k8s.io/cloud-provider"; "k8s.io/cloud-provider/app"; "k8s.io/cloud-provider/app/config"; "k8s.io/controller-manager/app"; "k8s.io/controller-manager/controller"; "k8s.io/klog/v2"; "k8s.io/kubernetes/cmd/kube-controller-manager/app/options"; "k8s.io/kubernetes/pkg/controller/nodeipam"; "k8s.io/kubernetes/pkg/controller/nodeipam/config"; "k8s.io/kubernetes/pkg/controller/nodeipam/ipam"; "k8s.io/utils/net" The receiver struct definitions of the function is not exist The parameter struct definition or not exist of the function is (ctx context.Context, initContext app.ControllerInitContext, ccmConfig *cloudcontrollerconfig.CompletedConfig, nodeIPAMConfig nodeipamconfig.NodeIPAMControllerConfiguration, controllerCtx genericcontrollermanager.ControllerContext, cloud cloudprovider.Interface) The return value struct definitions or not exist of the function is (controller.Interface, bool, error) The code snippets before the function is func (nodeIpamController *nodeIPAMController) StartNodeIpamControllerWrapper(initContext app.ControllerInitContext, completedConfig *cloudcontrollerconfig.CompletedConfig, cloud cloudprovider.Interface) app.InitFunc {
	allErrors := nodeIpamController.nodeIPAMControllerOptions.Validate()
	if len(allErrors) > 0 {
		klog.Fatal("NodeIPAM controller values are not properly set.")
	}
	nodeIpamController.nodeIPAMControllerOptions.ApplyTo(&nodeIpamController.nodeIPAMControllerConfiguration)

	return func(ctx context.Context, controllerContext genericcontrollermanager.ControllerContext) (controller.Interface, bool, error) {
		return startNodeIpamController(ctx, initContext, completedConfig, nodeIpamController.nodeIPAMControllerConfiguration, controllerContext, cloud)
	}
} And here is the function you are asked to complete func startNodeIpamController((ctx context.Context, initContext app.ControllerInitContext, ccmConfig *cloudcontrollerconfig.CompletedConfig, nodeIPAMConfig nodeipamconfig.NodeIPAMControllerConfiguration, controllerCtx genericcontrollermanager.ControllerContext, cloud cloudprovider.Interface)) ((controller.Interface, bool, error))
func startNodeIpamController(ctx context.Context, initContext app.ControllerInitContext, ccmConfig *cloudcontrollerconfig.CompletedConfig, nodeIPAMConfig nodeipamconfig.NodeIPAMControllerConfiguration, controllerCtx genericcontrollermanager.ControllerContext, cloud cloudprovider.Interface) (controller.Interface, bool, error) {
	var serviceCIDR *net.IPNet
	var secondaryServiceCIDR *net.IPNet

	// should we start nodeIPAM
	if !ccmConfig.ComponentConfig.KubeCloudShared.AllocateNodeCIDRs {
		return nil, false, nil
	}

	// Cannot run cloud ipam controller if cloud provider is nil (--cloud-provider not set or set to 'external')
	if cloud == nil && ccmConfig.ComponentConfig.KubeCloudShared.CIDRAllocatorType == string(ipam.CloudAllocatorType) {
		return nil, false, errors.New("--cidr-allocator-type is set to 'CloudAllocator' but cloud provider is not configured")
	}

	// failure: bad cidrs in config
	clusterCIDRs, dualStack, err := processCIDRs(ccmConfig.ComponentConfig.KubeCloudShared.ClusterCIDR)
	if err != nil {
		return nil, false, err
	}

	// failure: more than one cidr but they are not configured as dual stack
	if len(clusterCIDRs) > 1 && !dualStack {
		return nil, false, fmt.Errorf("len of ClusterCIDRs==%v and they are not configured as dual stack (at least one from each IPFamily", len(clusterCIDRs))
	}

	// failure: more than cidrs is not allowed even with dual stack
	if len(clusterCIDRs) > 2 {
		return nil, false, fmt.Errorf("len of clusters is:%v > more than max allowed of 2", len(clusterCIDRs))
	}

	// service cidr processing
	if len(strings.TrimSpace(nodeIPAMConfig.ServiceCIDR)) != 0 { <MID> 	}

	if len(strings.TrimSpace(nodeIPAMConfig.SecondaryServiceCIDR)) != 0 {
		_, secondaryServiceCIDR, err = netutils.ParseCIDRSloppy(nodeIPAMConfig.SecondaryServiceCIDR)
		if err != nil {
			klog.ErrorS(err, "Unsuccessful parsing of service CIDR", "CIDR", nodeIPAMConfig.SecondaryServiceCIDR)
		}
	}

	// the following checks are triggered if both serviceCIDR and secondaryServiceCIDR are provided
	if serviceCIDR != nil && secondaryServiceCIDR != nil {
		// should be dual stack (from different IPFamilies)
		dualstackServiceCIDR, err := netutils.IsDualStackCIDRs([]*net.IPNet{serviceCIDR, secondaryServiceCIDR})
		if err != nil {
			return nil, false, fmt.Errorf("failed to perform dualstack check on serviceCIDR and secondaryServiceCIDR error:%v", err)
		}
		if !dualstackServiceCIDR {
			return nil, false, fmt.Errorf("serviceCIDR and secondaryServiceCIDR are not dualstack (from different IPfamiles)")
		}
	}

	nodeCIDRMaskSizes, err := setNodeCIDRMaskSizes(nodeIPAMConfig, clusterCIDRs)
	if err != nil {
		return nil, false, err
	}

	nodeIpamController, err := nodeipamcontroller.NewNodeIpamController(
		ctx,
		controllerCtx.InformerFactory.Core().V1().Nodes(),
		cloud,
		controllerCtx.ClientBuilder.ClientOrDie(initContext.ClientName),
		clusterCIDRs,
		serviceCIDR,
		secondaryServiceCIDR,
		nodeCIDRMaskSizes,
		ipam.CIDRAllocatorType(ccmConfig.ComponentConfig.KubeCloudShared.CIDRAllocatorType),
	)
	if err != nil {
		return nil, true, err
	}
	go nodeIpamController.Run(ctx)
	return nil, true, nil
} Ensure that only missing codes marked as <MID> are returned<|im_end|>
<|im_start|>assistant
		_, <ATTN>serviceCIDR</ATTN>, err = <ATTN>netutils.ParseCIDRSloppy</ATTN>(<ATTN>nodeIPAMConfig.ServiceCIDR</ATTN>)
		if err != nil {
			<ATTN>klog.ErrorS</ATTN>(err, "Unsuccessful parsing of service CIDR", "CIDR", <ATTN>nodeIPAMConfig.ServiceCIDR</ATTN>)
		}<|im_end|>
```

```go
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
This is a go programming task on some code contents. Given task: The task is to fill in the missing part of a go function according to the provided code   content. Below is the package path:github.com/kubernetes/Kubernetes/vendor/github.com/gogo/protobuf/proto Below is the code repository: github.com/kubernetes/Kubernetes/ Below is the imported package path "errors"; "fmt"; "math"; "reflect"; "sort"; "strconv"; "strings"; "sync"; "sync/atomic"; "unicode/utf8" The receiver struct definitions of the function is not exist The parameter struct definition or not exist of the function is pb Message The return value struct definitions or not exist of the function is (*ast.ArrayType, error) The code snippets before the function is func (u *marshalInfo) sizeMessageSet(ext *XXX_InternalExtensions) int {
	m, mu := ext.extensionsRead()
	if m == nil {
		return 0
	}
	mu.Lock()

	n := 0
	for id, e := range m {
		n += 2                          // start group, end group. tag = 1 (size=1)
		n += SizeVarint(uint64(id)) + 1 // type_id, tag = 2 (size=1)

		if e.value == nil || e.desc == nil {
			// Extension is only in its encoded form.
			msgWithLen := skipVarint(e.enc) // skip old tag, but leave the length varint
			siz := len(msgWithLen)
			n += siz + 1 // message, tag = 3 (size=1)
			continue
		}

		// We don't skip extensions that have an encoded form set,
		// because the extension value may have been mutated after
		// the last time this function was called.

		ei := u.getExtElemInfo(e.desc)
		v := e.value
		p := toAddrPointer(&v, ei.isptr)
		n += ei.sizer(p, 1) // message, tag = 3 (size=1)
	}
	mu.Unlock()
	return n
}

func (u *marshalInfo) appendMessageSet(b []byte, ext *XXX_InternalExtensions, deterministic bool) ([]byte, error) {
	m, mu := ext.extensionsRead()
	if m == nil {
		return b, nil
	}
	mu.Lock()
	defer mu.Unlock()

	var err error
	var nerr nonFatal

	// Fast-path for common cases: zero or one extensions.
	// Don't bother sorting the keys.
	if len(m) <= 1 {
		for id, e := range m {
			b = append(b, 1<<3|WireStartGroup)
			b = append(b, 2<<3|WireVarint)
			b = appendVarint(b, uint64(id))

			if e.value == nil || e.desc == nil {
				// Extension is only in its encoded form.
				msgWithLen := skipVarint(e.enc) // skip old tag, but leave the length varint
				b = append(b, 3<<3|WireBytes)
				b = append(b, msgWithLen...)
				b = append(b, 1<<3|WireEndGroup)
				continue
			}

			// We don't skip extensions that have an encoded form set,
			// because the extension value may have been mutated after
			// the last time this function was called.

			ei := u.getExtElemInfo(e.desc)
			v := e.value
			p := toAddrPointer(&v, ei.isptr)
			b, err = ei.marshaler(b, p, 3<<3|WireBytes, deterministic)
			if !nerr.Merge(err) {
				return b, err
			}
			b = append(b, 1<<3|WireEndGroup)
		}
		return b, nerr.E
	}

	// Sort the keys to provide a deterministic encoding.
	keys := make([]int, 0, len(m))
	for k := range m {
		keys = append(keys, int(k))
	}
	sort.Ints(keys)

	for _, id := range keys {
		e := m[int32(id)]
		b = append(b, 1<<3|WireStartGroup)
		b = append(b, 2<<3|WireVarint)
		b = appendVarint(b, uint64(id))

		if e.value == nil || e.desc == nil {
			// Extension is only in its encoded form.
			msgWithLen := skipVarint(e.enc) // skip old tag, but leave the length varint
			b = append(b, 3<<3|WireBytes)
			b = append(b, msgWithLen...)
			b = append(b, 1<<3|WireEndGroup)
			continue
		}

		// We don't skip extensions that have an encoded form set,
		// because the extension value may have been mutated after
		// the last time this function was called.

		ei := u.getExtElemInfo(e.desc)
		v := e.value
		p := toAddrPointer(&v, ei.isptr)
		b, err = ei.marshaler(b, p, 3<<3|WireBytes, deterministic)
		b = append(b, 1<<3|WireEndGroup)
		if !nerr.Merge(err) {
			return b, err
		}
	}
	return b, nerr.E
}

func (u *marshalInfo) sizeV1Extensions(m map[int32]Extension) int {
	if m == nil {
		return 0
	}

	n := 0
	for _, e := range m {
		if e.value == nil || e.desc == nil {
			// Extension is only in its encoded form.
			n += len(e.enc)
			continue
		}

		// We don't skip extensions that have an encoded form set,
		// because the extension value may have been mutated after
		// the last time this function was called.

		ei := u.getExtElemInfo(e.desc)
		v := e.value
		p := toAddrPointer(&v, ei.isptr)
		n += ei.sizer(p, ei.tagsize)
	}
	return n
}

func (u *marshalInfo) appendV1Extensions(b []byte, m map[int32]Extension, deterministic bool) ([]byte, error) {
	if m == nil {
		return b, nil
	}

	// Sort the keys to provide a deterministic encoding.
	keys := make([]int, 0, len(m))
	for k := range m {
		keys = append(keys, int(k))
	}
	sort.Ints(keys)

	var err error
	var nerr nonFatal
	for _, k := range keys {
		e := m[int32(k)]
		if e.value == nil || e.desc == nil {
			// Extension is only in its encoded form.
			b = append(b, e.enc...)
			continue
		}

		// We don't skip extensions that have an encoded form set,
		// because the extension value may have been mutated after
		// the last time this function was called.

		ei := u.getExtElemInfo(e.desc)
		v := e.value
		p := toAddrPointer(&v, ei.isptr)
		b, err = ei.marshaler(b, p, ei.wiretag, deterministic)
		if !nerr.Merge(err) {
			return b, err
		}
	}
	return b, nerr.E
}

func Size(pb Message) int {
	if m, ok := pb.(newMarshaler); ok {
		return m.XXX_Size()
	}
	if m, ok := pb.(Marshaler); ok {
		// If the message can marshal itself, let it do it, for compatibility.
		// NOTE: This is not efficient.
		b, _ := m.Marshal()
		return len(b)
	}
	// in case somehow we didn't generate the wrapper
	if pb == nil {
		return 0
	}
	var info InternalMessageInfo
	return info.Size(pb)
} And here is the function you are asked to complete func Marshal(pb Message) ((*ast.ArrayType, error))
func Marshal(pb Message) ([]byte, error) {
	if m, ok := pb.(newMarshaler); ok { <MID> 	}
	if m, ok := pb.(Marshaler); ok {
		// If the message can marshal itself, let it do it, for compatibility.
		// NOTE: This is not efficient.
		return m.Marshal()
	}
	// in case somehow we didn't generate the wrapper
	if pb == nil {
		return nil, ErrNil
	}
	var info InternalMessageInfo
	siz := info.Size(pb)
	b := make([]byte, 0, siz)
	return info.Marshal(b, pb, false)
} Ensure that only missing codes marked as <MID> are returned<|im_end|>
<|im_start|>assistant
		siz := <ATTN>m.XXX_Size()</ATTN>
		b := make([]byte, 0, siz)
		return <ATTN>m.XXX_Marshal(b, false)</ATTN><|im_end|>
```

```go
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
This is a go programming task on some code contents. Given task: The task is to fill in the missing part of a go function according to the provided code   content. Below is the package path:github.com/kubernetes/Kubernetes/vendor/go.etcd.io/etcd/server/v3/internal/clientv2 Below is the code repository: github.com/kubernetes/Kubernetes/ Below is the imported package path "bytes"; "context"; "encoding/json"; "net/http"; "net/url"; "path" The receiver struct definitions of the function is type authUserAPIList struct{}

// Methods:
- func (list *authUserAPIList) HTTPRequest(ep url.URL) *http.Request
 The parameter struct definition or not exist of the function is ep url.URL The return value struct definitions or not exist of the function is *http.Request The code snippets before the function is func (s *httpAuthAPI) Disable(ctx context.Context) error {
	return s.enableDisable(ctx, &authAPIAction{http.MethodDelete})
}

func (s *httpAuthAPI) enableDisable(ctx context.Context, req httpAction) error {
	resp, body, err := s.client.Do(ctx, req)
	if err != nil {
		return err
	}
	if err = assertStatusCode(resp.StatusCode, http.StatusOK, http.StatusCreated); err != nil {
		var sec authError
		err = json.Unmarshal(body, &sec)
		if err != nil {
			return err
		}
		return sec
	}
	return nil
}

func (e authError) Error() string {
	return e.Message
}

func NewAuthUserAPI(c Client) AuthUserAPI {
	return &httpAuthUserAPI{
		client: c,
	}
} And here is the function you are asked to complete func (list *authUserAPIList) HTTPRequest(ep url.URL) *http.Request
func (list *authUserAPIList) HTTPRequest(ep url.URL) *http.Request { <MID> } Ensure that only missing codes marked as <MID> are returned<|im_end|>
<|im_start|>assistant
	u := <ATTN>v2AuthURL</ATTN>(ep, <ATTN>"users"</ATTN>, <ATTN>""</ATTN>)
	req, _ := http.<ATTN>NewRequest</ATTN>(<ATTN>http.MethodGet</ATTN>, u.<ATTN>String</ATTN>(), nil)
	req.Header.Set(<ATTN>"Content-Type"</ATTN>, <ATTN>"application/json"</ATTN>)
	return req<|im_end|>
```

