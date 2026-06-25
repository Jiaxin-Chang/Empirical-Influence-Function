package main

func (w drainWatcher) enabled() bool {
	return w != nil
}

func makeDrainWatcher(flowCtx *execinfra.FlowCtx) (w drainWatcher, _ func()) {
	if !aggregatorEmitsShutdownCheckpoint.Get(&flowCtx.Cfg.Settings.SV) {
		// Drain watcher disabled
		return nil, func() {}
	}

	if cfKnobs, ok := flowCtx.TestingKnobs().Changefeed.(*TestingKnobs); ok && cfKnobs != nil && cfKnobs.OnDrain != nil {
		return cfKnobs.OnDrain(), func() {}
	}
	return flowCtx.Cfg.JobRegistry.OnDrain()
}

) (_ execinfra.Processor, retErr error) {
	// Setup monitoring for this node drain.
	drainWatcher, drainDone := makeDrainWatcher(flowCtx)
	memMonitor := execinfra.NewMonitor(ctx, flowCtx.Mon, mon.MakeName("changeagg-mem"))
	ca := &changeAggregator{
		spec:              spec,
		memAcc:            memMonitor.MakeBoundAccount(),
		checkForNodeDrain: drainWatcher.checkForNodeDrain,
	}

	defer func() {
		if retErr != nil {
			ca.close()
			drainDone()
		}
	}()

	if err := ca.Init(
		ctx,
		ca,
		post,
		changefeedResultTypes,
		flowCtx,
		processorID,
		memMonitor,
		execinfra.ProcStateOpts{
			TrailingMetaCallback: func() (producerMeta []execinfrapb.ProducerMetadata) {
				defer drainDone()

				if drainWatcher.enabled() {
					var meta execinfrapb.ChangefeedMeta
					if err := drainWatcher.checkForNodeDrain(); err != nil {
						// This node is draining.  Indicate so in the trailing metadata.
						nodeID, _ := flowCtx.Cfg.NodeID.OptionalNodeID()
						meta.DrainInfo = &execinfrapb.ChangefeedMeta_DrainInfo{NodeID: nodeID}
					}
					ca.computeTrailingMetadata(&meta)
					producerMeta = []execinfrapb.ProducerMetadata{{Changefeed: &meta}}
				}

				if ca.agg != nil {
					meta := bulkutil.ConstructTracingAggregatorProducerMeta(ctx,
						ca.FlowCtx.NodeID.SQLInstanceID(), ca.FlowCtx.ID, ca.agg)
					producerMeta = append(producerMeta, *meta)
				}

				ca.close()
				return producerMeta
			},
		},
	); err != nil {
		return nil, err
	}

	opts := changefeedbase.MakeStatementOptions(ca.spec.Feed.Opts)

	// MinCheckpointFrequency controls how frequently the changeAggregator flushes the sink
	// and checkpoints the local frontier to changeFrontier. It is used as a rough
	// approximation of how latency-sensitive the changefeed user is. For a high latency
	// user, such as cloud storage sink where flushes can take much longer, it is often set
	// as the sink's flush frequency so as not to negate the sink's batch config.
	//
	// If a user does not specify a 'min_checkpoint_frequency' duration, we instead default
	// to 30s, which is hopefully long enough to account for most possible sink latencies we
	// could see without falling too behind.
	//
	// NB: As long as we periodically get new span-level resolved timestamps
	// from the poller (which should always happen, even if the watched data is
	// not changing), then this is sufficient and we don't have to do anything
	// fancy with timers.
	// // TODO(casper): add test for OptMinCheckpointFrequency.
	checkpointFreq, err := opts.GetMinCheckpointFrequency()
	if err != nil {
		return nil, err
	}
	if checkpointFreq != nil {
		ca.flushFrequency = *checkpointFreq
	} else {
		ca.flushFrequency = changefeedbase.DefaultMinCheckpointFrequency
	}

	ca.frontierFlushLimiter, err = newSaveRateLimiter(saveRateConfig{
		name: "frontier",
		intervalName: func() redact.SafeValue {
			return redact.SafeString(changefeedbase.OptMinCheckpointFrequency)
		},
		interval: func() time.Duration {
			return ca.flushFrequency
		},
		jitter: func() float64 {
			return aggregatorFlushJitter.Get(&ca.FlowCtx.Cfg.Settings.SV)
		},
	}, timeutil.DefaultTimeSource{})
	if err != nil {
		return nil, err
	}

	return ca, nil
}

func (ca *changeAggregator) MustBeStreaming() bool {
	return true
}

) (metricsRecorder, error) {
	details := ca.spec.Feed
	jobID := ca.spec.JobID
	description := ca.spec.Description
	// This code exists so that the old behavior is preserved if the spec is created
	// in a mixed-version cluster on a node without the new Description field.
	// It can be deleted once we no longer need to interoperate with binaries that
	// are version 24.3 or earlier.
	if description == "" {
		// Don't emit telemetry messages for core changefeeds without a description.
		if ca.isSinkless() {
			return recorder, nil
		}
		job, err := ca.FlowCtx.Cfg.JobRegistry.LoadJob(ctx, jobID)
		if err != nil {
			return nil, err
		}
		description = job.Payload().Description
	}

	recorderWithTelemetry, err := wrapMetricsRecorderWithTelemetry(ctx, details, description, jobID, ca.FlowCtx.Cfg.Settings, recorder, ca.knobs, targets)
	if err != nil {
		return ca.sliMetrics, err
	}
	ca.closeTelemetryRecorder = recorderWithTelemetry.close

	return recorderWithTelemetry, nil
}


func (ca *changeAggregator) Start(ctx context.Context) {
	// Derive a separate context so that we can shutdown the poller.
	ctx, ca.cancel = ca.FlowCtx.Stopper().WithCancelOnQuiesce(ctx)

	if ca.spec.JobID != 0 {
		ctx = logtags.AddTag(ctx, "job", ca.spec.JobID)
	}
	ctx = logtags.RemoveTag(ctx, changeFrontierLogTag)
	ctx = logtags.AddTag(ctx, changeAggregatorLogTag, nil /* value */)

	ca.agg = tracing.TracingAggregatorForContext(ctx)
	if ca.agg != nil {
		ca.aggTimer.Reset(tracingAggTimerInterval)
	}

	ctx = ca.StartInternal(ctx, changeAggregatorProcName, ca.agg)

	spans, err := ca.setupSpansAndFrontier()
	if err != nil {
		log.Changefeed.Warningf(ca.Ctx(), "moving to draining due to error setting up spans and frontier: %v", err)
		ca.MoveToDraining(err)
		ca.cancel()
		return
	}

	execCfg := ca.FlowCtx.Cfg.ExecutorConfig.(*sql.ExecutorConfig)
	if ca.knobs.OverrideExecCfg != nil {
		execCfg = ca.knobs.OverrideExecCfg(execCfg)
	}
	targetTS := ca.spec.GetSchemaTS()
	ca.targets, err = AllTargets(ctx, ca.spec.Feed, execCfg, targetTS)
	if err != nil {
		log.Changefeed.Warningf(ca.Ctx(), "moving to draining due to error getting targets: %v", err)
		ca.MoveToDraining(err)
		ca.cancel()
		return
	}

	feed, err := makeChangefeedConfigFromJobDetails(ca.spec.Feed, ca.targets)
	if err != nil {
		log.Changefeed.Warningf(ca.Ctx(), "moving to draining due to error making changefeed config: %v", err)
		ca.MoveToDraining(err)
		ca.cancel()
		return
	}
	opts := feed.Opts

	timestampOracle := &changeAggregatorLowerBoundOracle{
		sf:                         ca.frontier,
		initialInclusiveLowerBound: feed.ScanTime,
	}

	if cfKnobs, ok := ca.FlowCtx.TestingKnobs().Changefeed.(*TestingKnobs); ok {
		ca.knobs = *cfKnobs
	}

	// The job registry has a set of metrics used to monitor the various jobs it
	// runs. They're all stored as the `metric.Struct` interface because of
	// dependency cycles.
	ca.metrics = ca.FlowCtx.Cfg.JobRegistry.MetricsStruct().Changefeed.(*Metrics)
	scope, _ := opts.GetMetricScope()
	ca.sliMetrics, err = ca.metrics.getSLIMetrics(scope)
	if err != nil { 		log.Changefeed.Errorf(ca.Ctx(), "failed to get SliMetrics: %v", err)
		return 	}
	ca.sliMetricsID = ca.sliMetrics.claimId()

	recorder := metricsRecorder(ca.sliMetrics)
	recorder, err = ca.wrapMetricsRecorderWithTelemetry(ctx, recorder, ca.targets)

	if err != nil {
		log.Changefeed.Warningf(ca.Ctx(), "moving to draining due to error wrapping metrics controller: %v", err)
		ca.MoveToDraining(err)
		ca.cancel()
	}

	ca.sink, err = getEventSink(ctx, ca.FlowCtx.Cfg, ca.spec.Feed, timestampOracle,
		ca.spec.User(), ca.spec.JobID, recorder, ca.targets)
	if err != nil {
		err = changefeedbase.MarkRetryableError(err)
		log.Changefeed.Warningf(ca.Ctx(), "moving to draining due to error getting sink: %v", err)
		ca.MoveToDraining(err)
		ca.cancel()
		return
	}

	// This is the correct point to set up certain hooks depending on the sink
	// type.
	if b, ok := ca.sink.(*bufferSink); ok {
		ca.changedRowBuf = &b.buf
	}

	// If the initial scan was disabled the highwater would've already been forwarded
	needsInitialScan := ca.frontier.Frontier().IsEmpty()

	// The "HighWater" of the KVFeed is the timestamp it will begin streaming
	// change events from.  When there's an inital scan, we want the scan to cover
	// data up to the StatementTime and change events to begin from that point.
	kvFeedHighWater := ca.frontier.Frontier()
	if needsInitialScan {
		kvFeedHighWater = ca.spec.Feed.StatementTime
	}

	// TODO(yevgeniy): Introduce separate changefeed monitor that's a parent
	// for all changefeeds to control memory allocated to all changefeeds.
	pool := ca.FlowCtx.Cfg.BackfillerMonitor
	if ca.knobs.MemMonitor != nil {
		pool = ca.knobs.MemMonitor
	}
	limit := changefeedbase.PerChangefeedMemLimit.Get(&ca.FlowCtx.Cfg.Settings.SV)
	ca.eventProducer, ca.kvFeedDoneCh, ca.errCh, err = ca.startKVFeed(ctx, spans, kvFeedHighWater, needsInitialScan, feed, pool, limit, opts)
	if err != nil {
		log.Changefeed.Warningf(ca.Ctx(), "moving to draining due to error starting kv feed: %v", err)
		ca.MoveToDraining(err)
		ca.cancel()
		return
	}
	ca.sink = &errorWrapperSink{wrapped: ca.sink}
	ca.eventConsumer, ca.sink, err = newEventConsumer(
		ctx, ca.FlowCtx.Cfg, ca.spec, feed, ca.frontier, kvFeedHighWater,
		ca.sink, ca.metrics, ca.sliMetrics, ca.knobs)
	if err != nil {
		log.Changefeed.Warningf(ca.Ctx(), "moving to draining due to error creating event consumer: %v", err)
		ca.MoveToDraining(err)
		ca.cancel()
		return
	}

	// Init heartbeat timer.
	ca.lastPush = timeutil.Now()

	// Generate expensive checkpoint only after we ran for a while.
	ca.lastSpanFlush = timeutil.Now()
}