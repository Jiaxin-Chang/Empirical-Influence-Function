package main

	reply *structs.IndexedPreparedQueries) error {
	if done, err := p.srv.ForwardRPC("PreparedQuery.Get", args, reply); done {
		return err
	}

	return p.srv.blockingQuery(
		&args.QueryOptions,
		&reply.QueryMeta,
		func(ws memdb.WatchSet, state *state.Store) error {
			index, query, err := state.PreparedQueryGet(ws, args.QueryID)
			if err != nil {
				return err
			}
			if query == nil {
				return structs.ErrQueryNotFound
			}

			// If no prefix ACL applies to this query, then they are
			// always allowed to see it if they have the ID. We still
			// have to filter the remaining object for tokens.
			reply.Index = index
			reply.Queries = structs.PreparedQueries{query}
			if _, ok := query.GetACLPrefix(); !ok {
				return p.srv.filterACL(args.Token, &reply.Queries[0])
			}

			// Otherwise, attempt to filter it the usual way.
			if err := p.srv.filterACL(args.Token, reply); err != nil {
				return err
			}

			// Since this is a GET of a specific query, if ACLs have
			// prevented us from returning something that exists,
			// then alert the user with a permission denied error.
			if len(reply.Queries) == 0 {
				p.logger.Warn("Request to get prepared query denied due to ACLs", "query", args.QueryID)
				return acl.ErrPermissionDenied
			}

			return nil
		})
}

func (p *PreparedQuery) List(args *structs.DCSpecificRequest, reply *structs.IndexedPreparedQueries) error {
	if done, err := p.srv.ForwardRPC("PreparedQuery.List", args, reply); done {
		return err
	}

	return p.srv.blockingQuery(
		&args.QueryOptions,
		&reply.QueryMeta,
		func(ws memdb.WatchSet, state *state.Store) error {
			index, queries, err := state.PreparedQueryList(ws)
			if err != nil {
				return err
			}

			reply.Index, reply.Queries = index, queries
			return p.srv.filterACL(args.Token, reply)
		})
}

	reply *structs.PreparedQueryExplainResponse) error {
	if done, err := p.srv.ForwardRPC("PreparedQuery.Explain", args, reply); done {
		return err
	}
	defer metrics.MeasureSince([]string{"prepared-query", "explain"}, time.Now())

	// We have to do this ourselves since we are not doing a blocking RPC.
	p.srv.SetQueryMeta(&reply.QueryMeta, args.Token)
	if args.RequireConsistent {
		if err := p.srv.ConsistentRead(); err != nil {
			return err
		}
	}

	// Try to locate the query.
	state := p.srv.fsm.State()
	_, query, err := state.PreparedQueryResolve(args.QueryIDOrName, args.Agent)
	if err != nil {
		return err
	}
	if query == nil {
		return structs.ErrQueryNotFound
	}

	// Place the query into a list so we can run the standard ACL filter on
	// it.
	queries := &structs.IndexedPreparedQueries{
		Queries: structs.PreparedQueries{query},
	}
	if err := p.srv.filterACL(args.Token, queries); err != nil {
		return err
	}

	// If the query was filtered out, return an error.
	if len(queries.Queries) == 0 {
		p.logger.Warn("Explain on prepared query denied due to ACLs", "query", query.ID)
		return acl.ErrPermissionDenied
	}

	reply.Query = *(queries.Queries[0])
	return nil
}

	reply *structs.PreparedQueryExecuteResponse) error {
	if done, err := p.srv.ForwardRPC("PreparedQuery.Execute", args, reply); done {
		return err
	}
	defer metrics.MeasureSince([]string{"prepared-query", "execute"}, time.Now())

	// We have to do this ourselves since we are not doing a blocking RPC.
	if args.RequireConsistent {
		if err := p.srv.ConsistentRead(); err != nil {
			return err
		}
	}

	// Try to locate the query.
	state := p.srv.fsm.State()
	_, query, err := state.PreparedQueryResolve(args.QueryIDOrName, args.Agent)
	if err != nil {
		return err
	}
	if query == nil {
		return structs.ErrQueryNotFound
	}

	// If we have a sameness group, it controls the initial query and
	// subsequent failover if required (Enterprise Only)
	if query.Service.SamenessGroup != "" {
		wrapper := newQueryServerWrapper(p.srv, p.ExecuteRemote)
		if err := querySameness(wrapper, *query, args, reply); err != nil {
			return err
		}
	} else {
		// Execute the query for the local DC.
		if err := p.execute(query, reply, args.Connect); err != nil {
			return err
		}

		// If they supplied a token with the query, use that, otherwise use the
		// token passed in with the request.
		token := args.Token
		if query.Token != "" {
			token = query.Token
		}
		if err := p.srv.filterACL(token, reply); err != nil {
			return err
		}

		// TODO (slackpad) We could add a special case here that will avoid the
		// fail over if we filtered everything due to ACLs. This seems like it
		// might not be worth the code complexity and behavior differences,
		// though, since this is essentially a misconfiguration.

		// We have to do this ourselves since we are not doing a blocking RPC.
		p.srv.SetQueryMeta(&reply.QueryMeta, token)

		// Shuffle the results in case coordinates are not available if they
		// requested an RTT sort.
		reply.Nodes.Shuffle()

		// Build the query source. This can be provided by the client, or by
		// the prepared query. Client-specified takes priority.
		qs := args.Source
		if qs.Datacenter == "" {
			qs.Datacenter = args.Agent.Datacenter
		}
		if query.Service.Near != "" && qs.Node == "" {
			qs.Node = query.Service.Near
		}

		// Respect the magic "_agent" flag.
		if qs.Node == "_agent" {
			qs.Node = args.Agent.Node
		} else if qs.Node == "_ip" {
			if args.Source.Ip != "" {
				_, nodes, err := state.Nodes(nil, structs.NodeEnterpriseMetaInDefaultPartition(), structs.TODOPeerKeyword)
				if err != nil {
					return err
				}

				for _, node := range nodes {
					if args.Source.Ip == node.Address {
						qs.Node = node.Node
						break
					}
				}
			} else {
				p.logger.Warn("Prepared Query using near=_ip requires " +
					"the source IP to be set but none was provided. No distance " +
					"sorting will be done.")

			}

			// Either a source IP was given, but we couldn't find the associated node
			// or no source ip was given. In both cases we should wipe the Node value
			if qs.Node == "_ip" {
				qs.Node = ""
			}
		}

		// Perform the distance sort
		err = p.srv.sortNodesByDistanceFrom(qs, reply.Nodes)
		if err != nil {
			return err
		}

		// If we applied a distance sort, make sure that the node queried for is in
		// position 0, provided the results are from the same datacenter.
		if qs.Node != "" && reply.Datacenter == qs.Datacenter {
			for i, node := range reply.Nodes {
				if strings.EqualFold(node.Node.Node, qs.Node) {
					reply.Nodes[0], reply.Nodes[i] = reply.Nodes[i], reply.Nodes[0]
					break
				}

				// Put a cap on the depth of the search. The local agent should
				// never be further in than this if distance sorting was applied.
				if i == 9 {
					break
				}
			}
		}

		// Apply the limit if given.
		if args.Limit > 0 && len(reply.Nodes) > args.Limit {
			reply.Nodes = reply.Nodes[:args.Limit]
		}

		// In the happy path where we found some healthy nodes we go with that
		// and bail out. Otherwise, we fail over and try remote DCs, as allowed
		// by the query setup.
		if len(reply.Nodes) == 0 {
			wrapper := newQueryServerWrapper(p.srv, p.ExecuteRemote)
			if err := queryFailover(wrapper, *query, args, reply); err != nil {
				return err
			}
		}
	}

	return nil
}

	reply *structs.PreparedQueryExecuteResponse) error {
	if done, err := p.srv.ForwardRPC("PreparedQuery.ExecuteRemote", args, reply); done {
		return err
	}
	defer metrics.MeasureSince([]string{"prepared-query", "execute_remote"}, time.Now())

	// We have to do this ourselves since we are not doing a blocking RPC.
	if args.RequireConsistent {
		if err := p.srv.ConsistentRead(); err != nil {
			return err
		}
	}

	// Run the query locally to see what we can find.
	if err := p.execute(&args.Query, reply, args.Connect); err != nil {
		return err
	}

	// If they supplied a token with the query, use that, otherwise use the
	// token passed in with the request.
	token := args.Token
	if args.Query.Token != "" {
		token = args.Query.Token
	}
	if err := p.srv.filterACL(token, reply); err != nil {
		return err
	}

	// We have to do this ourselves since we are not doing a blocking RPC.
	p.srv.SetQueryMeta(&reply.QueryMeta, token)

	// We don't bother trying to do an RTT sort here since we are by
	// definition in another DC. We just shuffle to make sure that we
	// balance the load across the results.
	reply.Nodes.Shuffle()

	// Apply the limit if given.
	if args.Limit > 0 && len(reply.Nodes) > args.Limit {
		reply.Nodes = reply.Nodes[:args.Limit]
	}

	return nil
}

function.
	f := state.CheckServiceNodes
	if query.Service.Connect || forceConnect {
		f = state.CheckConnectServiceNodes
	}

	_, nodes, err := f(nil, query.Service.Service, &query.Service.EnterpriseMeta, query.Service.Peer)
	if err != nil {
		return err
	}

	// Filter out any unhealthy nodes.
	filterType := structs.HealthFilterExcludeCritical
	if query.Service.OnlyPassing { 		filterType = structs.HealthFilterIncludeOnlyPassing
 	}

	nodes = nodes.Filter(structs.CheckServiceNodeFilterOptions{FilterType: filterType,
		IgnoreCheckIDs: query.Service.IgnoreCheckIDs})

	// Apply the node metadata filters, if any.
	if len(query.Service.NodeMeta) > 0 {
		nodes = nodeMetaFilter(query.Service.NodeMeta, nodes)
	}

	// Apply the service metadata filters, if any.
	if len(query.Service.ServiceMeta) > 0 {
		nodes = serviceMetaFilter(query.Service.ServiceMeta, nodes)
	}

	// Apply the tag filters, if any.
	if len(query.Service.Tags) > 0 {
		nodes = tagFilter(query.Service.Tags, nodes)
	}

	// Capture the nodes and pass the DNS information through to the reply.
	reply.Service = query.Service.Service
	reply.EnterpriseMeta = query.Service.EnterpriseMeta
	reply.Nodes = nodes
	reply.DNS = query.DNS

	// Stamp the result with its this datacenter or peer.
	if peerName := query.Service.Peer; peerName != "" {
		reply.PeerName = peerName
		reply.Datacenter = ""
	} else {
		reply.Datacenter = p.srv.config.Datacenter
	}

	return nil
}