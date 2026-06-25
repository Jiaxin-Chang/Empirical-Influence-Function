package main

func DefaultSource() Source {
	cfg := consul.DefaultConfig()
	serfLAN := cfg.SerfLANConfig.MemberlistConfig
	serfWAN := cfg.SerfWANConfig.MemberlistConfig

	return FileSource{
		Name:   "default",
		Format: "hcl",
		Data: `
		acl = {
			token_ttl = "30s"
			policy_ttl = "30s"
			default_policy = "allow"
			down_policy = "extend-cache"
		}
		bind_addr = "0.0.0.0"
		bootstrap = false
		bootstrap_expect = 0
		check_output_max_size = ` + strconv.Itoa(checks.DefaultBufSize) + `
		check_update_interval = "5m"
		client_addr = "127.0.0.1"
		datacenter = "` + consul.DefaultDC + `"
		default_query_time = "300s"
		disable_coordinates = false
		disable_host_node_id = true
		disable_remote_exec = true
		domain = "consul."
		enable_central_service_config = true
		encrypt_verify_incoming = true
		encrypt_verify_outgoing = true
		log_level = "INFO"
		max_query_time = "600s"
		primary_gateways_interval = "30s"
		protocol = ` + strconv.Itoa(consul.DefaultRPCProtocol) + `
		retry_interval = "30s"
		retry_interval_wan = "30s"

		# segment_limit is the maximum number of network segments that may be declared. Default 64 is highly encouraged
		segment_limit = 64

		server = false
		server_rejoin_age_max = "168h"
		syslog_facility = "LOCAL0"

		tls = {
			defaults = {
				tls_min_version = "TLSv1_2"
			}
		}

		// TODO (slackpad) - Until #3744 is done, we need to keep these
		// in sync with agent/consul/config.go.
		autopilot = {
			cleanup_dead_servers = true
			last_contact_threshold = "200ms"
			max_trailing_logs = 250
			server_stabilization_time = "10s"
		}
		gossip_lan = {
			gossip_interval = "` + serfLAN.GossipInterval.String() + `"
			gossip_nodes = ` + strconv.Itoa(serfLAN.GossipNodes) + `
			retransmit_mult = ` + strconv.Itoa(serfLAN.RetransmitMult) + `
			probe_interval = "` + serfLAN.ProbeInterval.String() + `"
			probe_timeout = "` + serfLAN.ProbeTimeout.String() + `"
			suspicion_mult = ` + strconv.Itoa(serfLAN.SuspicionMult) + `
		}
		gossip_wan = {
			gossip_interval = "` + serfWAN.GossipInterval.String() + `"
			gossip_nodes = ` + strconv.Itoa(serfLAN.GossipNodes) + `
			retransmit_mult = ` + strconv.Itoa(serfLAN.RetransmitMult) + `
			probe_interval = "` + serfWAN.ProbeInterval.String() + `"
			probe_timeout = "` + serfWAN.ProbeTimeout.String() + `"
			suspicion_mult = ` + strconv.Itoa(serfWAN.SuspicionMult) + `
		}
		dns_config = {
			allow_stale = true
			a_record_limit = 0
			udp_answer_limit = 3
			max_stale = "87600h"
			recursor_timeout = "2s"
		}
		limits = {
			http_max_conns_per_client = 200
			https_handshake_timeout = "5s"
			request_limits = {
				mode = "disabled"
				read_rate = -1
				write_rate = -1
			}
			rpc_handshake_timeout = "5s"
			rpc_client_timeout = "60s"
			rpc_rate = -1
			rpc_max_burst = 1000
			rpc_max_conns_per_client = 100
			kv_max_value_size = ` + strconv.FormatInt(raft.SuggestedMaxDataSize, 10) + `
			txn_max_req_len = ` + strconv.FormatInt(raft.SuggestedMaxDataSize, 10) + `
		}
		performance = {
			leave_drain_time = "5s"
			raft_multiplier = ` + strconv.Itoa(int(consul.DefaultRaftMultiplier)) + `
			rpc_hold_timeout = "7s"
			grpc_keepalive_interval = "30s"
			grpc_keepalive_timeout = "20s"
			enable_xds_load_balancing = true
		}
		ports = {
			dns = 8600
			http = 8500
			https = -1
			grpc = -1
			serf_lan = ` + strconv.Itoa(consul.DefaultLANSerfPort) + `
			serf_wan = ` + strconv.Itoa(consul.DefaultWANSerfPort) + `
			server = ` + strconv.Itoa(consul.DefaultRPCPort) + `
			proxy_min_port = 20000
			proxy_max_port = 20255
			sidecar_min_port = 21000
			sidecar_max_port = 21255
			expose_min_port = 21500
			expose_max_port = 21755
		}
		raft_protocol = 3
		telemetry = {
			metrics_prefix = "consul"
			filter_default = true
			prefix_filter = []
			retry_failed_connection = true
		}
		raft_snapshot_threshold = ` + strconv.Itoa(int(cfg.RaftConfig.SnapshotThreshold)) + `
		raft_snapshot_interval =  "` + cfg.RaftConfig.SnapshotInterval.String() + `"
		raft_trailing_logs = ` + strconv.Itoa(int(cfg.RaftConfig.TrailingLogs)) + `
		raft_prevote_disabled =  false
		raft_logstore {
			wal {
				segment_size_mb = 64
			}
		}
		xds {
			update_max_per_second = 250
		}

		connect = {
			enabled = true
		}

		peering = {
			enabled = true
		}
	`,
	}
}

func DevSource() Source {
	return FileSource{
		Name:   "dev",
		Format: "hcl",
		Data: `
		bind_addr = "127.0.0.1"
		disable_anonymous_signature = true
		disable_keyring_file = true
		enable_debug = true
		ui_config {
			enabled = true
		}
		log_level = "DEBUG"
		server = true

		gossip_lan = {
			gossip_interval = "100ms"
			probe_interval = "100ms"
			probe_timeout = "100ms"
			suspicion_mult = 3
		}
		gossip_wan = {
			gossip_interval = "100ms"
			probe_interval = "100ms"
			probe_timeout = "100ms"
			suspicion_mult = 3
		}
		connect = {
			enabled = true
		}

		peering = {
			enabled = true
		}

		performance = {
			raft_multiplier = 1
		}
		ports = {
			grpc = 8502
		}
		experiments = []
	`,
	}
}


func NonUserSource() Source {
	return FileSource{
		Name:   "non-user",
		Format: "hcl",
		Data: `
		check_deregister_interval_min = "1m"
		check_reap_interval = "30s"
		ae_interval = "1m"
		sync_coordinate_rate_target = 64
		sync_coordinate_interval_min = "15s"

		# SegmentNameLimit is the maximum segment name length.
		segment_name_limit = 64
 		segment_id_limit = 256
		ports = {
			grpc = 8502
		}
		experiments 
		peering = {
			# We use peer registration for various testing
			test_allow_peer_registrations = false
		}
	`,
	}
}