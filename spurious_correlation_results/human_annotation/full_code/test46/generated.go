package main

func NewVaultProvider(logger hclog.Logger) *VaultProvider {
	return &VaultProvider{
		stopWatcher: func() {},
		logger:      logger,
	}
}

func vaultTLSConfig(config *structs.VaultCAProviderConfig) *vaultapi.TLSConfig {
	return &vaultapi.TLSConfig{
		CACert:        config.CAFile,
		CAPath:        config.CAPath,
		ClientCert:    config.CertFile,
		ClientKey:     config.KeyFile,
		Insecure:      config.TLSSkipVerify,
		TLSServerName: config.TLSServerName,
	}
}

func (v *VaultProvider) Configure(cfg ProviderConfig) error {
	config, err := ParseVaultCAConfig(cfg.RawConfig, v.isPrimary)
	if err != nil {
		return err
	}

	clientConf := &vaultapi.Config{
		Address: config.Address,
	}
	err = clientConf.ConfigureTLS(vaultTLSConfig(config))
	if err != nil {
		return err
	}
	client, err := vaultapi.NewClient(clientConf)
	if err != nil {
		return err
	}

	// We don't want to set the namespace if it's empty to prevent potential
	// unknown behavior (what does Vault do with an empty namespace). The Vault
	// client also makes sure the inputs are not empty strings so let's do the
	// same.
	if config.Namespace != "" {
		client.SetNamespace(config.Namespace)
		v.baseNamespace = config.Namespace
	}

	if config.AuthMethod != nil {
		loginResp, err := vaultLogin(client, config.AuthMethod)
		if err != nil {
			return err
		}
		config.Token = loginResp.Auth.ClientToken
	}
	client.SetToken(config.Token)

	v.config = config
	v.client = client
	v.isPrimary = cfg.IsPrimary
	v.clusterID = cfg.ClusterID
	v.spiffeID = connect.SpiffeIDSigningForCluster(v.clusterID)

	// Look up the token to see if we can auto-renew its lease.
	secret, err := client.Auth().Token().LookupSelf()
	if err != nil {
		return err
	} else if secret == nil {
		return fmt.Errorf("could not look up Vault provider token: not found")
	}
	var token struct {
		Renewable bool
		TTL       int
	}
	if err := mapstructure.Decode(secret.Data, &token); err != nil {
		return err
	}

	// Set up a renewer to renew the token automatically, if supported.
	if token.Renewable || config.AuthMethod != nil {
		lifetimeWatcher, err := client.NewLifetimeWatcher(&vaultapi.LifetimeWatcherInput{
			Secret: &vaultapi.Secret{
				Auth: &vaultapi.SecretAuth{
					ClientToken:   config.Token,
					Renewable:     token.Renewable,
					LeaseDuration: secret.LeaseDuration,
				},
			},
			Increment:     token.TTL,
			RenewBehavior: vaultapi.RenewBehaviorIgnoreErrors,
		})
		if err != nil {
			return fmt.Errorf("error beginning Vault provider token renewal: %v", err)
		}

		ctx, cancel := context.WithCancel(context.Background())
		if v.stopWatcher != nil {
			// stop the running watcher loop if we are re-configuring
			v.stopWatcher()
		}
		v.stopWatcher = cancel
		// NOTE: Any codepaths after v.renewToken(...) which return an error
		// _must_ call v.stopWatcher() to prevent the renewal goroutine from
		// leaking when the CA initialization fails and gets retried later.
		go v.renewToken(ctx, lifetimeWatcher)
	}

	// Update the intermediate (managed) PKI mount and role
	if err := v.setupIntermediatePKIPath(); err != nil {
		if v.stopWatcher != nil {
			v.stopWatcher()
		}
		return err
	}

	return nil
}


func (v *VaultProvider) ValidateConfigUpdate(prevRaw, nextRaw map[string]interface{}) error {
	prev, err := ParseVaultCAConfig(prevRaw, v.isPrimary)
	if err != nil {
		return fmt.Errorf("failed to parse existing CA config: %w", err)
	}
	next, err := ParseVaultCAConfig(nextRaw, v.isPrimary)
	if err != nil {
		return fmt.Errorf("failed to parse new CA config: %w", err)
	}

	if prev.RootPKIPath != next.RootPKIPath {
		return nil
	}

	if prev.PrivateKeyType != "" && prev.PrivateKeyType != connect.DefaultPrivateKeyType { 		if prev.PrivateKeyType != next.PrivateKeyType {
			return fmt.Errorf("cannot update the privateKeyType field without changingRootPKIPath")
		} 	}

	if prev.PrivateKeyBits != 0 && prev.PrivateKeyBits != connect.DefaultPrivateKeyBits {
		if prev.PrivateKeyBits != next.PrivateKeyBits {
			return fmt.Errorf("cannot update the PrivateKeyBits field without changing RootPKIPath")
		}
	}
	return nil
}