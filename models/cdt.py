from osrl.algorithms.cdt import CDT, CDTTrainer

def build(env, model_config, trainer_config, device):
    """We build the CDT model directly from OSRL, we use this function to wrap it with our params"""
    action_dim = env.action_space.shape[0]

    model = CDT(
        state_dim=env.observation_space.shape[0],
        action_dim=action_dim,
        max_action=float(env.action_space.high[0]),
        episode_len=env.spec.max_episode_steps,
        target_entropy=-action_dim,
        **model_config,
    ).to(device)

    if model.stochastic:
        model.log_temperature = model.log_temperature.detach().to(device).requires_grad_(True)

    trainer = CDTTrainer(model=model, env=env, device=device, **trainer_config)
    return model, trainer