"""Default Hyperparameter configuration."""

import ml_collections

def get_default_configs():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # Data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'mnist'

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 512
    training.n_iters = 10000
    training.n_jitted_steps = 5
    training.log_freq = 1

    # Model
    config.model = model = ml_collections.ConfigDict()
    model.latent_dim = 4
    model.hidden_dims = [500, 784]
    model.distribution = 'bernoulli'
    model.architecture = 'mlp'

    # Sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.num_mcmc_warmups = 200
    sampling.num_mcmc_samples = 200
    sampling.step_size = 0.1

    # Opimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = 'Adam'
    optim.learning_rate = 3e-4

    config.seed = 1234

    return config