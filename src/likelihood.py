import jax.numpy as jnp

def prior_ll(z):
    return - jnp.sum(jnp.square(z) / 2)


def get_emission_ll(distribution):
    if distribution == 'gaussian':
        def emission_ll(pred, true):
            return - jnp.sum(jnp.square(pred - true) / 2)

        return emission_ll

    elif distribution == 'bernoulli':
        def emission_ll(pred, true):
            return jnp.sum(true * jnp.log(pred + 1e-10) + (1 - true) * jnp.log(1 - pred + 1e-10))

        return emission_ll

    else:
        raise NotImplementedError