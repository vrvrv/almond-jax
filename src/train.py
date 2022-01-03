import os
import jax
from jax import random, numpy as jnp
from flax.core import freeze
from flax.metrics import tensorboard
from flax.training import train_state
import flax.jax_utils as flax_utils

import optax
import tensorflow as tf

from .model import get_model
from .dataset import get_dataset
from .likelihood import prior_ll, get_emission_ll
from collections import namedtuple
from absl import logging

import numpyro
from numpyro.infer import MCMC
from functools import partial
import ml_collections


def rescale(tensor):
    return 0.5 * 0.5 * tensor


def create_train_state(model, rng, config):
    """Creates initial `TrainState`."""
    params_init = model.init(rng, jnp.ones([1, config.model.latent_dim]))

    if config.optim.optimizer == 'Adam':
        tx = optax.adam(config.optim.learning_rate)

    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params_init['params'], tx=tx
    )

    params_lst = []
    for lyr, p in params_init['params'].items():
        params_lst.extend(['_'.join([lyr, p_name]) for p_name in p.keys()])

    return params_init, state, params_lst


def grad_tree_to_lst(grad_tree, params_lst):
    grad_lst = []
    for param_name in params_lst:
        lyr, p_name = '_'.join(param_name.split('_')[:-1]), param_name.split('_')[-1]
        grad_lst.append(grad_tree[lyr][p_name])
    return grad_lst


def train(config: ml_collections.ConfigDict,
          workdir: str):
    rng = jax.random.PRNGKey(config.seed + 1)

    # Init dataset
    train_ds, test_ds = get_dataset(config)
    train_iter = iter(train_ds)
    test_iter = iter(test_ds)

    # Init model
    decoder = get_model(config.model)

    # Define likelihood function
    emission_ll = get_emission_ll(distribution=config.model.distribution)

    def log_likelihood(params, z, x):
        recon_x = decoder.apply({'params': params}, z)

        log_pz = prior_ll(z)
        log_pxz = emission_ll(recon_x, x)

        return jnp.sum(log_pz + log_pxz)

    # Init logger
    if jax.process_index() == 0:
        tf.io.gfile.makedirs(workdir)
        summary_writer = tensorboard.SummaryWriter(workdir)
        summary_writer.hparams(dict(config.sampling))

    # Create checkpoint directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    tf.io.gfile.makedirs(checkpoint_dir)

    rng, init_rng = jax.random.split(rng)
    # params, tx, opt_state, params_lst = create_train_state(decoder, init_rng, config)
    params, state, params_lst = create_train_state(decoder, init_rng, config)

    AlmondState = namedtuple(
        "AlmondState", ["u", *params_lst, "rng_key"]
    )

    class LangevinSampler(numpyro.infer.mcmc.MCMCKernel):
        sample_field = "u"

        def __init__(self, joint_ll, step_size=0.1):
            self.joint_ll = joint_ll
            self.step_size = step_size

        def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
            return AlmondState(*init_params, rng_key)

        def sample(self, state, model_args, model_kwargs):
            rng_key, key_eps = random.split(state.rng_key, 2)

            score = jax.grad(self.joint_ll, argnums=1)(model_args[0], state.u)

            u_new = state.u + self.step_size * score + jnp.sqrt(2 * self.step_size) * random.normal(key_eps,
                                                                                                    state.u.shape)

            grad_tree = jax.grad(self.joint_ll, argnums=0)(model_args[0], state.u)
            grad_lst = grad_tree_to_lst(grad_tree, params_lst)

            return AlmondState(u_new, *grad_lst, rng_key)

    def apply_model(carry, x):
        """Computes gradients, loss and accuracy for a single batch."""
        rng, state = carry
        rng, z_init_rng = jax.random.split(rng)


        # define log p(z, x = x)
        log_likelihood_fn = partial(
            log_likelihood, x=x
        )
        kernel = LangevinSampler(log_likelihood_fn, step_size=config.sampling.step_size)

        mcmc = MCMC(
            kernel,
            num_warmup=config.sampling.num_mcmc_warmups,
            num_samples=config.sampling.num_mcmc_samples,
            # num_chains=config.sampling.num_mcmc_chains,
            chain_method='parallel',
            progress_bar=False
        )

        # Initialize MCMC init parameter
        # Init param = (z0, grad0)

        z = random.normal(
            z_init_rng, (config.model.latent_dim,)
        )

        grads_tree = jax.grad(log_likelihood_fn, argnums=0)(
            state.params, jnp.zeros(config.model.latent_dim)
        )

        grad_lst = []
        for grad in grad_tree_to_lst(grads_tree, params_lst):
            grad_lst.append(grad)

        mcmc_init_param = (z, *grad_lst)

        rng, mcmc_rng = jax.random.split(rng)

        # Run Langevin MCMC
        mcmc.run(
            mcmc_rng,
            state.params,
            init_params=mcmc_init_param,
            extra_fields=params_lst
        )

        posterior_samples = mcmc.get_samples()
        grads = mcmc.get_extra_fields()

        unfreezed_grads = grads_tree.unfreeze()
        for lyr, p in unfreezed_grads.items():
            for p_name in p.keys():
                unfreezed_grads[lyr][p_name] = - grads[lyr + '_' + p_name].mean(0)
        grads = freeze(unfreezed_grads)

        # update parameters
        new_state = state.apply_gradients(grads=grads)

        new_carry_state = (rng, new_state)

        return new_carry_state, posterior_samples

    # Pmap (and jit-compile) multiple training steps together for faster running
    p_apply_model = jax.pmap(partial(jax.lax.scan, apply_model))

    # Replicate the training states to run on multiple devices
    p_state = flax_utils.replicate(state)
    rng = jax.random.fold_in(rng, jax.process_index())

    for step in range(config.training.n_iters):
        batch = jax.tree_map(lambda x: x._numpy(), next(train_iter))
        rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
        next_rng = jnp.array(next_rng)

        (_, p_state), p_z = p_apply_model((next_rng, p_state), batch['image'])

        if step % config.training.log_freq == 0 and jax.process_index() == 0:
            state = flax_utils.unreplicate(p_state)

            xhat = decoder.apply({'params': state.params}, p_z[:, :, -1, :])

            recon_loss = float(jnp.square(xhat - batch['image']).mean())

            # log images
            recon_image = jnp.reshape(xhat[0], (-1, 28, 28, 1))
            origin_image = jnp.reshape(batch['image'][0], (-1, 28, 28, 1))

            image_pair = jnp.concatenate([origin_image, recon_image], axis=-2)

            summary_writer.image(
                'reconstruction',
                image_pair, max_outputs=4, step=step
            )
            summary_writer.scalar('training_loss', recon_loss, step=step)

            logging.info(
                "step: %d, recon_loss: %.4f" % (step, recon_loss)
            )

    summary_writer.flush()
    return
