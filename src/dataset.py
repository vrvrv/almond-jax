import jax

import tensorflow as tf
import tensorflow_datasets as tfds
import jax.numpy as jnp


def prepare_image(x):
    x = tf.cast(x['image'], tf.float32)
    x = tf.reshape(x, (-1,))
    return x


def preprocess_fn(ds):
    img = tf.reshape(tf.image.convert_image_dtype(ds['image'], tf.float32), (784, ))
    return dict(image=img, label=ds.get('label', None))


def create_dataset(dataset_builder, split, config):
    per_device_batch_size = config.training.batch_size // jax.local_device_count()
    batch_dims = [jax.local_device_count(), per_device_batch_size]

    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.experimental_threading.private_threadpool_size = 48
    dataset_options.experimental_threading.max_intra_op_parallelism = 1
    read_config = tfds.ReadConfig(options=dataset_options)
    dataset_builder.download_and_prepare()

    ds = dataset_builder.as_dataset(
        split=split,
        shuffle_files=(split == 'train'),
        read_config=read_config
    )
    ds = ds.repeat(count=None)
    ds = ds.shuffle(10000)

    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    for batch_size in reversed(batch_dims):
        ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(tf.data.experimental.AUTOTUNE)


def get_dataset(config):

    if config.training.batch_size % jax.local_device_count() != 0:
        raise ValueError(f'Batch sizes ({config.training.batch_size}) must be devided by'
                         f'the number of devices ({jax.local_device_count()})')

    if config.data.dataset == 'mnist':
        """Load MNIST train and test datasets into memory."""
        ds_builder = tfds.builder('mnist')

        train_ds = create_dataset(ds_builder, 'train', config)
        test_ds = create_dataset(ds_builder, 'test', config)
        return train_ds, test_ds

    else:
        raise NotImplementedError
