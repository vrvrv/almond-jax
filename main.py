from absl import app
from absl import flags
from absl import logging

from src import train, evaluate
import jax
import numpyro
import tensorflow as tf
from ml_collections import config_flags

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
flags.DEFINE_enum("mode", None, ["train", "eval"], "Running mode: train or eval")
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True
)
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.mark_flags_as_required(['config', 'workdir', 'mode'])

def main(argv):
    tf.config.experimental.set_visible_devices([], 'GPU')

    logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())

    if FLAGS.mode == 'train':
        train(FLAGS.config, FLAGS.workdir)

    elif FLAGS.mode == 'eval':
        evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)

    else:
        raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
    app.run(main)