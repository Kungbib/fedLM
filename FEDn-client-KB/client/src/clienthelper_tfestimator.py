import tensorflow as tf
import time
import os
from .electra.run_pretraining import train_or_eval
from .electra import configure_pretraining
import json


def load_weights_to_model(weights, data_dir, modelname):

    t0 = time.time()

    CHECKPOINT_DIR = data_dir + '/models/' + modelname

    if not os.path.isdir(CHECKPOINT_DIR):
        create_graph(data_dir, modelname)
    checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
        saver.restore(sess, checkpoint.model_checkpoint_path)

        old_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        weights_updated = []
        for w in old_weights:
            try:
                weights_updated += [w.assign(weights[w.name])]
            except Exception as e:
                print(f"found Exception {e} while updating weights in load_weights_to_model")
                pass

        # Load global step separately and ensure it is loaded to the model as an int.
        try:
            global_step = weights['global_step:0']
            global_step_ = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == 'global_step:0'][0]
            weights_updated += [global_step_.assign(int(global_step))]
            data = {
                "global_step": int(global_step)
            }
            with open("metadata.json", "w") as fh:
                fh.write(json.dumps(data))
        except Exception as e:
            print(f"found Exception {e} while further updating weights in load_weights_to_model")
            pass

        r = sess.run(weights_updated)
        saver.save(sess, checkpoint.model_checkpoint_path)
        sess.close()

    t1 = time.time()
    print("time: ", t1 - t0)

    return 0


def get_weights_from_model(data_dir, modelname, hparams_fn):
    t0 = time.time()
    tf.reset_default_graph()

    CHECKPOINT_DIR = data_dir + '/models/' + modelname

    if not os.path.isdir(CHECKPOINT_DIR):
        create_graph(data_dir, modelname, hparams_fn)
    checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
        saver.restore(sess, checkpoint.model_checkpoint_path)

        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        weights_ = sess.run(weights)
        global_step_ = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == 'global_step:0'][0]
        global_step = sess.run(global_step_)

    stored_weights = {}
    for i in range(len(weights)):
        stored_weights[weights[i].name] = weights_[i]
    stored_weights['global_step:0'] = global_step
    t1 = time.time()
    print("time: ", t1 - t0)
    return stored_weights


def create_graph(data_dir, model_name, hparams_fn):
    """Creates checkpoints and model dependent files to initate the electra model."""
    import os
    arr = os.listdir(data_dir)
    print(arr)
    with open(hparams_fn) as fh:
        hparams = json.load(fh)

    tf.logging.set_verbosity(tf.logging.ERROR)
    train_or_eval(configure_pretraining.PretrainingConfig(
        model_name, data_dir, **hparams))


def get_global_step(data_dir, modelname):

    try:
        with open("metadata.json") as json_file:
            data = json.load(json_file)
            global_step = data["global_step"]
    except Exception as e:
        print(f"loading metadata.json in get_global_step causing Exception {e}")
        print("This is expected and I continue...")
        CHECKPOINT_DIR = data_dir + '/models/' + modelname

        if not os.path.isdir(CHECKPOINT_DIR):
            create_graph(data_dir, modelname)
        checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
            saver.restore(sess, checkpoint.model_checkpoint_path)

            global_step_ = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == 'global_step:0'][0]
            global_step = sess.run(global_step_)
            data = {
                "global_step": int(global_step)
            }
            with open("metadata.json", "w") as fh:

                fh.write(json.dumps(data))
    return global_step
