# import tensorflow.compat.v1 as tf
import tensorflow  as tf
import time
import os
import sys
import json

sys.path.append('src/electra')
sys.path.append('client/src/electra')
from run_pretraining import train_or_eval
import configure_pretraining

GLOBAL = True  # if True then federate also optimizer variables
if GLOBAL:
    PASSABLE_VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
else:
    PASSABLE_VARIABLES = tf.GraphKeys.TRAINABLE_VARIABLES


def load_weights_to_model(weights, settings):

    t0 = time.time()
    data_dir = settings["data_dir"]
    model_name = settings["model_name"]
    CHECKPOINT_DIR = data_dir + '/models/' + model_name

    if not os.path.isdir(CHECKPOINT_DIR):
        create_graph(settings)
    checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
        saver.restore(sess, checkpoint.model_checkpoint_path)

        old_weights = tf.get_collection(PASSABLE_VARIABLES)
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
            with open(CHECKPOINT_DIR + "/" + "metadata.json", "w") as fh:
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


def get_weights_from_model(settings):
    t0 = time.time()
    tf.reset_default_graph()

    data_dir = settings["data_dir"]
    model_name = settings["model_name"]
    CHECKPOINT_DIR = data_dir + '/models/' + model_name

    if not os.path.isdir(CHECKPOINT_DIR):
        create_graph(settings)
    checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
        saver.restore(sess, checkpoint.model_checkpoint_path)

        weights = tf.get_collection(PASSABLE_VARIABLES)
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


def create_graph(settings):
    """Creates checkpoints and model dependent files to initate the electra model."""
    # import os
    # arr = os.listdir(data_dir)
    # print(arr)
    with open(settings["hparams"]) as fh:
        hparams = json.load(fh)
    for setting in settings:
        if setting in hparams:
            hparams[setting] = settings[setting]
    hparams["num_train_steps"] = 1
    data_dir = settings["data_dir"]
    model_name = settings["model_name"]
    tf.logging.set_verbosity(tf.logging.ERROR)
    train_or_eval(configure_pretraining.PretrainingConfig(
        model_name, data_dir, **hparams))


def get_global_step(settings):
    data_dir = settings["data_dir"]
    model_name = settings["model_name"]
    CHECKPOINT_DIR = data_dir + '/models/' + model_name
    print(f"getting global step from {CHECKPOINT_DIR}")

    try:
        with open(CHECKPOINT_DIR + "/" + "metadata.json") as json_file:
            data = json.load(json_file)
            global_step = data["global_step"]
    except Exception as e:
        print(f"loading metadata.json in get_global_step causing Exception {e}")
        print("This is expected because there was no metadata.json and I continue...")

        if not os.path.isdir(CHECKPOINT_DIR):
            create_graph(settings)
        checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
            saver.restore(sess, checkpoint.model_checkpoint_path)

            global_step_ = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == 'global_step:0'][0]
            global_step = sess.run(global_step_)
            data = {
                "global_step": int(global_step)
            }
            with open(CHECKPOINT_DIR + "/" + "metadata.json", "w") as fh:

                fh.write(json.dumps(data))
    return global_step
