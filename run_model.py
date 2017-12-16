import tensorflow as tf
import drink_data_reader as dr
import model as mod
import numpy as np
import time
import math
import os
import sys
import random


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay", 0.99, "Learning rate decays by this")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this")
tf.app.flags.DEFINE_integer("batch_size", 64, "Learning batch size")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each layer")
tf.app.flags.DEFINE_integer("from_size", 678, "Number of ingredients")
tf.app.flags.DEFINE_integer("to_size", 1071, "Number of words in drink names")
#tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "How many training steps per checkpoint")
tf.app.flags.DEFINE_boolean("decode", False, "Set to true to decode interactively")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "directory where training takes place")


FLAGS = tf.app.flags.FLAGS

_buckets = [(6, 3), (4, 7), (9, 5)
            ]
NUMSHUFFLES = 4
TRAIN_DATA_FRACTION = 0.9


def prepare_data():
    from_data, to_data = dr.prepare_data()
    data_set = [[] for _ in _buckets]
    shuffled = [[] for _ in _buckets]
    for index, val in enumerate(from_data):
        for bucket, (from_size, to_size) in enumerate(_buckets):
            if len(val) < from_size and len(to_data[index]) < to_size:
                data_set[bucket].append([val, to_data[index]])
    training_set = [random.sample(data, int(len(data)*TRAIN_DATA_FRACTION)) for data in data_set]
    cv_set = [[entry for entry in data if entry not in training_set[i]] for i, data in enumerate(data_set)]
    for index, bucket in enumerate(training_set):
        for entry in bucket:
            for _ in range(NUMSHUFFLES):
                copy = [v for v in entry]
                random.shuffle(copy[0])
                shuffled[index].append(copy)
        training_set[index].extend(shuffled[index])
    return training_set, cv_set


def create(session, forward_only):
    model = mod.Model(FLAGS.from_size,
                          FLAGS.to_size,
                          _buckets,
                          FLAGS.size,
                          FLAGS.num_layers,
                          FLAGS.max_gradient_norm,
                          FLAGS.batch_size,
                          FLAGS.learning_rate,
                          FLAGS.learning_rate_decay,
                          forward_only=forward_only,
                          dtype=tf.float32)
    chk = tf.train.get_checkpoint_state(FLAGS.train_dir)
    #if chk and tf.train.checkpoint_exists(chk.model_checkpoint_path):
    #print("loading from existing checkpoint")
    #    model.saver.restore(session, chk.model_checkpoint_path)
    #else:
    print("making new model from scratch")
    session.run(tf.global_variables_initializer())
    return model


def calc_perplexity(loss):
    return math.exp(float(loss)) if loss < 300 else float("inf")




def train(num_steps):
    with tf.Session() as sess:
        model = create(sess, False)
        data, cv = prepare_data()
        train_buckets_sizes = [len(data[b]) for b in range(len(_buckets))]
        train_total_size = float(sum(train_buckets_sizes))
        train_buckets_scale = [sum(train_buckets_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_buckets_sizes))]

        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_loss = []
        errors = {}
        while current_step < num_steps:
            rand = np.random.random_sample()
            bucket = min([i for i in range(len(train_buckets_scale))
                          if train_buckets_scale[i] > rand])
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                data, bucket
            )
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket, False)

            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            if current_step % FLAGS.steps_per_checkpoint == 0:
                perplexity = math.exp(float(loss) if loss < 300 else float("inf"))
                print("global step %d learning rate %.4f step-time %.2f perplexity " 
                      "%.2f" % (model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))
                if len(previous_loss) > 2 and loss > max(previous_loss[-3:]):
                    sess.run(model.learning_rate_decay)
                previous_loss.append(loss)
                checkpoint_path = os.path.join(FLAGS.train_dir, "drink.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)

                step_time, loss = 0.0, 0.0
                errors[current_step] = []
                for bucket in range(len(_buckets)):
                    if len(data[bucket]) == 0:
                        print("empty bucket %d" % (bucket))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(data, bucket)
                    cv_encoder, cv_decoder, cv_weights = model.get_batch(cv, bucket)
                    _, cv_loss, _ = model.step(sess, cv_encoder, cv_decoder, cv_weights, bucket, True)
                    cv_ppx = calc_perplexity(cv_loss)
                    errors[current_step].append(cv_ppx)
                    print(" Cross Validation: bucket %d perplexity %.2f" % (bucket, cv_ppx))
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket, True)
                    eval_ppx = calc_perplexity(eval_loss)
                    print(" eval: bucket %d perplexity %.2f" % (bucket, eval_ppx))
                sys.stdout.flush()
        return errors


def decode():
    with tf.Session() as sess:
        model = create(sess, True)
        vocab, words = dr.vocab_matching(dr.all_words())
        sys.stdout.write('> ')
        sys.stdout.flush()
        ingredients = sys.stdin.readline()
        while ingredients:
            ids = [int(w) for w in ingredients.split(',')]
            bucket_id = len(_buckets) - 1
            for i, bucket in enumerate(_buckets):
                if bucket[0] >= len(ids):
                    bucket_id = i
                    break

            encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id : [(ids, [])]}, bucket_id)
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)

            outputs = [int(np.argmax(logit, axis=1)[0]) for logit in output_logits]

            if dr._EOS_ID in outputs:
                outputs = outputs[:outputs.index(dr._EOS_ID)]

            print(" ".join([tf.compat.as_str(words[output]) for output in outputs]))
            print("> ", end="")
            sys.stdout.flush()
            ingredients = sys.stdin.readline()


def write_error(data, path='.',):
    with open(os.path.join(path, ['Error.txt'])) as errfile:
        for index, value in enumerate(data):
            errfile.write(str(index) + '\n')
            for step in value.items():
                errfile.write('step :' + str(step[0]) + ' ')
                for bucket_err in step[1]:
                    errfile.write(str(bucket_err) + ' ')
                errfile.write('\n')


def main(_):
    if FLAGS.decode:
        decode()
    else:
        num_steps = 10000
        num_layers = [1, 2, 3]
        layer_size = [256, 512, 1024]
        max_gradient_norm = [1.0, 5.0, 10.0]
        cv_errors = []
        for num in num_layers:
            for size in layer_size:
                for norms in max_gradient_norm:

                    FLAGS.num_layers = num
                    FLAGS.size = size
                    FLAGS.max_gradient_norm = norms
                    print('num layers: ' + str(FLAGS.num_layers))
                    print('layer size: ' + str(FLAGS.size))
                    print('max gradient norm: ' + str(FLAGS.max_gradient_norm))
                    with tf.variable_scope('model' + str(size) + str(num), reuse=tf.AUTO_REUSE) as scope:
                        cv_errors.append(train(num_steps))
        write_error(cv_errors)
        #train(num_steps)


if __name__ == "__main__":
    tf.app.run()