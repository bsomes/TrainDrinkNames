import tensorflow as tf
import numpy as np
import random

import drink_data_reader as reader


class Model(object):


    def __init__(self,
                 source_vocab_size,
                 target_vocab_size,
                 buckets,
                 size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay,
                 num_samples = 512,
                 forward_only = False,
                 dtype = tf.float32):
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay = self.learning_rate.assign(self.learning_rate * learning_rate_decay)
        self.global_step = tf.Variable(0, trainable=False)

        output_projection = None
        softmax_loss_function = None
        if num_samples > 0 and num_samples < self.target_vocab_size:
            w_t = tf.get_variable('proj_w', [self.target_vocab_size, size], dtype=dtype)
            w = tf.transpose(w_t)
            b = tf.get_variable('proj_b', [self.target_vocab_size], dtype=dtype)
            output_projection = (w, b)

            def sampled_loss(labels, logits):
                labels = tf.reshape(labels, [-1, 1])
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(logits, tf.float32)
                return tf.cast(
                    tf.nn.sampled_softmax_loss(
                        weights=local_w_t,
                        biases=local_b,
                        labels = labels,
                        inputs = local_inputs,
                        num_sampled = num_samples,
                        num_classes=self.target_vocab_size),
                    dtype)
            softmax_loss_function = sampled_loss


        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            if num_layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(size) for _ in range(num_layers)])
            else:
                cell = tf.contrib.rnn.GRUCell(size)
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                encoder_inputs,
                decoder_inputs,
                cell,
                num_encoder_symbols=source_vocab_size,
                num_decoder_symbols=target_vocab_size,
                embedding_size=size,
                output_projection=output_projection,
                feed_previous=do_decode,
                dtype=dtype
            )

        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in range(buckets[-1][0]):
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))

        for i in range(buckets[-1][0] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(dtype, shape=[None], name="weight{0}".format(i)))

        targets = [self.decoder_inputs[i + 1] for i in range(len(self.decoder_inputs) - 1)]


        if forward_only:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets, self.target_weights, buckets,
                lambda x, y: seq2seq_f(x, y, True),
                softmax_loss_function=softmax_loss_function
            )
            if output_projection is not None:
                for b in range(len(buckets)):
                    self.outputs[b] = [
                        tf.matmul(output, output_projection[0]) + output_projection[1]
                        for output in self.outputs[b]
                    ]
        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs,
                self.decoder_inputs,
                targets,
                self.target_weights,
                buckets,
                lambda x, y: seq2seq_f(x, y, False),
                softmax_loss_function=softmax_loss_function
            )

        params = tf.trainable_variables()
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            for b in range(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params), global_step=self.global_step
                ))
        self.saver = tf.train.Saver(tf.global_variables())


    def step(self, session, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only):
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder input must be same size as bucket")
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder input must be same size as bucket")
        if len(target_weights) != decoder_size:
            raise ValueError("weights size must be same size as bucket")

        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        if not forward_only:
            output_feed = [self.updates[bucket_id],
                           self.gradient_norms[bucket_id],
                           self.losses[bucket_id]]
        else:
            output_feed = [self.losses[bucket_id]]
            for l in range(decoder_size):
                output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None
        else:
            return None, outputs[0], outputs[1:]


    def get_batch(self, data, bucket_id):
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []

        for _ in range(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])
            encoder_pad = [reader._PAD_ID]  * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([reader._GO_ID] + decoder_input + [reader._PAD_ID] * decoder_pad_size)

        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        for ind in range(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][ind]
                          for batch_idx in range(self.batch_size)], dtype=np.int32)
            )

        for ind in range(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][ind]
                          for batch_idx in range(self.batch_size)], dtype=np.int32)
            )

            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in range(self.batch_size):
                if ind < decoder_size - 1:
                    target = decoder_inputs[batch_idx][ind + 1]
                if ind == decoder_size - 1 or target == reader._PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights