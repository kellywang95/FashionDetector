from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

tf.logging.set_verbosity(tf.logging.INFO)

slim = tf.contrib.slim


# DATA_DIR = '/home/ubuntu/kelly/fashion_rgb/'
DATA_DIR = '/Users/kellywang/Downloads/fabric_attr/'



class Attribute_model:

    def __init__(self, is_training, n_attri):
        self.is_training = is_training
        self.n_attri = n_attri
        self.weight_decay = 0.0005

    def forward(self, features):
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                weights_regularizer=slim.l2_regularizer(self.weight_decay)):

            net = slim.repeat(features, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            # import IPython
            # IPython.embed()
            # net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            # net = slim.max_pool2d(net, [2, 2], scope='pool5')

            logits = slim.conv2d(net, self.n_attri, [1, 1], scope='fc6')
            logits = tf.squeeze(logits, [1, 2], name='fc6/squeezed')
            probabilities = tf.nn.softmax(logits)
            # output = tf.argmax(logits, axis=-1)

        return logits, probabilities



def get_model_fn(features, labels, mode, params):
    """Get model_fn for tf.estimator.Estimator
    tf.estimator.EstimatorSpec args (basic):
        - mode
        - predictions
        - loss
        - train_op
        - eval_metric_ops
    Args:
        features: default features  (batch, 32, 32, 3)
        labels: default labels      (batch)
        mode: default tf.estimator.ModeKeys
        params: default HParams object
    Return:
        tf.estimator.EstimatorSpec
    """
    # get custom model
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    model = Attribute_model(is_training, 202)
    logits, probabilities = model.forward(features)
    
    if mode != tf.estimator.ModeKeys.PREDICT:
        # loss

        # loss = tf.nn.sigmoid_cross_entropy_with_logits(
        #     labels=tf.cast(labels, tf.float32),
        #     logits=logits
        # )

        loss = tf.losses.sigmoid_cross_entropy(
		    multi_class_labels = tf.cast(labels, tf.int32),
		    logits = logits,
		    loss_collection=tf.GraphKeys.LOSSES,
		    reduction= tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
		)

        # train_op
        decay_steps = 1000
        lr = tf.train.exponential_decay(
            0.001,
            tf.train.get_or_create_global_step(),
            decay_steps,
            0.95, staircase=True
        )
        train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=tf.train.get_or_create_global_step())

        # eval_metric_ops
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(
                labels=labels,
                predictions=logits,
                name='accuracy'
            )
        }
    else:
        loss = None
        train_op = None
        eval_metric_ops=None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"logits":logits, "probabilities": probabilities},
        # predictions={"predictions":predictions,"logits":logits},
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)

def topk_acc(probabilities, labels,k):
    correct = 0
    for i, each in enumerate(probabilities):
        top_indices = each.argsort()[-k:][::-1]
        if labels[i] in top_indices:
            correct += 1
    return correct/float(len(labels))


def evaluate_model():
	print('evaluating')
	predictions = list(fashion_tagger.predict(input_fn = eval_input_fn))
	ct = 0
	for i, each in enumerate(predictions):
		prob = each['probabilities']
		# import IPython
		# IPython.embed()
		# zero_idx = np.argwhere(logits<0.5)
		one_idx = np.argwhere(prob>0.5)
		print('one_idx:', one_idx)
		prediction = np.zeros(202)
		prediction[one_idx] = 1
		if np.array_equal(prediction,eval_labels[i]):
			ct += 1
	acc = float(ct) / len(eval_labels)
	print('accuracy:', acc)

def main(unused_argv):
    # Load training and eval data
    # mnist = input_data.read_data_sets(DATA_DIR, one_hot=False, validation_size=0)
    # train_data = mnist.train.images  # Returns np.array

    train_data = np.load(DATA_DIR+'train_set.npy')
    eval_data = np.load(DATA_DIR + 'valid_set.npy')
    train_labels = np.load(DATA_DIR + 'train_labels.npy')
    eval_labels = np.load(DATA_DIR + 'valid_labels.npy')
    print('length of train_labels:', len(train_labels[0]))
    # print(train_data.dtype)
    train_data = np.asarray(train_data,dtype= np.float32)
    # train_labels = np.asarray([x for x in train_labels], dtype=np.int32)
    train_data, train_labels = shuffle(train_data, train_labels)
    # eval_data = mnist.test.images  # Returns np.array
    eval_data = np.asarray(eval_data, dtype = np.float32)
    eval_data, eval_labels = shuffle(eval_data, eval_labels)
    # eval_data, eval_labels = eval_data[:100], eval_labels[:100]

    # Create the Estimator
    fashion_tagger = tf.estimator.Estimator(
        model_fn=get_model_fn, model_dir="/tmp/fc_model_fabric")

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=train_data,
        y=train_labels,
        batch_size=400,
        num_epochs=None,
        shuffle=True)

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x= eval_data,
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    for j in range(100):
        fashion_tagger.train(
            input_fn=train_input_fn,
            steps=400)

        # eval_results = fashion_tagger.evaluate(input_fn=eval_input_fn)
    	# print(eval_results)
    	evaluate_model()
    	
    	
# import IPython
# IPython.embed()

        # pred_probabilities = np.asarray([p['logits'] for p in predictions])
        # top1 = topk_acc(pred_probabilities,eval_labels,1)

        # print("top-1 accuracy:", top1)
        # print("top-3 accuracy:", top3)
        # print("top-5 accuracy:", top5)


if __name__ == "__main__":
    tf.app.run()


