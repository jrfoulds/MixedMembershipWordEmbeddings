# Code to fit mixed membership word embeddings using the topic model pre-clustering as input.
# Based on tutorial word embedding code by the authors of TensorFlow, with modifications by
# James Foulds to fit the mixed membership skip-gram embeddings. 
# Copyright 2018 James Foulds. All Rights Reserved.
#
# The original tensorflow code is available at https://www.tensorflow.org/tutorials/word2vec,
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

#SET THESE FOR YOUR DATA!
vocabulary_size = 13649 
numTopics = 2000 
dataDir = 'data/' #text data
topicDataDir = './' #output of java MMSGTM algorithm
saveDir = './';
textFilename = 'NIPS.txt'
topicAssignmentsFilename = 'MMskipGramTopicModel_topicAssignments.txt'

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 5       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

num_steps = 1000000 #used for MMR experiments on small datasets
#num_steps = 10000000 #used for classification experiments on medium sized datasets
learningRate = 0.1   #Reducing the learning rate improves gradient explosion problem. If you get NaN's, consider reducing this.
num_sampled = 64    # Number of negative examples to sample.



def read_data(filename, topicAssignmentFilename):
  data = list()
  topicAssignments = list()
  #Load all documents into one array, like in the word2vec_basic demo.
  #This leads to some words being included when they shouldn't at document boundaries,
  #but should be negligible for large documents (and acts as a regularizer :p)
  f = open(filename, 'r')
  for line in f:
   spl = line.split()
   for word in spl:
    data.append(int(word))
  f.close()
  
  f = open(topicAssignmentFilename, 'r')
  for line in f:
   spl = line.split()
   for word in spl:
     topicAssignments.append(int(word))
  f.close()
  return data, topicAssignments

data, topicAssignments = read_data(dataDir + textFilename, topicDataDir + topicAssignmentsFilename)
print('Data size', len(data))
print('Topic assignments size', len(topicAssignments))
data_index = 0


# Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  buffer2 = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    buffer2.append(topicAssignments[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer. Note that this is now a topic assignment, stored in buffer2
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer2[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    buffer2.append(topicAssignments[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

# Build and train a skip-gram model.
batch, labels = generate_batch(batch_size=batch_size, num_skips=num_skips, skip_window=skip_window)


graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([numTopics, embedding_size], -1.0, 1.0))
		#tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, inputs=embed, labels=train_labels,
                     num_sampled=num_sampled, num_classes=vocabulary_size))

  # Construct the SGD optimizer
  optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm

  # Add variable initializer.
  init = tf.initialize_all_variables()

# Step 5: Begin training.
with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print("Initialized")

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", step, ": ", average_loss)
      average_loss = 0
      #DEBUG-JIMMY! Use this if you hit problems and suspect the issue is exploding gradients. (If so, consider reducing the learning rate)
      #norms = norm.eval()
      #print(norms)


  #get parameter values as numpy arrays
  final_embeddings = normalized_embeddings.eval()
  embeddings_ = embeddings.eval()
  nce_weights_ = nce_weights.eval()
  nce_biases_ = nce_biases.eval()
  
#save the parameters to disk as CSV
np.savetxt(saveDir + 'MMembeddings.txt', embeddings_)
np.savetxt(saveDir + 'MMnormalizedEmbeddings.txt', final_embeddings)
np.savetxt(saveDir + 'MMnce_weights.txt', nce_weights_)
np.savetxt(saveDir + 'MMnce_biases.txt', nce_biases_)  

