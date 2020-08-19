import tensorflow as tf

import os
import numpy as np

# downloading the shakespeare dataset
path_to_file = tf.keras.utils.get_file('shakespeare.txt',
                                       'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# length of text is the number of characters in it
print ('Length of text: {} characters'.format(len(text)))

print(text[:250])

# Let's see the unique characters in the file
vocab = sorted(set(text))    # will make a set of all characters in text so that there would be no duplicates
print ('There are {} unique characters'.format(len(vocab)))


# ['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'
# , 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
# 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# Process the text
# Step 1: Vectorize the text : Before training we need to map string to a numerical representation
char2idx = {u:i for i, u in enumerate(vocab)}
# {'\n': 0, ' ': 1, '!': 2, '$': 3, '&': 4, "'": 5, ',': 6, '-': 7, '.': 8, '3': 9, ':': 10, ';': 11,
# '?': 12, 'A': 13, 'B': 14, 'C': 15, 'D': 16, 'E': 17, 'F': 18, 'G': 19, 'H': 20, 'I': 21, 'J': 22, 'K': 23, 'L': 24,
# 'M': 25, 'N': 26, 'O': 27, 'P': 28, 'Q': 29, 'R': 30, 'S': 31, 'T': 32, 'U': 33, 'V': 34, 'W': 35, 'X': 36,
# 'Y': 37, 'Z': 38, 'a': 39, 'b': 40, 'c': 41, 'd': 42, 'e': 43, 'f': 44, 'g': 45, 'h': 46, 'i': 47, 'j': 48, 'k': 49,
# 'l': 50, 'm': 51, 'n': 52, 'o': 53, 'p': 54, 'q': 55, 'r': 56, 's': 57, 't': 58, 'u': 59, 'v': 60, 'w': 61, 'x': 62, 'y': 63, 'z': 64}

idx2char = np.array(vocab)

# now that we have a numerical representation for each character, let's convert our main text into integers
# looping through our dict of char2idx we can change all character in main text to numerical representation
text_as_int = np.array([char2idx[c] for c in text])
# this is what we get after applying the above code [18 47 56 ... 45  8  0]

# example
# First Citizen' ---- characters mapped to int ---- > [18 47 56 57 58  1 15 47 58 47 64 43 52]



# The prediction task
# Given a character, or a sequence of characters, what is the most probable next character?
# This is the task we're training the model to perform.
# The input to the model will be a sequence of characters, and we train the model to predict the output—the following character at each time step.

# Since RNNs maintain an internal state that depends on the previously seen elements, given all the characters computed until this moment
# what is the next character?
# Create training examples and targets
# Next divide the text into example sequences. Each input sequence will contain *seq_length* characters from the text.
# For each input sequence, the corresponding targets contain the same length of text, except shifted one character to the right.
# So break the text into chunks of seq_length+1. For example, say seq_length is 4 and our text is "Hello".
# The input sequence would be "Hell", and the target sequence "ello".
# To do this first use the tf.data.Dataset.from_tensor_slices function to convert the text vector into a stream of character indices.


seq_length = 100  # The maximum length sentence we want for a single input in characters
examples_per_epoch = len(text)//(seq_length+1)  #this will divide the whole text data into sequence of 100

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
print(char_dataset)
for i in char_dataset.take(5):
   print(idx2char[i.numpy()]) # through this we can extract what character was at which place in the array(idx2char)
                              # eg: the char_dataset gives the value 18 and in idx2char 18 index is of "F"
                             # so for 5 such cases we get F,i,r,s,t

#we could also use "batch" which will convert individual words to a sequence of desired size
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

for item in sequences.take(5):
  print(repr(''.join(idx2char[item.numpy()]))) #this will print sequence of characters

# For each sequence, duplicate and shift it to form the input and target text by using the map method to apply a simple function to each batch
def split_input_target(chunk):
  input_text = chunk[:-1]
  target_text = chunk[1:]
  return input_text, target_text

dataset = sequences.map(split_input_target)

for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
  print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))

# ------------------------------------------------------------------------------------------------------------------
# Create training batches
# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

print(dataset)


# ------------------------------------------------------------------------------------------------------------------
# Build The Model

# Use "tf.keras.Sequential" to define the model. For this simple example three layers are used to define our model:
# "tf.keras.layers.Embedding": The input layer. A trainable lookup table that will map the numbers of
#                              each character to a vector with embedding_dim dimensions;
# "tf.keras.layers.LSTM": A type of RNN with size units=rnn_units
# "tf.keras.layers.Dense": The output layer, with vocab_size outputs.

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(
    vocab_size = len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

# For each character the model looks up the embedding, runs the GRU one timestep with the embedding as input,
# and applies the dense layer to generate logits predicting the log-likelihood of the next character
model.summary()

for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
  sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()

print("sampled_indices" , sampled_indices)


print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("Next Char Predictions Decode of sample indices: \n", repr("".join(idx2char[sampled_indices ])))

# ---------------------------------------------------------------------------------------------------------------------
# Train the model

# Attach an optimizer, and a loss function
# The standard tf.keras.losses.sparse_categorical_crossentropy loss function works in this case because
# it is applied across the last dimension of the predictions.
# Because our model returns logits, we need to set the from_logits flag.

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())

model.compile(optimizer='adam', loss=loss)


# Configure checkpoints
# Use a tf.keras.callbacks.ModelCheckpoint to ensure that checkpoints are saved during training

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


EPOCHS=10

#history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

tf.train.latest_checkpoint(checkpoint_dir)

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

model.summary()

def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 1000

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
    predictions = model(input_eval)
    # remove the batch dimension
    predictions = tf.squeeze(predictions, 0)

    # using a categorical distribution to predict the character returned by the model
    predictions = predictions / temperature
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

    # We pass the predicted character as the next input to the model
    # along with the previous hidden state
    input_eval = tf.expand_dims([predicted_id], 0)

    text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))


print(generate_text(model, start_string=u"ROMEO: "))
