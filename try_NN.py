# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 14:42:55 2021

@author: Jesús S. Alegre
"""

import tensorflow as tf
import os
import tensorflow_datasets as tfds
from official.vision.image_classification.resnet import common

batch_size = 1024

def build_model():
  """Constructs the ML model used to predict handwritten digits."""

  image = tf.keras.layers.Input(shape=(28, 28, 1))

  y = tf.keras.layers.Conv2D(filters=32,
                             kernel_size=5,
                             padding='same',
                             activation='relu')(image)
  y = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2),
                                   padding='same')(y)
  y = tf.keras.layers.Conv2D(filters=32,
                             kernel_size=5,
                             padding='same',
                             activation='relu')(y)
  y = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2),
                                   padding='same')(y)
  y = tf.keras.layers.Flatten()(y)
  y = tf.keras.layers.Dense(1024, activation='relu')(y)
  y = tf.keras.layers.Dropout(0.4)(y)

  probs = tf.keras.layers.Dense(10, activation='softmax')(y)

  model = tf.keras.models.Model(image, probs, name='mnist')

  return model

@tfds.decode.make_decoder(output_dtype=tf.float32)
def decode_image(example, feature):
  """Convert image to float32 and normalize from [0, 255] to [0.0, 1.0]."""
  return tf.cast(feature.decode_example(example), dtype=tf.float32) / 255


# load dataset
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# assert x_train.shape == (60000, 28, 28)
# assert x_test.shape == (10000, 28, 28)
# assert y_train.shape == (60000,)
# assert y_test.shape == (10000,)

# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# y_train = y_train.astype('float32')
# y_test = y_test.astype('float32')

# x_train = x_train.reshape(-1,28,28,1)
# x_test = x_test.reshape(-1,28,28,1)

mnist = tfds.builder("mnist",data_dir=r'C:\Users\jesus\Documents\python projects\Abbott DHR\image_classification\$DATA_DIR')

mnist_train, mnist_test = mnist.as_dataset(
      split=['train', 'test'],
      decoders={'image': decode_image()},  # pylint: disable=no-value-for-parameter
      as_supervised=True)

train_input_dataset = mnist_train.cache().repeat().shuffle(
      buffer_size=50000).batch(batch_size)
eval_input_dataset = mnist_test.cache().repeat().batch(batch_size)


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.05, decay_steps=100000, decay_rate=0.96)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

model = build_model()
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'])


num_train_examples = mnist.info.splits['train'].num_examples
train_steps = num_train_examples
train_epochs = 1
batch_size = 1024
model_dir = r'C:\Users\jesus\Documents\python projects\Abbott DHR\DIR'
ckpt_full_path = os.path.join(model_dir, 'model.ckpt-{epoch:04d}')
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        ckpt_full_path, save_weights_only=True),
    tf.keras.callbacks.TensorBoard(log_dir=model_dir),
]
  
  
num_eval_examples = mnist.info.splits['test'].num_examples
num_eval_steps = num_eval_examples

train_input_dataset = mnist_train.cache().repeat().shuffle(
buffer_size=50000).batch(batch_size)
eval_input_dataset = mnist_test.cache().repeat().batch(batch_size)

history = model.fit(
    train_input_dataset,
    epochs=train_epochs,
    steps_per_epoch=train_steps,
    callbacks=callbacks,
    validation_steps=num_eval_steps,
    validation_data=eval_input_dataset,
    validation_freq=True)

export_path = os.path.join(model_dir, 'saved_model2')
model.save(export_path, include_optimizer=False)
#tf.saved_model.save(model,export_path)

eval_output = model.evaluate(eval_input_dataset, steps=num_eval_steps, verbose=2)

stats = common.build_stats(history, eval_output, callbacks)

