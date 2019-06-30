import os
import sys
import time

import tensorflow as tf

tf.enable_eager_execution()
import cv2 as cv
import json
import numpy as np

TRAIN_SIZE = 100
OUTPUT_CHANNELS = 3
SIZE = 256

IMGS_PATH = "Hanabi/images_full/"
TAGS_PATH = "Hanabi/tags_full/"

imgs = []
tags = []

alltags = []
for i in range(TRAIN_SIZE):
    img = cv.imread(IMGS_PATH + "{}.png".format(i))
    img = cv.resize(img, (SIZE, SIZE))
    imgs.append(img)
    tag = json.load(open(TAGS_PATH + "{}.json".format(i)))
    tags.append(tag)
    alltags += tag

alltags = list(set(list(alltags)))
N_TAGS = len(alltags)

tags_train = []
for tag in tags:
    tags_ones = np.zeros((N_TAGS))
    tagsind = []
    for tag1 in tag:
        index = alltags.index(tag1)
        tagsind.append(index)
    tags_ones[tagsind] = 1
    tags_train.append(tags_ones)

tags_train = np.float32(tags_train)
imgs_train = np.float32(imgs) / 127.5 - 1


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def Generator():
    inputs = tf.keras.layers.Input(shape=[N_TAGS])

    dense_stack = [
        # tf.keras.layers.Dense(1000, activation='relu'),
        # tf.keras.layers.Dense(512, activation='relu'),
    ]

    reshape = tf.keras.layers.Reshape((1, 1, N_TAGS))

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
        # upsample(32, 4),  # (bs, 256, 256, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (bs, 256, 256, 3)

    x = inputs

    for dense in dense_stack:
        x = dense(x)

    x = reshape(x)

    # Upsampling and establishing the skip connections
    for up in up_stack:
        x = up(x)

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  tar = tf.keras.layers.Input(shape=[SIZE, SIZE, OUTPUT_CHANNELS], name='target_image2')

  down1 = downsample(64, 4, False)(tar) # (bs, 128, 128, 64)
  down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
  down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

  return tf.keras.Model(inputs=tar, outputs=last)

def Encoder():
    input = tf.keras.layers.Input(shape=[SIZE, SIZE, OUTPUT_CHANNELS], name='target_image')

    down_stack = [
        downsample(64, 4, False),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]

    flatten = tf.keras.layers.Flatten()

    dense_stack = [
        # tf.keras.layers.Dense(512, activation='relu'),
        # tf.keras.layers.Dense(1000, activation='relu')
    ]

    last = tf.keras.layers.Dense(N_TAGS, activation='sigmoid')

    x = input

    for down in down_stack:
        x = down(x)

    x = flatten(x)

    for dense in dense_stack:
        x = dense(x)

    x = last(x)

    return tf.keras.Model(inputs=input, outputs=x)

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

def encoder_loss(enc_real_output, target_tags):
    return loss_object(target_tags, enc_real_output)

LAMBDA = 100
def generator_loss(disc_generated_output, enc_generated_output, gen_output, target_image, target_tags):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    enc_loss = loss_object(target_tags, enc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target_image - gen_output))

    total_gen_loss = gan_loss + enc_loss + (LAMBDA * l1_loss)

    return total_gen_loss

generator = Generator()
discriminator = Discriminator()
encoder = Encoder()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
encoder_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = "drawer_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

def generate_images(model, test_input, tar, title_def="Predicted Image"):
  prediction = model(test_input, training=True)

  display_list = [tar[0], prediction[0]]
  title = ['Ground Truth', title_def]

  for i in range(2):
    cv.imshow(title[i], np.float32(display_list[i]*0.5 + 0.5))
  cv.waitKey(100)

@tf.function
def train_step(input, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as enc_tape:
        gen_output = generator(input, training=True)

        enc_real_output = encoder(target)
        enc_gen_output = encoder(gen_output)

        disc_real_output = discriminator(target, training=True)
        disc_generated_output = discriminator(gen_output, training=True)

        gen_loss = generator_loss(disc_generated_output, enc_gen_output, gen_output, target, input)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
        enc_loss = encoder_loss(enc_real_output, input)

    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    encoder_gradients = enc_tape.gradient(enc_loss, encoder.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))
    encoder_optimizer.apply_gradients(zip(encoder_gradients, encoder.trainable_variables))


def train(tags, images, epochs):
    for epoch in range(epochs):
        start = time.time()

        for step in range(TRAIN_SIZE):
            input = tags[step:step+1]
            target = images[step:step+1]
            train_step(input, target)
            sys.stdout.write("\r" + str(step) + " ")
            sys.stdout.flush()

            ind = np.random.randint(0, TRAIN_SIZE-1)
            generate_images(generator, tags[ind:ind+1], images[ind:ind+1])

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time() - start))

        if epoch % 1 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


train(tags_train, imgs_train, 10)


img1 = generator.predict(tf.random_normal((1, N_TAGS)))[0]
img1 = img1 * 0.5 + 0.5
cv.imshow("Image", img1)
cv.waitKey()
