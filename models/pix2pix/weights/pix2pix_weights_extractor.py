import time
import tensorflow as tf
import collections
import numpy as np
import cv2

Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake")

class ArgumentSettings():
    ngf = 64
    ndf = 64

a = ArgumentSettings()
EPS = 1e-12

def conv(batch_input, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv

def lrelu(x, a):
    with tf.name_scope("lrelu"):
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        # INPUTS.append(input)
        input = tf.identity(input)
        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        # WEIGHTS.append((mean,variance))
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized


def deconv(batch_input, out_channels):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
        return conv


def create_generator(generator_inputs, generator_outputs_channels):
    layers = []

    with tf.variable_scope("encoder_1"):
        output = conv(generator_inputs, a.ngf, stride=2)
        layers.append(output)

    layer_specs = [
        a.ngf * 2,
        a.ngf * 4,
        a.ngf * 8,
        a.ngf * 8,
        a.ngf * 8,
        a.ngf * 8,
        a.ngf * 8,
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            convolved = conv(rectified, out_channels, stride=2)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (a.ngf * 8, 0.5),
        (a.ngf * 8, 0.5),
        (a.ngf * 8, 0.5),
        (a.ngf * 8, 0.0),
        (a.ngf * 4, 0.0),
        (a.ngf * 2, 0.0),
        (a.ngf, 0.0),
    ]

    is_auto_encoder = False

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0 or is_auto_encoder:
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            output = deconv(rectified, out_channels)
            output = batchnorm(output)

            # if dropout > 0.0:
            #     output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    with tf.variable_scope("decoder_1"):
        if is_auto_encoder:
            input = layers[-1]
        else:
            input = tf.concat([layers[-1], layers[0]], axis=3)

        rectified = tf.nn.relu(input)
        output = deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]

def create_model(inputs, targets):
    def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []

        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        with tf.variable_scope("layer_1"):
            convolved = conv(input, a.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.ndf * min(2**(i+1), 8)
                stride = 1 if i == n_layers - 1 else 2
                convolved = conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]

    with tf.variable_scope("generator") as scope:
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)

    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            predict_real = create_discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            predict_fake = create_discriminator(inputs, outputs)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        outputs=outputs,
    )

def preprocess(img2):
    img_ = tf.image.convert_image_dtype(img2, dtype=tf.float32)
    img_ = (img_ / 127.5) - 1
    img_ = tf.image.resize(img_, [512, 512], method=tf.image.ResizeMethod.BILINEAR)

    return img_

def postprocess(final_output):
    final_output = tf.image.convert_image_dtype((final_output + 1) / 2, dtype=tf.uint8, saturate=True)
    return final_output

if __name__ == "__main__":
    tf.set_random_seed(1)

    # 소이넷 웨이트를 저장 할 경로
    weights_path="pix2pix.weights"

    # 저장한 모델 weight
    model_path="./lung_seg/"

    input_image = tf.placeholder(dtype=tf.float32, shape=(1, 512, 512, 1))
    target_image = tf.placeholder(dtype=tf.float32, shape=(1, 512, 512, 1))

    model = create_model(input_image, target_image)

    print(model.outputs)

    with tf.Session() as sess:
        # load_weights
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(model_path))

        weight_list = dict()

        for op in tf.global_variables():
            weight_list[op.name] = sess.run(op)
            print(op.name, op.shape)
            if len(weight_list[op.name].shape) == 4:
                weight_list[op.name] = np.transpose(weight_list[op.name], (3, 2, 0, 1))

        with open(weights_path, 'wb') as f:
            # weight_list = [(key, value) for (key, value) in x.items()]
            dumy = np.array([0] * 10, dtype=np.float32)
            dumy.tofile(f)
            weight_list["generator/encoder_1/conv/filter:0"].tofile(f)

            for i in range(2, 9):
                weight_list[f"generator/encoder_{i}/conv/filter:0"].tofile(f)
                weight_list[f"generator/encoder_{i}/batchnorm/scale:0"].tofile(f)
                weight_list[f"generator/encoder_{i}/batchnorm/offset:0"].tofile(f)

            for i in range(8, 0, -1):
                weight_list[f"generator/decoder_{i}/deconv/filter:0"].tofile(f)
                if i == 1:
                    break
                weight_list[f"generator/decoder_{i}/batchnorm/scale:0"].tofile(f)
                weight_list[f"generator/decoder_{i}/batchnorm/offset:0"].tofile(f)
        print("Done!")