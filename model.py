import keras
from keras import backend as K
from keras import layers


def conv_bn_relu(inputs, filters):
    conv = layers.Conv2D(filters, (3, 3), padding="same")
    bn = layers.BatchNormalization()
    relu = layers.ReLU()

    return relu(bn(conv(inputs)))

def encoder_block(inputs, filters):
    h = conv_bn_relu(inputs, filters)
    pool = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    return h, pool(h)

def decoder_block(input_, feature_map, filters, output_channels=None):
    concat = layers.Concatenate()
    deconv = layers.Conv2DTranspose(filters // 2, (3, 3), strides=(2, 2), padding="same")

    inputs = concat([input_, feature_map])

    if output_channels is not None:
        filters = output_channels

    h = conv_bn_relu(inputs, filters)
    output = deconv(h)

    return output, h

def encoder(inputs, layers_filters):
    pooled = inputs
    feature_maps = []

    for filters in layers_filters:
        feature_map, pooled = encoder_block(pooled, filters)
        feature_maps.append(feature_map)

    return pooled, feature_maps

def decoder(inputs, feature_maps, layers_filters):
    for filters, feature_map in zip(layers_filters, feature_maps):
        inputs, h = decoder_block(inputs, feature_map, filters)

    return h

def q(input_, context_input, encoder_layers_filters, latent_size):
    concat = layers.Concatenate()
    flatten = layers.Flatten()
    dense1 = layers.Dense(latent_size)
    dense2 = layers.Dense(latent_size)

    inputs = concat([input_, context_input])
    h, _ = encoder(inputs, encoder_layers_filters)
    flat = flatten(h)
    mean = dense1(flat)
    log_var = dense2(flat)

    return mean, log_var

def p(z, context_input, encoder_layers_filters, decoder_layers_filters, latent_size):
    concat = layers.Concatenate()
    flatten = layers.Flatten()
    dense = layers.Dense(latent_size)
    reshape = layers.Reshape((1, 1, latent_size))
    deconv = layers.Conv2DTranspose(1024, (3, 3), strides=(2, 2), padding="same")

    pooled, feature_maps = encoder(context_input, encoder_layers_filters)
    flat = flatten(pooled)
    merged = concat([flat, z])
    inputs = reshape(dense(merged))
    output = decoder(deconv(inputs), reversed(feature_maps), decoder_layers_filters)

    return output

def sample_z(args):
    (noise, mean, log_var) = args
    return K.exp(log_var / 2) * noise + mean

def kl(truth, pred):
    mean = pred[:, 0, :]
    log_var = pred[:, 1, :]
    kl = 0.5 * K.sum(K.exp(log_var) + K.square(mean) - 1. - log_var, axis=1)
    return kl


class VAE(object):
    def __init__(self, lr=0.001):
        latent_size = 512
        input_shape = (256, 256, 3)
        context_shape = (256, 256, 6)

        encoder_layers_filters = [64, 128, 256, 512, 512, 512, 512, 512]
        decoder_layers_filters = encoder_layers_filters[::-1]
        decoder_layers_filters[-1] = 3

        input_ = layers.Input(shape=input_shape)
        context_input = layers.Input(shape=context_shape)
        noise = layers.Input(shape=(latent_size,))

        mean, log_var = q(input_, context_input, encoder_layers_filters, latent_size)
        z = layers.Lambda(sample_z)([noise, mean, log_var])
        pred = p(z, context_input, encoder_layers_filters, decoder_layers_filters, latent_size)
        z_param = layers.Concatenate(axis=-2)([
            layers.Reshape((1, latent_size))(mean),
            layers.Reshape((1, latent_size))(log_var)
        ])

        model = keras.models.Model(inputs=[input_, noise, context_input],
            outputs=[pred, z_param])
        # model.summary()

        losses = [keras.losses.mean_squared_error, kl]

        optimizer = keras.optimizers.Adam(lr)
        model.compile(optimizer=optimizer, loss=losses)


vae = VAE()
