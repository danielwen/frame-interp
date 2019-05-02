import keras
from keras import backend as K
from keras import layers
import criterion


def identity1(context_input):
    return context_input[:, :, :, :3]

def identity2(context_input):
    return context_input[:, :, :, 3:]

def interpolate(context_input):
    prevs, nexts = context_input[:, :, :, :3], context_input[:, :, :, 3:]
    return (prevs + nexts) / 2


class Identity1(object):
    def __init__(self, lr=0.001):
        latent_size = 512
        context_shape = (256, 256, 6)

        context_input = layers.Input(shape=context_shape, name="input_ctx")
        z_test = layers.Input(shape=(latent_size,), name="z_test")

        pred_test = layers.Lambda(identity1)(context_input)

        model_test = keras.models.Model(inputs=[z_test, context_input],
            outputs=[pred_test])

        metrics = [criterion.scaled_mse, criterion.psnr, criterion.ssim]

        optimizer = keras.optimizers.Adam(lr)
        model_test.compile(optimizer=optimizer, loss=criterion.mse, metrics=metrics)

        self.model_test = model_test


class Identity2(object):
    def __init__(self, lr=0.001):
        latent_size = 512
        context_shape = (256, 256, 6)

        context_input = layers.Input(shape=context_shape, name="input_ctx")
        z_test = layers.Input(shape=(latent_size,), name="z_test")

        pred_test = pred_test = layers.Lambda(identity2)(context_input)

        model_test = keras.models.Model(inputs=[z_test, context_input],
            outputs=[pred_test])

        metrics = [criterion.scaled_mse, criterion.psnr, criterion.ssim]

        optimizer = keras.optimizers.Adam(lr)
        model_test.compile(optimizer=optimizer, loss=criterion.mse, metrics=metrics)

        self.model_test = model_test


class Naive(object):
    def __init__(self, lr=0.001):
        latent_size = 512
        context_shape = (256, 256, 6)

        context_input = layers.Input(shape=context_shape, name="input_ctx")
        z_test = layers.Input(shape=(latent_size,), name="z_test")

        pred_test = layers.Lambda(interpolate)(context_input)

        model_test = keras.models.Model(inputs=[z_test, context_input],
            outputs=pred_test)

        metrics = [criterion.scaled_mse, criterion.psnr, criterion.ssim]

        optimizer = keras.optimizers.Adam(lr)
        model_test.compile(optimizer=optimizer, loss=criterion.mse, metrics=metrics)

        self.model_test = model_test
