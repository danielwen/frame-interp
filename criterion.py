import numpy as np
from keras import backend as K
import tensorflow as tf

MAX_VAL = 255

def kl(truth, pred):
    mean = pred[:, 0, :]
    log_var = pred[:, 1, :]
    kl = 0.5 * K.sum(K.exp(log_var) + K.square(mean) - 1. - log_var, axis=1)
    return kl

def l1(truth, pred):
    errors = K.abs(pred - truth)
    return K.mean(K.reshape(errors, [-1, np.prod(pred.shape[1:])]), axis=-1)

def mse(truth, pred):
    errors = K.square(pred - truth)
    return K.mean(K.reshape(errors, [-1, np.prod(pred.shape[1:])]), axis=-1)

def scaled_mse(truth, pred):
    return mse(MAX_VAL*truth, MAX_VAL*pred)

def psnr(truth, pred):
    return tf.image.psnr(MAX_VAL*truth, K.clip(MAX_VAL*pred, 0, MAX_VAL), MAX_VAL)

def ssim(truth, pred):
    return tf.image.ssim(MAX_VAL*truth, K.clip(MAX_VAL*pred, 0, MAX_VAL), MAX_VAL)

def neg_ssim(truth, pred):
    return -ssim(truth, pred)
