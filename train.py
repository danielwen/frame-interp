import keras
import tensorflow as tf
from keras import backend as K
from data_generator import DataGenerator
from model import VAE

batch_size = 32
image_size = 256
latent_size = 512
n_context = 2
epochs = 20

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))

train_data = DataGenerator("DAVIS_Train_Val", batch_size, image_size,
    latent_size, n_context)
val_data = DataGenerator("DAVIS_Dev", batch_size, image_size, latent_size,
    n_context, test=True)

vae = VAE()
best_loss = float("inf")

for epoch in range(1, epochs + 1):
    print("Epoch %d" % epoch)
    vae.model_train.fit_generator(train_data, steps_per_epoch=train_data.steps)
    loss = vae.model_test.evaluate_generator(val_data, steps=val_data.steps)
    print("Val Loss: %.4f" % loss)

    if loss < best_loss:
        vae.save("model.h5")
