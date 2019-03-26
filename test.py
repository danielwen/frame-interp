import numpy as np
from scipy.misc import imsave
from data_generator import DataGenerator
from model import VAE

batch_size = 16
image_size = 256
latent_size = 512
n_context = 2

train_data = DataGenerator("DAVIS_Train_Val", batch_size, image_size,
    latent_size, n_context, test=True)
val_data = DataGenerator("DAVIS_Dev", batch_size, image_size, latent_size,
    n_context, test=True)
test_data = DataGenerator("DAVIS_Challenge", batch_size, image_size, latent_size,
    n_context, test=True)

vae = VAE()
vae.load("model.h5")

# print("Evaluating training...")
# train_loss = vae.model_test.evaluate_generator(train_data, steps=train_data.steps, verbose=1)
# print(train_loss)
# print("Evaluating validation...")
# val_loss = vae.model_test.evaluate_generator(val_data, steps=val_data.steps, verbose=1)
# print(val_loss)
# print("Evaluating test...")
# test_loss = vae.model_test.evaluate_generator(test_data, steps=val_data.steps, verbose=1)
# print(test_loss)

for name, data in zip(("train", "val", "test"), (train_data, val_data, test_data)):
    input_, _ = next(train_data)
    pred = vae.model_test.predict_on_batch(input_)
    result = np.clip(pred, 0, 1)

    for i in range(result.shape[0]):
        imsave("%s-%d.png" % (name, i), result[i])
