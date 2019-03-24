from data_generator import DataGenerator
from model import VAE

batch_size = 2
image_size = 256
latent_size = 512
n_context = 2

train_data = DataGenerator("DAVIS_Train_Val", batch_size, image_size,
    latent_size, n_context)
val_data = DataGenerator("DAVIS_Dev", batch_size, image_size, latent_size,
    n_context)

vae = VAE()
vae.model_train.fit_generator(train_data,
    steps_per_epoch=train_data.steps_per_epoch, verbose=1)

output = vae.model_test.evaluate_generator(val_data,
    steps_per_epoch=val_data.steps_per_epoch)
