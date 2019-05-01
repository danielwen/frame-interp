import keras
from data_generator import DataGenerator
from model import VAE

batch_size = 16
image_size = 256
latent_size = 512
n_context = 2
epochs = 40

train_data = DataGenerator("DAVIS_Train_Val", batch_size, image_size,
    latent_size, n_context)
val_data = DataGenerator("DAVIS_Dev", batch_size, image_size, latent_size,
    n_context, test=True)

vae = VAE()
best_loss = float("inf")
tensorboard = keras.callbacks.TensorBoard(write_graph=False)

for epoch in range(1, epochs + 1):
    print("Epoch %d" % epoch)
    vae.model_train.fit_generator(train_data, steps_per_epoch=train_data.steps,
        callbacks=[tensorboard])
    (_, mse, psnr, ssim) = vae.model_test.evaluate_generator(val_data, steps=val_data.steps)
    print("Validation | MSE: %.4f | PSNR: %.4f | SSIM: %.4f" % (mse, psnr, ssim))

    if mse < best_loss:
        vae.save("model.h5")
        best_loss = mse
