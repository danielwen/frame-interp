import keras
from data_generator import DataGenerator
from model import VAE
import sys

batch_size = 16
image_size = 256
latent_size = 512
n_context = 2
epochs = 20

if(len(sys.argv)==3):
    train_folder = sys.argv[1]
    val_folder = sys.argv[2]
else:
    train_folder = 'DAVIS_Train_Val'
    val_folder = 'DAVIS_Dev'
train_data = DataGenerator(train_folder, batch_size, image_size,
    latent_size, n_context)
val_data = DataGenerator(val_folder, batch_size, image_size, latent_size,
    n_context, test=True)

vae = VAE()
best_loss = float("inf")
tensorboard = keras.callbacks.TensorBoard(write_graph=False, update_freq="batch")

for epoch in range(1, epochs + 1):
    print("Epoch %d" % epoch)
    vae.model_train.fit_generator(train_data, steps_per_epoch=train_data.steps,
        callbacks=[tensorboard])
    loss = vae.model_test.evaluate_generator(val_data, steps=val_data.steps)
    print("Val Loss: %.4f" % loss)

    if loss < best_loss:
        vae.save("model.h5")
        best_loss = loss
