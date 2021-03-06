import sys
import numpy as np
import imageio
from data_generator import DataGenerator
from model import VAE
from naive import Identity1, Identity2, Naive

prefix = sys.argv[1]

seed = 1
batch_size = 16
image_size = 256
latent_size = 512
n_context = 2

train_data = DataGenerator("DAVIS_Train_Val", batch_size, image_size,
    latent_size, n_context, test=True, seed=seed)
val_data = DataGenerator("DAVIS_Dev", batch_size, image_size, latent_size,
    n_context, test=True, seed=seed)
test_data = DataGenerator("DAVIS_Challenge", batch_size, image_size, latent_size,
    n_context, test=True, seed=seed)

# model = VAE()
# model.load("model.h5")
model = Naive()

for name, data in zip(("train", "val", "test"), (train_data, val_data, test_data)):
    input_, (truth,) = next(data)
    (_, contexts) = input_
    prevs, nexts = contexts[:, :, :, :3], contexts[:, :, :, 3:]
    pred = model.model_test.predict_on_batch(input_)

    for label, batch in zip(("1", "2", "truth", "pred"), (prevs, nexts, truth, pred)):
        result = np.round(np.clip(255*batch, 0, 255)).astype("uint8")

        for i in range(result.shape[0]):
            imageio.imwrite("%s_%s_%d_%s.png" % (prefix, name, i, label), result[i])

    # print("Evaluating %s" % name)
    # data.reset()
    # (_, mse, psnr, ssim) = model.model_test.evaluate_generator(data, steps=data.steps, verbose=1)
    # print("MSE: %.4f | PSNR: %.4f | SSIM: %.4f" % (mse, psnr, ssim))
