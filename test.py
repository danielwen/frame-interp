import numpy as np
from scipy.misc import imsave
from data_generator import DataGenerator
from model import VAE
import sys
import os
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr


c1 = (255*0.01)**2
c2 = (255*0.03)**2

batch_size = 16
image_size = 256
latent_size = 512
n_context = 2

if(len(sys.argv)==3):
    model_name = sys.argv[1]
    test_folder = sys.argv[2]
else:
    model_name = 'model.h5'
    test_folder = 'DAVIS_Challenge'

model_prefix = model_name.split('.')[0]

train_data = DataGenerator("DAVIS_Train_Val", batch_size, image_size,
    latent_size, n_context, test=True)
val_data = DataGenerator("DAVIS_Dev", batch_size, image_size, latent_size,
    n_context, test=True)
test_data = DataGenerator(test_folder, batch_size, image_size, latent_size,
    n_context, test=True)

vae = VAE()
vae.load(model_name)

# print("Evaluating training...")
# train_loss = vae.model_test.evaluate_generator(train_data, steps=train_data.steps, verbose=1)
# print(train_loss)
# print("Evaluating validation...")
# val_loss = vae.model_test.evaluate_generator(val_data, steps=val_data.steps, verbose=1)
# print(val_loss)
# print("Evaluating test...")
# test_loss = vae.model_test.evaluate_generator(test_data, steps=val_data.steps, verbose=1)
# print(test_loss)

# for name, data in zip(("train", "val", "test"), (train_data, val_data, test_data)):
input_, originals = next(test_data)
pred = vae.gen_model_test.predict_on_batch(input_)
result = np.clip(pred, 0, 1)
if(not os.path.isdir('./test_imgs/%s' % (model_prefix))):
    os.mkdir('./test_imgs/%s' % (model_prefix))
for i in range(result.shape[0]):
    imsave("./test_imgs/%s/%s-%d.png" % (model_prefix, 'test', i), result[i])

steps = test_data.N//test_data.batch_size
avg_ssim = 0.0
avg_psnr = 0.0
for s in range(steps):
    
    pred_64 = pred.astype(np.float64)
    for i in range(batch_size):
        cur_ssim = ssim(originals[0][i], pred_64[i], multichannel=True)
        avg_ssim += cur_ssim
        cur_psnr = psnr(originals[0][i], pred_64[i])
        avg_psnr += cur_psnr
    input_, originals = next(test_data)
    pred = vae.gen_model_test.predict_on_batch(input_)
print('avg ssim = %f' % (avg_ssim/test_data.batch_size/steps))
print('avg psnr = %f' % (avg_psnr/test_data.batch_size/steps))