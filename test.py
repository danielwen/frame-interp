import numpy as np
from scipy.misc import imsave
from data_generator import DataGenerator
from model import VAE

c1 = (255*0.01)**2
c2 = (255*0.03)**2

def ssim(pred, groundtruth):
    m1 = np.mean(pred)
    m2 = np.mean(groundtruth)
    v1 = np.var(pred)
    v2 = np.var(groundtruth)
    # print(m1, m2, v1, v2)
    v12 = np.cov([np.reshape(pred,[-1]), np.reshape(groundtruth, [-1])])
    t1 = (2*m1*m2+c1)*(2*v12+c2)
    t2 = (m1**2+m2**2+c1)*(v1**2+v2**2+c2)
    # print(t1, t2)
    return t1/t2

def ssim_2(img1, img2, max_val=255, filter_size=11,
                       filter_sigma=1.5, k1=0.01, k2=0.03):

  if img1.shape != img2.shape:
    raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                       img1.shape, img2.shape)
  if img1.ndim != 4:
    raise RuntimeError('Input images must have four dimensions, not %d',
                       img1.ndim)

  img1 = img1.astype(np.float64)
  img2 = img2.astype(np.float64)
  _, height, width, _ = img1.shape

  # Filter size can't be larger than height or width of images.
  size = min(filter_size, height, width)

  # Scale down sigma if a smaller filter size is used.
  sigma = size * filter_sigma / filter_size if filter_size else 0

  if filter_size:
    window = np.reshape(_FSpecialGauss(size, sigma), (1, size, size, 1))
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
    sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
    sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
  else:
    # Empty blur kernel so no need to convolve.
    mu1, mu2 = img1, img2
    sigma11 = img1 * img1
    sigma22 = img2 * img2
    sigma12 = img1 * img2

  mu11 = mu1 * mu1
  mu22 = mu2 * mu2
  mu12 = mu1 * mu2
  sigma11 -= mu11
  sigma22 -= mu22
  sigma12 -= mu12

  # Calculate intermediate values used by both ssim and cs_map.
  c1 = (k1 * max_val) ** 2
  c2 = (k2 * max_val) ** 2
  v1 = 2.0 * sigma12 + c2
  v2 = sigma11 + sigma22 + c2
  ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
  cs = np.mean(v1 / v2)
  return ssim, cs

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
if(len(sys.argv)==1):
    model_name = sys.argv[1]
else:
    model_name = 'model.h5'
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
input_, _ = next(test_data)
pred = vae.model_test.predict_on_batch(input_)
result = np.clip(pred, 0, 1)

for i in range(result.shape[0]):
    imsave("%s-%d.png" % ('test', i), result[i])
