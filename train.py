import keras
from data_generator import DataGenerator
from model import VAE
import sys
import numpy as np
from tqdm import tqdm
import json
import csv

batch_size = 8
image_size = 256
latent_size = 512
n_context = 2
epochs = 7

if(len(sys.argv)==2):
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = json.load(f)
    train_folder = config['train_folder']
    val_folder = config['val_folder']
    model_name = config['model_name']
    load_model = config['load_model']
else:
    train_folder = 'DAVIS_Train_Val'
    val_folder = 'DAVIS_Dev'
    model_name = 'model.h5'
    load_model = False
train_data = DataGenerator(train_folder, batch_size, image_size,
    latent_size, n_context)
val_data = DataGenerator(val_folder, batch_size, image_size, latent_size,
    n_context, test=True)
# train_data_test = DataGenerator(train_folder, batch_size, image_size,
#     latent_size, n_context, test=True)
vae = VAE()
model_prefix = model_name.split('.')[0]
if(load_model):
    vae.load(model_name)
    with open('%s.csv' % model_prefix, 'r') as f:
        best_loss = float(list(csv.reader(f))[-1][0])
else:
    best_loss = float("inf")
tensorboard = keras.callbacks.TensorBoard(write_graph=False, update_freq="batch")

with open('%s.csv' % model_prefix, 'a') as f:
    writer = csv.writer(f)
    for epoch in range(1, epochs + 1):
        print("Epoch %d" % epoch)
        if('disc' in model_prefix):
            half_batch = batch_size // 2
            #train discriminator:
            for step in tqdm(range(train_data.steps)):
                (frames, noises, contexts), _ = train_data.__next__()
                fake_frames = vae.gen_model_test.predict([noises[0 : half_batch], contexts[half_batch : ]])
                # print(fake_frames.shape, contexts.shape, vae.disc_model.outputs[0].shape, len(vae.disc_model.outputs))
                # assert(False)
                vae.disc_model.train_on_batch(frames[0 : half_batch], np.ones((half_batch, 1)))
                vae.disc_model.train_on_batch(fake_frames, np.zeros((half_batch, 1)))
            
            train_data.reset()

            #train the generator
            validity = np.ones(batch_size)  #the generator wants discriminator to say fake images are true
            for step in tqdm(range(train_data.steps)):
                inputs, (frames, dummy, _) = train_data.__next__()
                
                vae.combined_model.train_on_batch(inputs, [frames, dummy, frames, validity])

            train_data.reset()
            
        elif('vgg' in model_prefix):
            vae.model_l1_vgg.fit_generator(train_data, steps_per_epoch=train_data.steps,
                callbacks=[tensorboard])
        else:
            vae.model_l1.fit_generator(train_data, steps_per_epoch=train_data.steps,
                callbacks=[tensorboard])
            
        loss = vae.gen_model_test.evaluate_generator(val_data, steps=val_data.steps)
        
        print("Val Loss: %.4f" % loss)
        writer.writerow([loss])
        if loss < best_loss:
            vae.save(model_name)
            best_loss = loss
