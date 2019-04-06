import keras
from keras import backend as K
from keras import layers, Model
from keras.applications import vgg16, mobilenet_v2
# from discriminator import Discriminator

class ConvBnRelu(object):
    def __init__(self, filters):
        self.conv = layers.Conv2D(filters, (3, 3), padding="same")
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def __call__(self, inputs):
        return self.relu(self.bn(self.conv(inputs)))

class EncoderBlock(object):
    def __init__(self, filters):
        self.conv_bn_relu = ConvBnRelu(filters)
        self.pool = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

    def __call__(self, inputs):
        h = self.conv_bn_relu(inputs)
        return h, self.pool(h)

class DecoderBlock(object):
    def __init__(self, filters):
        self.concat = layers.Concatenate()
        self.deconv = layers.Conv2DTranspose(filters // 2, (3, 3), strides=(2, 2), padding="same")
        self.conv_bn_relu = ConvBnRelu(filters)

    def __call__(self, input_, feature_map):
        inputs = self.concat([input_, feature_map])
        h = self.conv_bn_relu(inputs)
        output = self.deconv(h)

        return output, h

class Encoder(object):
    def __init__(self, layers_filters):
        self.encoder_blocks = []

        for filters in layers_filters:
            self.encoder_blocks.append(EncoderBlock(filters))

    def __call__(self, inputs):
        pooled = inputs
        feature_maps = []

        for encoder_block in self.encoder_blocks:
            feature_map, pooled = encoder_block(pooled)
            feature_maps.append(feature_map)

        return pooled, feature_maps

class Decoder(object):
    def __init__(self, layers_filters):
        self.decoder_blocks = []

        for filters in layers_filters:
            self.decoder_blocks.append(DecoderBlock(filters))

    def __call__(self, inputs, feature_maps):
        for feature_map, decoder_block in zip(feature_maps, self.decoder_blocks):
            inputs, h = decoder_block(inputs, feature_map)

        return h

class Q(object):
    def __init__(self, encoder_layers_filters, latent_size):
        self.concat = layers.Concatenate()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(latent_size, name="mean")
        self.dense2 = layers.Dense(latent_size, name="log_var")
        self.encoder = Encoder(encoder_layers_filters)

    def __call__(self, input_, context_input):
        inputs = self.concat([input_, context_input])
        h, _ = self.encoder(inputs)
        flat = self.flatten(h)
        mean = self.dense1(flat)
        log_var = self.dense2(flat)

        return mean, log_var

class P(object):
    def __init__(self, encoder_layers_filters, decoder_layers_filters, latent_size):
        self.concat = layers.Concatenate()
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(latent_size)
        self.reshape = layers.Reshape((1, 1, latent_size))
        self.deconv = layers.Conv2DTranspose(1024, (3, 3), strides=(2, 2), padding="same")
        self.encoder = Encoder(encoder_layers_filters)
        self.decoder = Decoder(decoder_layers_filters)

    def __call__(self, z, context_input):
        pooled, feature_maps = self.encoder(context_input)
        flat = self.flatten(pooled)
        merged = self.concat([flat, z])
        inputs = self.reshape(self.dense(merged))
        output = self.decoder(self.deconv(inputs), reversed(feature_maps))

        return output

def sample_z(args):
    (noise, mean, log_var) = args
    return K.exp(log_var / 2) * noise + mean

def kl(truth, pred):
    mean = pred[:, 0, :]
    log_var = pred[:, 1, :]
    kl = 0.5 * K.sum(K.exp(log_var) + K.square(mean) - 1. - log_var, axis=1)
    return kl

def get_vgg_loss():
    vgg_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(256,256,3))
    loss_model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block3_conv3').output) 
    # mnet = mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False, input_shape=(256,256,3))
    # loss_model = Model(inputs=mnet.input, outputs=mnet.get_layer('block3_conv3').output) 
    loss_model.trainable = False
    def vgg_loss(truth, pred):    
        pred_feature = loss_model(pred)
        truth_feature = loss_model(truth)
        return K.mean(K.square(pred_feature - truth_feature)) 
    return vgg_loss
    
def get_discriminator():
    vgg = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(256,256,3))
    flat = layers.Flatten()(vgg.output)
    # mnet = mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False, input_shape=(256,256,3))
    # flat = layers.Flatten()(mnet.output)
    fc1_out = layers.Dense(4096, activation='relu')(flat)
    fc2_out = layers.Dense(4096, activation='relu')(fc1_out)
    valid_score = layers.Dense(1, activation='sigmoid')(fc2_out)
    return Model(inputs=vgg.input, outputs=valid_score)

class VAE(object):
    def __init__(self, lr=0.001):
        latent_size = 512
        input_shape = (256, 256, 3)
        context_shape = (256, 256, 6)

        encoder_layers_filters = [64, 128, 256, 512, 512, 512, 512, 512]
        decoder_layers_filters = encoder_layers_filters[::-1]
        decoder_layers_filters[-1] = 3

        q = Q(encoder_layers_filters, latent_size)
        p = P(encoder_layers_filters, decoder_layers_filters, latent_size)

        input_ = layers.Input(shape=input_shape, name="input_frame")
        context_input = layers.Input(shape=context_shape, name="input_ctx")
        noise = layers.Input(shape=(latent_size,), name="input_noise")
        z_test = layers.Input(shape=(latent_size,), name="z_test")

        mean, log_var = q(input_, context_input)
        z = layers.Lambda(sample_z, name="z")([noise, mean, log_var])

        pred_train = p(z, context_input)
        z_param = layers.Concatenate(axis=-2, name="z_params")([
            layers.Reshape((1, latent_size))(mean),
            layers.Reshape((1, latent_size))(log_var)
        ])

        pred_test = p(z_test, context_input)

        # keras.utils.plot_model(model_train, to_file="model.png")
        # model_train.summary()

        gen_model_test = keras.models.Model(inputs=[z_test, context_input],
            outputs=[pred_test])

        optimizer = keras.optimizers.Adam(lr)
        gen_model_test.compile(optimizer=optimizer, loss=keras.losses.mean_absolute_error)

        disc_model = get_discriminator()
        disc_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        #Freeze discriminator in combined model. combined model is used to train the generator
        disc_model.trainable = False
        valid_score = disc_model(pred_train)
        combined_model = Model(inputs=[input_, noise, context_input], \
            outputs=[pred_train, z_param, pred_train, valid_score])
        losses = [keras.losses.mean_absolute_error, kl, get_vgg_loss(), keras.losses.binary_crossentropy]
        combined_model.compile(optimizer=optimizer, loss=losses, loss_weights=[0.3, 0.3, 0.3, 0.1])

        #only l1 loss
        model_l1 = Model(inputs=[input_, noise, context_input], \
            outputs=[pred_train, z_param])
        losses = [keras.losses.mean_absolute_error, kl]
        model_l1.compile(optimizer=optimizer, loss=losses)

        #vgg+l1 loss
        model_l1_vgg = Model(inputs=[input_, noise, context_input], \
            outputs=[pred_train, z_param, pred_train])
        losses = [keras.losses.mean_absolute_error, kl, get_vgg_loss()]
        model_l1_vgg.compile(optimizer=optimizer, loss=losses)


        self.disc_model = disc_model
        self.combined_model = combined_model
        self.gen_model_test = gen_model_test
        self.model_l1 = model_l1
        self.model_l1_vgg = model_l1_vgg

    def save(self, filename):
        if('disc' in filename):
            self.combined_model.save_weights(filename)
        elif('vgg' in filename):
            self.model_l1_vgg.save_weights(filename)
        else:
            self.model_l1.save_weights(filename)

    def load(self, filename):
        self.gen_model_test.load_weights(filename, by_name=True)
        if('disc' in filename):
            self.combined_model.load_weights(filename)
            self.disc_model.load_weights(filename, by_name=True)
        elif('vgg' in filename):
            self.model_l1_vgg.load_weights(filename)
        else:
            self.model_l1.save_weights(filename)

if __name__ == "__main__":
    vae = VAE()
    for l in vae.combined_model.layers:
        ws = l.get_weights()
        for w in ws:
            print(w.shape)