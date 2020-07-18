
import tensorflow as tf 
from customlayers import DenseSN,ConvSN2D,ConvSN2DTranspose

weight_init_std = 0.2
weight_init_mean = 0.0
leaky_relu_slope = 0.2
downsize_factor = 2
dropout_rate = 0.5
weight_initializer = tf.keras.initializers.TruncatedNormal(stddev=weight_init_std, mean=weight_init_mean, seed=42)




def transposed_conv(model, out_channels, ksize, stride_size, ptype='same'):
    model.add(tf.keras.layers.Conv2DTranspose(out_channels, (ksize, ksize),
                              strides=(stride_size, stride_size), padding=ptype, 
                              kernel_initializer=weight_initializer, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    return model

def transposed_convSN(model, out_channels, ksize, stride_size, ptype='same'):
    model.add(ConvSN2DTranspose(out_channels, (ksize, ksize), 
                              strides=(stride_size, stride_size), padding=ptype, 
                              kernel_initializer=weight_initializer, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    return model

def convSN(model, out_channels, ksize, stride_size):
    model.add(ConvSN2D(out_channels, (ksize, ksize), strides=(stride_size, stride_size), padding='same',
                     kernel_initializer=weight_initializer, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=leaky_relu_slope))
    #model.add(Dropout(dropout_rate))
    return model

def conv(model, out_channels, ksize, stride_size):
    model.add(tf.keras.layers.Conv2D(out_channels, (ksize, ksize), strides=(stride_size, stride_size), padding='same',
                     kernel_initializer=weight_initializer, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=leaky_relu_slope))
    #model.add(Dropout(dropout_rate))
    return model


def make_generator_model(original_w, noise_dim):
    channels_z = 16
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(original_w* original_w * 128//(8*8) ,   input_shape=(noise_dim,), kernel_initializer=weight_initializer))
 
    #model.add(LeakyReLU(alpha=leaky_relu_slope))
    model.add(tf.keras.layers.Reshape((original_w//8, original_w//8, 128)))   

    model = transposed_conv(model, channels_z*16, ksize=3, stride_size=1)
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model = transposed_conv(model, channels_z*8, ksize=3, stride_size=2)
    model.add(tf.keras.layers.Dropout(dropout_rate))
    print("GA >>", model.output_shape)
    if model.output_shape[1] <  original_w:
       model = transposed_conv(model, channels_z*4, ksize=3, stride_size=2)
    if model.output_shape[1] <  original_w:
       model = transposed_conv(model, channels_z*2, ksize=3, stride_size=2)
    if model.output_shape[1] <  original_w:
       model = transposed_conv(model, channels_z, ksize=3, stride_size=2)    
    
    print("GX >>", model.output_shape)

    model.add(tf.keras.layers.Dense(3, activation='tanh', kernel_initializer=weight_initializer))
    print("G3 >>", model.output_shape)
    return model

def make_discriminator_model(original_w, spectral_normalization=True):
    model = tf.keras.Sequential()
    if spectral_normalization:
        model.add(ConvSN2D(64, (3, 3), strides=(1,1), padding='same', use_bias=False,
                         input_shape=[original_w, original_w, 3], 
                         kernel_initializer=weight_initializer))
        #model.add(BatchNormalization(epsilon=BN_EPSILON, momentum=BN_MOMENTUM))
        model.add(tf.keras.layers.LeakyReLU(alpha=leaky_relu_slope))
        #model.add(Dropout(dropout_rate))
        
        print("DA >>", model.output_shape)

        model = convSN(model, 64, ksize=3, stride_size=2)
        #model = convSN(model, 128, ksize=3, stride_size=1)
        model = convSN(model, 128, ksize=3, stride_size=2)
        #model = convSN(model, 256, ksize=3, stride_size=1)
        model = convSN(model, 256, ksize=3, stride_size=2)
        #model = convSN(model, 512, ksize=3, stride_size=1)
        #model.add(Dropout(dropout_rate))
        print("DX >>", model.output_shape)

        model.add(tf.keras.layers.Flatten())

        print("D3 >>", model.output_shape)

        model.add(DenseSN(1, activation='sigmoid'))
        print("D3 >>", model.output_shape)
    else:
        model.add(tf.keras.layers.Conv2D(64, (4, 4), strides=(2,2), padding='same', use_bias=False,
                         input_shape=[original_w, original_w, 3], 
                         kernel_initializer=weight_initializer))
        #model.add(BatchNormalization(epsilon=BN_EPSILON, momentum=BN_MOMENTUM))
        model.add(tf.keras.layers.LeakyReLU(alpha=leaky_relu_slope))
        #model.add(Dropout(dropout_rate))
 
        print("DsA >>", model.output_shape)
        model = conv(model, 64, ksize=4, stride_size=2)
        #model = convSN(model, 128, ksize=3, stride_size=1)
        model = conv(model, 128, ksize=4, stride_size=2)
        #model = convSN(model, 256, ksize=3, stride_size=1)
        model = conv(model, 256, ksize=4, stride_size=2)
        #model = convSN(model, 512, ksize=3, stride_size=1)
        print("DsX >>", model.output_shape)
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model


