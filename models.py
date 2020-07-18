 
import tensorflow as tf 
 
def make_generator_model(original_w, noise_dim):
    g = 16 * 32//original_w
    s16 = original_w // 16
    k = (3, 3)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense((g * 8 * s16 * s16), use_bias=False, input_shape=(noise_dim,)))

    # model.add(tf.keras.layers.Dense((g * 4 * s16 * s16),use_bias=False, activation=tf.nn.leaky_relu))
    # model.add(tf.keras.layers.Dense((g * 4 * s16 * s16),use_bias=False, activation=tf.nn.leaky_relu))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Reshape([s16, s16, 8 * g], input_shape=(noise_dim,)))
    print(">>", model.output_shape)

    model.add(     tf.keras.layers.Conv2DTranspose(filters=8 * g, kernel_size=k, strides=(2, 2), padding="same", use_bias=False,                                       kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    print(model.output_shape)

    model.add( tf.keras.layers.Conv2DTranspose(filters=4 * g, kernel_size=k, strides=(2, 2), padding="same", use_bias=False,                                       kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(0.3))
    print(model.output_shape)

    if model.output_shape[1] < original_w  :
        model.add( tf.keras.layers.Conv2DTranspose(filters=2 * g, kernel_size=k, strides=(2, 2), padding="same", use_bias=False,                                        kernel_initializer='he_uniform'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(0.2))
        print(model.output_shape)

    if model.output_shape[1] < original_w  :
        model.add(tf.keras.layers.Conv2DTranspose(filters=g, kernel_size=k, strides=(2, 2), padding="same", use_bias=False,                                              kernel_initializer='he_uniform'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(0.2))
        print(model.output_shape)

    if model.output_shape[1] < original_w  :
        model.add(tf.keras.layers.Conv2DTranspose(filters=g, kernel_size=k, strides=(2, 2), padding="same", use_bias=False,                                              kernel_initializer='he_uniform'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(0.2))
        print(model.output_shape)



    model.add(tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(1, 1), strides=(1, 1), padding="same",                                              kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Activation(activation=tf.nn.tanh))

    print(model.output_shape)
    assert model.output_shape == ( None, original_w, original_w, 3)
    return model


def make_discriminator_model(original_w, noise_var ):
    g = 512
    model = tf.keras.Sequential()
    k = (3, 3)

    model.add(tf.keras.layers.Reshape([original_w, original_w, 3], input_shape=(  original_w, original_w, 3)))
 
    print("D ",model.output_shape)

    model.add(tf.keras.layers.Conv2D(g // 8, k, strides=(2, 2), padding='same', use_bias=False,
                                     input_shape=[  original_w, original_w, 3]))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    #model.add(tf.keras.layers.Dropout(0.2))
    print("D ",model.output_shape)

    # model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.Conv2D(g // 4, k, strides=(2, 2), use_bias=False, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    #model.add(tf.keras.layers.Dropout(0.2))
    print("D ",model.output_shape)

    if model.output_shape[1] > 4 :
        # model.add(tf.keras.layers.MaxPool2D())
        model.add(tf.keras.layers.Conv2D(g // 4, k, strides=(2, 2), use_bias=False, padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.01))
        print("D ",model.output_shape)

    if model.output_shape[1] > 4 :
        # model.add(tf.keras.layers.MaxPool2D())
        model.add(tf.keras.layers.Conv2D(g//4, k, strides=(2, 2), use_bias=False, padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.01))
        print("D ",model.output_shape)

    
    model.add(tf.keras.layers.Conv2D(g//2, k, strides=(1, 1), use_bias=False, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.01))
    print("D ",model.output_shape)

    model.add(tf.keras.layers.Flatten())
    #model.add(tf.keras.layers.Dense(g // 4, activation=tf.nn.leaky_relu))
    #model.add(tf.keras.layers.LeakyReLU())
    #model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(g, activation=tf.nn.leaky_relu))
    model.add(tf.keras.layers.Dense(1))
    # model.add(  tf.keras.layers.Activation(activation=tf.nn.sigmoid))
    return model
