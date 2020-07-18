from datetime import time

import numpy as np
import tensorflow as tf

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import generateDataset
 

#from modelsSN import make_discriminator_model,make_generator_model
from models import make_discriminator_model,make_generator_model

original_w = 16
batch_size = 64000 //(original_w * original_w)
print(f"batch_size {batch_size}")
EPOCHS = 900
num_examples_to_generate = 25
noise_dim = 200
decay_step = 10
lr_initial_g = 0.0002
lr_decay_steps = 1000
replay_step = 32

seed = tf.random.normal([num_examples_to_generate, noise_dim])

noise_var = tf.Variable(initial_value=0.005, trainable=False, name="noiseIn")
noise_var.assign(0.005)

generator = make_generator_model(original_w, noise_dim)
discriminator = make_discriminator_model(original_w, noise_var)

if True:
    noise = tf.random.normal([1, noise_dim])
    generated_image = generator(noise, training=False)
    decision = discriminator(generated_image)
    print(decision)
    # plt.imshow( generated_image[0] *0.5 + 0.5  )
    # plt.show()
generator.summary()
discriminator.summary()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def invert_if(v1):
    n = v1.shape[0]
    q = []
    for k in range(n):
        if np.random.rand() > 0.01:
            q.append(v1[k])
        else:
            q.append(1.0 - v1[k])
    return np.array(q)


 

 


generator_optimizer = tf.keras.optimizers.Adam(lr_initial_g, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2*lr_initial_g, beta_1=0.5)
#discriminator_optimizer = tf.keras.optimizers.SGD(lr_initial_g )


print("Start Loading Dataset")
training_dataset = generateDataset.getImageDataSet(original_w).batch(batch_size)
print("End Loading Dataset")


 
# Label smoothing -- technique from GAN hacks, instead of assigning 1/0 as class labels, we assign a random integer in range [0.7, 1.0] for positive class
# and [0.0, 0.3] for negative class

def smooth_positive_labels(y):
    return y - 0.3 + (np.random.random(y.shape) * 0.5)

def smooth_negative_labels(y):
    return y + np.random.random(y.shape) * 0.3



# randomly flip some labels
def noisy_labels(y, p_flip):
    # determine the number of labels to flip
    n_select = int(p_flip * int(y.shape[0]))
    # choose labels to flip
    flip_ix = np.random.choice([i for i in range(int(y.shape[0]))], size=n_select)
    
    op_list = []
    # invert the labels in place
    #y_np[flip_ix] = 1 - y_np[flip_ix]
    for i in range(int(y.shape[0])):
        if i in flip_ix:
            op_list.append(tf.subtract(1, y[i]))
        else:
            op_list.append(y[i])
    
    outputs = tf.stack(op_list)
    return outputs



def discriminator_loss(real_output, fake_output,  apply_label_smoothing=True, label_noise=True):
    if label_noise and apply_label_smoothing:
        real_output_noise = noisy_labels(tf.ones_like(real_output), 0.05)
        fake_output_noise = noisy_labels(tf.zeros_like(fake_output), 0.05)
        real_output_smooth = smooth_positive_labels(real_output_noise)
        fake_output_smooth = smooth_negative_labels(fake_output_noise)        
        real_loss = cross_entropy(tf.ones_like(real_output_smooth), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output_smooth), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    elif label_noise and not apply_label_smoothing:
        real_output_noise = noisy_labels(tf.ones_like(real_output), 0.05)
        fake_output_noise = noisy_labels(tf.zeros_like(fake_output), 0.05)        
        real_loss = cross_entropy(tf.ones_like(real_output_noise), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output_noise), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    elif apply_label_smoothing and not label_noise:
        real_output_smooth = smooth_positive_labels(tf.ones_like(real_output))
        fake_output_smooth = smooth_negative_labels(tf.zeros_like(fake_output))        
        real_loss = cross_entropy(tf.ones_like(real_output_smooth), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output_smooth), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss    
    else:        
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss




def generator_loss(real_output, fake_output,  apply_label_smoothing=True):
    if apply_label_smoothing:
        fake_output_smooth = smooth_negative_labels(tf.ones_like(fake_output))
        return cross_entropy(tf.ones_like(fake_output_smooth), fake_output) 
    else:                   
        return cross_entropy(tf.ones_like(fake_output), fake_output)


#@tf.function
def train_step(images ):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(real_output, fake_output,  apply_label_smoothing=True)
        disc_loss = discriminator_loss(real_output, fake_output,    apply_label_smoothing=True, label_noise=True)
 
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss





 





def get_mix_images(_real_images):
    noise = tf.random.normal([batch_size, noise_dim])
    generated_images = generator(noise, training=True)


def getError(model, test_input, _real_images ):
    predictions = model(test_input, training=False)
    fake_values = np.mean(discriminator(predictions, training=False))
    real_values = np.mean(discriminator(_real_images, training=False))
    #print(fake_values,real_values)
    return fake_values,real_values


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    fake_values = np.mean(discriminator(predictions, training=False))
    print("FV ",fake_values)

    fig = plt.figure(figsize=(5, 5))
    for i in range(predictions.shape[0]):
        plt.subplot(5, 5, i + 1)
        plt.imshow(predictions[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.close('all')

    # plt.show()


#generator.load_weights("generator31_f/generator")
#discriminator.load_weights("discriminator31_f/discriminator")


def train_sets(dataset):
    dataset.shuffle( 1000) #muda a ordem dos batchs
    s = 0
    p_images = 0
    Gs= False
    Ds = False   # considera que os dois estao ruims
    for image_batch in dataset:
        p_images += batch_size
        if (p_images > 20000):
            p_images = 0
            print("Progress " + str(s * batch_size) )
            generate_and_save_images(generator, 999, seed)
        s = s +1
        #treina por 4 batchs e verifica se esta OK
        if (s % 4) == 3 :
            #verifica se esta ok
            e, d = getError(generator, noise, image_batch)
            Gs = True
            Ds = True
            if e < -3.0 :
                Gs = False
                print("ED ",e, d)
            if   ( e > d) :
                print("ED ",e, d)
                Ds = False   
        train_step(image_batch)
 
               

def train(dataset, epochs):
    generate_and_save_images(generator, 0, seed)
    for epoch in range(epochs):
        print("Start Epoch", epoch)
        bb = 0
        train_sets(dataset)
        generator.save_weights("generator{}/generator".format(epoch))
        discriminator.save_weights("discriminator{}/discriminator".format(epoch))
        generate_and_save_images(generator, epoch, seed)
        print('End Epoch', epoch)
        new_lr_g = 0.9*generator_optimizer.learning_rate
        new_lr_d= 0.9*discriminator_optimizer.learning_rate
        generator_optimizer.learning_rate = new_lr_g
        discriminator_optimizer.learning_rate = new_lr_d



 

train(training_dataset, EPOCHS)
