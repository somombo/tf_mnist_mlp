#%% Imports

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Create the model
import tensorflow as tf

from matplotlib import pyplot as plt

import numpy as np
import math

#%%

# data = {
#     'train': mnist.validation,
#     'validation': mnist.test,
#     'test': mnist.train
# }

data = {
    'train': mnist.train,
    'validation': mnist.validation,
    'test': mnist.test
}





FEATURES_count = data['train'].images.shape[1] # 784 pixels
CLASSES_count = data['train'].labels.shape[1] # 10 digits
EXAMPLES_count = data['train'].images.shape[0] # 5000 images (val), 55000 images (train), 10000 images (test)


BATCH_SIZE = EXAMPLES_count

LAMBDA = 1
HIDDEN_UNITS = 1024
EPOCHS = 10
ALPHA = 0.08
IMAGE_COLOR = 'bwr'
DISPLAY_EVERY = 1

def disp(images):  
    def _disp_one(image):
        n = math.sqrt(image.shape[0])
        # print(image.shape)
        pixels = np.reshape(image, (math.ceil(n),math.floor(n)))
        img = plt.imshow(pixels, cmap=IMAGE_COLOR, interpolation='nearest')

        plt.axis('off')  
        
    fig = plt.figure()
    
    m = math.sqrt(images.shape[0])
    # print(images.shape)
    for i, img in enumerate(images): 
        a=fig.add_subplot(math.ceil(m),math.floor(m),i+1)
        _disp_one(img)

    plt.show()

#%%

# disp(data['train'].images[0:4,:])    

#%%
X = tf.placeholder(tf.float32, [None, FEATURES_count])
Y = tf.placeholder(tf.float32, [None, CLASSES_count])

# EPS = 1
# randInit = lambda shape: tf.random_uniform(shape, -EPS,EPS)
# randInit = lambda shape: tf.random_normal(shape, stddev=EPS)
randInit = tf.contrib.layers.xavier_initializer(uniform=False, seed=None)

link = lambda z: tf.nn.softmax(z)

# Theta1 = tf.Variable(randInit([FEATURES_count, HIDDEN_UNITS]))
# biases1 = tf.Variable(randInit([HIDDEN_UNITS]))

# Theta2 = tf.Variable(randInit([HIDDEN_UNITS, CLASSES_count]))
# biases2 = tf.Variable(randInit([CLASSES_count]))

Theta1 = tf.get_variable("Theta1", [FEATURES_count, HIDDEN_UNITS], initializer=randInit)
biases1 = tf.get_variable("biases1", [HIDDEN_UNITS], initializer=randInit)
Theta2 = tf.get_variable("Theta2", [HIDDEN_UNITS, CLASSES_count], initializer=randInit)
biases2 = tf.get_variable("biases2", [CLASSES_count], initializer=randInit)



Theta = tf.transpose(tf.matmul(Theta1, Theta2))

A = link(tf.matmul(X, Theta1) + biases1)
H = link(tf.matmul(A, Theta2) + biases2)


cost =  tf.reduce_mean(-tf.reduce_sum(Y * tf.log(H), reduction_indices=[1]))
accuracy = 100*tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(H, 1)), tf.float32))

loss = cost
regularization = tf.constant(0)


if (LAMBDA != 0):
    l2 =  tf.contrib.layers.l2_regularizer(float(LAMBDA/BATCH_SIZE))

    regularization = tf.add(l2(Theta1), l2(Theta2)) 
    loss =  tf.add(cost, regularization) 

train_step = tf.train.AdamOptimizer(ALPHA).minimize(loss)


#%%
init = tf.global_variables_initializer()

saver = tf.train.Saver({
    "Theta1": Theta1,
    "biases1": biases1,
    "Theta2": Theta2,
    "biases2": biases2,
})

#%%
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    try:
        saver.restore(sess, "./saved_params/model.ckpt")
    except: 
        pass    

    def report(dataset, description='DATASET', prepend='', append=''):
        print(
            '  ' + 
            description + 
            ' ... ' + 
            prepend + 
            '\tCost: {:018.16f} \tAccuracy: {:06.3f}%  '.format( *sess.run([cost,accuracy], feed_dict={X: dataset.images, Y: dataset.labels}) ) + 
            append  
        ) 
    
    report(data['train'], 'PRE-TRAIN')
    print()
    # Train
    for epoch in range(EPOCHS):
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        batch={X: batch_xs, Y: batch_ys}
        sess.run(train_step, feed_dict=batch)
        sess.run(train_step, feed_dict=batch)
        if (epoch % DISPLAY_EVERY) == 0:
            report(data['train'], 'TRAINING ', 'Epoch: {:2d}'.format(epoch+1), ' \tRegularization: {:018.16f}'.format( sess.run(regularization, feed_dict=batch) ))

    print("\n**Done Training**\n")

    # Evaluate trained model
    # print('\nVALIDATION:\n\t:-{:3.16f} {:.3f}%'.format( *sess.run([cost,accuracy], feed_dict={X: mnist.validation.images, Y: mnist.validation.labels}) ) ) # 0.3287599980831146
    report(data['validation'], 'VALIDATION ..', 'Lambda: {:04.2f}'.format(LAMBDA))


    # Test trained model
    report(data['test'], 'TESTING .....')
    # print('Test Accuracy: {:.3f}%'.format( sess.run(accuracy, feed_dict={X: mnist.train.images, Y: mnist.train.labels}) ) ) # 91.050%

    save_path = saver.save(sess, "./saved_params/model.ckpt")
    print("\n***Model saved in file: %s" % save_path)

    disp(sess.run(Theta))

