#%% Imports

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Create the model
import tensorflow as tf

from display import disp

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
EXAMPLES_count = data['train'].images.shape[0] # 55000 images (train), 5000 images (valid), 10000 images (test)


BATCH_SIZE = EXAMPLES_count

LAMBDA = 0
HIDDEN_UNITS = 1024
EPOCHS = 0
ALPHA = 0.01
DISPLAY_EVERY = 1



#%%
X = tf.placeholder(tf.float32, [None, FEATURES_count])
Y = tf.placeholder(tf.float32, [None, CLASSES_count])

# EPS = 1
# randInit = lambda shape: tf.random_uniform(shape, -EPS,EPS)
# randInit = lambda shape: tf.random_normal(shape, stddev=EPS)
randInit = tf.contrib.layers.xavier_initializer(uniform=False, seed=None)

softmax = lambda z: tf.nn.softmax(z)
relu = lambda z: tf.nn.relu(z)

Theta1 = tf.get_variable("Theta1", [FEATURES_count, HIDDEN_UNITS], initializer=randInit)
biases1 = tf.get_variable("biases1", [HIDDEN_UNITS], initializer=randInit)

Theta2 = tf.get_variable("Theta2", [HIDDEN_UNITS, CLASSES_count], initializer=randInit)
biases2 = tf.get_variable("biases2", [CLASSES_count], initializer=randInit)

Theta = tf.transpose(tf.matmul(Theta1, Theta2))

Z1 = tf.add(tf.matmul(X, Theta1), biases1)
A = softmax(Z1) # Todo: try relu here

Z2 = tf.add(tf.matmul(A, Theta2), biases2)
H = softmax(Z2)

# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(H), reduction_indices=[1]))
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Z2))

accuracy = 100*tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(Z2, 1)), tf.float32))

loss = cost
regularization = tf.constant(0)


if (LAMBDA > 0 and BATCH_SIZE >= 1):
    l2 =  lambda t: float(LAMBDA/BATCH_SIZE)*tf.nn.l2_loss(t)

    regularization = tf.add(l2(Theta1), l2(Theta2)) 
    loss =  tf.add(cost, regularization) 

# train_step = tf.train.AdamOptimizer(ALPHA).minimize(loss)
train_step = tf.train.GradientDescentOptimizer(ALPHA).minimize(loss)


init = tf.global_variables_initializer()

saver = tf.train.Saver({
    "Theta1": Theta1,
    "biases1": biases1,
    "Theta2": Theta2,
    "biases2": biases2,
})


with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    try:
        saver.restore(sess, "./saved_params/model.ckpt")
    except: 
        pass    

    def report(dataset, description='DATASET', prepend='', append=''):
        print(
            '   {:.<14}'.format( description ) +  
            '{}'.format( prepend ) + 
            '\tCost: {:018.16f} \tAccuracy: {:07.4f}%  '.format( *sess.run([cost,accuracy], feed_dict={X: dataset.images, Y: dataset.labels}) ) + 
            '{}'.format( append )
        ) 
    
    print()
    report(data['train'], 'TRAINED ')
    if(EPOCHS>0): print()
    
    # Train
    for epoch in range(EPOCHS):
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        batch={X: batch_xs, Y: batch_ys}

        _, reg = sess.run([train_step, regularization], feed_dict=batch)

        if (epoch % DISPLAY_EVERY) == 0:
            report(data['train'], 'TRAINING ', '\tEpoch: {:2d}'.format(epoch+1), ' \tRegularization: {:018.16f}'.format( reg ))

    if(EPOCHS>0): print("\n**Done Training**\n")

    # Evaluate trained model
    report(data['validation'], 'VALIDATION ', append='\tLambda: {:04.2f}'.format(LAMBDA))


    # Test trained model
    report(data['test'], 'TEST ')

    if(EPOCHS>0): 
        save_path = saver.save(sess, "./saved_params/model.ckpt")
        print("\n***Model saved in file: {}".format(save_path))

    disp(sess.run(Theta))

    # disp(sess.run(A, feed_dict={X: mnist.test.images[0:16,:], Y: mnist.test.labels[0:16,:]}) )