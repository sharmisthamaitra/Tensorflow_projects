#from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys


from six.moves import urllib
import pickle


##Create a folder cifar-10 under working directory
CIFAR_10_FILE_NAME = 'cifar-10/cifar-10-python.tar.gz'



#CIFAR-10 contains 60,000 color images of animals and inanimate objects, total 10 categories. Each image is 32 x 32 pixels, 3 channels (RGB).
urllib.request.urlretrieve('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', CIFAR_10_FILE_NAME)


#Extract cifar-10-python.tar.gz into cifar-10 folder, under working directory. 
import tarfile
with tarfile.open('cifar-10/cifar-10-python.tar.gz') as tar:
   tar.extractall()
   tar.close()
    


#The tar.gz file will be extracted into 5 batches for training and 1 batch for tesing into cifar-10-batches-py folder under working directory
CIFAR10_DATASET_FOLDER = "cifar-10-batches-py"



#Helper function to load a particular training batch (eg data_batch_1, data_batch_2) into feature and labels.
def load_cifar10_batch(batch_id):
    with open(CIFAR10_DATASET_FOLDER + '/data_batch_' + str(batch_id), mode='rb') as file:
        
        batch = pickle.load(file, encoding='latin1')
        
        #Reshape and transpose fetched data into a format acceptable by the convolutional layers 
        features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        
        labels = batch['labels']
        
        return features, labels
 
features, labels = load_cifar10_batch(1)


#Check length of features
print("length of features:", len(features))

#Print the first 5 labels 
print("First 5 labels:", labels[0:5])



##Helper function to translate 0-9 numeric labels into word names
lookup = ['airplane',
          'automobile',
          'bird',
          'cat',
          'deer',
          'dog',
          'frog',
          'horse',
          'ship',
          'truck']
          
def display_feature_label(features, labels, index):
    if index >= len(features):
        raise Error("index out of range")
        
    plt.imshow(features[index])
    
 
display_feature_label(features, labels, 7) #output: cat



#Every image in cifar dataset is 32 x 32 pixels with 3 channels
img_height = 32
img_width = 32
img_channels = 3
img_size_pixels = img_height * img_width



#Building Tensorflow graph. 
tf.reset_default_graph()

#X is the placeholder for input images. 
x = tf.placeholder(tf.float32, shape=[None, img_height, img_width, img_channels], name="x")

#Placeholder for the training flag, dault value is 'False'
training = tf.placeholder_with_default(False, shape=(), name='training')

#Placeholder for labels. 
y = tf.placeholder(tf.int32, shape=[None], name="y")



#Use dropout just after input layer to turn off a percentage(30%) of neurons to prevent overfitting while training
#Dropout will happen only in the training phase . 
dropout_rate = 0.3
x_drop = tf.layers.dropout(x, dropout_rate, training=training)





##THE CNN ARCHITECTURE
# input_images -> input_images_after_30%_neuron_dropout -> conv1 -> conv2 -> pool3 -> conv4 -> pool5 -> pool5_flat -> fully_conn1 -> fully_conn2 -> logit layer for prediction

print("input data x:", x.shape)
print("input data after 30% dropout:", x_drop.shape)


#Input data: x(?,32,32,3)
#FIRST CONVOLUTIONAL LAYER, Shape is (None, 32, 32, 32)
conv1 = tf.layers.conv2d(x_drop, filters=32,
                        kernel_size=(3,3),
                         strides=(1,1), padding="SAME",
                         activation=tf.nn.relu, name="conv1")

print("conv1 shape:", conv1.shape)
          
                                                                                  

#SECOND CONVOLUTIONAL LAYER, shape is (None, 16, 16, 64)
#Kernel(filter) size is 3x3. Strides 2,2 in horizontal and vertical direction.
conv2 = tf.layers.conv2d(conv1, filters=64,
                         kernel_size=(3,3),
                         strides=(2,2), padding="SAME",
                         activation=tf.nn.relu, name="conv2")

print("conv2 shape:", conv2.shape)




#FIRST POOLING LAYER, shape is (None, 8, 8, 64) 
#Pooling makes the image smaller, but has no effect on depth
pool3 = tf.nn.max_pool(conv2,
                       ksize=[1,2,2,1],
                       strides=[1,2,2,1],
                       padding="VALID") #means no zero padding. pixels at the edges might be ignored.
                       
print("pool3 shape:", pool3.shape)




#THIRD CONVOLUTIONAL LAYER, shape is (None, 3 ,3, 128) 
conv4 = tf.layers.conv2d(pool3, filters=128,
                         kernel_size=(4,4),
                         strides=(3,3), padding="SAME",
                         activation=tf.nn.relu, name="conv4")
                         
print("conv4 shape:", conv4.shape)



# SECOND POOLING LAYER, Shape is (None, 2, 2, 128).
pool5 = tf.nn.max_pool(conv4,
                       ksize=[1,2,2,1],
                       strides=[1,1,1,1],
                       padding="VALID") #means no zero padding. pixels at edges can be ignored
                       
print("pool5 shape:", pool5.shape)




#FLATTEN OUTPUT OF LAST POOL LAYER, shape is a vector with 512 elements
pool5_flat = tf.reshape(pool5, shape=[-1,128 * 2 * 2])
print("pool5_flat:", pool5_flat)




#FULLY CONNECTED LAYER 1 , Shape is (none, 128)
fullyconn1 = tf.layers.dense(pool5_flat, 128,
                             activation=tf.nn.relu, name="fc1")
print("fullyconn1:", fullyconn1)   

   
                                                                                                 

#FULLY CONNECTED LAYER 2, Shape is (None, 64)
fullyconn2 = tf.layers.dense(fullyconn1, 64,
                             activation=tf.nn.relu, name="fc2")
print("fullyconn2:", fullyconn2)     

                                                  

#logits softmax prediction layer . 10 neurons for prediction. Shape is (None, 10). An image can belong to one of 10 classes 
logits = tf.layers.dense(fullyconn2, 10, name="output")
print("logits:", logits)



#CROSS ENTROPY layer comparing the predicted labels(logits) with actual labels(y) 
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, 
                                                          labels=y)



#CALCULATE LOSS 
#Target is to reduce xentropy, measured as loss. Loss needs to be low.
loss = tf.reduce_mean(xentropy)




# Use Adam Optimizer to reduce loss
optimizer = tf.train.AdamOptimizer()




#Run the Adam optimizer to minimize loss
training_op = optimizer.minimize(loss)




#Check the accuracy of predicted labels(logits) vs actual labels(y)
correct = tf.nn.in_top_k(logits, y, 1)



#Calculate accuracy as a percentage of correctly predicted labels (1) out of total predicted labels, incorrect(0) and correct(1) ones. 
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))



#Instantiate Tensorflow
init = tf.global_variables_initializer()


#Following 80-20 approach for splitting data into training and test. Batch_size is chosen as 128(lower batch size better trains the model).
#Total images 10,000 in dataset. Training size (train_size) is 8000 images. Test size is 2000 images. 
train_size = int(len(features) * 0.8)
n_epochs = 20
batch_size = 128


#Helper Function to get the next batch of images from a particular training dataset. Total 5 training datasets(data_batch_1 thru data_batch_5).
def get_next_batch(features, labels, train_size, batch_index, batch_size):
    training_images = features[:train_size, :, :]
    training_labels = labels[:train_size]
    
    test_images = features[train_size:, :, :]
    test_labels = labels[train_size:]
    
    start_index = batch_index * batch_size
    end_index = start_index + batch_size
    
    return features[start_index:end_index,:,:], labels[start_index:end_index], test_images, test_labels



with tf.Session() as sess:
    init.run()
    
    
    for epoch in range(n_epochs):
        
        #Looping thru 5 times to train the model on all 5 training datasets.
        for batch_id in range(1,6):
            batch_index = 0
            
            features, labels = load_cifar10_batch(batch_id)
            train_size = int(len(features) * 0.8)
        
            #train the model for train_size//batch_size number of iterations 
            #Load training images in x_batch, training labels in y_batch, Load test images in test_images, test labels in test_labels
            for iteration in range(train_size // batch_size):
                x_batch, y_batch, test_images, test_labels = get_next_batch(features,
                                                                    labels,
                                                                    train_size,
                                                                    batch_index,
                                                                    batch_size)
                                                                    
                batch_index += 1
             
                sess.run(training_op, feed_dict={x: x_batch, y: y_batch, training: True})
           
           
           
         
        #Evaluate accuracy of training, percentage of correct predictions vs total number of predictions on training data
        acc_train = accuracy.eval(feed_dict={x: x_batch, y: y_batch})
        
        
        #Evaluate accuracy of test, percentage of correct predictions vs total number of predictions on test data data
        acc_test = accuracy.eval(feed_dict={x: test_images, y: test_labels})
        
        
        print(epoch, "Train accuracy:", acc_train, "Test_accuracy:", acc_test)
        
        
      
        
    #Write the event file with the tensorflow graph to tf_project directory under working folder        
    writer = tf.summary.FileWriter('./tf_project', sess.graph)
 
    writer.close()   
    
           
        