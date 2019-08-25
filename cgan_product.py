# -*- coding: utf-8 -*-
"""
CGAN Single Digit
@SNT
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
#from utils import *
from pathlib import Path
#from examples.single_digit.single_digit_data import load_data

HOME = str(Path.home())
flags = tf.app.flags
FLAGS = flags.FLAGS

class CGAN_PRODUCT(object):
    def __init__(self,dataset):
        self.dataset  = dataset
        self.z_dim    = FLAGS.znum
        self.mb_size  = FLAGS.bsize
        
        self.rows= FLAGS.width
        self.cols= FLAGS.height

        self.s_dim    = FLAGS.s_dim

        self.x        = tf.placeholder(tf.float32, shape=[None, self.rows,self.cols,3])
        self.s        = tf.placeholder(tf.float32, shape=[None, self.s_dim])
        self.z        = tf.placeholder(tf.float32, shape=[None, self.z_dim])

    def build_model(self):
        fake = self.generator()
        p_real = self.discriminator(self.x,reuse=False)
        p_fake = self.discriminator(fake,reuse=True)

        rlabs   = tf.ones([self.mb_size,1],dtype=tf.float32)
        flabs   = tf.zeros([self.mb_size,1],dtype=tf.float32)

        gen_cost = tf.reduce_mean(tf.keras.backend.binary_crossentropy(rlabs,p_fake))
        dis_cost = (tf.reduce_mean(tf.keras.backend.binary_crossentropy(rlabs,p_real)) + tf.reduce_mean(tf.keras.backend.binary_crossentropy(flabs,p_fake)))
        #gen_cost  = -tf.reduce_mean(tf.log(p_fake))
        #dis_cost  = -tf.reduce_mean(tf.log(p_real)) - tf.reduce_mean(tf.log(1-p_fake))
        #fake = 0.5*fake + 0.5
        return gen_cost,dis_cost,fake

    def generator(self):
        with tf.variable_scope("gen") as scope:
            # embedding --> multiply??
            i = tf.concat([self.z,self.s],axis=1)
            print('this is a checkpoint')
            print(i.get_shape)
            i = tf.layers.dense(i,1024*16*1*16)
            i = tf.reshape(i,[-1,16,16,1024])
            print(i.get_shape)
            i = tf.image.resize_nearest_neighbor(i, (2*16,2*16))
           
            i = tf.layers.conv2d(i,1024,5,(1,1),padding="SAME")
            i = tf.layers.batch_normalization(i,momentum=0.9)
            i = tf.nn.leaky_relu(i,alpha=0.2) 
            #i = tf.image.resize_nearest_neighbor(i, (2*14,2*14))
            
            i = tf.layers.conv2d(i,512,5,(1,1),padding="SAME")
            i = tf.layers.batch_normalization(i,momentum=0.9)
            i = tf.nn.leaky_relu(i,alpha=0.2)

            i = tf.layers.conv2d_transpose(i,512,4,strides=2,padding='same')
            i = tf.nn.leaky_relu(i,alpha=0.2)
            
            i = tf.layers.conv2d(i,256,3,(1,1),padding="SAME")
            i = tf.layers.batch_normalization(i,momentum=0.9)
            i = tf.nn.leaky_relu(i,alpha=0.2)


            i = tf.layers.conv2d(i,256,3,(1,1),padding="SAME")
            i = tf.layers.batch_normalization(i,momentum=0.9)
            i = tf.nn.leaky_relu(i,alpha=0.2)

            i = tf.layers.conv2d(i,128,3,(1,1),padding="SAME")
            i = tf.layers.batch_normalization(i,momentum=0.9)
            i = tf.nn.leaky_relu(i,alpha=0.2)

           
            i = tf.layers.conv2d(i,3,3,(1,1),padding="SAME")
            fake_image = tf.nn.sigmoid(i,name="fake_image")
           
            
        return fake_image


    def conv_cond_concat(self,x,y,bsize):
        x_shapes = x.get_shape()
        y_shapes = y.get_shape()
        
        #print(x_shapes,y_shapes)
        y_label = y*tf.ones([x_shapes[1], x_shapes[2], y_shapes[3]])
        y_label = tf.reshape(y_label, [-1, x_shapes[1], x_shapes[2], y_shapes[3]])
        return tf.concat([x, y_label], 3)
        #return tf.concat([x, y*tf.ones([bsize, x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


    def discriminator(self,input_,reuse=False):
        with tf.variable_scope("dis") as scope:
            if reuse == True:
                scope.reuse_variables()
            print(input_.get_shape)
            s = tf.reshape(self.s,[-1,1,1,self.s_dim])
            i = self.conv_cond_concat(input_,s,self.mb_size)
            print(i.get_shape)
            i = tf.layers.conv2d(i,64,4,2,padding="SAME")
            i = tf.nn.leaky_relu(i)
            
            i = tf.layers.conv2d(i,128,4,2,padding="SAME")
            i = tf.layers.batch_normalization(i,momentum=0.8)
            i = tf.nn.leaky_relu(i)
            
            i = tf.layers.flatten(i)
            
            i = tf.layers.dense(i,1024)
            i = tf.layers.batch_normalization(i,momentum=0.8)
            i = tf.nn.leaky_relu(i)
            
            i = tf.layers.dense(i,1,activation="sigmoid")
            
        return i
    
    

        
    
    
    def train(self):        
        # losses
        gen_cost,dis_cost,x_samples = self.build_model()
        gen_vars  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"gen")
        dis_vars  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"dis")
        
        dis_op = tf.train.AdamOptimizer(learning_rate=FLAGS.lr, beta1=0.5).minimize(dis_cost, var_list=dis_vars)
        gen_op = tf.train.AdamOptimizer(learning_rate=FLAGS.lr, beta1=0.5).minimize(gen_cost, var_list=gen_vars)
    

        session = tf.Session()
        session.run(tf.global_variables_initializer())
        if not os.path.exists('out/'):
            os.makedirs('out/')
        
        #s = np.zeros(shape=[FLAGS.bsize,self.s_dim])
        s = np.zeros(shape=[FLAGS.sample_per_label*self.s_dim, self.s_dim])

        for j in range(self.s_dim):
        	s[j*5:(j+1)*5,j] = 1              

        rs_dir = "./results/lr%f_z%d"%(FLAGS.lr,self.z_dim)
        if not os.path.exists(rs_dir):
            os.makedirs(rs_dir)

        rs = None
        for i in range(1,FLAGS.iter+1):
            #train_dir = '/Users/ywei9/work/retail_prod_reg-master/data_generation/images'
            #CAPACITY=1000
            #image_list,label_list = self.dataset.get_files()
            #x,s_mb = self.dataset.get_batch(image_list,label_list,self.mb_size,CAPACITY)
            inds = np.random.randint(0, self.dataset.train_size, self.mb_size)
            x,s_mb = self.dataset.get_batch(inds) # get random set of images and their labels
            
            #print(x.get_shape())

            z = np.random.randn(x.shape[0], self.z_dim)
            _, dloss = session.run([dis_op, dis_cost], feed_dict={self.x: x, self.s: s_mb, self.z:z})
            #'''s_mb is the class number? if s_mb is s, how to deal with array of batch'''
            z = np.random.randn(x.shape[0], self.z_dim)
            _, gloss = session.run([gen_op, gen_cost], feed_dict={self.s: s_mb, self.z:z})
                   
            print("[Epoch %d] Dis loss: %.7f  Gen Loss %.7f"%(i,dloss,gloss)) # get the code run to this point
            
            if i%100 ==0 or i>=FLAGS.iter-1:
                # Save samples
                samples = session.run(x_samples,
                                      feed_dict={self.z: np.random.randn(5*self.s_dim, self.z_dim), self.s: s})
                
                for j in range(25):
                    imgname=str(i)+'_'+str(j)
                    
                    pltimg = samples[j,:,:,:]
                    pltimg = cv2.cvtColor(pltimg,cv2.COLOR_RGB2BGR)
                    pltimg=pltimg*255
                    cv2.imwrite('out/{}.png'.format(imgname),pltimg)
                    #plt.imshow(x)
                    #plt.show()
                '''fig     = plt.plot(samples,19,10,digit_num=1,flatten=False)
                plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                plt.close(fig)'''
            
