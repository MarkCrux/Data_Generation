# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
import cv2
label_mtx = np.eye(5)

class GROCERY_IMGS():
    def __init__(self,file_dir):
        self.s_size = 5 # number of labels
        self.file_dir=file_dir
        #define the path and lable variables list
        file_dir=self.file_dir
        zeroclass = []
        label_zeroclass = []
        oneclass = []
        label_oneclass = []
        twoclass = []
        label_twoclass = []
        threeclass = []
        label_threeclass = []
        fourclass = []
        label_fourclass = []
        fiveclass = []
        label_fiveclass = []
        #count=1
        # get the image name and path，store the lable
        for file in os.listdir(file_dir+'AppleBraeburn/'):#AppleBraeburn
            path=file_dir+'AppleBraeburn'
            path=os.path.join(path,file)
            zeroclass.append(path)
            #count=count+1
            #print(file)
            #zeroclass.append(file_dir+'AppleBraeburn/'+file)
            label_zeroclass.append(0)
        for file in os.listdir(file_dir+'Pineapple'):
            oneclass.append(file_dir+'Pineapple/'+file)
            label_oneclass.append(1)
        for file in os.listdir(file_dir+'Strawberry'):
            twoclass.append(file_dir+'Strawberry/'+file)
            label_twoclass.append(2)
        for file in os.listdir(file_dir+'Banana'):
            threeclass.append(file_dir+'Banana/'+file)
            label_threeclass.append(3)
        for file in os.listdir(file_dir+'Walnut'):
            fourclass.append(file_dir+'Walnut/'+file)
            label_fourclass.append(4)
        
        #print(count)
        #image_list = zeroclass
        #label_list = label_zeroclass
        image_list = np.hstack((zeroclass,oneclass,twoclass,threeclass,fourclass))
        label_list = np.hstack((label_zeroclass,label_oneclass,label_twoclass,label_threeclass,label_fourclass))
        #combine the data by horizontal
        temp = np.array([image_list,label_list])
        temp = temp.transpose()
        np.random.shuffle(temp)
        image_list = list(temp[:,0])
        label_list = list(temp[:,1])
        label_list = [int(i) for i in label_list]
        self.image_list = image_list
        #print(self.image_list)
        self.label_list = np.array(label_list)
        #print(self.label_list)
    
    @property
    def train_size(self):
        return len(self.image_list)
    
    """ def get_batch(self,batch_size=10):
        
        #image,label,batch_size,capacity):
        image = tf.cast(self.image_list,tf.string)
        label = tf.cast(self.label_list,tf.int32)
        #tf.cast() for transfering the format
        input_queue = tf.train.slice_input_producer([image,label])
        #add to the queue
        label = input_queue[1]
        image_contents = tf.read_file(input_queue[0])
        image = tf.image.decode_jpeg(image_contents,channels=3)
    
        #jpeg/jpg images use decode_jpeg
        image = tf.image.resize_image_with_crop_or_pad(image,28,28)
        #resize
        image = tf.image.per_image_standardization(image)
        #standardize the images
        image_batch,label_batch = tf.train.batch([image,label],batch_size = batch_size,num_threads=16,capacity = 1000)
        label_batch = tf.reshape(label_batch,[batch_size])
        return image_batch,label_batch
    """

    
    def get_batch(self,inds=None):
        if inds is None:
            return None,None
        #print(inds)
        lablist = self.label_list[inds]
        
        images = None
        for inx in inds:
            #print(self.image_list[inx])
            img = cv2.imread(self.image_list[inx])

            #print(img)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = img/255
            #noise=np.random.normal(0,0.01,img.shape)
            #img=img+noise
            img = cv2.resize(img,(64,64))[np.newaxis,:,:,:]
          
       
            if images is None:
                images = img
            else:
                images = np.append(images,img,axis=0)

        labels = label_mtx[lablist,:]
        return images,labels



if __name__=="__main__":
    home = str(Path.home())
    #print(home)
    dataset = GROCERY_IMGS(os.path.join(home,"work/Grocery_GAN/grocery-image-by-gan-master/images/"))
    """
    for ipath,l in zip(dataset.image_list,dataset.label_list):
        img = cv2.imread(ipath)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()
    """
    inds = np.random.randint(0, dataset.train_size, 10)
    x_mb,y_mb = dataset.get_batch(inds)
    
    #print(x_mb.shape)
    #print(x_mb)
    #print(type(x_mb))
    
    for i in range(10):
        x = x_mb[i,:,:,:]
        plt.imshow(x)
        y = y_mb[i,:]
        print(y)
        plt.show()


            
