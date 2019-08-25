# -*- coding: utf-8 -*-
import tensorflow as tf
from pathlib import Path
import os

#from data_generation.grocery_images import GROCERY_IMGS
from grocery_images import GROCERY_IMGS
from gan_product import GAN_PRODUCT as GAN
#from gan_product_params import GAN_PRODUCT as GAN

home = str(Path.home())
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("iter",50000,"max iteration")
flags.DEFINE_float("lr",0.000002,"learning rate")

flags.DEFINE_integer("znum",500,"z units")
flags.DEFINE_integer("bsize",16,"mini-batch size")

flags.DEFINE_integer("width",64,"im_width")
flags.DEFINE_integer("height",64,"im_height")
flags.DEFINE_integer("s_dim",5,"class_num")
flags.DEFINE_integer("sample_per_label",5,"img_num")

#flags.DEFINE_integer("sample_per_label",100,"sample per label")

if __name__=="__main__":
    dataset = GROCERY_IMGS(os.path.join(home,"work/Grocery_GAN/grocery-image-by-gan-master/images/"))
    model = GAN(dataset)
    model.train()
    #model.reload()
