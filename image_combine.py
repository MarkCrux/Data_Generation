# -*- coding: utf-8 -*-
from os import listdir
from PIL import Image

#get all png images
#imgs = [Image.open(fn) for fn in listdir() if fn.endswith('.png')]
path = 'Walnut/Walnut/'
img_format = ['.png']
imgs = [name for name in listdir(path) for item in img_format]
img_row = 3
img_column = 3
# image size
#width, height = imgs[0].size
width = 64
height = 64
imgs.sort(key= lambda x:int(x[:-4]))
#create blank image
#result = Image.new(imgs[0].mode, (width*3, height*3))
#print(imgs)
 
loop = len(imgs)/9
loop = int(loop)

#print(loop)
#combine images
for i in range(1, loop+1):
    result = Image.new('RGB', (width*3, height*3))
    img_name = str(i*100)
    for j in range(img_column):
        for k in range(img_row):
            from_image = Image.open(path+imgs[img_column*j+k+(i-1)*9])
            #from_image = imgs[img_column*j+k+(i-1)*9]
            result.paste(from_image, (k*width,j*height))
    result.save('out/{}.png'.format(img_name))

