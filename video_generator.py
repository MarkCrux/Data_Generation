import os
import cv2

path = 'out/'
fps = 18    #保存视频的FPS，可以适当调整

#可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter('saveVideo.avi',fourcc,fps,(64*3,64*3))#最后一个是保存图片的尺寸

for i in range(500):
    frame = cv2.imread(path+str((i+1)*100)+'.png')
    videoWriter.write(frame)
videoWriter.release()

