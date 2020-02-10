#coding:utf-8
import re
import cv2
import os
import numpy as np
import cv2 as cv
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
#得到一共多少个样本
import shutil
import math
np.set_printoptions(threshold=np.inf)

def detect_line(canny):
	point = 50
	while (1):
		if (point < 50) or (point >300): 
			 return None
		lines = cv2.HoughLines(canny, 1, np.pi/180, point)
		if (lines is None): 
			point = point - 1
		elif (len(lines)>=1) and (len(lines)<=10): 
			print(lines.tolist())
			print(point)
			print(len(lines))
			return lines 
		else:
			point = point + 1

			
def clr_line(lines,img,canny,h,w,pre_path):
	if lines is not None:
		for line in lines:  
			if line is None:
				pass
			else:
				rho, theta = line[0]  
				a = np.cos(theta)  
				b = np.sin(theta)  
				x0 = a * rho  
				y0 = b * rho  
				x1 = int(x0+1000*(-b))  
				y1 = int(y0+1000*(a))  
				x2 = int(x0-1000*(-b))  
				y2 = int(y0-1000*(a)) 
				if b==0:
					if x0>w/2:
						for y in range(int(x0),w+1):
							for x in range(0,h+1):
								try:
									canny[x][y] = 0
								except:
									pass
					else:
						for y in range(0,int(x0+1)):
							for x in range(0,h+1):
								try:
									canny[x][y] = 0
								except:
									pass
				else:
					k = -a/b
					b1 = rho/b
					if abs(k*w/2+b1-h/2)/math.sqrt(k*k+1) < h/10:
						flag = 2
					elif k*w+b1 >= h - b1:
						flag = 0       #消除直线下面
					else:
						flag = 1
					
					if flag == 1:      #消除直线上面
						for y in range(0,w+1):
							for x in range(0,int(k*y+b1+1)):
								try: 
									canny[x][y] = 0
								except:
									pass
					elif flag == 0:
						for y in range(0,w+1):
							for x in range(int(k*y+b1),h+1):
								try:
									canny[x][y] = 0
								except:
									pass
				cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  
	return img,canny
  
def file2matric(filename):      #get the number of lines in the file
    imagename = []       #prepare matrix to return
    classLabelVector = []                       #prepare labels return   
    fr = open(filename,'r')
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split(' ')
        imagename.append(listFromLine[0])
        classLabelVector.append(listFromLine[-1])
    return imagename,classLabelVector

def file2matric1(filename):      #get the number of lines in the file
    imagename = []       #prepare matrix to return
    classLabelVector = []                       #prepare labels return   
    fr = open(filename,'r')
    for line in fr.readlines():
        line = line.strip()
        imagename.append(line)
    return imagename

def image_dir(imagename,classLabelVector):
    image_data = []
    for i in range(len(imagename)):
        pre_path = r'F:\eclipse-workspace\brand_classfication\datasets\train\{}'.format(imagename[i])
        # new_path = r'F:\eclipse-workspace\brand_classfication\train1\{}'.format(classLabelVector[i])
       # print(new_path)
        # if not os.path.exists(new_path):
            # os.makedirs(new_path)
        img = cv2.imread(pre_path)
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
        #cv2.imshow('Canny1', img)    
        gray1 = cv2.GaussianBlur(gray,(3,3),0) #高斯平滑处理原图像降噪   
        canny = cv2.Canny(gray1, 50, 150)     #apertureSize默认为3  
        #cv2.imshow('Canny', canny) 
        lines = cv2.HoughLines(canny, 1, np.pi/180, int(w/3) )
        img,canny1 = clr_line(lines,img,canny,h,w,pre_path)
        res=cv2.resize(canny1,(192,128),interpolation=cv2.INTER_CUBIC)
        image_data.append(res)
        # cv2.imwrite(new_path+r'/{}'.format(imagename[i]), res)
    return image_data,classLabelVector

def image_dir1(imagename):
    image_data = []
    for i in range(len(imagename)):
        pre_path = r'F:\eclipse-workspace\brand_classfication\datasets\test\{}'.format(imagename[i])
        # new_path = r'F:\eclipse-workspace\brand_classfication\train1\{}'.format(classLabelVector[i])
       # print(new_path)
        # if not os.path.exists(new_path):
            # os.makedirs(new_path)
        img = cv2.imread(pre_path)
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
        #cv2.imshow('Canny1', img)    
        gray1 = cv2.GaussianBlur(gray,(3,3),0) #高斯平滑处理原图像降噪   
        canny = cv2.Canny(gray1, 50, 150)     #apertureSize默认为3  
        #cv2.imshow('Canny', canny) 
        lines = cv2.HoughLines(canny, 1, np.pi/180, int(w/3) )
        img,canny1 = clr_line(lines,img,canny,h,w,pre_path)
        res=cv2.resize(canny1,(192,128),interpolation=cv2.INTER_CUBIC)
        image_data.append(res)
        # cv2.imwrite(new_path+r'/{}'.format(imagename[i]), res)
    return image_data

	

def getnum(file_path):
    pathDir = os.listdir(file_path)
    i = 0
    for allDir in pathDir:
        i +=1
    return i
#制作数据集
def data_label(path,count):
    data = np.empty((count,1,128,192),dtype = 'float32')#建立空的四维张量类型32位浮点
    label = np.empty((count,),dtype = 'uint8')
    i = 0
    pathDir = os.listdir(path)
    print(pathDir)
    for each_image in pathDir:
        all_path = os.path.join('%s%s' % (path,each_image))#路径进行连接
        image = cv2.imread(all_path,0)
        # des_image = cv.CreateImage((width_scale,height_scale),image.depth,1)
        # cv.Resize(image,des_image,cv2.INTER_AREA)
        res=cv2.resize(image,(192,128),interpolation=cv2.INTER_CUBIC)
        mul_num = re.findall(r"\d",all_path)#寻找字符串中的数字，由于图像命名为300.jpg 标签设置为0
        num = int(mul_num[0])-3
#        print num,each_image
#        cv2.imshow("fad",image)
#        print child
        array = np.asarray(res,dtype='float32')
        print(array)
        array -= np.min(array)
        array /= np.max(array)
        data[i,:,:,:] = array
        label[i] = int(num)
        i += 1
    return data,label
#构建卷积神经网络
def cnn_model(train_data,train_label,test_data,test_label):
    model = Sequential()
#卷积层 12 × 120 × 120 大小
    model.add(Convolution2D(
        nb_filter = 12,
        nb_row = 3,
        nb_col = 3,
        border_mode = 'valid',
        dim_ordering = 'th',
        input_shape = (1,128,192)))
    model.add(Activation('relu'))#激活函数使用修正线性单元
#池化层12 × 60 × 60
    model.add(MaxPooling2D(
        pool_size = (2,2),
        strides = (2,2),
        border_mode = 'valid'))
#卷积层 24 * 58 * 58
    model.add(Convolution2D(
        24,
        3,
        3,
        border_mode = 'valid',
        dim_ordering = 'th'))
    model.add(Activation('relu'))
#池化层 24×29×29
    model.add(MaxPooling2D(
        pool_size = (2,2),
        strides = (2,2),
        border_mode = 'valid'))
    model.add(Convolution2D(
        48,
        3,
        3,
        border_mode = 'valid',
        dim_ordering = 'th'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(
        pool_size = (2,2),
        strides =(2,2),
        border_mode = 'valid'))
    model.add(Flatten())
    model.add(Dense(20))
    model.add(Activation(LeakyReLU(0.3)))
    model.add(Dropout(0.5))
    model.add(Dense(20))
    model.add(Activation(LeakyReLU(0.3)))
    model.add(Dropout(0.4))
    model.add(Dense(5,init = 'normal'))
    model.add(Activation('softmax'))
    adam = Adam(lr = 0.001)
    model.compile(optimizer = adam,loss =  'categorical_crossentropy',metrics = ['accuracy'])
    print ('----------------training-----------------------')
    model.fit(train_data,train_label,batch_size = 20,nb_epoch = 50,shuffle = True,validation_split = 0.1)
    print ('----------------testing------------------------')
    
    test_label = model.predict(test_data) 
    # loss,accuracy = model.evaluate(test_data,test_label)
    # print ('\n test loss:',loss)
    # print ('\n test accuracy',accuracy)
    return test_label

if __name__=='__main__':
    train_imagename,train_label = file2matric(r'F:\eclipse-workspace\brand_classfication\datasets\train.txt')
    test_imagename = file2matric1(r'F:\eclipse-workspace\brand_classfication\datasets\test.txt')
    #print(train_imagename,train_label)
    train_data,train_label = image_dir(train_imagename,train_label)
    # print(train_data)
    # print(train_label)
    test_data = image_dir1(test_imagename)
    test_label = []
    train_label = np_utils.to_categorical(train_label,100)
    test_label = cnn_model(train_data,train_label,test_data,test_label)
    print(test_label)
    # train_length = len(train_label)
    # train_data = image_data[:int(7*length/10)]
    # train_label = train_label[:int(7*length/10)]
    # test_data = image_data[int(7*length/10):]
    # test_label = train_label[int(7*length/10):]
    print('OK')

# train_path = 'F:/pythonex/all_data/train/'
# test_path = 'F:/pythonex/all_data/test/'
# train_count = getnum(train_path)
# test_count = getnum(test_path)
# train_data,train_label = data_label(train_path,train_count)
# print(train_data)
# print(train_label)
# test_data,test_label = data_label(test_path,test_count)
# train_label = np_utils.to_categorical(train_label,5)
# test_label = np_utils.to_categorical(test_label,5)
# cnn_model(train_data,train_label,test_data,test_label)


#print getnum('/home/zhanghao/data/classification/test_scale/')
#data_label('/home/zhanghao/data/classification/test_scale/',1)
#cv.WaitKey(0)
