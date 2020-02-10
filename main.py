'''
Created on 2018年6月2日

@author: Administrator
'''
from numpy import *
import os
import shutil
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

def image_dir(imagename,classLabelVector):
    for i in range(len(classLabelVector)):
        pre_path = r'F:\eclipse-workspace\brand_classfication\datasets\train\{}'.format(imagename[i])
        new_path = r'F:\eclipse-workspace\brand_classfication\train\{}'.format(classLabelVector[i])
       # print(new_path)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        shutil.copy(pre_path, new_path)
        
if __name__=='__main__':
    imagename,classLabelVector = file2matric(r'F:\eclipse-workspace\brand_classfication\datasets\train.txt')
    print(imagename,classLabelVector)
    image_dir(imagename,classLabelVector)
    print('OK')
    
    
    
    
    
    