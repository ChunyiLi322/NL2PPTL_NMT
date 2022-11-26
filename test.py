# from shutil import copy2
# from sklearn.model_selection import train_test_split
#
# import tensorflow as tf
# import os
#
# import unicodedata
# import re
# import numpy as np
# import os
# import io
# import time
#
# new_location = os.getcwd()
#
# path_to_zip = tf.keras.utils.get_file(
#     'fra-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/cmn-eng.zip',
#     cache_subdir=new_location + '/assets',
#     extract=True)
#
# path_to_file = os.path.dirname(path_to_zip) + "/cmn-eng"
#
# try:
#     os.mkdir(path_to_file)
# except:
#     pass
#
# path_to_file = copy2(new_location + '/assets/cmn.txt', path_to_file)

#1.使用python random模块的choice方法随机选择某个元素
import random
foo = ['a', 'b', 'c', 'd', 'e']
from random import choice
print(choice(foo))
print(foo)
 
#2.使用python random模块的sample函数从列表中随机选择一组元素
list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
#设置种子使得每次抽样结果相同
random.seed(10)
slice = random.sample(list, 5)  #从list中随机获取5个元素，作为一个片断返回  
print(slice)  
print(list) #原有序列并没有改变。