import os
import random
import numpy as np
import os
import io
import re

_path_to_file = "/home/lcy/PPTLGeneration/seqtoseq_nmt-master/assets/datahence/nlp5.txt"
_path_to_file_2 = "/home/lcy/PPTLGeneration/seqtoseq_nmt-master/assets/nlp6.txt"

def read_sample_dataset(_path_to_file):
    file = open(_path_to_file, 'r', encoding='utf-8')
    lines_1 = []
    lines_2 = []
    for line in file:
        lines_1.append(line.split('\t')[0])
        lines_2.append(line.split('\t')[1])
    file.close()
    print(lines_1,  lines_2)
    return    lines_1,  lines_2

datalist_1, datalist_2 = read_sample_dataset(_path_to_file)

new_datalist_1 = []
new_datalist_2 = []

for index_list in range(0,len(datalist_1)):
    new_datalist_1.append(datalist_1[index_list])
    new_datalist_2.append(datalist_2[index_list])    
    for index_list_2 in range(0,len(datalist_1)-1):
        new_datalist_1.append(datalist_1[index_list].strip('.')+" and "+datalist_1[index_list_2])
        new_datalist_2.append(datalist_2[index_list].replace('\n', '')+" && "+datalist_2[index_list_2])
    
def record_input_dataset(_path_to_file_2, input_lang, output_lang):
    sentences = []
    count = 0 
    for i in range(len(input_lang)):
        sentences.append(input_lang[i]+'\t'+output_lang[i])
        count = count + 1
    str = '\n'
    f=open(_path_to_file_2,"w")
    f.write(str.join(sentences))
    print("记录完毕")
    f.close()

record_input_dataset(_path_to_file_2,new_datalist_1,new_datalist_2)