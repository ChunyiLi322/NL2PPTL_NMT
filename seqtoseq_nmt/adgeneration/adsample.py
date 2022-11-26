from stanfordcorenlp import StanfordCoreNLP
import numpy as np
import random
from collections import Counter



def record_input_dataset(_path_to_file):
    file = open(_path_to_file, 'r', encoding='utf-8')
    sentences = []
    count = 0 
    for line in file:
        input_batch = line.split('\t')[0]
        sentences.append(input_batch)
        count = count + 1
        #print(input_batch)
    file.close()
    str = '\n'
    f=open("traintest.txt","w")
    f.write(str.join(sentences))
    f.close()
    return  count
    
def adsample_generation(number_input):    

    nlp=StanfordCoreNLP(r'/home/lcy/adsample/NCR2Code/')
    
    fin=open('traintest.txt','r',encoding='utf8')
    fner=open('nerCops.txt','w',encoding='utf8')
    ftag=open('pos_tag.txt','w',encoding='utf8')
    
    word_dict=[]
    word_tag =[]
    
    line_count = 0 
    batch_size = 64
    
    ls = [random.randint(0,number_input) for i in range(batch_size)]
     
    for line in fin:
        line=line.strip()
        if len(line)<1:
            continue
        for i in ls:
            if(i == line_count) :
                fner.write(" ".join([each[0]+"/"+each[1] for each in nlp.ner(line) if len(each)==2 ])+"\n")
                ftag.write(" ".join([each[0]+"/"+each[1] for each in nlp.pos_tag(line) if len(each)==2 ]) +"\n")
                #word_dict.append(i)
                #word_tag.append(i)
                for each in nlp.ner(line):
                    string_word = each
                    word_dict.append(string_word[0])
                for each in nlp.pos_tag(line):
                    word_tag.append(each[1])
        line_count = line_count + 1
    
    adsample_comput_important(word_tag)
    NN_word_tag =[]    
    for m,n in zip(word_dict,word_tag):
        if n == 'NN' or n == 'NNS' or n == 'NNP':
           NN_word_tag.append(m)
    # dict_NN_word_tag = {}
    # for i in NN_word_tag:
    #  if i not in dict_NN_word_tag.keys():
    #   dict_NN_word_tag[i] = NN_word_tag.count(i)
    # print(dict_NN_word_tag)
    
    collection_words = Counter(NN_word_tag)
    most_counterNum = collection_words.most_common(3)
    
    
    print(word_dict)
    print("--------------------")
    print(word_tag)
    print("--------------------")
    print(len(word_dict))
    print(len(word_tag))
    print("---------NN_word_tag-----------")
    print(NN_word_tag)
    print(most_counterNum)
    print(most_counterNum[0][0])
    
    fner.close()
    ftag.close()
    
    return most_counterNum[0][0],most_counterNum[1][0],most_counterNum[2][0]


def adsample_comput_important(word_tag): 
        count = 0
        for n in word_tag:
            if n == 'NN' or n == 'NNS' or n == 'NNP':
               count = count + 1;
        print("显著性",count/len(word_tag))
        return count/len(word_tag)



