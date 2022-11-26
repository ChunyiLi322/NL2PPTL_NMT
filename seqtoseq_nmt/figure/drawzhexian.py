import pandas as pd
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#��ȡ����
data = pd.read_excel('matplotlib.xlsx')
 
plt.figure(figsize=(10,5))#���û����ĳߴ�
# plt.title('Training loss',fontsize=20)#���⣬���趨�ֺŴ�С
plt.xlabel(u'x-epochs',fontsize=14)#����x�ᣬ���趨�ֺŴ�С
plt.ylabel(u'y-loss',fontsize=14)#����y�ᣬ���趨�ֺŴ�С
 
#color����ɫ��linewidth���߿�linestyle���������ͣ�label��ͼ����marker�����ݵ������
in1, = plt.plot(data['loss'],data['cmn-eng'],color="silver",linewidth=2,linestyle=':', marker='3')
in2, = plt.plot(data['loss'],data['spa-eng'],color="mediumslateblue",linewidth=1,linestyle='--', marker='+')
in3, = plt.plot(data['loss'],data['fra-eng'],color="khaki",linewidth=1.5,linestyle='-', marker='*')
in4, = plt.plot(data['loss'],data['eng-pptl'],color="palegreen",linewidth=1.5,linestyle='-.', marker='4')
 
plt.legend(handles = [in1,in2,in3,in4],labels=['cmn-eng','spa-eng','fra-eng','eng-pptl'],loc=2)
plt.savefig('./zhexian.pdf')
