import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h



a = range(10,50)
print("上下区间",mean_confidence_interval(a))



jddf = pd.read_csv('interval.csv', sep=',', header=None,
                   names=['id', 'time', 'bar1', 'bar2', 'line1', 'line2', 'line3'])


# 设置X轴和Y轴取值
x = np.arange(1,6)
y1 = jddf['bar1']
y2 = jddf['bar2']
# y3 = jddf['bar3']
index = jddf['id'].tolist()


# 柱形图
plt.figure(figsize=(6,4))
bar_width = 0.05  # 设置柱状图的宽度
bar1 = plt.bar(x-bar_width, y1, bar_width, color=(0/256, 114/256, 189/256))
# bar1 = plt.bar(x, y1, bar_width, color=(0/256, 114/256, 189/256))
bar2 = plt.bar(x, y2, bar_width, color=(217/256, 83/256, 25/256))
# bar3 = plt.bar(x+bar_width, y3, bar_width, color=(237/256, 177/256, 32/256))
# bar1 = plt.plot(np.arange(1,6), jddf['bar1'], color='green', lw=0.5, ls='-', marker='o', ms=4)
# bar2 = plt.plot(np.arange(1,6), jddf['bar2'], color='blue', lw=0.5, ls='-', marker='o', ms=4)

# 折线图
line1, = plt.plot(np.arange(1,6), jddf['line1'], color='purple', lw=0.5, ls='-', marker='o', ms=4)
# line2, = plt.plot(np.arange(1,17), jddf['line2'], color='green', lw=0.5,  marker='^', ms=4)
# line3, = plt.plot(np.arange(1,17), jddf['line3'], color='blue', lw=0.5, ls='-.', marker='s', ms=4)

# 折线图置信区间
plt.fill_between(np.arange(1,6), jddf['line2'], jddf['line3'], color=(229/256, 204/256, 249/256), alpha=0.9)
# plt.fill_between(np.arange(1,17), jddf['line2'] - 2, jddf['line2'] + 2, color=(204/256, 236/256, 223/256), alpha=0.9)
# plt.fill_between(np.arange(1,17), jddf['line3'] - 2, jddf['line3'] + 2, color=(191/256, 191/256, 255/256), alpha=0.9)

 # X轴和y轴坐标文字设置
plt.xticks(rotation=20)
plt.xticks(x,index, horizontalalignment='right')
plt.ylabel('Weight fluctuation range')

# 图例设置
# plt.legend([bar1, bar2, bar3, line1, line2, line3], ["Aviation BC number emission", "Aviation BC mass emission", "Aviation fuel consumption", "Geometric mean diameter(GMD)", "Geometric standard deviation(GSD)", "Average fleet EI(BC)"], loc='upper right')
plt.legend([bar1, bar2, line1], ["Weight fluctuation minimum", "Weight fluctuation maximum", "Confidence interval"], loc='upper right')

# 保存数据并画图
plt.savefig("interval.png",dpi=500,bbox_inches = 'tight')#解决图片不清晰，不完整的问题
