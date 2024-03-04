import pandas as pd
import jieba
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from PIL import Image
import numpy as np
from matplotlib import colors

data = pd.read_csv('danmu.csv', header=0, encoding="utf-8-sig")

print(data.shape[0])  # 数量
print(data.head())  # 数据内容,只打印了头部的前4个信息

# 分词
# 词云是按照词来进行统计的，这个使用jieba自动进行词频统计
text = ''.join(data['comments'])  # 此处将所有的评论进行了整合连成一个字符串
print(text)
cut = jieba.cut(text)  # 将一个字符串进行分割
words = list(jieba.cut(text))
ex_sw_words = []
# 下方是对目前的一些字符串进行筛选，将一些没有意义的词语进行清除
stop_words = [x.strip() for x in open('stopwords.txt', encoding="utf-8")]
for word in words:
    if len(word) > 1 and (word not in stop_words):
        ex_sw_words.append(word)
print(ex_sw_words)
print(cut)  # 返回cut是一个对象
string = ' '.join(ex_sw_words)  # 此处将其对象cut变成字符串，可在下方显示，#' '.join(cut)  以指定字符串空格‘ ’作为分隔符，将 cut 中所有的元素(的字符串表示)合并为一个新的字符串

print(string)  # 此时可以打印如下
print(len(string))  # 个词，要对这些词进行统计

# 可以自己找图建议轮廓清晰
img = Image.open(r'aixin.png')  # 打开遮罩图片
img_arry = np.array(img)  # 将图片转换为数组，有了数组即可做词云的封装了

# 建立颜色数组，可更改颜色
color_list = ['#CD853F', '#DC143C', '#00FF7F', '#FF6347', '#8B008B', '#00FFFF', '#0000FF', '#8B0000', '#FF8C00',
              '#1E90FF', '#00FF00', '#FFD700', '#008080', '#008B8B', '#8A2BE2', '#228B22', '#FA8072', '#808080']

# 调用
colormap = colors.ListedColormap(color_list)
wc = WordCloud(
    background_color='white',  # 背景必须是白色
    mask=img_arry,  # 传入遮罩的图片，必须是数组
    font_path="STXINGKA.TTF",  # 设置字体
    colormap=colormap,  # 设置文字颜色
    max_font_size=150,  # 设置字体最大值
    random_state=18  # 设置有多少种随机生成状态，即有多少种配色方案
)

wc.generate_from_text(string)  # 从哪个文本生成wc,这个文本必须是切好的词

# 绘制图片
fig = plt.figure(1)  # 1表示第一个位置绘制
plt.imshow(wc)  # 按照wc词云的规格显示
plt.axis('off')  # 是否显示坐标轴
# plt.show()  # 显示生成的词云图片
plt.savefig(r'danmu2.jpg', dpi=400)  # 输出词云图片到文件,默认清晰度是400
