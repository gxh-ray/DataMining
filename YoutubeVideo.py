#!usr/bin/env python
# coding: UTF-8

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

USvideo = pd.DataFrame(pd.read_csv('USvideos.csv', low_memory=False))
CAvideo = pd.DataFrame(pd.read_csv('CAvideos.csv', low_memory=False))
INvideo = pd.DataFrame(pd.read_csv('INvideos.csv', low_memory=False))
DEvideo = pd.DataFrame(pd.read_csv('DEvideos.csv', low_memory=False))


#category_id标称属性，每个可能取值的频数
print(USvideo['category_id'].value_counts())
print(CAvideo['category_id'].value_counts())
print(INvideo['category_id'].value_counts())
print(DEvideo['category_id'].value_counts())

#title标称属性，每个可能取值的频数
print(USvideo['title'].value_counts())
print(CAvideo['title'].value_counts())
print(INvideo['title'].value_counts())
print(MXvideo['title'].value_counts())

#channel_title标称属性，每个可能取值的频数
print(USvideo['channel_title'].value_counts())
print(CAvideo['channel_title'].value_counts())
print(INvideo['channel_title'].value_counts())
print(MXvideo['channel_title'].value_counts())


#UScategory_id属性直方图
plt.hist(x=USvideo['category_id'], bins=50, edgecolor='black')
# 添加x轴和y轴标签
plt.xlabel('category_id')
plt.ylabel('frequency')
# 添加标题
plt.title('USVideo_Category distribution')
plt.xticks(rotation=90)
plt.tick_params(labelsize=6)
plt.savefig('./videoResult/US/UScategory_distribution_hist.png')
plt.show()

#CAcategory_id属性直方图
plt.hist(x=CAvideo['category_id'], bins=50, edgecolor='black')
# 添加x轴和y轴标签
plt.xlabel('category_id')
plt.ylabel('frequency')
# 添加标题
plt.title('CAVideo_Category distribution')
plt.xticks(rotation=90)
plt.tick_params(labelsize=6)
plt.savefig('./videoResult/CA/CAcategory_distribution_hist.png')
plt.show()

#INcategory_id属性直方图
plt.hist(x=INvideo['category_id'], bins=50, edgecolor='black')
# 添加x轴和y轴标签
plt.xlabel('category_id')
plt.ylabel('frequency')
# 添加标题
plt.title('INVideo_Category distribution')
plt.xticks(rotation=90)
plt.tick_params(labelsize=6)
plt.savefig('./videoResult/IN/INcategory_distribution_hist.png')
plt.show()

#DEcategory_id属性直方图
plt.hist(x=DEvideo['category_id'], bins=50, edgecolor='black')
# 添加x轴和y轴标签
plt.xlabel('category_id')
plt.ylabel('frequency')
# 添加标题
plt.title('DEVideo_Category distribution')
plt.xticks(rotation=90)
plt.tick_params(labelsize=6)
plt.savefig('./videoResult/DE/DEcategory_distribution_hist.png')
plt.show()



#######################
#views数值属性五数概括
np.set_printoptions(suppress=True)
print(USvideo['views'].dropna().astype(int).describe())
print(CAvideo['views'].dropna().astype(int).describe())
print(INvideo['views'].dropna().astype(int).describe())
print(DEvideo['views'].dropna().astype(int).describe())

#likes数值属性五数概括
np.set_printoptions(suppress=True)
print(USvideo['likes'].dropna().astype(int).describe())
print(CAvideo['likes'].dropna().astype(int).describe())
print(INvideo['likes'].dropna().astype(int).describe())
print(DEvideo['likes'].dropna().astype(int).describe())

#dislikes数值属性五数概括
np.set_printoptions(suppress=True)
print(USvideo['dislikes'].dropna().astype(int).describe())
print(CAvideo['dislikes'].dropna().astype(int).describe())
print(INvideo['dislikes'].dropna().astype(int).describe())
print(DEvideo['dislikes'].dropna().astype(int).describe())

#comment_count数值属性五数概括
np.set_printoptions(suppress=True)
print(USvideo['comment_count'].dropna().astype(int).describe())
print(CAvideo['comment_count'].dropna().astype(int).describe())
print(INvideo['comment_count'].dropna().astype(int).describe())
print(DEvideo['comment_count'].dropna().astype(int).describe())


#USvideo缺省值情况
print("NAN:")
print(USvideo.isna().sum())

#CAvideo缺省值情况
print("NAN:")
print(CAvideo.isna().sum())

#INvideo缺省值情况
print("NAN:")
print(INvideo.isna().sum())

#DEvideo缺省值情况
print("NAN:")
print(DEvideo.isna().sum())


#views直方图
plt.hist(USvideo['views'].dropna().astype(int), bins=50)
# 添加x轴和y轴标签
plt.xlabel('views')
plt.ylabel('frequency')
# 添加标题
plt.title('US-views_distribution')
plt.savefig('./videoResult/US/USviews_hist.png')
plt.show()

plt.hist(CAvideo['views'].dropna().astype(int), bins=50)
# 添加x轴和y轴标签
plt.xlabel('views')
plt.ylabel('frequency')
# 添加标题
plt.title('CA-views_distribution')
plt.savefig('./videoResult/CA/CAviews_hist.png')
plt.show()

plt.hist(INvideo['views'].dropna().astype(int), bins=50)
# 添加x轴和y轴标签
plt.xlabel('views')
plt.ylabel('frequency')
# 添加标题
plt.title('IN-views_distribution')
plt.savefig('./videoResult/IN/INviews_hist.png')
plt.show()

plt.hist(DEvideo['views'].dropna().astype(int), bins=50)
# 添加x轴和y轴标签
plt.xlabel('views')
plt.ylabel('frequency')
# 添加标题
plt.title('DE-views_distribution')
plt.savefig('./videoResult/DE/DEviews_hist.png')
plt.show()

#likes直方图
plt.hist(USvideo['likes'].dropna().astype(int), bins=50)
# 添加x轴和y轴标签
plt.xlabel('likes')
plt.ylabel('frequency')
# 添加标题
plt.title('US-likes_distribution')
plt.savefig('./videoResult/US/USlikes_hist.png')
plt.show()

plt.hist(CAvideo['likes'].dropna().astype(int), bins=50)
# 添加x轴和y轴标签
plt.xlabel('likes')
plt.ylabel('frequency')
# 添加标题
plt.title('CA-likes_distribution')
plt.savefig('./videoResult/CA/CAlikes_hist.png')
plt.show()

plt.hist(INvideo['likes'].dropna().astype(int), bins=50)
# 添加x轴和y轴标签
plt.xlabel('likes')
plt.ylabel('frequency')
# 添加标题
plt.title('IN-likes_distribution')
plt.savefig('./videoResult/IN/INlikes_hist.png')
plt.show()

plt.hist(DEvideo['likes'].dropna().astype(int), bins=50)
# 添加x轴和y轴标签
plt.xlabel('likes')
plt.ylabel('frequency')
# 添加标题
plt.title('DE-likes_distribution')
plt.savefig('./videoResult/DE/DElikes_hist.png')
plt.show()


#views属性Q-Q图
sorted_ = np.sort(USvideo['views'].dropna().astype(int))
yvals = np.arange(len(sorted_))/float(len(sorted_))
x_label = stats.norm.ppf(yvals)
plt.scatter(x_label, sorted_)
stats.probplot(USvideo['views'].dropna().astype(int), dist="norm", plot=plt)
plt.title('US-views Q-Q')
plt.savefig('./videoResult/US/USviews_qq.png')
plt.show()

sorted_ = np.sort(CAvideo['views'].dropna().astype(int))
yvals = np.arange(len(sorted_))/float(len(sorted_))
x_label = stats.norm.ppf(yvals)
plt.scatter(x_label, sorted_)
stats.probplot(CAvideo['views'].dropna().astype(int), dist="norm", plot=plt)
plt.title('CA-views Q-Q')
plt.savefig('./videoResult/CA/CAviews_qq.png')
plt.show()

sorted_ = np.sort(INvideo['views'].dropna().astype(int))
yvals = np.arange(len(sorted_))/float(len(sorted_))
x_label = stats.norm.ppf(yvals)
plt.scatter(x_label, sorted_)
stats.probplot(INvideo['views'].dropna().astype(int), dist="norm", plot=plt)
plt.title('IN-views Q-Q')
plt.savefig('./videoResult/IN/INviews_qq.png')
plt.show()

sorted_ = np.sort(DEvideo['views'].dropna().astype(int))
yvals = np.arange(len(sorted_))/float(len(sorted_))
x_label = stats.norm.ppf(yvals)
plt.scatter(x_label, sorted_)
stats.probplot(DEvideo['views'].dropna().astype(int), dist="norm", plot=plt)
plt.title('DE-views Q-Q')
plt.savefig('./videoResult/DE/DEviews_qq.png')
plt.show()

#likes属性Q-Q图
sorted_ = np.sort(USvideo['likes'].dropna().astype(int))
yvals = np.arange(len(sorted_))/float(len(sorted_))
x_label = stats.norm.ppf(yvals)
plt.scatter(x_label, sorted_)
stats.probplot(USvideo['likes'].dropna().astype(int), dist="norm", plot=plt)
plt.title('US-likes Q-Q')
plt.savefig('./videoResult/US/USlikes_qq.png')
plt.show()

sorted_ = np.sort(CAvideo['likes'].dropna().astype(int))
yvals = np.arange(len(sorted_))/float(len(sorted_))
x_label = stats.norm.ppf(yvals)
plt.scatter(x_label, sorted_)
stats.probplot(CAvideo['likes'].dropna().astype(int), dist="norm", plot=plt)
plt.title('CA-likes Q-Q')
plt.savefig('./videoResult/CA/CAlikes_qq.png')
plt.show()

sorted_ = np.sort(INvideo['likes'].dropna().astype(int))
yvals = np.arange(len(sorted_))/float(len(sorted_))
x_label = stats.norm.ppf(yvals)
plt.scatter(x_label, sorted_)
stats.probplot(INvideo['likes'].dropna().astype(int), dist="norm", plot=plt)
plt.title('IN-likes Q-Q')
plt.savefig('./videoResult/IN/INlikes_qq.png')
plt.show()

sorted_ = np.sort(DEvideo['likes'].dropna().astype(int))
yvals = np.arange(len(sorted_))/float(len(sorted_))
x_label = stats.norm.ppf(yvals)
plt.scatter(x_label, sorted_)
stats.probplot(DEvideo['likes'].dropna().astype(int), dist="norm", plot=plt)
plt.title('DE-likes Q-Q')
plt.savefig('./videoResult/DE/DElikes_qq.png')
plt.show()


#views属性盒图
plt.boxplot(USvideo['views'].dropna().astype(int))
plt.ylabel('views')
plt.title('US-views_Boxplot')
plt.savefig('./videoResult/US/USviews_box.png')
plt.show()

plt.boxplot(CAvideo['views'].dropna().astype(int))
plt.ylabel('views')
plt.title('CA-views_Boxplot')
plt.savefig('./videoResult/CA/CAviews_box.png')
plt.show()

plt.boxplot(INvideo['views'].dropna().astype(int))
plt.ylabel('views')
plt.title('IN-views_Boxplot')
plt.savefig('./videoResult/IN/INviews_box.png')
plt.show()

plt.boxplot(DEvideo['views'].dropna().astype(int))
plt.ylabel('views')
plt.title('DE-views_Boxplot')
plt.savefig('./videoResult/DE/DEviews_box.png')
plt.show()

#likes属性盒图
plt.boxplot(USvideo['likes'].dropna().astype(int))
plt.ylabel('likes')
plt.title('US-likes_Boxplot')
plt.savefig('./videoResult/US/USlikes_box.png')
plt.show()

plt.boxplot(CAvideo['likes'].dropna().astype(int))
plt.ylabel('likes')
plt.title('CA-likes_Boxplot')
plt.savefig('./videoResult/CA/CAlikes_box.png')
plt.show()

plt.boxplot(INvideo['likes'].dropna().astype(int))
plt.ylabel('likes')
plt.title('IN-likes_Boxplot')
plt.savefig('./videoResult/IN/INlikes_box.png')
plt.show()

plt.boxplot(DEvideo['likes'].dropna().astype(int))
plt.ylabel('likes')
plt.title('DE-likes_Boxplot')
plt.savefig('./videoResult/DE/DElikes_box.png')
plt.show()
'''


'''
#最高频率
#直方图
plt.hist(USvideo['likes'].fillna(USvideo['likes'].interpolate(missing_values='NaN', strategy='mode', axis=0, verbose=0, copy=True)),bins=100)
plt.savefig('./videoResult/likes_mode_hist.png')
plt.show()

#QQ图

sorted_ = np.sort(USvideo['likes'].fillna(USvideo['likes'].interpolate(missing_values='NaN', strategy='mode', axis=0, verbose=0, copy=True)))
yvals = np.arange(len(sorted_))/float(len(sorted_))
x_label = stats.norm.ppf(yvals)
plt.scatter(x_label, sorted_)
stats.probplot(USvideo['likes'], dist="norm", plot=plt)
plt.savefig('./videoResult/likes_mode_qq.png')
plt.show()

#盒图
plt.boxplot(USvideo['likes'].fillna(USvideo['likes'].interpolate(missing_values='NaN', strategy='mode', axis=0, verbose=0, copy=True)))
plt.ylabel('likes')
plt.legend()
plt.savefig('./videoResult/likes_mode_box.png')
plt.show()

#通过属性的相关关系来填补缺失值
#直方图
plt.hist(USvideo['likes'].interpolate(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True), bins=100)
plt.savefig('./videoResult/likes_means_hist.png')
plt.show()

#QQ图
sorted_ = np.sort(USvideo['likes'].interpolate(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True))
yvals = np.arange(len(sorted_))/float(len(sorted_))
x_label = stats.norm.ppf(yvals)
plt.scatter(x_label, sorted_)
stats.probplot(USvideo['likes'].interpolate(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True), dist="norm", plot=plt)
plt.savefig('./videoResult/likes_means_qq.png')
plt.show()

#盒图
plt.boxplot(USvideo['likes'].interpolate(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True))
plt.ylabel('likes')
plt.legend()
plt.savefig('./videoResult/likes_means_box.png')
plt.show()

#通过数据对象之间的相似性来填补缺失值
USvideo = USvideo[USvideo['views'].notnull()]
known_price = USvideo[USvideo['likes'].notnull()].sample(frac=0.1)
unknown_price = USvideo[USvideo['likes'].isnull()]
x = known_price[['views']]
y = known_price[['likes']]
t_x = unknown_price[['views']]
fc = RandomForestClassifier()
fc.fit(x, y)
pr = fc.predict(t_x)
USvideo.loc[USvideo.likes.isnull(), 'likes'] = pr

#直方图
plt.hist(USvideo['likes'].astype(int), bins=100)
plt.savefig('./videoResult/likes_relative_hist.png')
plt.show()

#QQ图
sorted_ = np.sort(USvideo['likes'])
yvals = np.arange(len(sorted_))/float(len(sorted_))
x_label = stats.norm.ppf(yvals)
plt.scatter(x_label, sorted_)
stats.probplot(USvideo['likes'])
plt.savefig('./videoResult/likes_relative_qq.png')
plt.show()

#盒图
plt.boxplot(USvideo['likes'])
plt.ylabel('likes')
plt.legend()
plt.savefig('./videoResult/likes_relative_box.png')
plt.show()

