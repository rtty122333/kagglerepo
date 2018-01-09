import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import string 
from wordcloud import WordCloud

df_train = pd.read_csv(r'.\data\train.tsv\train.tsv', sep='\t')
df_test = pd.read_csv(r'.\data\test.tsv\test.tsv', sep='\t')
##查看价格分布情况表
plt.figure(figsize=(20, 15))
plt.hist(df_train['price'], bins=50, range=[0,250], label='price')
plt.title('Train "price" distribution', fontsize=15)
plt.xlabel('Price', fontsize=15)
plt.ylabel('Samples', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.show()