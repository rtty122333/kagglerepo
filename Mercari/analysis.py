import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import string 
from wordcloud import WordCloud


def check_price_layout(dataset):
    '''
    check_price_layout
    '''
    print(dataset.describe())
    plt.figure(figsize=(20, 15))
    plt.hist(dataset['price'], bins=50, range=[0,250], label='price')
    plt.title('Train "price" distribution', fontsize=15)
    plt.xlabel('Price', fontsize=15)
    plt.ylabel('Samples', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.show()
    
def check_price_shipping(dataset):
    '''
    check price over shipping type distribution
    '''
    plt.figure(figsize=(20, 15))
    plt.hist(dataset[dataset['shipping']==1]['price'], \
    bins=50, normed=True,range=[0,250], alpha=0.6, \
    label='price when shipping is 1')
    plt.hist(dataset[dataset['shipping']==0]['price'], \
    bins=50, normed=True,range=[0,250], alpha=0.6, \
    label='price when shipping is 0')
    plt.title('Train Price over shipping type distribution', \
    fontsize=15)
    plt.xlabel('Price', fontsize=15)
    plt.ylabel('Normalized Samples', fontsize=15)
    plt.legend(fontsize=15)
    plt.legend(fontsize=15)
    plt.legend(fontsize=15)
    plt.show()
    
def check_desc_word(dataset):
    '''
    check word distribution in descriptions
    '''
    cloud = WordCloud(width=1920,height=1080).generate( \
    " ".join(dataset['item_description'].astype(str)))
    plt.figure(figsize=(30,15))
    plt.imshow(cloud)
    plt.axis('off')
    plt.show()


def show_description_length(train_dataset, test_dataset):
    train_ds = pd.Series(train_dataset['item_description'].tolist()).astype(str)
    test_ds = pd.Series(test_dataset['item_description'].tolist()).astype(str)
    plt.figure(figsize=(20,15))
    plt.hist(train_ds.apply(len), bins=100, range=[0,600], label='train')
    plt.hist(test_ds.apply(len), bins=100, alpha=0.6, range=[0,600], label='test')
    plt.title('Histogram of charactor count', fontsize=15)
    plt.xlabel('Charactors Number', fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.show()
    
def compute_tf_idf(description):
    '''
    to be fixed
    '''
    description = str(description)
    description.translate(string.punctuation)
    
def tf_idf(dataset):
    '''
    use tf_idf handle item_description
    '''
    tfidf = TfidfVectorizer(min_df=5, strip_accents='unicode', \
    lowercase=True, analyzer='word', token_pattern=r'\w+', ngram_range=(1,3), \
    use_idf=True, smooth_idf=True, subliner_tf=True, stop_words='english')
    tfidf.fit_transform(dataset['item_description'].apply(str))
    tfidf_dict = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
    

    
def main():
    df_train = pd.read_csv(r'.\train.tsv\train_small.tsv', sep='\t')
    df_test = pd.read_csv(r'.\test.tsv\test_small.tsv', sep='\t')
#    check_price_layout(df_train)
#    check_price_shipping(df_train)
#   check_desc_word(df_train)
    show_description_length(df_train,df_test)
    
if __name__ == '__main__':
    main()