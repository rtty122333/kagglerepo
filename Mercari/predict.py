import numpy as np
import pandas as pd
import lightgbm as lgb
import time

from scipy.sparse import csr_matrix,hstack

from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score


def main():
    start_time = time.time()
    
    train = pd.read_table('train.tsv/train.tsv', engine='c')
    test = pd.read_table('test.tsv/test.tsv', engine='c')
    print('[{}] Finished to load data'.format(time.time()-start_time))
    print('Train shape: ',train.shape)
    print('Test shape: ',test.shape)
    print(test.describe())
    
if __name__ == '__main__':
    main()