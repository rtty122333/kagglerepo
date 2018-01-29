# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import gc
import time
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
import lightgbm as lgb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

def cut_brand(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:4000]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category = dataset['category_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:4000]
    dataset.loc[~dataset['category_name'].isin(pop_category), 'category_name'] = 'missing'

def main():
    start_time = time.time()

    train = pd.read_table(r'../input/train.tsv', engine='c')
    test = pd.read_table(r'../input/test.tsv', engine='c')

    nrow_train = train.shape[0]
    y = np.log1p(train["price"])
    merge = pd.DataFrame(pd.concat([train, test]))
    submission = test[['test_id']]

    del train
    del test
    gc.collect()







#deal with missing
    merge['category_name'].fillna(value='missing', inplace=True)
    merge['brand_name'].fillna(value='missing', inplace=True)
    merge['item_description'].fillna(value='missing', inplace=True)

#leave 5000 brands
    cut_brand(merge)

#change to category
    merge['category_name'] = merge['category_name'].astype('category')
    merge['brand_name'] = merge['brand_name'].astype('category')
    merge['item_condition_id'] = merge['item_condition_id'].astype('category')

    cv = CountVectorizer(min_df=10)
    X_name = cv.fit_transform(merge['name'])
#    print(cv.vocabulary_)

    cv = CountVectorizer()
    X_category = cv.fit_transform(merge['category_name'])

    tv = TfidfVectorizer(max_features=35000,
                         ngram_range=(1, 3),
                         stop_words='english')
    X_description = tv.fit_transform(merge['item_description'])

    lb = LabelBinarizer(sparse_output=True)
    X_brand = lb.fit_transform(merge['brand_name'])

    X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                          sparse=True).values)

    sparse_merge = hstack((X_dummies, X_description, X_brand, X_category, X_name)).tocsr()

    X = sparse_merge[:nrow_train]
    X_test = sparse_merge[nrow_train:]
    
    d_train = lgb.Dataset(X, label=y)
    
    params = {
        'learning_rate': 0.75,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 100,
        'verbosity': -1,
        'metric': 'RMSE',
    }


    model = lgb.train(params, train_set=d_train, num_boost_round=3335,  \
    verbose_eval=100) 
    preds = 0.6*model.predict(X_test)

    model = Ridge(solver="sag", fit_intercept=True, random_state=205)
    model.fit(X, y)
    print('[{}] Finished to train ridge'.format(time.time() - start_time))
    preds += 0.4*model.predict(X=X_test)
    print('[{}] Finished to predict ridge'.format(time.time() - start_time))


    submission['price'] = np.expm1(preds)
    submission.to_csv("submission_lgbm.csv", index=False)


if __name__ == '__main__':
    main()