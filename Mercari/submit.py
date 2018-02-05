import lightgbm as lgbm
from sklearn.metrics import mean_squared_error
from scipy import sparse as ssp
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import time
import re
import collections
import gc

NAME_MIN_DF=30
MAX_FEATURES_ITEM_DESCRIPTION=20000
NFOLDS = 4

def split_cat(text):
    try:
        cat_nm=text.split("/")
        if len(cat_nm)>=3:
            return cat_nm[0],cat_nm[1],cat_nm[2]
        if len(cat_nm)==2:
            return cat_Nm[0],cat_nm[1],'missing'
        if len(cat_nm)==1:
            return cat_nm[0],'missing','missing'
    except: return ("missing", "missing", "missing")
    
def handle_missing(dataset):
    dataset['category_name'].fillna('other', inplace=True)
    dataset['brand_name'].fillna('missing', inplace=True)
    dataset['item_description'].fillna('none', inplace=True)
    
def handle_nm_word_len(dataset):
    dataset['nm_word_len']=list(map(lambda x: len(x.split()), dataset['name'].tolist()))

def handle_nm_len(dataset):
    dataset['nm_len']=list(map(lambda x: len(x),dataset['name'].tolist()))
    
def handle_desc_word_len(dataset):
    dataset['desc_word_len']=list(map(lambda x: len(x.split()), dataset['item_description'].tolist()))

def handle_desc_len(dataset):
    dataset['desc_len']=list(map(lambda x: len(x), dataset['item_description'].tolist()))

def handle_laberencoder(train,test,label):
    for item in label:
        newlist = train[item].append(test[item])
        le = LabelEncoder()
        le.fit(newlist)
        train[item] = le.transform(train[item])
        test[item] = le.transform(test[item])

def handle_onehot(train,test,label):
    encoder = OneHotEncoder()
    encoder.fit(train[label].append(test[label]))
    X_cat = encoder.transform(train[label])
    X_test_cat = encoder.transform(test[label])
    return X_cat,X_test_cat
    
def handle_category(dataset):
    dataset['subcat_0'], dataset['subcat_1'], dataset['subcat_2'] = \
        zip(*dataset['category_name'].apply(lambda x: split_cat(x)))
def main():
    start_time = time.time()
    #  stop-word, can add any wording I want to replace
    stopwords=set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
                'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
                'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 
                'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
                'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
                'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 
                'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
                'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
                'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
                'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 
                've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 
                'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn',
                '&','brand new','new','[rm]','free ship.*?',
                'rm','price firm','no description yet'               
                ])
              
    pattern = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')
    train = pd.read_csv('./train.tsv/train.tsv', sep="\t",encoding='utf-8',
                        converters={'item_description':lambda x:  pattern.sub('',x.lower()),
                                'name':lambda x:  pattern.sub('',x.lower())}
                    )
    print("finished to load train file : {}".format(time.time()-start_time))
    test = pd.read_csv('./test.tsv/test.tsv', sep="\t",encoding='utf-8',
                        converters={'item_description':lambda x:  pattern.sub('',x.lower()),
                                'name':lambda x:  pattern.sub('',x.lower())}
                        )
    print("finished to load test file : {}".format(time.time()-start_time))
    train_label = np.log1p(train['price'])
    train_texts = train['name'].tolist()
    test_texts = test['name'].tolist()
    handle_missing(train)
    handle_missing(test)
    handle_nm_word_len(train)
    handle_nm_word_len(test)
    handle_desc_word_len(train)
    handle_desc_word_len(test)
    handle_nm_len(train)
    handle_nm_len(test)
    handle_desc_len(train)
    handle_desc_len(test)
#    print(train.describe())
    nrow_train = train.shape[0] 
    handle_category(train)
    handle_category(test)
    count = CountVectorizer(min_df=NAME_MIN_DF)
    X_name_mix = count.fit_transform(train['name'].append(test['name']))
    X_name=X_name_mix[:nrow_train]
    X_t_name = X_name_mix[nrow_train:]
    tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,ngram_range=(1,3))
    X_description_mix = tv.fit_transform(train['item_description'].append(test['item_description']))
    X_description=X_description_mix[:nrow_train]
    X_t_description = X_description_mix[nrow_train:]
 #handle label encoder
    cat_features=['subcat_2','subcat_1','subcat_0','brand_name','category_name','item_condition_id','shipping']
    handle_laberencoder(train,test,cat_features)
    X_cat,X_test_cat = handle_onehot(train,test,cat_features)
#   print(train.describe()) 
    print("finished to label encoder : {}".format(time.time()-start_time))
    
    train_feature=['desc_word_len','nm_word_len','desc_len','nm_len']
    train_list = [train[train_feature].values,X_description,X_name,X_cat]
    test_list = [test[train_feature].values,X_t_description,X_t_name,X_test_cat]
    X = ssp.hstack(train_list).tocsr()
    X_test = ssp.hstack(test_list).tocsr()
    print("finished to handle features : {}".format(time.time()-start_time))
    
    kfold =KFold(n_splits=NFOLDS, shuffle=True, random_state=128)
    
    learning_rate = 0.8
    num_leaves =128
    min_data_in_leaf = 1000
    feature_fraction = 0.5
    bagging_fraction=0.9
    bagging_freq=1000
    num_boost_round = 1000
    params = {"objective": "regression",
            "boosting_type": "gbdt",
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "feature_fraction": feature_fraction, 
            "bagging_freq": bagging_freq,
            "bagging_fraction": bagging_fraction,
            "verbosity": 0,
            "metric": "l2_root",
            "nthread": 4,
            "subsample": 0.9
            }
            
    test_id=test['test_id']
    cv_pred = np.zeros(len(test_id))
    
    kf = kfold.split(X)
    for i, (train_fold, test_fold) in enumerate(kf):
        train_t0 = time.time()
        X_train, X_validate, label_train, label_validate = \
                X[train_fold, :], X[test_fold, :], train_label[train_fold], train_label[test_fold]
        dtrain = lgbm.Dataset(X_train, label_train)
        dvalid = lgbm.Dataset(X_validate, label_validate, reference=dtrain)
        bst = lgbm.train(params, dtrain, num_boost_round, valid_sets=dvalid, verbose_eval=100,early_stopping_rounds=100)
        cv_pred += bst.predict(X_test, num_iteration=bst.best_iteration)
        print ('training & predict time',time.time()-train_t0)
        gc.collect()
    cv_pred /= NFOLDS
    cv_pred = np.expm1(cv_pred)
    submission = test[["test_id"]]
    submission["price"] = cv_pred
    submission.to_csv("./submission.csv", index=False)
    print ('done',time.time()-start_time)
    
if __name__ == '__main__':
    main()