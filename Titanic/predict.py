import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv(r'data\train.csv')
test = pd.read_csv(r'data\test.csv')

#print (train.head())
#print (test.head())

print ("Train size : %d" % len(train))
print ("Train Survived size : %d" % len(train[train['Survived']==1]))

print ("% of men who survived", 100*np.mean(train['Survived'][train['Sex']=='male']))
print ("% of women who survived", 100*np.mean(train['Survived'][train['Sex']=='female']))

train['Sex'] = train['Sex'].apply(lambda x: 1 if x =='male' else 0)
train['Age'] = train['Age'].fillna(np.mean(train['Age']))
train['Fare'] = train['Fare'].fillna(np.mean(train['Fare']))

test['Sex'] = test['Sex'].apply(lambda x: 1 if x =='male' else 0)
test['Age'] = train['Age'].fillna(np.mean(test['Age']))
test['Fare'] = test['Fare'].fillna(np.mean(test['Fare']))

train = train[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare']]

X = train.drop('Survived', axis = 1)
y = train['Survived']

fix_test = test[['Pclass','Sex','Age','SibSp','Parch','Fare']]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)

#classifier = DecisionTreeClassifier(max_depth=3)
#classifier.fit(X_train, y_train)

#svmclassifier = SVC()
#svmclassifier.fit(X_train, y_train)

classifier = RandomForestClassifier(max_depth=5, random_state=0)
classifier.fit(X_train, y_train)

print ("Training accuracy :",accuracy_score(y_train,classifier.predict(X_train)))
print ("Validation accuracy :",accuracy_score(y_test,classifier.predict(X_test)))

result = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived":classifier.predict(fix_test)})
result.to_csv(r'data\result.csv', index=False)

