import numpy as np
import pandas as pd
import scipy.stats as st

from azureml import Workspace
ws = Workspace(
    workspace_id='e0dd9dba46274dbbae0f6577f68c5470',
    authorization_token='9d81fba3477040039cda2d7f75e15e35',
    endpoint='https://studioapi.azureml.net'
)
ds = ws.datasets['train.csv']
frame = ds.to_dataframe()
frame["PassengerId"]=pd.Categorical(frame["PassengerId"],ordered=False)
frame["Survived"]=pd.Categorical(frame["Survived"],ordered=False)
frame["Pclass"]=pd.Categorical(frame["Pclass"],ordered=False)
frame["Name"]=pd.Categorical(frame["Name"],ordered=False)
frame["Sex"]=pd.Categorical(frame["Sex"],ordered=False)
frame["Ticket"]=pd.Categorical(frame["Ticket"],ordered=False)
frame["Cabin"]=frame["Cabin"].fillna("No")
frame["Cabin"] = np.where(frame["Cabin"]!="No", 1, 0)
frame["Cabin"]=pd.Categorical(frame["Cabin"],ordered=False)
frame["Embarked"]=frame["Embarked"].fillna("N/A")
frame["Embarked"]=pd.Categorical(frame["Embarked"],ordered=False)
frame["Age"]=frame["Age"].fillna(frame["Age"].median())
frame["Age"]=frame["Age"].astype(int)
frame["Fare"]=frame["Fare"].astype(int)
frame.dtypes
frame.head()
df_q=frame.drop(["Age","Fare","PassengerId","Name","Ticket"],axis=1)
df1=pd.get_dummies(df_q[["Survived","Pclass","Sex","Cabin","Embarked"]])
df1=df1.drop(["Survived_0","Sex_female","Cabin_0"],axis=1)
df1.head()
df2=frame[["Age","Fare","SibSp","Parch"]]
df2.head()
df_c=pd.concat([df1,df2],axis=1)
df_c.head()
var=df_c.drop(["Survived_1"],axis=1)
reponses=df_c["Survived_1"]

import sklearn
from sklearn.cross_validation import train_test_split
var_train, var_test, rep_train, rep_test= train_test_split(var,reponses,test_size=0.1,random_state=10)
var_train.head()

from sklearn.grid_search import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier()
digit_tree=tree.fit(var_train,rep_train)
1-digit_tree.score(var_test,rep_test)

param=[{"max_depth":list(range(2,10))}]
titan_tree=GridSearchCV(DecisionTreeClassifier(),param,cv=5,n_jobs=-1)
titan_opt=titan_tree.fit(var_train,rep_train)
titan_opt.best_params_
1-titan_opt.score(var_test,rep_test)

rep_tree=titan_opt.predict(var_test)
table=pd.crosstab(rep_test,rep_tree)

from azureml import Workspace
ws = Workspace(
    workspace_id='e0dd9dba46274dbbae0f6577f68c5470',
    authorization_token='9d81fba3477040039cda2d7f75e15e35',
    endpoint='https://studioapi.azureml.net'
)
ds = ws.datasets['test.csv']
test_set = ds.to_dataframe()
test_set["PassengerId"]=pd.Categorical(test_set["PassengerId"],ordered=False)
test_set["Pclass"]=pd.Categorical(test_set["Pclass"],ordered=False)
test_set["Name"]=pd.Categorical(test_set["Name"],ordered=False)
test_set["Sex"]=pd.Categorical(test_set["Sex"],ordered=False)
test_set["Ticket"]=pd.Categorical(test_set["Ticket"],ordered=False)
test_set["Cabin"]=test_set["Cabin"].fillna("No")
test_set["Cabin"] = np.where(test_set["Cabin"]!="No", 1, 0)
test_set["Cabin"]=pd.Categorical(test_set["Cabin"],ordered=False)
test_set["Embarked"]=test_set["Embarked"].fillna("N/A")
test_set["Embarked"]=pd.Categorical(test_set["Embarked"],ordered=False)
test_set["Age"]=test_set["Age"].fillna(test_set["Age"].median())
test_set["Age"]=test_set["Age"].astype(int)
test_set["Fare"]=test_set["Fare"].fillna(test_set["Fare"].median())
test_set["Fare"]=test_set["Fare"].astype(int)
test_q=test_set.drop(["Age","Fare","PassengerId","Name","Ticket"],axis=1)
test1=pd.get_dummies(test_q[["Pclass","Sex","Cabin","Embarked"]])
test1=test1.drop(["Sex_female","Cabin_0"],axis=1)
test2=test_set[["Age","Fare","SibSp","Parch"]]
test_fin=pd.concat([test1,test2],axis=1)
test_fin.insert(6, 'Embarked_N/A', 0)
rep_fin=titan_opt.predict(test_fin)
test_fin.insert(0,'Survived',rep_fin)


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(
    n_estimators=500,
    criterion='gini',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='auto',
    max_leaf_nodes=None,
    bootstrap=True,
    oob_score=True)
var_train, var_test, rep_train, rep_test= train_test_split(var,reponses,test_size=0.1,random_state=10)
forest=forest.fit(var_train,rep_train)
print(1-forest.oob_score_)
print(1-forest.score(var_test,rep_test))

test_fin2=pd.concat([test1,test2],axis=1)
test_fin2.insert(6, 'Embarked_N/A', 0)
rep_fin2=titan_opt.predict(test_fin2)
test_fin2.insert(0,'Survived',rep_fin2)



from azureml import DataTypeIds

dataset = ws.datasets.add_from_raw_data(
    raw_data=test_fin2,
    data_type_id=DataTypeIds.GenericCSV,
    name='Titanic Random Forest',
    description=''
)
