import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import cPickle as pickle

data = pd.read_csv('final_data.csv')
y = data['Label']
X = data.drop(['Q_ID','U_ID','Label'],axis=1)
v = pd.read_csv('final_test_data.csv')

test = v.drop(['qid','uid','label'],axis=1)

clf = RandomForestClassifier(n_estimators=500,n_jobs=-1)
clf.fit(X,y)

f = open('rf_500_final.pkl','wb')
pickle.dump(clf,f)

probabilities = clf.predict_proba(test)
probability_dataFrame = pd.DataFrame(probabilities)
v = pd.concat([v,probability_dataFrame],axis=1)

v.to_csv('rf_500_final.csv',index=False)






