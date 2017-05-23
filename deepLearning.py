from hw_utils import *
import pandas as pd
from lasange import *
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

X_train = pd.read_csv('final_data.csv')
X_train = X_train.drop(['Q_ID','U_ID','Label'],axis =1)
Y_train = pd.read_csv('Y.csv')
X_test = pd.read_csv('new_validate_final.csv')
X_test = X_test.drop(['qid','uid','label'],axis =1)

X_train = pd.read_csv('Xtrain_SVD_500.csv')
X_test = pd.read_csv('Xtrain_SVD_500.csv')
Y_train = pd.read_csv('Y.csv')
X_train = X_train.values
Y_train = Y_train.values
X_test = X_test.values
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
din = len(X_train[0])
dout = len(Y_train[0])

Y_train = Y_train.astype(np.int32)
Y_train = Y_train.reshape((245752,))

pred = NNet(X_train,Y_train,X_test)

archs = [din,800,500,300,200,dout]

decay = 1e-5 #ASSUMPTION
reg =5e-7 #ASSUMPTION
moms = 0.99 #ASSUMPTION

clf = testmodels(X_train, Y_train, archs, actfn='relu', last_act='sigmoid', reg_coeff=0,
				num_epoch=500, batch_size=1000, sgd_lr=0.0001, sgd_decay=0, sgd_mom=moms,
					sgd_Nesterov=True, EStop=False, verbose=1)
deep_learn = clf.predict_proba(X_test)
pred = pd.DataFrame(deep_learn)
#
validate = pd.read_csv('new_validate_final.csv')
validate = validate[['qid','uid']]
validate = pd.concat([validate,pred],axis=1)
validate.columns = ['qid','uid','label']
validate.to_csv('NN_full_dataset.csv',index=False)
