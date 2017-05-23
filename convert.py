validate_nolabel = pd.read_csv('validate_nolabel.txt')
validate = pd.read_csv('NN_first.csv')
latest_NN_final = pd.merge(validate_nolabel,validate,how="left", on=['qid','uid'])
#drop
latest_NN_final = latest_NN_final.drop('label_x',axis=1)
latest_NN_final.columns = ['qid','uid','label']
latest_NN_final.to_csv('NN_500_noreg.csv',index=False)
