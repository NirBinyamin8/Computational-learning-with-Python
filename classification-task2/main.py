import time
import json
import pandas as pd

from sklearn.metrics import accuracy_score
from typing import List
from sklearn.preprocessing import StandardScaler
from collections import Counter
from scipy.spatial.distance import euclidean




def pred_for_test(r,Test,x_train,y_train,defualt_class)->list:
    res=[]
    for per in Test:
        pred_class=preClass(r,per,x_train,y_train,defualt_class)
        res.append(pred_class)
    return res
def preClass(r,per,x,y,defualt_class):
    pred_class=[]

    for i in range (len(x)):
        d=euclidean(per,x[i])
        if(d<r):
            pred_class.append(y[i])
    if(len(pred_class)!=0):
        counts = Counter(pred_class)
        most_common = counts.most_common(1)[0][0]
    else:
        most_common=defualt_class


    return(most_common)





def trainModel(x,y,defualt_class)->float:
    count=0
    res=[]
    rads = [i-0.1 for i in range(2,30)]
    for r in rads:
        for i in range(len(x)):
            mostcommon_class=preClass(r,x[i],x,y,defualt_class)
            if(mostcommon_class==y[i]):
                count+=1
        res.append((r, count / len(x)))
        count = 0
    max = res[0]
    for i in range(len(res)):
        if (max[1] < res[i][1]):
            max = res
    return max[0]




def classify_with_NNR(data_trn, data_vld, data_tst) -> List:
    start = time.time()
    print(f'starting classification with {data_trn}, {data_vld}, and {data_tst}')
    df_train = pd.read_csv(data_trn)
    df_valid=pd.read_csv(data_vld)
    df_test=pd.read_csv(data_tst)
    x=df_train.drop(['class'], axis=1)
    y=df_train['class']
    X_valid = df_valid.drop(['class'], axis=1)
    y_valid = df_valid['class']
    x_test=df_test.drop(['class'], axis=1)
    y_test=df_test['class']
    scaler=StandardScaler()

    x_train_scaled=pd.DataFrame(scaler.fit_transform(x))
    X_valid_scaled=pd.DataFrame(scaler.fit_transform(X_valid))
    X_test_scaled=pd.DataFrame(scaler.fit_transform(x_test))

    list_ofx_valid=X_valid_scaled.values.tolist()
    list_ofy_valid=list(y_valid)

    list_ofx_train=x_train_scaled.values.tolist()
    list_ofy_train=list(y)

    list_ofx_test=X_test_scaled.values.tolist()
    list_ofy_test = list(y_test)

    counts = Counter(list_ofy_valid)
    defualt_class= counts.most_common(1)[0][0]
    r=trainModel(list_ofx_valid,list_ofy_valid,defualt_class)

    predictions = pred_for_test(r,list_ofx_test,list_ofx_train,list_ofy_train,defualt_class)  # todo: return a list of your predictions for test instances
    print(f'1 -total time: {round(time.time()-start, 0)} sec')

    return predictions

if __name__ == '__main__':
    start = time.time()

    with open('config.json', 'r', encoding='utf8') as json_file:
        config = json.load(json_file)

    predicted = classify_with_NNR(config['data_file_train'],
                                  config['data_file_validation'],
                                  config['data_file_test'])

    df = pd.read_csv(config['data_file_test'])
    labels = df['class'].values

    if not predicted:  # empty prediction, should not happen in your implementation
        predicted = list(range(len(labels)))

    assert(len(labels) == len(predicted))  # make sure you predict label for all test instances
    print(f'test set classification accuracy: {accuracy_score(labels, predicted)}')

    print(f'1 -total time: {round(time.time()-start, 0)} sec')
