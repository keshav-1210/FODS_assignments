import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
data = pd.read_csv('./House_Price_Prediction.csv')
#preprocess
bool_columns = ['has_basement', 'perfect_condition', 'has_lavatory', 'single_floor','renovated','nice_view']
for col in bool_columns:
    data[col] = data[col].astype(int)

data['year'] = pd.to_datetime(data['date']).dt.year
data['day']=pd.to_datetime(data['date']).dt.day
data.drop(columns=['date'], inplace=True)

# print(data.isna().sum())
Y=data['price']
scaler=MinMaxScaler()
for c in data.columns:
    if c!='price' and c not in bool_columns:
        data[c]=scaler.fit_transform(data[[c]])

def best_weight(X, Y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    w = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)
    return w

def mse(X, Y, w):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    Y_pred = np.dot(X_b, w)
    return np.mean((Y - Y_pred) ** 2)

def train_test(X):
    X = X.sample(frac=1,random_state=42).reset_index(drop=True)
    rows = X.shape[0]
    train_index = int(rows * 0.8)
    Y = X["price"]
    X = X.drop(['price'], axis=1)
    Y_train = Y[:train_index]
    X_train = X[:train_index]
    Y_test = Y[train_index:]
    X_test = X[train_index:]
    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test=train_test(data)
# print(data)
spears_man ={}
for c in data.columns:
        if c!='price':
            spears_man[c]=stats.spearmanr(data[c],data['price']).statistic

key=list(spears_man.keys()) 
sort_sp=np.argsort(list(spears_man.values()))
sorted_f=[key[i] for i in sort_sp]
sorted_f=sorted_f[::-1]
current_feat=[]
selected_feat=[]
last_mse=float('inf')
for f in sorted_f:
    current_feat.append(f)
    w=best_weight(X_train[current_feat],Y_train)
    Y_pred=np.dot(np.c_[np.ones((X_test.shape[0], 1)), X_test[current_feat]], w)
    mse_f=mse(X_test[current_feat],Y_test,w)
    if mse_f<last_mse:  
        selected_feat=current_feat
        last_mse=mse_f
    # else:
    #     break
print(f"Selected features: {selected_feat} with len ={len(selected_feat)} mse {last_mse}")
