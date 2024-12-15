import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
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
scaler=StandardScaler()
for c in data.columns:
    if c!='price' and c not in bool_columns:
        data[c]=scaler.fit_transform(data[[c]])

def best_weight(X, Y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    w = np.linalg.inv(X_b.T@X_b)@X_b.T@Y
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

rem_feauture=list(X_train.columns)
selected_feat=[]
last_mse=float('inf')
# model=LinearRegression()
while rem_feauture:

    best_feat=None
    for f in rem_feauture:
        curr_feat=selected_feat+[f]
        w=best_weight(X_train[curr_feat],Y_train)
        Y_pred=np.dot(np.c_[np.ones((X_test.shape[0], 1)), X_test[curr_feat]], w)
        mse_f=mse(X_test[curr_feat],Y_test,w)
        if mse_f<last_mse:
                best_feat=f
                last_mse=mse_f
    if best_feat:
            selected_feat=selected_feat+[best_feat]
            rem_feauture.remove(best_feat)
    else:
         break
# print(last_mse)
print(f"selected feat for forw are {len(selected_feat)} with mse {last_mse} ")
for i in selected_feat: 
    print(i)


#now we will do reverse feature selection
back_feat=list(X_train.columns)
last_mse=float('inf')
while back_feat:
     current_mse=float('inf')
     for f in back_feat:
          curr_feat=back_feat.copy()
          curr_feat.remove(f)
          w=best_weight(X_train[curr_feat],Y_train)
          Y_pred=np.dot(np.c_[np.ones((X_test.shape[0], 1)), X_test[curr_feat]], w)
          mse_f=mse(X_test[curr_feat],Y_test,w)
          if current_mse>mse_f:
                best_feat=f
                current_mse=mse_f

     if current_mse<last_mse:
            last_mse=current_mse
            back_feat.remove(best_feat)
     else:
         break
print(f"selected feat for bck are {len(back_feat)} with mse {last_mse} ")
for i in back_feat: 
    print(i)