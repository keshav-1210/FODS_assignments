import numpy as np
import pandas as pd
def entropy(X):
    uniq = set(X)
    ans = 0.0
    for x in uniq:
        count = X.count(x)
        prob = count / len(X)
        ans += prob * np.log2(prob)
    return -ans

def feat_selec(X, Y, k):
    X=pd.DataFrame(X)
    ans_dict = {}
    for i, column in enumerate(X.columns):
        ig = entropy(list(Y)) + entropy(list(X[column])) - entropy(list(zip(X[column], Y)))
        ans_dict[i + 1] = ig 
    sorted_features = sorted(ans_dict, key=ans_dict.get, reverse=True)[:k]
    return sorted_features


first_line = input().strip().split()
n, m, k = int(first_line[0]), int(first_line[1]), int(first_line[2])

X_data = []
for _ in range(n):
    row = list(map(int, input().strip().split()))
    X_data.append(row)
X = pd.DataFrame(X_data)

Y = list(map(int, input().strip().split()))
selected_features = feat_selec(X, Y, k)
print(selected_features)

# X = np.array([
#     [1,0,1,0],
#     [0,1,1,1],
#     [1,1,0,0],
#     [0,0,1,1],
# ])
# y = np.array([0,1,1,0])
# k=2

