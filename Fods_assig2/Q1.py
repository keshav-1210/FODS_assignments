import numpy as np
p = []
q = []
fl=input()
i=0
while sum(p) < float(1):
    x = float(fl.strip().split()[i])
    if x<0:
        raise ValueError("non negative only")
    p.append(x)
    i=i+1
if sum(p) != 1:
    raise ValueError("sum not 1")
    
sl=input()
i=0
while sum(q) <float(1):
    x = float(sl.strip().split()[i])
    q.append(x)
    i=i+1
if sum(q) != 1:
    print("Invalid input")
p=np.array(p)
q=np.array(q)
# p = [0.1, 0.3, 0.4, 0.1, 0.1]
# q = [0.2, 0.2, 0.3, 0.2, 0.1]
def KL_div(p,q):
    ans=0
    for x, y in zip(p,q):
        if x!=0:
            ans+=x*np.log2(x/y)
    return ans

print(f"{KL_div(p,q):.4f}")