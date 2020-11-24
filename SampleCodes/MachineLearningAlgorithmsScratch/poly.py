# Author: Abouzar Ghavami
# Email: ghavamip@gmail.com
# This code is protected by copyright laws in US.
# Please do not reuse in any format without permission of Abouzar Ghavami.

## Equivalent with:
## from sklearn.preprocessing import PolynomialFeatures
## p = PolynomialFeatures(degree = n)  # e.g.: n = 3
## y = p.fit_transform(X)
def poly(k, n):
    if k == 1:
        return [[n]]
    if k > 0:
        s = []
        for i in range(n + 1):
            r = poly(k - 1, n - i)
            for y in r:
                x = y + [i]
                s.append(x)
        return s

def xpoly(X, n):
    k = len(X[0])
    y = []
    for x in X:
        s = []
        for i in range(n + 1):
            p = poly(k, i)
            for r in p:
                b = [x[j] ** r[j] for j in range(k)]
                a = 1
                for t in b:
                    a = a * t
                s.append(a)
        y.append(s)
    return y

print(poly(2, 3))
X = [[1, 2], [5, 6]]
print(xpoly(X, 3))
