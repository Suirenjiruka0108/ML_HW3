import numpy
import random
import math


def data_raed(filename):
    f = open(filename, 'r')
    data = []
    y = []
    num = 0
    for line in f.readlines():
        D = []
        D.extend(map(numpy.double, line.split('	')))
        if D[10] == 1.0:
            y.append([1.0])
        else:
            y.append([-1.0])
        D.pop()
        data.append(D)
    num = len(data)
    return num, data, y

def data_transform(data, num, Q):
    Tdata = []
    for i in range(num):
        D = []
        D.append(1)
        for j in range(1, Q + 1):
            for k in range(10):
                D.append(math.pow(data[i][k], j))
        Tdata.append(D)
    return Tdata

def data_fullorder_transform(data, num):
    Tdata = []
    for i in range(num):
        D = []
        D.append(1)
        for j in range(10):
            D.append(data[i][j])
        for j in range(10):
            for k in range(j, 10):
                D.append(data[i][j] * data[i][k])
        Tdata.append(D)
    return Tdata

def down_transform(data, num, Q):
    Tdata = []
    for i in range(num):
        D = []
        D.append(1)
        for j in range(Q):
            D.append(data[i][j])
        Tdata.append(D)
    return Tdata

def random_down_transform(data, num, random):
    Tdata = []
    for i in range(num):
        D = []
        D.append(1)
        for j in random:
            D.append(data[i][j])
        Tdata.append(D)
    return Tdata

def shuffle():
    possible = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in range(3):
        for j in range(10):
            r = random.randint(0, 100) % 10
            temp = possible[r]
            possible[r] = possible[j]
            possible[j] = temp
    rand = possible[4:9]
    return rand

def liner_regression(data, y):
    MT = numpy.transpose(data)          #Ein liner regression
    X = numpy.dot(MT, data)
    X = numpy.linalg.inv(X)
    W = numpy.dot(X, MT)
    W =numpy.dot(W, y)
    predict = numpy.dot(data, W)
    return W, predict


def error_zo(data, y):
    error = 0
    for i in range(len(data)):
        #error += (predict[i][0] - yin[i][0]) * (predict[i][0] - yin[i][0])   13題
        if data[i][0] * y[i][0] < 0:       #14題
            error += 1
    E = error / len(data)
    return E


Ein = 0
Eout = 0
ELout = 0
num, raw_data, y= data_raed("./ML_hw3/hw3_train.dat")
transform_data = data_transform(raw_data, num, 2)   #Q(最後一個變數調整transform的次方)
#transform_data = data_fullorder_transform(raw_data, num)   #fullorder 2 polynomial transformation
test_num, out_data, yout= data_raed("./ML_hw3/hw3_test.dat")
transform_out_data = data_transform(out_data, test_num, 2)
#transform_out_data = data_fullorder_transform(out_data, test_num)
#以上是獲以上是獲取test 和 training data，並對其trnasform

Win, predict = liner_regression(transform_data, y)
predictout = numpy.dot(transform_out_data, Win)


Ein = error_zo(predict, y)
Eout = error_zo(predictout, yout)
print(Ein, Eout, abs(Ein - Eout))


min = 0                                       #以下是15題的10個transform找最小的計算
min_value = 1
for i in range(1, 11):
    Tdata = down_transform(raw_data, num, i)
    out_Tdata = down_transform(out_data, test_num, i)
    W, P = liner_regression(Tdata, y)
    Pout = numpy.dot(out_Tdata, W)
    Ein = error_zo(P, y)
    Eout =error_zo(Pout, yout)
    if abs(Ein - Eout) < min_value:
        min = i
        min_value = abs(Ein - Eout)

print(min, min_value)

avg_E = 0
for i in range(200):
    r = shuffle()                #產生隨機不重複5個數
    Tdata = random_down_transform(raw_data, num, r)
    out_Tdata = random_down_transform(out_data, test_num, r)
    W, P = liner_regression(Tdata, y)
    Pout = numpy.dot(out_Tdata, W)
    Ein = error_zo(P, y)
    Eout =error_zo(Pout, yout)
    avg_E += abs(Ein - Eout)
avg_E = avg_E / 200
print(avg_E)

#12. (b)0.32
#13. (d)0.45
#14. (a)0.33
#15. (c)3
#16. (d)0.21