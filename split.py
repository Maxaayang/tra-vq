import pickle

train =[]
for i in range(697):
    # print(i)
    train.append(str(i) + '.pkl')

with open ("./new_pickles/train_pieces.pkl", 'wb') as f: #打开文件
    pickle.dump(train, f) #用 dump 函数将 Python 对象转成二进制对象文件
# pickle
# print(train)

test = []
val = []

for i in range(40):
    test.append(str(i + 696) + '.pkl')
    val.append(str(i + 735) + '.pkl')
    
with open ("./new_pickles/test_pieces.pkl", 'wb') as f: #打开文件
    pickle.dump(test, f) #用 dump 函数将 Python 对象转成二进制对象文件

with open ("./new_pickles/val_pieces.pkl", 'wb') as f: #打开文件
    pickle.dump(val, f) #用 dump 函数将 Python 对象转成二进制对象文件