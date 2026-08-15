import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv('tourism_project/data/tourism.csv')
df.drop(columns = ['Unnamed: 0','CustomerID'],inplace = True)

cat_col = list(df.select_dtypes(['object']).columns)
num = list(df.select_dtypes(['number']).columns)
df[num] = df[num].astype('float32')
df = pd.get_dummies(df,cat_col,drop_first = True, dtype=int)

X = df.drop(columns = ['ProdTaken'])
y = df['ProdTaken']
xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size = 0.2,random_state = 42)

xtrain.to_csv('tourism_project/data/xtrain.csv',index=False)
xtest.to_csv('tourism_project/data/xtest.csv',index=False)
ytrain.to_csv('tourism_project/data/ytrain.csv',index=False)
ytest.to_csv('tourism_project/data/ytest.csv',index=False)

print('data prepared')
