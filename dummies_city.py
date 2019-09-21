import pandas as pd
df = pd.read_csv('city.csv')
print(df)

dummies=pd.get_dummies(df.town)
print(dummies)

marged=pd.concat([df,dummies],axis='columns')
print(marged)

#drop method
marged.drop('town',axis='columns',inplace=True)


marged.drop('west windsor',axis='columns',inplace=True)


marged.drop('Unnamed: 3',axis='columns',inplace=True)
print(marged)

ss

#train the data

X = marged.drop('price',axis='columns')
y = marged.price

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,y)

#predication

print(model.predict([[2800,0,1]])) ## 2800 sqr ft home in robbinsville
print(model.predict([[3400,0,0]])) # 3400 sqr ft home in west windsor

