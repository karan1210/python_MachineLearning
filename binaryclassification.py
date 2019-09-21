import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
df = pd.read_csv('insurance.csv')
print(df)

#plot
plt.scatter(df.age,df.bought_insurance,marker='+',color='red')
plt.show()

#train import

x_train, x_test, y_train, y_test = train_test_split(df[['age']],df.bought_insurance,train_size=0.9)
print('x test',x_test)
print('x train',x_train)
print('y test',y_test)
print('y train',y_train)

model = LogisticRegression()
model.fit(x_train, y_train)


print('output of x_test:',model.predict(x_test))
print('accuracy',model.score(x_test,y_test))
print('probability :',model.predict_proba(x_test))#prob of not buy  product or buy product.
print(model.predict([[56]]))#when thw errorr like model is not run or dimenation error you can simply solve this by adding brakets.
#it consider that 56 age and that's why index array reqaire.
