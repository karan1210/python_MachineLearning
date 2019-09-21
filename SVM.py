import pandas as pd
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
iris = load_iris()

print(iris.feature_names)
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()


df['target'] = iris.target
df.head()


df['flower_name'] =df.target.apply(lambda x: iris.target_names[x])
df.head()

df0=df[:50]
df1=df[50:100]
df2=df[100:150]

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color="green",marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],color="blue",marker='.')


plt.xlabel('petal Length')
plt.ylabel('petal Width')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'],color="green",marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'],color="blue",marker='.')


from sklearn.model_selection import train_test_split
x=df.drop(['target','flower_name'],axis=1)
y=df.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.svm import SVC
model=SVC(gamma=0.22)



print(model.fit(x_train, y_train))
print(model.score(x_test,y_test))
print(model.predict([[4.8,3.0,1.5,0.3]]))
