##read the data

import pandas as pd
df = pd.read_csv('titanic.csv')
print(df)
print(dir(df))

#modify the data
inputs = df.drop(['PassengerId', 'Survived', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked', 'Unnamed: 12'], axis = 1)
print(inputs)
target = df['Survived']
print(target)



#train data

from sklearn.preprocessing import LabelEncoder
le_Pclass = LabelEncoder()
le_Sex=LabelEncoder()
le_Age=LabelEncoder()
le_Fare=LabelEncoder()

###fit the data in to structure
inputs['Pclass_n'] = le_Pclass.fit_transform(inputs['Pclass'])
inputs['Fare_n'] = le_Fare.fit_transform(inputs['Fare'])
inputs['Sex_n'] = le_Sex.fit_transform(inputs['Sex'])
inputs['Age_n'] = le_Age.fit_transform(inputs['Age'])
####print('output is heere',inputs.head())
inputs_n = inputs.drop(['Pclass','Sex' , 'Fare', 'Sex' ,'Age'], axis = 1)
print(inputs_n)



from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(inputs_n, target)


print(model.predict([[0,0,0,0]]))
