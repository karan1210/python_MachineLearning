import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression

df = pd.read_csv('car_price.csv')
print(df)


df.drop('Unnamed: 3',axis='columns',inplace=True)
print(df)


#plot
plt.xlabel('Mileage')
plt.ylabel('Sell Price($)')
plt.scatter(df['Mileage'],df['Sell Price($)'])
plt.show()
plt.scatter(df['Age(yrs)'],df['Sell Price($)'])
plt.show()


#for train data
x = df[['Mileage','Age(yrs)']]#seprate the train data
y = df[['Sell Price($)']]#input perameter

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4)#split train & test, 0.3 is ratio of test size so, here 30% test size 70% train.
#we get max accuracy at 0.4 ration approximetly 92.5.ss


print(x_train)
print(x_test)
print(y_test)
print(y_train)

reg = LinearRegression()
reg = reg.fit(x_train, y_train)

print('output :',reg.predict(x_test))
print('accurcy :',reg.score(x_test, y_test))




