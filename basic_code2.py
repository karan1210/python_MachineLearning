#import libraries
import pandas as pd
import numpy as np
from sklearn import linear_model
import math


#read file
df = pd.read_csv('homeprice.csv')
print(df)

#cmplt the data
k= math.floor(df.bedrooms.median())#mean is equal to 4.
print('mediann of the raw',k)

df.bedrooms = df.bedrooms.fillna(k)#fill the blank with median always remember that df. replace the frame
print(df)

#train model

reg = linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)


#coef and inter
print(reg.coef_)
print(reg.intercept_)

#input

print('home price with 3 bedrooms',reg.predict([[3000,3,40]]))
print('home price with 4 bedrooms',reg.predict([[2500,4,5]]))      
