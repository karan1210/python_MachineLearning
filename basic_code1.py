import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

#read the data
df = pd.read_csv("book1.csv")
print(df) # course is not say abt the print function bt we have to write.


#ploat the data in to scatter points
plt.xlabel('area')
plt.ylabel('price')
print(plt.scatter(df.area,df.price,color='red',marker='+'))
#df.plot() this function is show grap not point.
plt.show()



#machine learning train model
reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)# fit cmd for train command

k=reg.predict([[3300]])#where 3300 is input
print(k)

#math y= mx + b (m is coef and intercept is b)

print(reg.coef_)

print(reg.intercept_)


