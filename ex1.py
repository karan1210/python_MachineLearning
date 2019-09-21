import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

#read file
df = pd.read_csv("canada_income.csv")
print(df)

#plot

plt.xlabel('per_cap_income')
plt.ylabel('year')
plt.scatter(df.per_cap_income,df.year,color='red',marker='+')
plt.show()

#train
new_df = df.drop('per_cap_income',axis='columns')
print(new_df)

price = df.per_cap_income
print(price)

reg = linear_model.LinearRegression()
reg.fit(new_df,price)
#.fit is for train model df.[['year']] is only input and other side output is df.per_cap_income so that's why reg.fit(df.[['year']],df.per_cap_income).
#if u get errorr in this syntex you can seprate data frame like as above.
############### when you apply .fit for seprate ==== no df. model use.

k=reg.predict([[2020]])
print(k)

print(reg.coef_)
print(reg.intercept_)
