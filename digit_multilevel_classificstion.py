import matplotlib.pyplot as plt
from sklearn.datasets import load_digits 
import pandas as pd

digits = load_digits()
print(dir(digits))# show how many columns are there with there name.

print(digits.data[0])#to show the data

#plt.gray()#to show the image 1 to 5
#for i in range(5):
#     plt.matshow(digits.images[i])
#     plt.show()

    
print(digits.target[0:15])# it is show wich number is there.
#so because of that we know data and target are our train model.

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(digits.data,digits.target)


from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()


reg.fit(x_train, y_train)
print('accuracy',reg.score(x_test, y_test))
print('output:',reg.predict(digits.data[0:5]))#predict data 0 to 5


#################################    TEST OF THE MODEL  ##################################################
plt.matshow(digits.images[43])#pic a rendom number. like my rendom number is 43.
plt.show()

print(digits.target[43])# for cross check what is the real answer.

print('output of 43:',reg.predict([digits.data[43]]))

