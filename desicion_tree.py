import pandas as pd



####get data
df = pd.read_csv("desicion_tree.csv")
#print(df)



#####set the data like inputs[drop the output line]  &  target[only output] output which we need 
inputs = df.drop('salary_more_then_100k',axis='columns')
target = df['salary_more_then_100k']
##print(inputs)
##print(target)


from sklearn.preprocessing import LabelEncoder
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()


inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])
print(inputs.head())
inputs_n = inputs.drop(['company','job','degree','Unnamed: 4'],axis='columns')
print(inputs_n)
####train the model
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(inputs_n, target)
###########################################[2,1,1] means 2 is for company 1 is for job and another 1 is for degree.
#print(model.predict([[2,1,1]]))

