
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import utils
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

RSEED = 123

df = pd.read_csv("votersdata.csv")

# Q2
# getting to know the data
print(df.head())
print(df.shape)
print(df.isnull().sum())
print('-------------------------------------------')

# cross tab with vote
passTime_cross_vote = pd.crosstab(df['passtime'], df['vote'], normalize=True)
passPlot = passTime_cross_vote.plot.bar(stacked=True, title="passtime")
plt.show()

status_cross_vote = pd.crosstab(df['status'], df['vote'], normalize=True)
statPlot = status_cross_vote.plot.bar(stacked=True, title="status")
plt.show()

# Q2b

df.boxplot(column=['age'], by='vote', grid=False)
plt.show()

df.boxplot(column=['salary'], by='vote', grid=False)
plt.show()

df.boxplot(column=['volunteering'], by='vote', grid=False)
plt.show()

# Q3
# fill the missing values
df['age'] = df['age'].fillna(value=df.age.mean())
df['salary'] = df['salary'].fillna(value=df.salary.mean())
# drop missing values
df = df[df['passtime'].notna()]

df = utils.shuffle(df, random_state=RSEED)

# Q4

# we need to encode the categorical columns
le_vote = LabelEncoder()
le_vote.fit(df['vote'])
df['new_vote'] = le_vote.transform(df['vote'])

le = LabelEncoder()
le.fit(df['status'])
df['new_status'] = le.transform(df['status'])

le.fit(df['passtime'])
df['new_passtime'] = le.transform(df['passtime'])

le.fit(df['sex'])
df['new_sex'] = le.transform(df['sex'])

# drop the old columns and build new data frame
df2 = df.drop(['passtime', 'sex', 'status', 'vote'], axis=1)

# split the data to features and target
X = df2.drop('new_vote', axis=1)
y = df2['new_vote']

# split the data to train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RSEED)

# Q5
# build the model
model = DecisionTreeClassifier(random_state=RSEED)
model.fit(X_train, y_train)

# plot a tree
plt.figure(figsize=(15, 20), dpi=100)
plot_tree(model, filled=True, feature_names=X.columns, class_names=le_vote.inverse_transform(model.classes_),
          fontsize=8)
plt.show()

# Q6
y_pred_test = model.predict(X_test)
# build a confusion matrix for our test set data
cm_test = pd.crosstab(y_test, y_pred_test, colnames=['pred'], margins=True)
print("Q6 confusion matrix")
print(cm_test)
plot_confusion_matrix(model, X_test, y_test)
plt.show()

# show the results for the test set
print("Test set results:")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_test))
print("Precision:", metrics.precision_score(y_test, y_pred_test))
print("recall:", metrics.recall_score(y_test, y_pred_test))
print('-------------------------------------------')

y_pred_train = model.predict(X_train)
# build a confusion matrix for our train set data
cm_train = pd.crosstab(y_train, y_pred_train, colnames=['pred'], margins=True)
print(cm_train)

# show the results for the train set
print("Train set results:")
print("Accuracy:", metrics.accuracy_score(y_train, y_pred_train))
print("Precision:", metrics.precision_score(y_train, y_pred_train))
print("recall:", metrics.recall_score(y_train, y_pred_train))
print('-------------------------------------------')

# Q7
# there is overfitting in this model. the Accuracy,Precision,recall is 1 in the train set
# and in the test set this values are lower. There is a big difference between those values.

# Q8
# limit the tree
model = DecisionTreeClassifier(max_depth=5, min_samples_split=40)
model.fit(X_train, y_train)
y_pred_test2 = model.predict(X_test)
y_pred_train2 = model.predict(X_train)
plt.figure(figsize=(8, 5), dpi=100)

# plot the new tree after limitation
plot_tree(model, filled=True, feature_names=X.columns, class_names=le_vote.inverse_transform(model.classes_),
          fontsize=8)
plt.show()

# 8A
# the tree depth is 5

# 8B
# the tree has 8 leaves

# 8C
# the best feature for the tree splits is volunteering

# 8D
# The features that are not in the model: sex , passtime

# 8E
col = np.column_stack((y_test, y_pred_test2))
print('observation number 68:')
print(col[68])
print('-------------------------------------------')

# the 68 observation classification is correct.

# Q9
# prediction for test set
y_pred_test2 = model.predict(X_test)
# build a confusion matrix for our model
cm_test2 = pd.crosstab(y_test, y_pred_test2, colnames=['pred'], margins=True)
print("the test matrix is:")
print(cm_test2)
print('-------------------------------------------')

# show the results for the test set
print("New model Test set results:")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_test2))
print("Precision:", metrics.precision_score(y_test, y_pred_test2))
print("recall:", metrics.recall_score(y_test, y_pred_test2))
print('-------------------------------------------')

# prediction for train set
y_pred_train2 = model.predict(X_train)
# build a confusion matrix for our model

# show the results for the train set
print("New model Train set results:")
print("Accuracy:", metrics.accuracy_score(y_train, y_pred_train2))
print("Precision:", metrics.precision_score(y_train, y_pred_train2))
print("recall:", metrics.recall_score(y_train, y_pred_train2))
print('-------------------------------------------')
cm_train2 = pd.crosstab(y_train, y_pred_train2, colnames=['pred'], margins=True)
print("the train matrix is:")
print(cm_train2)
print('-------------------------------------------')

# Q10
# from Maor's prediction we can conclude a few things:
# 1. in this model there is no overfitting because in the train set there are values that are
# lower than 1, and the difference between the test and the train measures is very low
# 2. the recall measure is 0.94 which means that out of all the positives results the model found
# almost all the results as positives


# Decision tree - Multiclass

# we need to encode the categorical columns
le_status = LabelEncoder()
le_status.fit(df['status'])
df['new_status'] = le_status.transform(df['status'])

le2 = LabelEncoder()
le2.fit(df['vote'])
df['new_vote'] = le2.transform(df['vote'])
le2.fit(df['passtime'])
df['new_passtime'] = le2.transform(df['passtime'])
le2.fit(df['sex'])
df['new_sex'] = le2.transform(df['sex'])

# drop the old columns and build new data frame
df2 = df.drop(['passtime', 'sex', 'status', 'vote'], axis=1)

# split the data to features and target
s_X = df2.drop('new_status', axis=1)
s_y = df2['new_status']

# split the data to train set and test set
s_X_train, s_X_test, s_y_train, s_y_test = train_test_split(s_X, s_y, test_size=0.3, random_state=RSEED)

# Q10
model.fit(s_X_train, s_y_train)
plt.figure(figsize=(15, 20), dpi=100)
# plot a tree
plot_tree(model, filled=True, feature_names=s_X.columns, class_names=le_status.inverse_transform(model.classes_),
          fontsize=8)
plt.show()

s_y_pred = model.predict(s_X_test)
print("the accuracy on test set is:", accuracy_score(s_y_test, s_y_pred))

s_y_pred = model.predict(s_X_train)
print("the accuracy on train set is:", accuracy_score(s_y_train, s_y_pred))
print('-------------------------------------------')

# the accuracy on test set is: 0.5315315315315315
# the accuracy on train set is: 0.6640926640926641
# the model will not predict the status as he should
# cause he have low accuracy

y_pred_test2 = model.predict(s_X_test)
# build a confusion matrix for our model
cm_test3 = pd.crosstab(s_y_test, y_pred_test2, colnames=['pred'], margins=True)
print("the test matrix is:")
print(cm_test3)

# Q11
# single = 2
tp = cm_test3.iloc[2, 2]
totalP = cm_test3.iloc[2, 2] + cm_test3.iloc[0, 2] + cm_test3.iloc[1, 2]
print("presicion:", tp / totalP)
