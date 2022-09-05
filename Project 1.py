
import pandas as pd
import numpy as np

df = pd.read_csv('customerData.csv')
print("Question #1A")
v1 = np.arange(0, 101, 10)  # vector from 1 to 100 included with steps of 10
print(v1)
print("-----------------------------------------------")
print("Question #1B")
v2 = np.linspace(1, 100, 8)    # vector from 1 to 100 included with 8 numbers
print(v2)
print("-----------------------------------------------")
print("Question #1C")
print(v2.reshape(2, 4))         # reshape the vector to mat
print("-----------------------------------------------")
print("Question #2")
print(df.describe())  # Q2      # statistic data
print(df.shape)  # Q2A          # size of the data frame

# Q2B
# numeric : custid , age , num_vehicles ,income
# Categorical : state_of_res , sex , is_employed , marital_stat , health_ins , housing_type , recent_move

# Q2C
# coustid is made in order to be the private key, to identify every row by unique number . it doesn't represent anything about the user

# Q3
print("-----------------------------------------------")
print("Question #3")
df2 = df.iloc[0::10, 0::2]      #The double columns and rows that are double of 10.
print(df2)

# Q4A
print("-----------------------------------------------")
print("Question #4A")
print(df2.shape)

# Q4B
print("-----------------------------------------------")
print("Question #4B")
print(df2.size)

# Q4C
# Shape represents the number of rows and columns of the table.
# Size represents the number of cells in the table that we get as a result of multiplication between rows and columns
# It is possible to calculate the size when given the shape by multiplication

# Q5
print("-----------------------------------------------")
print("Question #5")
df3 = df[(df.age > 38) & (df.age <= 50)]
print(df3)

# Q6A
print("-----------------------------------------------")
print("Question #6A")
df4 = df[df.age > 50]           #custumers older than 50
print(df4[['custid', 'age', 'num_vehicles', 'income']])

# Q6B
print("-----------------------------------------------")
print("Question #6B")
print(df4.iloc[0:, [0, 3, 8, 9]])

# Q7
print("-----------------------------------------------")
print("Question #7")
print(df4.iloc[0:100, [9]])     #100 first costomers , show only age


# Q8
print("-----------------------------------------------")
df8 = df[df.age <= 18 & df['marital_stat'].isin(['Married', 'Divorced/Separated'])].iloc[0:, [0]]
print("Question #8")
print(df8)

# Q9A
print("-----------------------------------------------")
print("Question #9A")
df9a = df[(df['state_of_res'] == 'Washington') & (df['income'] >= 16000)].age.mean()
print("The mean age is:", df9a)

# Q9B
print("-----------------------------------------------")
print("Question #9B")
df9b = df[(df['state_of_res'] == 'Washington') & (df['income'] >= 16000)].age.max()
print("The oldest age is:", df9b)

# Q9C
print("-----------------------------------------------")
print("Question #9C")
df9c = df[(df['state_of_res'] == 'Washington') & (df['income'] >= 16000)].income.min()
print("The minimum income is:", df9c)

# Q9D
print("-----------------------------------------------")
print("Question #9D")
df9d = len(df[(df['state_of_res'] == 'Washington') & (df['income'] >= 16000)])
print("The number of people in the group:", df9d)

# Q10
print("-----------------------------------------------")
print("Question #10A")

# build pivot to see the housing type for every sex
df10 = df.groupby(['sex', 'housing_type'])[['housing_type']].count()
df10.rename(columns={"housing_type": "amount"}, inplace=True)
df10 = df10.reset_index('sex')
df10 = df10.pivot(columns='sex', values='amount')

print(df10)

print("the most frequent housing type for females is:", df.groupby('sex')['housing_type'].describe().top[0])

print("Question #10B")
print("the most frequent housing type for males is:", df.groupby('sex')['housing_type'].describe().top[1])
