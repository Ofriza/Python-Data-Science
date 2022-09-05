# Ofri Zadok 208668269
# Maya David 209282532

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat

df = pd.read_csv('clubmed_HW2.csv')

# Q1a
plt.hist(df.age)
plt.xlabel("age")
plt.ylabel("frequency")
plt.title("ages")
plt.show()

# Q1b
# Build histogram with only 5 bins
plt.hist(df.age, bins=5)
plt.xlabel("age")
plt.ylabel("frequency")
plt.title("ages with 5 bins")
plt.show()

# Build histogram with 50 bins
plt.hist(df.age, bins=50)
plt.xlabel("age")
plt.ylabel("frequency")
plt.title("ages with 50 bins")
plt.show()

# From the first graph it can be learned that most people are between the ages of 35-50
# and there are very few people aged 85-100
# From the second graph you can learn about the real age of most of the guests -35
# It can also be seen that the elderly are aged 95-100

# Q2
count = df.club_member.value_counts()
x = np.array(["true", "false"])
plt.bar(x, count, color='r')
plt.title("club member count")
plt.xlabel("club membership")
plt.ylabel("count")
plt.show()

# Q3
# show the column before the log
plt.hist(df.nights)
plt.title("nights before log")
plt.show()
# show the column after the log
df['logNights'] = np.log10(df['nights'])
plt.hist(df.logNights)
plt.title("nights after log")
plt.show()
# The Log transformation did not help, because after that we got a graph with an abnormal distribution.
# In the new graph we have 2 peaks

# Q4a
cros_age_stat = pd.crosstab(df['sex'], df['status'],normalize=True)
print("----------------------------------")
print(cros_age_stat)

# Q4b
cros_stat_sex = pd.crosstab(df['status'], df['sex'], normalize=True)
print("----------------------------------")
print(cros_stat_sex)

# Q4c
barplot1 = cros_stat_sex.plot.bar(stacked =True, title="Distribution of marital status by sex")
plt.show()
barplot2 = cros_age_stat.plot.bar(stacked =True, title="Distribution of sex by marital status")
plt.show()

# In marital status couple the percentage of men is the largest.
# the quantity of men in the same status is not newcomer the highest if the percentage is the highest
# The most common marital status among women is couple
# Percentage of married women is 70%
# Percentage of men in total singles is 50%

# Q4d
cros4 = pd.crosstab(df['sex'], df['club_member'],normalize=True)
print("----------------------------------")
print(cros4)

barplot4 = cros4.plot.bar(rot=0, title="club member vs sex")
plt.show()
# most of the man have club member, most of the women dont have club member

# Q4e
# we can see that the sex vs club member ,
# males are more likely to have club member.
# most of the women on the other hand doesn't have a club member, and we can see that clear connection in the graph.

# Q5
x = np.array(df.minibar)
y = np.array(df.age)

plt.scatter(x, y, c='green')
plt.xlabel("mini bar")
plt.title("mini bar vs age")
plt.ylabel("age")
plt.show()

# Q6
df2 = df[df['room_price'].notna()]
q1 = df2.room_price.quantile(0.25)
q2 = df2.room_price.median()
q3 = df2.room_price.quantile(0.75)
print("the first quainter is:", q1)
print("the threed quainter is:", q3)
print(df2.room_price. mean())
iqr = q3 - q1
print("the IQR is:", iqr)
print("the STD is: ", df2.room_price.std())

# Q6b
count = 0
new_df = df2.room_price.copy()
for index, row in new_df.iteritems():
    if new_df[index] <= q2:
        count += 1
print("----------------------------------")
print("Number of values under the median is:", count)
# the new table contains 195 rows and the mid value is 98.
# our count value is deviate, because of the null values

# Q6c
plt.hist(df.room_price)
std1 = q2 + df.room_price.std()
std2 = q2 - df.room_price.std()
plt.axvline(x=std1, color="red")
plt.axvline(x=std2, color="red")
plt.axvline(x=q2, color="green")
plt.show()

# Q6d
# The distribution is approximately normal, with a tilt to the left

# Q6e
df.boxplot(column=['age'], by='ranking', grid=False)
plt.axhline(y=98)
plt.axhline(y=22)
plt.show()
# for ranking 2 the IQR is the biggest.

# Q6f
df.boxplot(column=['age'], by='visits5years', grid=False)
plt.show()
# The visit number is 8 ,the IQR is between 47-55 years old.

# Q6g
df.boxplot(column=['room_price'], by='visits5years', grid=False)
plt.show()
# The people in this category has the highest room price

# Q6h
x = np.array(df.ranking)
y = np.array(df.total_expenditure)
plt.scatter(x, y)
plt.xlabel("ranking vs total_expenditure")
plt.title("ranking")
plt.ylabel("total_expenditure")
plt.show()
# We cant see a clear trend connection because the highest spender is rank 4

# Q7
df['visit_in_2016'] = df['visits2016'].replace(to_replace=np.nan, value=-1)
df['visit_in_2016'] = df['visit_in_2016'].mask(df['visit_in_2016'] > 0, "visit")
df['visit_in_2016'] = df['visit_in_2016'].replace(to_replace=0, value="not visit")
df['visit_in_2016'] = df['visit_in_2016'].replace(to_replace=-1, value="not join yet")
print("----------------------------------")
print(df['visit_in_2016'].describe())

# Q8a
# replace the negative values with NaN.
df['total_expenditure2'] = df['total_expenditure'].mask(df['total_expenditure'] < 0)
# replace the missing values with the mean
df['total_expenditure2'] = df['total_expenditure2'].replace(to_replace=np.nan, value=df.total_expenditure2.mean())
# make the quantiles
q1 = df.total_expenditure2.quantile(0.25)
q2 = df.total_expenditure2.median()
q3 = df.total_expenditure2.quantile(0.75)
q4 = df.total_expenditure2.quantile(1)

bins = [df.total_expenditure2.min(), q1, q2, q3, q4]
labels = ["First", "second", "Third", "Fourth"]
df['total_expenditure_new'] = pd.cut(df['total_expenditure2'], bins=bins, labels=labels)

# Q8b
# In order to maintain the form of the distribution, it is advisable to replace the missing values in the mean
# Because the median is more affective from unusual values

# Q8c
the_std = df.total_expenditure2.std()
mean = df.total_expenditure2.mean()

std1 = mean + the_std
std2 = mean + (the_std * 2)
std3 = mean + (the_std * 3)
std4 = mean - the_std
std5 = mean - (the_std * 2)
std6 = mean - (the_std * 3)

bins = [std6, std5, std4, mean, std1, std2, std3]
labels = ["between -3 - -2 std", "between -2 - -1 std", "between -1 - 0", "between 0 - 1", "between 1 - 2", "between "
                                                                                                            "2 - 3"]
df['total_expenditure_new2'] = pd.cut(df['total_expenditure2'], bins=bins, labels=labels)
print("----------------------------------")
print(df['total_expenditure_new2'].describe())
# 2 guests wont been included in the partition

# Q9a
zscores = stat.zscore(df.minibar)
print("The STD before normalize", df.minibar.std())
print("The STD after normalize", zscores.std())

# Q9c
miniBarStd = df.minibar.mean() + df.minibar.std()
miniBarStd2 = df.minibar.mean() - df.minibar.std()
minibarBin = [miniBarStd2, miniBarStd]
minibarLabel = ["minibar"]
df["miniBarNORM"] = pd.cut(df["minibar"], labels=minibarLabel, bins=minibarBin)
print("The number of typical values is:", df["miniBarNORM"].count())
