#################################################################################################
# Railway Systems by Age Groups  - Segmentation
#################################################################################################

## Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

## Reading Data

data = pd.read_csv("yas_grubuna_göre_rayli_sistemler.csv")
df = data.copy()


## Descriptive Statistics

df.describe().T
df.info()
df.dtypes
df.columns
df.index
df.shape

## Preparing Data and Preprocessing

df.isnull().values.any()
df.isnull().sum()
df.dropna(inplace = True)

del_vars = ["LONGITUDE", "LATITUDE"]
df.drop(del_vars, axis = 1, inplace = True)
df.head()

df["GENDER"] = ["MALE" if i == 1 else "FEMALE" for i in df["GENDER"]]
df.head()

# del_rows = df.loc[(df["transaction_Year"] == 2020) & (df["transaction_Month"].isin([4,5])),:]
# df.drop(del_rows, axis = 0, inplace = True)

df1 = df.loc[(df["transaction_Year"] == 2020) & (df["transaction_Month"].isin([1,2,3])),:]
df2 = df.loc[df["transaction_Year"] == 2019]
df = pd.concat([df1, df2], ignore_index = True)
df.tail()


## Analyzing and Data Visualization

df["GENDER"].value_counts()
df["GENDER"].value_counts().plot(kind = "bar", rot = 20)
plt.show()

df.groupby(["transaction_Year", "transaction_Month"]).agg({"PASSENGER_COUNT": "mean"})

df.groupby(["GENDER","transaction_Month", "transaction_Year"]).agg({"PASSENGER_COUNT": ["mean", "sum"]})

### Correlation analysis

df.corr()
sns.pairplot(df)
sns.pairplot(df, kind = "reg")

### "Line" Frequencies

lines_bplot = sns.countplot(x = df["LINE"], data = df)
lines_bplot.set_xticklabels(lines_bplot.get_xticklabels(), rotation=45)
plt.show()

### "PASSENGER_COUNT" Normality and Outlier Detection Analyze

df["PASSENGER_COUNT"].describe().T
print(df["PASSENGER_COUNT"].skew())
print(df["PASSENGER_COUNT"].kurt())

plt.hist(df["PASSENGER_COUNT"])
plt.show()

plt.boxplot(df["PASSENGER_COUNT"])
plt.show()

sns.catplot(y = "PASSENGER_COUNT", kind = "violin", data = df)
plt.show()

### According to "Town" variable, detection of outliers for "PASSENGER_COUNT"

town_bplot = sns.boxplot(x = "TOWN", y = "PASSENGER_COUNT", data = df, orient = "v")
town_bplot.set_xticklabels(town_bplot.get_xticklabels(), rotation = 45)
plt.show()

### Detection of linear relationship for "transaction_Month" and "PASSENGER_COUNT"

sns.lmplot(x = "transaction_Month", y = "PASSENGER_COUNT", data = df)

### Time Series Analysis

tseries_df = df.copy()
tseries_df = tseries_df.dropna()
tseries_df.head(1)

tseries_df["Year-Month"] = [str(i[4]) + "-" + str(i[3]) for i in tseries_df.values]
tseries_df["Year-Month"] = pd.DatetimeIndex(tseries_df["Year-Month"])
sns.lineplot(x = "Year-Month", y = "PASSENGER_COUNT", data = tseries_df)
plt.xticks(fontsize = 14)
plt.show()

## Segmentation

df.columns
df.head(1)

agg_df = df[["GENDER", "TOWN", "age_group", "PASSENGER_COUNT"]]
agg_df.info()
agg_df.head()


agg_df["CARD"] = ["_".join(i[0:3]).upper() for i in agg_df.values]
agg_df.isnull().values.any()
agg_df.head()

## Identfying Segments and Values

agg_df_segmented = agg_df.pivot_table(index = "CARD", values = "PASSENGER_COUNT")
agg_df_segmented = agg_df_segmented.reset_index()

segment = pd.qcut(agg_df_segmented["PASSENGER_COUNT"], 8,
                  labels=["Eighth", "Seventh","Sixth", "Fifth", "Fourth", "Third", "Second", "First"])

agg_df_segmented["SEGMENT"] = segment
agg_df_segmented.head()

agg_df_segmented.groupby("SEGMENT").agg({"PASSENGER_COUNT": ["mean", "median", "max", "min"]})

## Prod System Usage

def segment_check():
    try:
        gender = input("Please enter the gender information").upper()
        town = input("Please enter the town information").upper()
        age_group = input("Please enter the age group information").upper()

        info_given = gender + "_" + town + "_" + age_group
        segment = agg_df_segmented.loc[agg_df_segmented["CARD"] == info_given].reset_index()
        print(" CARD = ",segment.values[0][1], "\n", "###########################","\n",
              "AVERAGE OF PASSENGER COUNT = ", segment.values[0][2],"\n", "###########################","\n",
              "SEGMENT INFORMATION = ", segment.values[0][3])
    except:
        print("###########################")
        print("Please enter correct inputs")
        print("###########################")

segment_check()
