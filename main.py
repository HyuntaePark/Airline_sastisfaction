# main.py

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import module

warnings.filterwarnings('ignore')

# print("pandas version: ", pd.__version__)
# pd.set_option('display.max_row', 500)
pd.set_option('display.max_columns', 100)

# read dataset
df = pd.read_csv('train.csv', encoding='utf-8')
df = df.drop(['Unnamed: 0'], 1)
df = df[(df['Age'] < 40) & (df['Age'] >= 20)]
# feature selection
# drop id column which is useless
df = df.drop(['id'], 1)

print(df.isnull().sum())
print()
print(df.head())
print()
print(df.describe())
print()
df = df.dropna(axis=0)
print(df.isnull().sum())

# Set target column name
target = 'satisfaction'

# dirty value detection
for i in df.columns:
    if len(np.unique(df[i])) > 10:
        continue
    print('{} : {}'.format(i, np.unique(df[i])))

sns.countplot(df[target], palette='Paired')
plt.show()

df = df.sample(20000)

# Arrival Delay in Minutes column has NaN values
# fill with mean value
df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].mean(), inplace=True)

# check null value
# print(df.isnull().sum())

# data curation
categorical_df = df.select_dtypes(include='object').drop([target], 1)
for i in categorical_df:
    sns.countplot(df[i], palette='Paired')
    plt.show()

# Show correlation heatmap
module.showHeatmap(df)

# Test your dataframe, target column, set test size and number of subsets to cross validate,
# and the parameters (list of Encoders, list of Scalers)
# For this module you can find out the best combination of which encoder and which scaler
# works well and can predict best and can have best result using diverse of algorithms.
module.diverseTrainingResult(df, target, 0.3, 10,
                             ['LabelEncoder', 'OneHotEncoder'],
                             ['MinMaxScaler', 'MaxAbsScaler','StandardScaler','RobustScaler'])



