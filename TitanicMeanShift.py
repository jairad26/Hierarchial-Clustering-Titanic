##https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


##pclass 1Passenger Class(1 = 1st; 2 = 2nd; 3 = 3rd)
##survived Survival (0 = No; 1 = Yes)
##name Name
##sex Sex
##age Age
##sibsp Number of Siblings/Spouses Aboard
##parch Number of Parents/Children Aboard
##ticket Ticket Number
##fare Passenger Fare (British pound)
##cabin Cabin
##embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
##boat Lifeboat
##body Body Identification Number
##home.dest Home/Destination

df = pd.read_excel('titanic.xls')
original_df = pd.DataFrame.copy(df)

df.drop(['pclass', 'name', 'sex', 'sibsp', 'parch', 'ticket', 'cabin', 'embarked',
         'boat', 'body', 'home.dest'], 1, inplace=True)
df._convert(numeric = True)
df.fillna(0, inplace = True)

def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

df = handle_non_numerical_data(df)
print(df.head())

##df.drop(['ticket','home.dest'], 1, inplace=True)

##X = np.array(df.drop(['survived'], 1).astype(float))
X = np.array(df)
##X = preprocessing.scale(X)
##y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan

for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i]

n_clusters_ = len(np.unique(labels))

survival_rates = {}
for i in range(n_clusters_):
    temp_df = original_df[(original_df['cluster_group']==float(i))]
    survival_cluster = temp_df[(temp_df['survived']==1)]
    survival_rate = len(survival_cluster)/len(temp_df)
    survival_rates[i] = survival_rate

print(survival_rates)




print(cluster_centers)
print("Number of estimated clusters:", n_clusters_)

colors = 10*['r','g','b','c','k','y','m']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(X)):
    ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]],marker='o')

ax.scatter(cluster_centers[:,0],cluster_centers[:,1],cluster_centers[:,2],
           marker = "x",color = 'k', s = 150,linewidths = 5, zorder = 10)

plt.show()




















































































