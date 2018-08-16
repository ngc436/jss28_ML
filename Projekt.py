import pandas as pd
import math
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from scipy.io import savemat
from mpl_toolkits import mplot3d
from sklearn.metrics import classification_report

COLUMN_NAMES = ['T_xacc', 'T_yacc', 'T_zacc', 'T_xgyro', 'T_ygyro', 'T_zgyro', 'T_xmag', 'T_ymag', 'T_zmag',
                'RA_xacc', 'RA_yacc', 'RA_zacc', 'RA_xgyro', 'RA_ygyro', 'RA_zgyro', 'RA_xmag', 'RA_ymag', 'RA_zmag',
                'LA_xacc', 'LA_yacc', 'LA_zacc', 'LA_xgyro', 'LA_ygyro', 'LA_zgyro', 'LA_xmag', 'LA_ymag', 'LA_zmag',
                'RL_xacc', 'RL_yacc', 'RL_zacc', 'RL_xgyro', 'RL_ygyro', 'RL_zgyro', 'RL_xmag', 'RL_ymag', 'RL_zmag',
                'LL_xacc', 'LL_yacc', 'LL_zacc', 'LL_xgyro', 'LL_ygyro', 'LL_zgyro', 'LL_xmag', 'LL_ymag', 'LL_zmag']

ADDITIONAL = COLUMN_NAMES + ['Action', 'Subject', 'Segment']


def create_df(dir_name):
    list_of_frames = []
    for subdir, dirs, files in os.walk(dir_name):
        for file in files:
            # print(subdir[6:8])
            df = pd.read_csv(os.path.join(subdir, file), header=None, names=COLUMN_NAMES)
            df['Action'] = subdir[6:8]
            df['Subject'] = subdir[10:11]
            df['Segment'] = file[1:3]
            print(file[1:3])
            list_of_frames.append(df)
    result = pd.concat(list_of_frames)
    result.apply(pd.to_numeric)
    return result


def save_to_csv(df, fname):
    df.to_csv(fname)


def main():
    new_df = create_df("data")
    save_to_csv(new_df, "prepared.csv")
    df = pd.read_csv('prepared.csv')
    print(df.head())


if __name__ == '__main__':
    main()

# CODE MICHAEL
df = pd.read_csv('prepared.csv')
df_train = df[['T_xacc', 'Action', 'Subject', 'Segment']]


# f_train.sort_values(['Action', 'Segment'], ascending=[1, 1])

# DTW
def DTWDistance(s1, s2):
    DTW = {}

    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

    return math.sqrt(DTW[len(s1) - 1, len(s2) - 1])


df_train
origin = df_train['T_xacc'][df['Action'] == 1][df['Subject'] == 1][df['Segment'] == 1]

ts = df_train['T_xacc'][df['Action'] == 1][df['Subject'] == 1][df['Segment'] == 2]
test = df_train['T_xacc'][df['Action'] == 3][df['Subject'] == 1][df['Segment'] == 2]
DTWDistance(origin.values, ts.values)
DTWDistance(origin.values, test.values)


def distance(df):
    origin = df['T_xacc'][df['Action'] == 1][df['Subject'] == 1][df['Segment'] == 1]
    df['distance'] = ''
    for i in range(1, 20):
        for j in range(1, 61):
            for k in range(1, 9):
                df['distance'][df['Action'] == i][df['Segment'] == j][df['Subject'] == k] = DTWDistance(origin.values,
                df['T_xacc'][df['Action'] == i][df['Subject'] == k][df['Segment'] == j].values)
    return df


def distance_one(df):
    origin = df['T_xacc'][df['Action'] == 1][df['Subject'] == 1][df['Segment'] == 1]
    distance = pd.DataFrame(index=list(range(0, 1140)), columns=['feature', 'distance'])
    for i in range(1, 20):
        for j in range(1, 61):
            test = df['T_xacc'][df['Action'] == i][df['Subject'] == 1][df['Segment'] == j]
            distance['distance'][(i - 1) * 60 + j - 1] = DTWDistance(origin.values, test.values)
            distance['feature'][(i - 1) * 60 + j - 1] = i
    return distance


df['distance'][df['Action'] == 1][df['Segment'] == 1][df['Subject'] == 1] = DTWDistance(origin.values, df['T_xacc'][
    df['Action'] == 10][df['Subject'] == 1][df['Segment'] == 10].values)

result = distance_one(df_train)
result['feature'] = np.mod(result['feature'], 100)
pd.DataFrame(columns=['aA', 'B'])
df_train['Segment'][10]

plt.scatter(result['distance'], result['feature'])  # arguments are passed to np.histogram
plt.show()

# three dimension
df_train = df[['T_xacc', 'T_yacc', 'T_zacc', 'Action', 'Subject', 'Segment']]
df_train

def distance_three(df):
    origin = df[['T_xacc', 'T_yacc', 'T_zacc']][df['Action'] == 1][df['Subject'] == 1][df['Segment'] == 1]
    distance = pd.DataFrame(index=list(range(0, 1140)),
                            columns=['observation', 'distance_x', 'distance_y', 'distance_z'])
    for i in range(1, 20):
        for j in range(1, 61):
            test = df['T_xacc'][df['Action'] == i][df['Subject'] == 1][df['Segment'] == j]
            distance['distance_x'][(i - 1) * 60 + j - 1] = DTWDistance(origin['T_xacc'].values, test.values)
            test = df['T_yacc'][df['Action'] == i][df['Subject'] == 1][df['Segment'] == j]
            distance['distance_y'][(i - 1) * 60 + j - 1] = DTWDistance(origin['T_yacc'].values, test.values)
            test = df['T_zacc'][df['Action'] == i][df['Subject'] == 1][df['Segment'] == j]
            distance['distance_z'][(i - 1) * 60 + j - 1] = DTWDistance(origin['T_zacc'].values, test.values)
            distance['observation'][(i - 1) * 60 + j - 1] = i
    return distance


result = distance_three(df_train)
ax = plt.axes(projection='3d')
colors = result['observation']
ax.scatter(result['distance_x'], result['distance_y'], result['distance_z'], c=colors)

# NORMALISED
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(result[['distance_x', 'distance_y', 'distance_z']])
result_normalised = pd.DataFrame(np_scaled)
result_normalised['observation'] = result['observation']
ax = plt.axes(projection='3d')
colors = result_normalised['observation']
ax.scatter(result_normalised[0], result_normalised[1], result_normalised[2], c=colors)

result_normalised.columns = ['x', 'y', 'z', 'observation']
savemat('result_normalised.mat', result_normalised)

pd.to_csv(result_normalised)

# Train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(result_normalised[[0, 1, 2]], result_normalised['observation'],
                                                    test_size=0.2, random_state=42)

# KNeighbors
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, list(y_train))
neigh.predict(X_test)
neigh.score(X_test, list(y_test))
list(y_test) - neigh.predict(X_test)
# Parameter tuning
from sklearn.neighbors import KNeighborsClassifier
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# The list of hyper-parameters we want to optimize. For each one we define the bounds,
# the corresponding scikit-learn parameter name, as well as how to sample values
# from that dimension (`'log-uniform'` for the learning rate)
space = [Integer(1, 5, name='n_'),
         Real(10 ** -5, 10 ** 0, "log-uniform", name='learning_rate'),
         Integer(1, n_features, name='max_features'),
         Integer(2, 100, name='min_samples_split'),
         Integer(1, 100, name='min_samples_leaf')]


# this decorator allows your objective function to receive a the parameters as
# keyword arguments. This is particularly convenient when you want to set scikit-learn
# estimator parameters
@use_named_args(space)
def objective(**params):
    reg.set_params(**params)

    return -np.mean(cross_val_score(reg, X, y, cv=5, n_jobs=-1,
                                    scoring="neg_mean_absolute_error"))

# first activity
person_data = {}
for i in range(8):
    person_data[i] = []
    for j in range(45):
        person_data[i] += [list(df[COLUMN_NAMES[j]][df['Action'] == 1][df['Subject'] == i].values)]


person_data[1][1]

origin = df[COLUMN_NAMES][df['Action'] == 1][df['Subject'] == 1]
origin[COLUMN_NAMES[0]]
origin.shape[0]

import vis

vis.plot_timeseries(person_data)

