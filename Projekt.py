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
    #new_df = create_df("data")
    #save_to_csv(new_df, "prepared.csv")
    df =  pd.read_csv('prepared.csv')
    print(df.head())



if __name__ == '__main__':
    main()

# CODE MICHAEL
    df =  pd.read_csv('prepared.csv')
    df_train = df[['T_xacc', 'Action', 'Subject', 'Segment']]
    #f_train.sort_values(['Action', 'Segment'], ascending=[1, 1])

# DTW
def DTWDistance(s1, s2):
    DTW={}

    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return math.sqrt(DTW[len(s1)-1, len(s2)-1])

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
                df['distance'][df['Action'] == i][df['Segment'] == j][df['Subject'] == k] = DTWDistance(origin.values, df['T_xacc'][df['Action'] == i][df['Subject'] == k][df['Segment'] == j].values)
    return df

def distance_one(df):
    origin = df['T_xacc'][df['Action'] == 1][df['Subject'] == 1][df['Segment'] == 1]
    distance = pd.DataFrame(index=list(range(0, 1140)), columns=['feature', 'distance'])
    for i in range(1, 20):
        for j in range(1, 61):
            test = df['T_xacc'][df['Action'] == i][df['Subject'] == 1][df['Segment'] == j]
            distance['distance'][(i-1)*60 + j-1] = DTWDistance(origin.values, test.values)
            distance['feature'][(i-1)*60 + j-1] = i
    return distance

df['distance'][df['Action'] == 1][df['Segment'] == 1][df['Subject'] == 1] = DTWDistance(origin.values, df['T_xacc'][df['Action'] == 10][df['Subject'] == 1][df['Segment'] == 10].values)

result = distance_one(df_train)
result['feature'] = np.mod(result['feature'], 100)
pd.DataFrame(columns=['aA', 'B'])
df_train['Segment'][10]

plt.scatter(result['distance'], result['feature'])  # arguments are passed to np.histogram
plt.show()

# three dimension
df_train = df[['T_xacc', 'T_yacc', 'T_zacc', 'Action', 'Subject', 'Segment']]

def distance_three(df):
    origin = df[['T_xacc', 'T_yacc', 'T_zacc']][df['Action'] == 1][df['Subject'] == 1][df['Segment'] == 1]
    distance = pd.DataFrame(index=list(range(0, 1140)), columns=['observation', 'distance_x', 'distance_y', 'distance_z'])
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
savemat('result.mat', result)

pd.to_csv(result_normalised)

# Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(result_normalised[[0, 1, 2]], result_normalised['observation'], test_size=0.2, random_state=42)
X_train.columns = ['x', 'y', 'z']

# KNeighbors
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, list(y_train))
neigh.predict(X_test)
neigh.score(X_test,list(y_test))
list(y_test) - neigh.predict(X_test)

# Parameter tuning
from sklearn.neighbors import KNeighborsClassifier
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from sklearn.model_selection import cross_val_score

# The list of hyper-parameters we want to optimize. For each one we define the bounds,
# the corresponding scikit-learn parameter name, as well as how to sample values
# from that dimension (`'log-uniform'` for the learning rate)
space = [Integer(1, 100, name='n_neighbors'),
         Categorical(['kd_tree', 'ball_tree', 'brute'], name='algorithm'),
         Categorical(['uniform', 'distance'], name='weights'),
         Integer(1, 3, name='p')]

# this decorator allows your objective function to receive a the parameters as
# keyword arguments. This is particularly convenient when you want to set scikit-learn
# estimator parameters
@use_named_args(space)
def objective(**params):
    reg = KNeighborsClassifier()
    reg.set_params(**params)

    return -np.mean(cross_val_score(reg, X_train[['x', 'y', 'z']], list(y_train), cv=5, n_jobs=-1,
                                    scoring="neg_mean_absolute_error"))

# Optimize
from skopt import gp_minimize
res_gp = gp_minimize(objective, space, n_calls=100, random_state=0)

"Best score=%.4f" % res_gp.fun

print("""Best parameters:
- max_depth=%d
- learning_rate=%.6f
- max_features=%d
- min_samples_split=%d
- min_samples_leaf=%d""" % (res_gp.x[0], res_gp.x[1],
                            res_gp.x[2], res_gp.x[3],
                            res_gp.x[4]))

# PLot of convergence
from skopt.plots import plot_convergence
plot_convergence(res_gp)

########
neigh = KNeighborsClassifier(n_neighbors=9, algorithm='kd_tree', weights='distance', p=1)
neigh.fit(X_train, list(y_train))
neigh.predict(X_test)
neigh.score(X_test,list(y_test))
# Score 0.8991228

################################3 Random Forest
from sklearn.ensemble import RandomForestClassifier
space = [Integer(1, 100, name='n_estimators'),
         Categorical(['gini', 'entropy'], name='criterion'),
         Integer(1, 100, name='max_depth'),
         Integer(2, 32, name='min_samples_split'),
         Integer(1, 4, name='min_samples_leaf'),
         Categorical(['False', 'True'], name='bootstrap')]

# this decorator allows your objective function to receive a the parameters as
# keyword arguments. This is particularly convenient when you want to set scikit-learn
# estimator parameters
@use_named_args(space)
def objective(**params):
    reg = RandomForestClassifier()
    reg.set_params(**params)

    return -np.mean(cross_val_score(reg, X_train[['x', 'y', 'z']], list(y_train), cv=5, n_jobs=-1,
                                    scoring="neg_mean_absolute_error"))

# Optimize
from skopt import gp_minimize
res_gp = gp_minimize(objective, space, n_calls=150, random_state=0)

"Best score=%.4f" % res_gp.fun

print("""Best parameters:
- max_depth=%d
- learning_rate=%.6f
- max_features=%d
- min_samples_split=%d
- min_samples_leaf=%d""" % (res_gp.x[0], res_gp.x[1],
                            res_gp.x[2], res_gp.x[3],
                            res_gp.x[4]))

# PLot of convergence
from skopt.plots import plot_convergence
plot_convergence(res_gp)

###########
rforest = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=69, min_samples_split=7, min_samples_leaf=1, bootstrap='True')
rforest.fit(X_train, list(y_train))
rforest.predict(X_test)
rforest.score(X_test,list(y_test))

################ Add more dimension / We should probably concentrate only on the accelerator
df.columns
df_train_9d = df[['T_xacc', 'T_yacc', 'T_zacc', 'RA_xacc', 'RA_yacc', 'RA_zacc','RL_xacc', 'RL_yacc', 'RL_zacc', 'Action', 'Subject', 'Segment']]

def distance_nine(df):
    origin = df[['T_xacc', 'T_yacc', 'T_zacc', 'RA_xacc', 'RA_yacc', 'RA_zacc', 'RL_xacc', 'RL_yacc', 'RL_zacc']][df['Action'] == 1][df['Subject'] == 1][df['Segment'] == 1]
    distance = pd.DataFrame(index=list(range(0, 1140)), columns=['Activity', 'distance_Tx', 'distance_Ty', 'distance_Tz', 'distance_Ax', 'distance_Ay', 'distance_Az', 'distance_Lx', 'distance_Ly', 'distance_Lz'])
    for i in range(1, 20):
        for j in range(1, 61):
            distance['distance_Tx'][(i - 1) * 60 + j - 1] = DTWDistance(origin['T_xacc'].values, df['T_xacc'][df['Action'] == i][df['Subject'] == 1][df['Segment'] == j].values)
            distance['distance_Ty'][(i - 1) * 60 + j - 1] = DTWDistance(origin['T_yacc'].values, df['T_yacc'][df['Action'] == i][df['Subject'] == 1][df['Segment'] == j].values)
            distance['distance_Tz'][(i - 1) * 60 + j - 1] = DTWDistance(origin['T_zacc'].values, df['T_zacc'][df['Action'] == i][df['Subject'] == 1][df['Segment'] == j].values)
            distance['distance_Ax'][(i - 1) * 60 + j - 1] = DTWDistance(origin['RA_xacc'].values, df['RA_xacc'][df['Action'] == i][df['Subject'] == 1][df['Segment'] == j].values)
            distance['distance_Ay'][(i - 1) * 60 + j - 1] = DTWDistance(origin['RA_yacc'].values, df['RA_yacc'][df['Action'] == i][df['Subject'] == 1][df['Segment'] == j].values)
            distance['distance_Az'][(i - 1) * 60 + j - 1] = DTWDistance(origin['RA_zacc'].values, df['RA_zacc'][df['Action'] == i][df['Subject'] == 1][df['Segment'] == j].values)
            distance['distance_Lx'][(i - 1) * 60 + j - 1] = DTWDistance(origin['RL_xacc'].values, df['RL_xacc'][df['Action'] == i][df['Subject'] == 1][df['Segment'] == j].values)
            distance['distance_Ly'][(i - 1) * 60 + j - 1] = DTWDistance(origin['RL_yacc'].values, df['RL_yacc'][df['Action'] == i][df['Subject'] == 1][df['Segment'] == j].values)
            distance['distance_Lz'][(i - 1) * 60 + j - 1] = DTWDistance(origin['RL_zacc'].values, df['RL_zacc'][df['Action'] == i][df['Subject'] == 1][df['Segment'] == j].values)
            distance['Activity'][(i - 1) * 60 + j - 1] = i
    return distance

result_9d = distance_nine(df_train_9d)
#ax = plt.axes(projection='3d')
#colors = result_9d['observation']
#ax.scatter(result_9d['distance_x'], result_9d['distance_y'], result_9d['distance_z'], c=colors)

# NORMALISED
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(result_9d[['distance_Tx', 'distance_Ty', 'distance_Tz', 'distance_Ax', 'distance_Ay', 'distance_Az', 'distance_Lx', 'distance_Ly', 'distance_Lz']])
result_normalised_9d = pd.DataFrame(np_scaled)
result_normalised_9d['Activity'] = result_9d['Activity']
#ax = plt.axes(projection='3d')
#colors = result_normalised_9d['Activity']
#ax.scatter(result_normalised_9d[0], result_normalised_9d[1], result_normalised_9d[2], c=colors)

# Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(result_normalised_9d[[0, 1, 2, 3, 4, 5, 6, 7, 8]], result_normalised_9d['Activity'], test_size=0.2, random_state=42)
X_train.columns = ['Tx', 'Ty', 'Tz', 'Ax', 'Ay', 'Az', 'Lx', 'Ly', 'Lz']

#### KNeighbors
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, list(y_train))
neigh.predict(X_test)
neigh.score(X_test,list(y_test))
list(y_test) - neigh.predict(X_test)

# Parameter tuning
from sklearn.neighbors import KNeighborsClassifier
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from sklearn.model_selection import cross_val_score

# The list of hyper-parameters we want to optimize. For each one we define the bounds,
# the corresponding scikit-learn parameter name, as well as how to sample values
# from that dimension (`'log-uniform'` for the learning rate)
space = [Integer(1, 100, name='n_neighbors'),
         Categorical(['kd_tree', 'ball_tree', 'brute'], name='algorithm'),
         Categorical(['uniform', 'distance'], name='weights'),
         Integer(1, 3, name='p')]

# this decorator allows your objective function to receive a the parameters as
# keyword arguments. This is particularly convenient when you want to set scikit-learn
# estimator parameters
@use_named_args(space)
def objective(**params):
    reg = KNeighborsClassifier()
    reg.set_params(**params)

    return -np.mean(cross_val_score(reg, X_train[['Tx', 'Ty', 'Tz', 'Ax', 'Ay', 'Az', 'Lx', 'Ly', 'Lz']], list(y_train), cv=5, n_jobs=-1,
                                    scoring="accuracy"))

# Optimize
from skopt import gp_minimize
res_gp_KN = gp_minimize(objective, space, n_calls=100, random_state=0)

"Best score=%.4f" % res_gp_KN.fun

print("""Best parameters:
- max_depth=%d
- learning_rate=%.6f
- max_features=%d
- min_samples_split=%d
- min_samples_leaf=%d""" % (res_gp_KN.x[0], res_gp_KN.x[1],
                            res_gp_KN.x[2], res_gp_KN.x[3],
                            res_gp_KN.x[4]))

# PLot of convergence
from skopt.plots import plot_convergence
plot_convergence(res_gp)

# On the test set
neigh = KNeighborsClassifier(n_neighbors=1, algorithm='brute', weights='distance', p=2)
neigh.fit(X_train, list(y_train))
neigh.predict(X_test)
neigh.score(X_test,list(y_test))


######### Random Forest
from sklearn.ensemble import RandomForestClassifier
space = [Integer(1, 100, name='n_estimators'),
         Categorical(['gini', 'entropy'], name='criterion'),
         Integer(1, 100, name='max_depth'),
         Integer(2, 32, name='min_samples_split'),
         Integer(1, 4, name='min_samples_leaf'),
         Categorical(['False', 'True'], name='bootstrap')]

# this decorator allows your objective function to receive a the parameters as
# keyword arguments. This is particularly convenient when you want to set scikit-learn
# estimator parameters
@use_named_args(space)
def objective(**params):
    reg = RandomForestClassifier()
    reg.set_params(**params)

    return -np.mean(cross_val_score(reg, X_train[['Tx', 'Ty', 'Tz', 'Ax', 'Ay', 'Az', 'Lx', 'Ly', 'Lz']], list(y_train), cv=5, n_jobs=-1, scoring="accuracy"))

# Optimize
from skopt import gp_minimize
res_gp_RF = gp_minimize(objective, space, n_calls=150, random_state=0)

"Best score=%.4f" % res_gp_RF.fun

print("""Best parameters:
- max_depth=%d
- learning_rate=%.6f
- max_features=%d
- min_samples_split=%d
- min_samples_leaf=%d""" % (res_gp_RF.x[0], res_gp_RF.x[1],
                            res_gp_RF.x[2], res_gp_RF.x[3],
                            res_gp_RF.x[4]))

# PLot of convergence
from skopt.plots import plot_convergence
plot_convergence(res_gp_RF)

# On the test set
rforest = RandomForestClassifier(n_estimators=36, criterion='gini', max_depth=56, min_samples_split=2, min_samples_leaf=1, bootstrap='False')
rforest.fit(X_train, list(y_train))
rforest.predict(X_test)
rforest.score(X_test,list(y_test))


##### Neural Netowork
from sklearn.neural_network import MLPClassifier
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from sklearn.model_selection import cross_val_score

# The list of hyper-parameters we want to optimize. For each one we define the bounds,
# the corresponding scikit-learn parameter name, as well as how to sample values
# from that dimension (`'log-uniform'` for the learning rate)
space = [Categorical(['identity', 'logistic', 'tanh', 'relu'], name='activation'),
         Categorical(['lbfgs', 'sgd', 'adam'], name='solver'),
         Categorical(['constant', 'invscaling', 'adaptive'], name='learning_rate'),
         Real(10 ** -6, 10 ** -2, "log-uniform", name='alpha'),
         Integer(2, 200, name='hidden_layer_sizes')]

# this decorator allows your objective function to receive a the parameters as
# keyword arguments. This is particularly convenient when you want to set scikit-learn
# estimator parameters
@use_named_args(space)
def objective(**params):
    reg = MLPClassifier()
    reg.set_params(**params)

    return -np.mean(cross_val_score(reg, X_train[['Tx', 'Ty', 'Tz', 'Ax', 'Ay', 'Az', 'Lx', 'Ly', 'Lz']], list(y_train), cv=5, n_jobs=7,
                                    scoring="accuracy"))

# Optimize
from skopt import gp_minimize
res_gp_NN = gp_minimize(objective, space, n_calls=100, random_state=0)

"Best score=%.4f" % res_gp_NN.fun

print("""Best parameters:
- max_depth=%d
- learning_rate=%.6f
- max_features=%d
- min_samples_split=%d
- min_samples_leaf=%d""" % (res_gp_NN.x[0], res_gp_NN.x[1],
                            res_gp_NN.x[2], res_gp_NN.x[3],
                            res_gp_NN.x[4]))

# PLot of convergence
from skopt.plots import plot_convergence
plot_convergence(res_gp_NN)

# On the test set
neural = MLPClassifier(activation='identity', solver='lbfgs', learning_rate='invscaling', alpha=0.01, hidden_layer_sizes=200)
neural.fit(X_train, list(y_train))
neural.predict(X_test)
neural.score(X_test,list(y_test))


#####
################ 15d - all accelerator divices
df.columnsdf_train_15d = df[['T_xacc', 'T_yacc', 'T_zacc', 'RA_xacc', 'RA_yacc', 'RA_zacc', 'RL_xacc', 'RL_yacc', 'RL_zacc', 'LA_xacc', 'LA_yacc', 'LA_zacc',  'LL_xacc', 'LL_yacc', 'LL_zacc', 'Action', 'Subject', 'Segment']]

def distance_fifteen(df, df_9d):
    origin = df[['T_xacc', 'T_yacc', 'T_zacc', 'RA_xacc', 'RA_yacc', 'RA_zacc', 'RL_xacc', 'RL_yacc', 'RL_zacc', 'LA_xacc', 'LA_yacc', 'LA_zacc',  'LL_xacc', 'LL_yacc', 'LL_zacc']][df['Action'] == 1][df['Subject'] == 1][df['Segment'] == 1]
    distance = pd.DataFrame(index=list(range(0, 1140)), columns=['distance_Ax', 'distance_Ay', 'distance_Az', 'distance_Lx', 'distance_Ly', 'distance_Lz'])
    for i in range(1, 20):
        for j in range(1, 61):
            distance['distance_Tx'][(i - 1) * 60 + j - 1] = DTWDistance(origin['T_xacc'].values, df['T_xacc'][df['Action'] == i][df['Subject'] == 1][df['Segment'] == j].values)
            distance['distance_Ty'][(i - 1) * 60 + j - 1] = DTWDistance(origin['T_yacc'].values, df['T_yacc'][df['Action'] == i][df['Subject'] == 1][df['Segment'] == j].values)
            distance['distance_Tz'][(i - 1) * 60 + j - 1] = DTWDistance(origin['T_zacc'].values, df['T_zacc'][df['Action'] == i][df['Subject'] == 1][df['Segment'] == j].values)
            distance['distance_Ax'][(i - 1) * 60 + j - 1] = DTWDistance(origin['RA_xacc'].values, df['RA_xacc'][df['Action'] == i][df['Subject'] == 1][df['Segment'] == j].values)
            distance['distance_Ay'][(i - 1) * 60 + j - 1] = DTWDistance(origin['RA_yacc'].values, df['RA_yacc'][df['Action'] == i][df['Subject'] == 1][df['Segment'] == j].values)
            distance['distance_Az'][(i - 1) * 60 + j - 1] = DTWDistance(origin['RA_zacc'].values, df['RA_zacc'][df['Action'] == i][df['Subject'] == 1][df['Segment'] == j].values)
            distance['distance_Lx'][(i - 1) * 60 + j - 1] = DTWDistance(origin['RL_xacc'].values, df['RL_xacc'][df['Action'] == i][df['Subject'] == 1][df['Segment'] == j].values)
            distance['distance_Ly'][(i - 1) * 60 + j - 1] = DTWDistance(origin['RL_yacc'].values, df['RL_yacc'][df['Action'] == i][df['Subject'] == 1][df['Segment'] == j].values)
            distance['distance_Lz'][(i - 1) * 60 + j - 1] = DTWDistance(origin['RL_zacc'].values, df['RL_zacc'][df['Action'] == i][df['Subject'] == 1][df['Segment'] == j].values)
            distance['Activity'][(i - 1) * 60 + j - 1] = i
    return distance

result_9d = distance_nine(df_train_9d)
#ax = plt.axes(projection='3d')
#colors = result_9d['observation']
#ax.scatter(result_9d['distance_x'], result_9d['distance_y'], result_9d['distance_z'], c=colors)

# NORMALISED
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(result_9d[['distance_Tx', 'distance_Ty', 'distance_Tz', 'distance_Ax', 'distance_Ay', 'distance_Az', 'distance_Lx', 'distance_Ly', 'distance_Lz']])
result_normalised_9d = pd.DataFrame(np_scaled)
result_normalised_9d['Activity'] = result_9d['Activity']
#ax = plt.axes(projection='3d')
#colors = result_normalised_9d['Activity']
#ax.scatter(result_normalised_9d[0], result_normalised_9d[1], result_normalised_9d[2], c=colors)

# Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(result_normalised_9d[[0, 1, 2, 3, 4, 5, 6, 7, 8]], result_normalised_9d['Activity'], test_size=0.2, random_state=42)
X_train.columns = ['Tx', 'Ty', 'Tz', 'Ax', 'Ay', 'Az', 'Lx', 'Ly', 'Lz']

#### KNeighbors
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, list(y_train))
neigh.predict(X_test)
neigh.score(X_test,list(y_test))
list(y_test) - neigh.predict(X_test)

# Parameter tuning
from sklearn.neighbors import KNeighborsClassifier
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from sklearn.model_selection import cross_val_score

# The list of hyper-parameters we want to optimize. For each one we define the bounds,
# the corresponding scikit-learn parameter name, as well as how to sample values
# from that dimension (`'log-uniform'` for the learning rate)
space = [Integer(1, 100, name='n_neighbors'),
         Categorical(['kd_tree', 'ball_tree', 'brute'], name='algorithm'),
         Categorical(['uniform', 'distance'], name='weights'),
         Integer(1, 3, name='p')]

# this decorator allows your objective function to receive a the parameters as
# keyword arguments. This is particularly convenient when you want to set scikit-learn
# estimator parameters
@use_named_args(space)
def objective(**params):
    reg = KNeighborsClassifier()
    reg.set_params(**params)

    return -np.mean(cross_val_score(reg, X_train[['Tx', 'Ty', 'Tz', 'Ax', 'Ay', 'Az', 'Lx', 'Ly', 'Lz']], list(y_train), cv=5, n_jobs=-1,
                                    scoring="accuracy"))

# Optimize
from skopt import gp_minimize
res_gp_KN = gp_minimize(objective, space, n_calls=100, random_state=0)

"Best score=%.4f" % res_gp_KN.fun

print("""Best parameters:
- max_depth=%d
- learning_rate=%.6f
- max_features=%d
- min_samples_split=%d
- min_samples_leaf=%d""" % (res_gp_KN.x[0], res_gp_KN.x[1],
                            res_gp_KN.x[2], res_gp_KN.x[3],
                            res_gp_KN.x[4]))

# PLot of convergence
from skopt.plots import plot_convergence
plot_convergence(res_gp)

# On the test set
neigh = KNeighborsClassifier(n_neighbors=1, algorithm='brute', weights='distance', p=2)
neigh.fit(X_train, list(y_train))
neigh.predict(X_test)
neigh.score(X_test,list(y_test))


######### Random Forest
from sklearn.ensemble import RandomForestClassifier
space = [Integer(1, 100, name='n_estimators'),
         Categorical(['gini', 'entropy'], name='criterion'),
         Integer(1, 100, name='max_depth'),
         Integer(2, 32, name='min_samples_split'),
         Integer(1, 4, name='min_samples_leaf'),
         Categorical(['False', 'True'], name='bootstrap')]

# this decorator allows your objective function to receive a the parameters as
# keyword arguments. This is particularly convenient when you want to set scikit-learn
# estimator parameters
@use_named_args(space)
def objective(**params):
    reg = RandomForestClassifier()
    reg.set_params(**params)

    return -np.mean(cross_val_score(reg, X_train[['Tx', 'Ty', 'Tz', 'Ax', 'Ay', 'Az', 'Lx', 'Ly', 'Lz']], list(y_train), cv=5, n_jobs=-1, scoring="accuracy"))

# Optimize
from skopt import gp_minimize
res_gp_RF = gp_minimize(objective, space, n_calls=150, random_state=0)

"Best score=%.4f" % res_gp_RF.fun

print("""Best parameters:
- max_depth=%d
- learning_rate=%.6f
- max_features=%d
- min_samples_split=%d
- min_samples_leaf=%d""" % (res_gp_RF.x[0], res_gp_RF.x[1],
                            res_gp_RF.x[2], res_gp_RF.x[3],
                            res_gp_RF.x[4]))

# PLot of convergence
from skopt.plots import plot_convergence
plot_convergence(res_gp_RF)

# On the test set
rforest = RandomForestClassifier(n_estimators=36, criterion='gini', max_depth=56, min_samples_split=2, min_samples_leaf=1, bootstrap='False')
rforest.fit(X_train, list(y_train))
rforest.predict(X_test)
rforest.score(X_test,list(y_test))


##### Neural Netowork
from sklearn.neural_network import MLPClassifier
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from sklearn.model_selection import cross_val_score

# The list of hyper-parameters we want to optimize. For each one we define the bounds,
# the corresponding scikit-learn parameter name, as well as how to sample values
# from that dimension (`'log-uniform'` for the learning rate)
space = [Categorical(['identity', 'logistic', 'tanh', 'relu'], name='activation'),
         Categorical(['lbfgs', 'sgd', 'adam'], name='solver'),
         Categorical(['constant', 'invscaling', 'adaptive'], name='learning_rate'),
         Real(10 ** -6, 10 ** -2, "log-uniform", name='alpha'),
         Integer(2, 200, name='hidden_layer_sizes')]

# this decorator allows your objective function to receive a the parameters as
# keyword arguments. This is particularly convenient when you want to set scikit-learn
# estimator parameters
@use_named_args(space)
def objective(**params):
    reg = MLPClassifier()
    reg.set_params(**params)

    return -np.mean(cross_val_score(reg, X_train[['Tx', 'Ty', 'Tz', 'Ax', 'Ay', 'Az', 'Lx', 'Ly', 'Lz']], list(y_train), cv=5, n_jobs=7,
                                    scoring="accuracy"))

# Optimize
from skopt import gp_minimize
res_gp_NN = gp_minimize(objective, space, n_calls=100, random_state=0)

"Best score=%.4f" % res_gp_NN.fun

print("""Best parameters:
- max_depth=%d
- learning_rate=%.6f
- max_features=%d
- min_samples_split=%d
- min_samples_leaf=%d""" % (res_gp_NN.x[0], res_gp_NN.x[1],
                            res_gp_NN.x[2], res_gp_NN.x[3],
                            res_gp_NN.x[4]))

# PLot of convergence
from skopt.plots import plot_convergence
plot_convergence(res_gp_NN)

# On the test set
neural = MLPClassifier(activation='identity', solver='lbfgs', learning_rate='invscaling', alpha=0.01, hidden_layer_sizes=200)
neural.fit(X_train, list(y_train))
neural.predict(X_test)
neural.score(X_test,list(y_test))







################### FOR ALL DATA / skip all data problem

# three dimension
df_train_all = df[['T_xacc', 'T_yacc', 'T_zacc', 'Action', 'Subject', 'Segment']]

origin = df[['T_xacc', 'T_yacc', 'T_zacc']][df['Action'] == 1][df['Subject'] == 1][df['Segment'] == 1]

def distance_three(df):
    distance = pd.DataFrame(index=list(range(0, 9120)), columns=['observation', 'distance_x', 'distance_y', 'distance_z'])
    for i in range(1, 20):
        for j in range(1, 61):
            for k in range(1, 9):
                test = df['T_xacc'][df['Action'] == i][df['Subject'] == k][df['Segment'] == j]
                distance['distance_x'][(i - 1) * 480 + (j - 1)*8 + (k-1)] = DTWDistance(origin['T_xacc'].values, test.values)
                test = df['T_yacc'][df['Action'] == i][df['Subject'] == k][df['Segment'] == j]
                distance['distance_y'][(i - 1) * 480 + (j - 1)*8 + (k-1)] = DTWDistance(origin['T_yacc'].values, test.values)
                test = df['T_zacc'][df['Action'] == i][df['Subject'] == k][df['Segment'] == j]
                distance['distance_z'][(i - 1) * 480 + (j - 1)*8 + (k-1)] = DTWDistance(origin['T_zacc'].values, test.values)
                distance['observation'][(i - 1) * 480 + (j - 1)*8 + (k-1)] = i
    return distance

origin = df[['T_xacc', 'T_yacc', 'T_zacc']][df['Action'] == 1][df['Subject'] == 1][df['Segment'] == 1]

def distance_three_test(df):
    distance = pd.DataFrame(index=list(range(0, 1140)), columns=['observation', 'distance_x', 'distance_y', 'distance_z'])
    for i in range(1, 20):
        for j in range(1, 61):
            test = df['T_xacc'][df['Action'] == i][df['Segment'] == j]
            distance['distance_x'][(i - 1) * 60 + j - 1] = DTWDistance(origin['T_xacc'].values, test.values)
            test = df['T_yacc'][df['Action'] == i][df['Segment'] == j]
            distance['distance_y'][(i - 1) * 60 + j - 1] = DTWDistance(origin['T_yacc'].values, test.values)
            test = df['T_zacc'][df['Action'] == i][df['Segment'] == j]
            distance['distance_z'][(i - 1) * 60 + j - 1] = DTWDistance(origin['T_zacc'].values, test.values)
            distance['observation'][(i - 1) * 60 + j - 1] = i
    return distance

import multiprocessing

def df_multiprocessing(df):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    list_of_data = []
    for i in range(1,9):
        list_of_data.append(df[['T_xacc', 'T_yacc', 'T_zacc', 'Action', 'Segment']][df['Subject'] == i])
    result = pool.map(distance_three_test, list_of_data)
    return list(result)

result_mult = df_multiprocessing(df)
result_mult = pd.concat(result_mult)
#################################
columns = ['T_xacc', 'T_yacc', 'T_zacc', 'RA_xacc', 'RA_yacc', 'RA_zacc', 'LA_xacc', 'LA_yacc', 'LA_zacc', 'RL_xacc', 'RL_yacc', 'RL_zacc', 'LL_xacc', 'LL_yacc', 'LL_zacc']

origin = df[columns][df['Action'] == 1][df['Subject'] == 1][df['Segment'] == 1]

def distance_15d(df):
    distance = pd.DataFrame(index=list(range(0, 1140)), columns=['Activity', 'Tx', 'Ty', 'Tz', 'RAx', 'RAy', 'RAz', 'LAx', 'LAy', 'LAz', 'RLx', 'RLy', 'RLz', 'LLx', 'LLy', 'LLz'])
    for i in range(1, 20):
        for j in range(1, 61):
            test = df['T_xacc'][df['Action'] == i][df['Segment'] == j]
            distance['Tx'][(i - 1) * 60 + j - 1] = DTWDistance(origin['T_xacc'].values, test.values)
            test = df['T_yacc'][df['Action'] == i][df['Segment'] == j]
            distance['Ty'][(i - 1) * 60 + j - 1] = DTWDistance(origin['T_yacc'].values, test.values)
            test = df['T_zacc'][df['Action'] == i][df['Segment'] == j]
            distance['Tz'][(i - 1) * 60 + j - 1] = DTWDistance(origin['T_zacc'].values, test.values)

            test = df['RA_xacc'][df['Action'] == i][df['Segment'] == j]
            distance['RAx'][(i - 1) * 60 + j - 1] = DTWDistance(origin['RA_xacc'].values, test.values)
            test = df['RA_yacc'][df['Action'] == i][df['Segment'] == j]
            distance['RAy'][(i - 1) * 60 + j - 1] = DTWDistance(origin['RA_yacc'].values, test.values)
            test = df['RA_zacc'][df['Action'] == i][df['Segment'] == j]
            distance['RAz'][(i - 1) * 60 + j - 1] = DTWDistance(origin['RA_zacc'].values, test.values)

            test = df['LA_xacc'][df['Action'] == i][df['Segment'] == j]
            distance['LAx'][(i - 1) * 60 + j - 1] = DTWDistance(origin['LA_xacc'].values, test.values)
            test = df['LA_yacc'][df['Action'] == i][df['Segment'] == j]
            distance['LAy'][(i - 1) * 60 + j - 1] = DTWDistance(origin['LA_yacc'].values, test.values)
            test = df['LA_zacc'][df['Action'] == i][df['Segment'] == j]
            distance['LAz'][(i - 1) * 60 + j - 1] = DTWDistance(origin['LA_zacc'].values, test.values)

            test = df['RL_xacc'][df['Action'] == i][df['Segment'] == j]
            distance['RLx'][(i - 1) * 60 + j - 1] = DTWDistance(origin['RL_xacc'].values, test.values)
            test = df['RL_yacc'][df['Action'] == i][df['Segment'] == j]
            distance['RLy'][(i - 1) * 60 + j - 1] = DTWDistance(origin['RL_yacc'].values, test.values)
            test = df['RL_zacc'][df['Action'] == i][df['Segment'] == j]
            distance['RLz'][(i - 1) * 60 + j - 1] = DTWDistance(origin['RL_zacc'].values, test.values)

            test = df['LL_xacc'][df['Action'] == i][df['Segment'] == j]
            distance['LLx'][(i - 1) * 60 + j - 1] = DTWDistance(origin['LL_xacc'].values, test.values)
            test = df['LL_yacc'][df['Action'] == i][df['Segment'] == j]
            distance['LLy'][(i - 1) * 60 + j - 1] = DTWDistance(origin['LL_yacc'].values, test.values)
            test = df['LL_zacc'][df['Action'] == i][df['Segment'] == j]
            distance['LLz'][(i - 1) * 60 + j - 1] = DTWDistance(origin['LL_zacc'].values, test.values)
            distance['Activity'][(i - 1) * 60 + j - 1] = i
    return distance

def df_multiprocessing(df):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    list_of_data = []
    for i in range(1, 9):
        list_of_data.append(df[['T_xacc', 'T_yacc', 'T_zacc', 'RA_xacc', 'RA_yacc', 'RA_zacc', 'LA_xacc', 'LA_yacc', 'LA_zacc', 'RL_xacc', 'RL_yacc', 'RL_zacc', 'LL_xacc', 'LL_yacc', 'LL_zacc', 'Action', 'Segment']][df['Subject'] == i])
    result = pool.map(distance_15d, list_of_data)
    return list(result)

result_15d = df_multiprocessing(df)
result_15d = pd.concat(result_15d)

# NORMALISED
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(result_15d[['Tx', 'Ty', 'Tz', 'RAx', 'RAy', 'RAz', 'LAx', 'LAy', 'LAz', 'RLx', 'RLy', 'RLz', 'LLx', 'LLy', 'LLz']])
result_normalised_15d = pd.DataFrame(np_scaled)
result_normalised_15d['Activity'] = result_9d['Activity']
#ax = plt.axes(projection='3d')
#colors = result_normalised_9d['Activity']
#ax.scatter(result_normalised_9d[0], result_normalised_9d[1], result_normalised_9d[2], c=colors)

result_15d.to_csv('data_15d.csv')
result_normalised_15d.to_csv('data_normalised_15d.csv')

# Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(result_normalised_9d[[0, 1, 2, 3, 4, 5, 6, 7, 8]], result_normalised_9d['Activity'], test_size=0.2, random_state=42)
X_train.columns = ['Tx', 'Ty', 'Tz', 'Ax', 'Ay', 'Az', 'Lx', 'Ly', 'Lz']

result_all = distance_three(df_train_all)
ax = plt.axes(projection='3d')
colors = result_all['observation']
ax.scatter(result_all['distance_x'], result_all['distance_y'], result_all['distance_z'], c=colors)

# NORMALISED
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(result_all[['distance_x', 'distance_y', 'distance_z']])
result_normalised_all = pd.DataFrame(np_scaled)
result_normalised_all['observation'] = result_all['observation']
ax = plt.axes(projection='3d')
colors = result_normalised_all['observation']
ax.scatter(result_normalised_all[0], result_normalised_all[1], result_normalised_all[2], c=colors)

# result_all, result_normalised_all
result_all.to_csv('data.csv')
result_normalised_all.to_csv('data_normalised.csv')

pd.save_to_csv('data', result)
pd.save_to_csv('data', result_normalised)

# Train test split for all
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(result_normalised_all[[0, 1, 2]], result_normalised_all['observation'], test_size=0.2, random_state=42)
X_train.columns = ['x', 'y', 'z']

# KNeighbors
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, list(y_train))
neigh.predict(X_test)
neigh.score(X_test,list(y_test))
list(y_test) - neigh.predict(X_test)

# Parameter tuning
from sklearn.neighbors import KNeighborsClassifier
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from sklearn.model_selection import cross_val_score

# The list of hyper-parameters we want to optimize. For each one we define the bounds,
# the corresponding scikit-learn parameter name, as well as how to sample values
# from that dimension (`'log-uniform'` for the learning rate)
space = [Integer(1, 100, name='n_neighbors'),
         Categorical(['kd_tree', 'ball_tree', 'brute'], name='algorithm'),
         Categorical(['uniform', 'distance'], name='weights'),
         Integer(1, 3, name='p')]

# this decorator allows your objective function to receive a the parameters as
# keyword arguments. This is particularly convenient when you want to set scikit-learn
# estimator parameters
@use_named_args(space)
def objective(**params):
    reg = KNeighborsClassifier()
    reg.set_params(**params)

    return -np.mean(cross_val_score(reg, X_train[['x', 'y', 'z']], list(y_train), cv=5, n_jobs=-1,
                                    scoring="neg_mean_absolute_error"))

# Optimize
from skopt import gp_minimize
res_gp = gp_minimize(objective, space, n_calls=100, random_state=0)

"Best score=%.4f" % res_gp.fun

print("""Best parameters:
- max_depth=%d
- learning_rate=%.6f
- max_features=%d
- min_samples_split=%d
- min_samples_leaf=%d""" % (res_gp.x[0], res_gp.x[1],
                            res_gp.x[2], res_gp.x[3],
                            res_gp.x[4]))

# PLot of convergence
from skopt.plots import plot_convergence
plot_convergence(res_gp)

neigh = KNeighborsClassifier(n_neighbors=10, algorithm='ball_tree', weights='distance', p=1)
neigh.fit(X_train, list(y_train))
neigh.predict(X_test)
neigh.score(X_test,list(y_test))


import multiprocessing

multiprocessing.cpu_count()


