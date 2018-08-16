import pandas as pd
import math
import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import multiprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize
from skopt.plots import plot_convergence
from sklearn.metrics import confusion_matrix

COLUMN_NAMES = ['T_xacc', 'T_yacc', 'T_zacc', 'T_xgyro', 'T_ygyro', 'T_zgyro', 'T_xmag', 'T_ymag', 'T_zmag',
                'RA_xacc', 'RA_yacc', 'RA_zacc', 'RA_xgyro', 'RA_ygyro', 'RA_zgyro', 'RA_xmag', 'RA_ymag', 'RA_zmag',
                'LA_xacc', 'LA_yacc', 'LA_zacc', 'LA_xgyro', 'LA_ygyro', 'LA_zgyro', 'LA_xmag', 'LA_ymag', 'LA_zmag',
                'RL_xacc', 'RL_yacc', 'RL_zacc', 'RL_xgyro', 'RL_ygyro', 'RL_zgyro', 'RL_xmag', 'RL_ymag', 'RL_zmag',
                'LL_xacc', 'LL_yacc', 'LL_zacc', 'LL_xgyro', 'LL_ygyro', 'LL_zgyro', 'LL_xmag', 'LL_ymag', 'LL_zmag']
ADDITIONAL = COLUMN_NAMES + ['Action', 'Subject', 'Segment']

# Create data
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

# CODE MICHAEL
df =  pd.read_csv('prepared.csv')

# DTW function
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

# Confusion matrix function
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

################################
################### FOR ALL DATA
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

# Multiprocessing function
def df_multiprocessing(df):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    list_of_data = []
    for i in range(1, 9):
        list_of_data.append(df[['T_xacc', 'T_yacc', 'T_zacc', 'RA_xacc', 'RA_yacc', 'RA_zacc', 'LA_xacc', 'LA_yacc', 'LA_zacc', 'RL_xacc', 'RL_yacc', 'RL_zacc', 'LL_xacc', 'LL_yacc', 'LL_zacc', 'Action', 'Segment']][df['Subject'] == i])
    result = pool.map(distance_15d, list_of_data)
    return list(result)

result_15d = df_multiprocessing(df)
result_15d = pd.concat(result_15d)

# Normalise data
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(result_15d[['Tx', 'Ty', 'Tz', 'RAx', 'RAy', 'RAz', 'LAx', 'LAy', 'LAz', 'RLx', 'RLy', 'RLz', 'LLx', 'LLy', 'LLz']])
result_normalised_15d = pd.DataFrame(np_scaled)
result_normalised_15d['Activity'] = result_15d['Activity'].values
#ax = plt.axes(projection='3d')
#colors = result_normalised_9d['Activity']
#ax.scatter(result_normalised_9d[0], result_normalised_9d[1], result_normalised_9d[2], c=colors)

# Save data
result_15d.to_csv('data_15d.csv')
result_normalised_15d.to_csv('data_normalised_15d.csv')

# Load data
result_15d = pd.read_csv('data_15d.csv')
result_normalised_15d['0'] = pd.read_csv('data_normalised_15d.csv')

# Train test split
X_train, X_test, y_train, y_test = train_test_split(result_normalised_15d[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']], result_normalised_15d['Activity'], test_size=0.2, random_state=42)
X_train.columns = ['Tx', 'Ty', 'Tz', 'RAx', 'RAy', 'RAz', 'LAx', 'LAy', 'LAz', 'RLx', 'RLy', 'RLz', 'LLx', 'LLy', 'LLz']


###### CLASSIFICATION
### KNeighbors
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, list(y_train))
neigh.score(X_test, list(y_test))

# Parameter tuning

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
    return -np.mean(cross_val_score(reg, X_train[['Tx', 'Ty', 'Tz', 'RAx', 'RAy', 'RAz', 'LAx', 'LAy', 'LAz', 'RLx', 'RLy', 'RLz', 'LLx', 'LLy', 'LLz']], list(y_train), cv=5, n_jobs=-1, scoring="accuracy"))

# Optimize
res_gp_KN_all = gp_minimize(objective, space, n_calls=100, random_state=0)

"Best score=%.4f" % res_gp_KN_all.fun

print("""Best parameters:
- max_depth=%d
- learning_rate=%.6f
- max_features=%d
- min_samples_split=%d
- min_samples_leaf=%d""" % (res_gp_KN_all.x[0], res_gp_KN_all.x[1],
                            res_gp_KN_all.x[2], res_gp_KN_all.x[3],
                            res_gp_KN_all.x[4]))

# PLot of convergence
from skopt.plots import plot_convergence
plot_convergence(res_gp_KN_all)

neigh = KNeighborsClassifier(n_neighbors=1, algorithm='kd_tree', weights='uniform', p=1)
neigh.fit(X_train, list(y_train))
neigh.predict(X_test)
neigh.score(X_test, list(y_test))

# Visualization of results / confusion matrix
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, neigh.predict(X_test))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], title='Confusion matrix')
plt.show()


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

    return -np.mean(cross_val_score(reg, X_train[
        ['Tx', 'Ty', 'Tz', 'RAx', 'RAy', 'RAz', 'LAx', 'LAy', 'LAz', 'RLx', 'RLy', 'RLz', 'LLx', 'LLy', 'LLz']],
                                    list(y_train), cv=5, n_jobs=-1, scoring="accuracy"))

# Optimize
from skopt import gp_minimize
res_gp_all_RF = gp_minimize(objective, space, n_calls=150, random_state=0)

"Best score=%.4f" % res_gp_all_RF.fun

print("""Best parameters:
- max_depth=%d
- learning_rate=%.6f
- max_features=%d
- min_samples_split=%d
- min_samples_leaf=%d""" % (res_gp_all_RF.x[0], res_gp_all_RF.x[1],
                            res_gp_all_RF.x[2], res_gp_all_RF.x[3],
                            res_gp_all_RF.x[4]))

# PLot of convergence
plot_convergence(res_gp_all_RF)

# On the test set
rforest = RandomForestClassifier(n_estimators=82, criterion='entropy', max_depth=72, min_samples_split=2, min_samples_leaf=1, bootstrap='False')
rforest.fit(X_train, list(y_train))
rforest.predict(X_test)
rforest.score(X_test, list(y_test))

# Visualization of results / confusion matrix
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, rforest.predict(X_test))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], title='Confusion matrix')
plt.show()


##### NEURAL NETWORK ########
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
    return -np.mean(cross_val_score(reg, X_train[['Tx', 'Ty', 'Tz', 'RAx', 'RAy', 'RAz', 'LAx', 'LAy', 'LAz', 'RLx', 'RLy', 'RLz', 'LLx', 'LLy', 'LLz']], list(y_train), cv=5, n_jobs=7,
                                    scoring="accuracy"))

# Optimize
res_gp_all_NN = gp_minimize(objective, space, n_calls=100, random_state=0)

# Scores and parameters
"Best score=%.4f" % res_gp_all_NN.fun
print(res_gp_all_NN.x)

# PLot of convergence
plot_convergence(res_gp_all_NN)

# On the test set
neural = MLPClassifier(activation='tanh', solver='lbfgs', learning_rate='constant', alpha=0.0004290666, hidden_layer_sizes=133)
neural.fit(X_train, list(y_train))
neural.score(X_test,list(y_test))

# Visualization of results / confusion matrix
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, neural.predict(X_test))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], title='Confusion matrix')
plt.show()





