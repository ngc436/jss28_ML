import numpy as np

from keras import losses, models, optimizers
from keras.activations import relu, softmax
from keras.callbacks import (EarlyStopping,
                             ModelCheckpoint, ReduceLROnPlateau)
from keras.layers import Dense, Dropout, Input, LSTM
from keras.utils import Sequence, to_categorical
from keras.backend import clear_session
from sklearn.model_selection import StratifiedKFold

'''
COLUMN_NAMES = ['T_xacc', 'T_yacc', 'T_zacc', 'T_xgyro', 'T_ygyro', 'T_zgyro', 'T_xmag', 'T_ymag', 'T_zmag',
                'RA_xacc', 'RA_yacc', 'RA_zacc', 'RA_xgyro', 'RA_ygyro', 'RA_zgyro', 'RA_xmag', 'RA_ymag', 'RA_zmag',
                'LA_xacc', 'LA_yacc', 'LA_zacc', 'LA_xgyro', 'LA_ygyro', 'LA_zgyro', 'LA_xmag', 'LA_ymag', 'LA_zmag',
                'RL_xacc', 'RL_yacc', 'RL_zacc', 'RL_xgyro', 'RL_ygyro', 'RL_zgyro', 'RL_xmag', 'RL_ymag', 'RL_zmag',
                'LL_xacc', 'LL_yacc', 'LL_zacc', 'LL_xgyro', 'LL_ygyro', 'LL_zgyro', 'LL_xmag', 'LL_ymag', 'LL_zmag']

df = pd.read_csv('prepared.csv')
X = np.asarray(df[COLUMN_NAMES])
Y = np.asarray(df['Action'])

segments = np.asarray(df['Segment'].values).astype(int)

X_final = []
Y_final = []

start = 0
end = 0
for i in range(len(segments)-1):
    if segments[i] != segments[i+1]:
        end = i+1
        X_final.append(X[start:end])
        Y_final.append(Y[start:end])
        start = i+1
    if i == len(segments) - 2:
        X_final.append(X[start:])
        Y_final.append(Y[start:])

X_final = np.asarray(X_final)
Y_final = np.asarray(Y_final)

print(X_final.shape)
print(Y_final.shape)


np.save('./final_full_data.npy', X_final)
np.save('./final_full_labels.npy', Y_final)
'''

#data generator
class DataGenerator(Sequence):
    def __init__(self, data, labels=None, batch_size=64, num_classes=19):
        self.labels = labels
        self.batch_size = batch_size
        self.data = data
        self.num_classes = num_classes
        self.indexes = np.arange(0, self.data.shape[0])
        #print('in init: len(indexes) = ', len(self.indexes))

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        part_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X = self.data[part_indexes]
        if self.labels is not None:
            #print(X.shape)
            #print('indexes', part_indexes)
            #print(len(self.labels))
            y = to_categorical(self.labels[part_indexes], self.num_classes)
            #print('y=',y)
            return X, y
        else:
            return X


def get_first_model():

    inp = Input(shape=(125, 45))
    print(inp.shape)
    x = LSTM(128)(inp)
    x = Dropout(rate=0.3)(x)

    x = Dense(64, activation=relu)(x)
    out = Dense(19, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model

N_fold = 4
batch_size = 5
'''
#training
#load data
X = np.load('./final_full_data.npy')
Y = np.load('./final_full_labels.npy')
Y = Y.mean(axis=1).astype(int)-1

#split on test and train
N = X.shape[0]
rand_ind = np.arange(0, N)
np.random.shuffle(rand_ind)
X = X[rand_ind]
Y = Y[rand_ind]

train_N = 7000
X_train = X[:train_N]
y_train = Y[:train_N]

X_test = X[train_N:]
y_test = Y[train_N:]

np.save('./preds/X_test.npy', X_test)
np.save('./preds/y_test.npy', y_test)

#build training process
skf = StratifiedKFold(n_splits=N_fold, shuffle=True)
for i, index in enumerate(skf.split(X_train, y_train)):
    #print("TRAIN:", index[0], "VALID:", index[1])
    X_tr, X_val = X_train[index[0]], X_train[index[1]]
    y_tr, y_val = y_train[index[0]], y_train[index[1]]
    clear_session()
    checkpoint = ModelCheckpoint('best_%d.h5' % i, monitor='val_loss', verbose=1, save_best_only=True)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=2, min_lr=0.0001)
    callbacks_list = [checkpoint, early, reduce_lr]

    model = get_first_model()
    print(model.summary())
    #train_generator = DataGenerator(X_tr, y_tr, batch_size)

    #val_generator = DataGenerator(X_val, y_val, batch_size)

    #history = model.fit_generator(train_generator, callbacks=callbacks_list, validation_data=val_generator,
    #                              epochs=10)  # , use_multiprocessing=True, workers=6, max_queue_size=20)

    model.load_weights('best_%d.h5' % i)

    test_generator = DataGenerator(data=X_test, batch_size=batch_size)
    predictions = model.predict_generator(test_generator)

    print('predictions.shape = ', predictions.shape)

    np.save('./preds/test_predictions_%d.npy' % i, predictions)

'''
#make final solution
pred_shape = np.load('./preds/test_predictions_0.npy').shape
final_pred = np.ones(pred_shape)
for i in range(N_fold):
    final_pred = final_pred*np.load('./preds/test_predictions_%d.npy' % i)

final_pred = final_pred**(1/N_fold)
np.save('./preds/final_test_pred.npy', final_pred)
print('final_pred.shape = ', final_pred.shape)
predicted_label = np.argsort(-final_pred, axis=1)

predicted_labels = predicted_label[:, :1]
true_labels = np.load('./preds/y_test.npy').reshape((pred_shape[0], 1))


#print('first prediction: ', predicted_label[110])
#print('first prediction: ', predicted_label[111])
#print('first prediction: ', predicted_label[112])
#print(predicted_label.shape)

#rint('first true label: ', true_labels[110])
#print('first true label: ', true_labels[111])
#print('first true label: ', true_labels[112])

score = 0
for i in range(pred_shape[0]):
    if true_labels[i] == predicted_labels[i]:
        score+=1
score/=pred_shape[0]
print('Total score:', score)

#print(predicted_label[:10])
#print(true_labels[:10])
