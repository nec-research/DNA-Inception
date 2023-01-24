#!/usr/bin/env python
# coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import json
import time
import random
import argparse
from pathlib import Path
from _utils import *
from collections import Counter
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.backend import shape
from keras import optimizers
from keras.models import Sequential,Model
from keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense
from keras.layers import Input, Conv1D, GlobalMaxPooling1D, MaxPooling1D , concatenate
from keras.callbacks import Callback,ModelCheckpoint, EarlyStopping
from keras.backend import clear_session
from sklearn.model_selection import train_test_split
from sklearn import metrics
print(sys.version)
print('keras v%s'%str(keras.__version__))
print('tensorflow v%s'%tf.__version__)
print('numpy v%s'%np.__version__)

np.random.seed(42)
maxlen = 1536
EPOCHS = 100
BATCH_SIZE = 64 
FILTER_SIZE  = 64
num_words = 5
#RUNS = 10

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, help='Input json file.', required=True)
    parser.add_argument('--model_path', type=str,default="./saved_models/dna_incep.h5", required=False,help='A path to save/load a trained moded to/from.')
    args, unknown = parser.parse_known_args()
    return args

## Credit to https://stackoverflow.com/questions/67271590/issue-with-custom-metric-auc-callback-for-keras
class scoreTarget(Callback):
    def __init__(self, target,metric):
        super(scoreTarget, self).__init__()
        self.target = target
        self.metric = metric

    def on_epoch_end(self, epoch, logs={}):
        acc = logs[self.metric]
        if acc >= self.target:
            self.model.stop_training = True

def dna_inception(max_len:int, embedding_dim:int, filters:int, activation:str="sigmoid", padding:str="same",vocab_size=6,metrics=METRICS):

    input_seq = Input(shape=(max_len,))  
    seq = Embedding(vocab_size, output_dim=128, input_length=max_len)(input_seq)        

    filters_16 = 16
    filters_32 = 32
    filters_64 = 64
    conv_1_3 = Conv1D(filters_32, 1, padding=padding, activation=activation, kernel_initializer='glorot_normal')(seq)
    conv_1_16 = Conv1D(filters_32, 1, padding=padding, activation=activation, kernel_initializer='glorot_normal')(seq)
    conv_1_64 = Conv1D(filters_32, 1, padding=padding, activation=activation, kernel_initializer='glorot_normal')(seq)

    max_pooling = MaxPooling1D(pool_size=2, strides=None, padding="same", data_format="channels_last")(seq)

    conv_2_1 = Conv1D(filters_64, 1, padding=padding, activation=activation, kernel_initializer='glorot_normal')(max_pooling)
    conv_2_3 = Conv1D(filters_64, 3, padding=padding, activation=activation, kernel_initializer='glorot_normal')(conv_1_3)
    conv_2_16 = Conv1D(filters_64, 16, padding=padding, activation=activation, kernel_initializer='glorot_normal')(conv_1_16)
    conv_2_64 = Conv1D(filters_64, 64, padding=padding, activation=activation, kernel_initializer='glorot_normal')(conv_1_64)

    conv_2_1 = GlobalMaxPooling1D()(conv_2_1)
    conv_2_3 = GlobalMaxPooling1D()(conv_2_3)
    conv_2_16 = GlobalMaxPooling1D()(conv_2_16)
    conv_2_64 = GlobalMaxPooling1D()(conv_2_64)

    cat = concatenate([conv_2_1,  conv_2_3, conv_2_16, conv_2_64])

    dense = Dense(32, activation=activation)(cat)
    out = Dense(2, activation='softmax')(dense)

    model = (Model(inputs=[input_seq], outputs=[out]))

    model.compile(loss='binary_crossentropy',
            metrics = metrics,
            optimizer=optimizers.Adam(lr=0.001,
                                        beta_1=0.9,
                                        beta_2=0.999,
                                        epsilon=1e-08,
                                        decay=0.0))
    return model

   
if __name__ == '__main__':
    params = vars(get_args())
    json_input_file = params['json']
    model_path = params['model_path']
    
    print('Preprocessing input sequences...')
    data = read_json_f(json_input_file)
    input_df = pd.DataFrame({'sentence': data[0],'label': data[1]})
    train_df, test_df = train_test_split(input_df, test_size=0.2)
    train_df, val_df = train_test_split(train_df, test_size=0.2)
    train_labels = np.array(train_df.pop('label'))
    val_labels = np.array(val_df.pop('label'))
    test_labels = np.array(test_df.pop('label'))
    train_sequences = np.array(train_df)
    val_sequences = np.array(val_df)
    test_sequences = np.array(test_df)
    bool_train_labels = train_labels != 1
    pos_sequences = train_sequences[bool_train_labels]
    neg_sequences = train_sequences[~bool_train_labels]
    pos_labels = train_labels[bool_train_labels]
    neg_labels = train_labels[~bool_train_labels]
    ids = np.arange(len(pos_sequences))
    choices = np.random.choice(ids, size=len(neg_sequences))
    res_pos_sequences = pos_sequences[choices]
    res_pos_labels = pos_labels[choices]
    resampled_sequences = np.concatenate([res_pos_sequences, neg_sequences], axis=0)
    resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis=0)
    order = np.arange(len(resampled_labels))
    np.random.shuffle(order)
    resampled_train_sequences = resampled_sequences[order]
    resampled_train_labels = resampled_labels[order]
    tokenizer = Tokenizer(num_words=num_words)#
    tokenizer.fit_on_texts(resampled_train_sequences[0][0])
    vocab_size = len(tokenizer.word_index) + 1

    label_train_cat = np.array(to_categorical(resampled_train_labels))
    label_valid_cat = np.array(to_categorical(val_labels))
    label_test_cat = np.array(to_categorical(test_labels))
    label_train_cat = label_train_cat[:,[1, 2]]
    label_valid_cat = label_valid_cat[:,[1, 2]]
    label_test_cat = label_test_cat[:,[1, 2]]
    X_train = [tokenizer.texts_to_sequences(sent[0]) for sent in resampled_train_sequences]
    X_valid = [tokenizer.texts_to_sequences(sent[0]) for sent in val_sequences]
    X_test = [tokenizer.texts_to_sequences(sent[0]) for sent in test_sequences]
    vocab_size = len(tokenizer.word_index) + 1
    X_train_pad = pad_sequences(X_train, maxlen=maxlen, padding='post', truncating='post')
    X_valid_pad = pad_sequences(X_valid, maxlen=maxlen, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test, maxlen=maxlen, padding='post', truncating='post')
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
    clear_session()
    print('Train a model...')
    model = dna_inception(maxlen, BATCH_SIZE, FILTER_SIZE,activation='relu')
    print(model.summary())
    model._get_distribution_strategy = lambda: None
    stopping_metric = 'val_recall' 
    early_stopping = EarlyStopping(
           monitor=stopping_metric,
           verbose=1,
           patience=15,
           mode='max',
           restore_best_weights=True)
    target = scoreTarget(0.95,stopping_metric)
    callbacks = [early_stopping,target]
    st = time.time()
    model_history = model.fit(X_train_pad.reshape(-1, maxlen), 
                   label_train_cat, 
                   batch_size=BATCH_SIZE, 
                   epochs=EPOCHS, callbacks=callbacks,
                   validation_data=(X_valid_pad.reshape(-1, maxlen), label_valid_cat))

    en = time.time()
    total_t = get_elapsed_time(st,en)
    print('\n\nTotal time -->',total_t)
    print('Save model to:',model_path)
    model.save(model_path)
    
    y_preds = model.predict(X_test_pad.reshape(-1,maxlen))
    y_pred_classes = np.argmax(y_preds, axis = 1)
    y_test = np.add(test_labels, -1) 
    f1 = metrics.f1_score(y_test, y_pred_classes, average="macro")
    precision = metrics.precision_score(y_test, y_pred_classes, average="macro")
    recall = metrics.recall_score(y_test, y_pred_classes, average="macro")
    precision_l, recall_l, _ = metrics.precision_recall_curve(y_test, y_preds[:, 1])
    auc_pr = metrics.auc(recall_l, precision_l)
    roc_auc = metrics.roc_auc_score(y_test, y_preds[:, 1], average="macro")
    mcc = metrics.matthews_corrcoef(y_test, y_pred_classes)
    conf_matrix = metrics.confusion_matrix(y_test, y_pred_classes)
    fpr, tpr, _ = metrics.roc_curve(y_test, y_preds[:, 1])

    print('F1 = %0.3f'%f1)
    print('Precision = %0.3f'%precision)
    print('Recall = %0.3f'%recall)
    print('AUC PR = %0.3f'%auc_pr)
    print('ROC AUC = %0.3f'%roc_auc)
    print ('MCC = %0.3f'%mcc)
    print('conf_matrix=',conf_matrix)
    #print('precision_l=',list(precision_l))
    #print('recall_l=',list(recall_l))