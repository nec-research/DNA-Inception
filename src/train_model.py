#!/usr/bin/env python

# coding=utf-8
#        ST-tau
#
#   File:     train_model.py
#   Authors:  Israa Alqassem israa.alqassem@neclab.eu
#             Filippo Grazioli filippo.grazioli@neclab.eu
#
#
# NEC Laboratories Europe GmbH, Copyright (c) 2021, All rights reserved.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#
#        PROPRIETARY INFORMATION ---
#
# SOFTWARE LICENSE AGREEMENT
#
# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
#
# BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
# LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
# DOWNLOAD THE SOFTWARE.
#
# This is a license agreement ("Agreement") between your academic institution
# or non-profit organization or self (called "Licensee" or "You" in this
# Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
# Agreement).  All rights not specifically granted to you in this Agreement
# are reserved for Licensor.
#
# RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
# ownership of any copy of the Software (as defined below) licensed under this
# Agreement and hereby grants to Licensee a personal, non-exclusive,
# non-transferable license to use the Software for noncommercial research
# purposes, without the right to sublicense, pursuant to the terms and
# conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
# LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
# Agreement, the term "Software" means (i) the actual copy of all or any
# portion of code for program routines made accessible to Licensee by Licensor
# pursuant to this Agreement, inclusive of backups, updates, and/or merged
# copies permitted hereunder or subsequently supplied by Licensor,  including
# all or any file structures, programming instructions, user interfaces and
# screen formats and sequences as well as any and all documentation and
# instructions related to it, and (ii) all or any derivatives and/or
# modifications created or made by You to any of the items specified in (i).
#
# CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
# proprietary to Licensor, and as such, Licensee agrees to receive all such
# materials and to use the Software only in accordance with the terms of this
# Agreement.  Licensee agrees to use reasonable effort to protect the Software
# from unauthorized use, reproduction, distribution, or publication. All
# publication materials mentioning features or use of this software must
# explicitly include an acknowledgement the software was developed by NEC
# Laboratories Europe GmbH.
#
# COPYRIGHT: The Software is owned by Licensor.
#
# PERMITTED USES:  The Software may be used for your own noncommercial
# internal research purposes. You understand and agree that Licensor is not
# obligated to implement any suggestions and/or feedback you might provide
# regarding the Software, but to the extent Licensor does so, you are not
# entitled to any compensation related thereto.
#
# DERIVATIVES: You may create derivatives of or make modifications to the
# Software, however, You agree that all and any such derivatives and
# modifications will be owned by Licensor and become a part of the Software
# licensed to You under this Agreement.  You may only use such derivatives and
# modifications for your own noncommercial internal research purposes, and you
# may not otherwise use, distribute or copy such derivatives and modifications
# in violation of this Agreement.
#
# BACKUPS:  If Licensee is an organization, it may make that number of copies
# of the Software necessary for internal noncommercial use at a single site
# within its organization provided that all information appearing in or on the
# original labels, including the copyright and trademark notices are copied
# onto the labels of the copies.
#
# USES NOT PERMITTED:  You may not distribute, copy or use the Software except
# as explicitly permitted herein. Licensee has not been granted any trademark
# license as part of this Agreement.  Neither the name of NEC Laboratories
# Europe GmbH nor the names of its contributors may be used to endorse or
# promote products derived from this Software without specific prior written
# permission.
#
# You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
# whole or in part, or provide third parties access to prior or present
# versions (or any parts thereof) of the Software.
#
# ASSIGNMENT: You may not assign this Agreement or your rights hereunder
# without the prior written consent of Licensor. Any attempted assignment
# without such consent shall be null and void.
#
# TERM: The term of the license granted by this Agreement is from Licensee's
# acceptance of this Agreement by downloading the Software or by using the
# Software until terminated as provided below.
#
# The Agreement automatically terminates without notice if you fail to comply
# with any provision of this Agreement.  Licensee may terminate this Agreement
# by ceasing using the Software.  Upon any termination of this Agreement,
# Licensee will delete any and all copies of the Software. You agree that all
# provisions which operate to protect the proprietary rights of Licensor shall
# remain in force should breach occur and that the obligation of
# confidentiality described in this Agreement is binding in perpetuity and, as
# such, survives the term of the Agreement.
#
# FEE: Provided Licensee abides completely by the terms and conditions of this
# Agreement, there is no fee due to Licensor for Licensee's use of the
# Software in accordance with this Agreement.
#
# DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
# OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
# FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
# BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
# RELATED MATERIALS.
#
# SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
# provided as part of this Agreement.
#
# EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
# permitted under applicable law, Licensor shall not be liable for direct,
# indirect, special, incidental, or consequential damages or lost profits
# related to Licensee's use of and/or inability to use the Software, even if
# Licensor is advised of the possibility of such damage.
#
# EXPORT REGULATION: Licensee agrees to comply with any and all applicable
# export control laws, regulations, and/or other laws related to embargoes and
# sanction programs administered by law.
#
# SEVERABILITY: If any provision(s) of this Agreement shall be held to be
# invalid, illegal, or unenforceable by a court or other tribunal of competent
# jurisdiction, the validity, legality and enforceability of the remaining
# provisions shall not in any way be affected or impaired thereby.
#
# NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
# or remedy under this Agreement shall be construed as a waiver of any future
# or other exercise of such right or remedy by Licensor.
#
# GOVERNING LAW: This Agreement shall be construed and enforced in accordance
# with the laws of Germany without reference to conflict of laws principles.
# You consent to the personal jurisdiction of the courts of this country and
# waive their rights to venue outside of Germany.
#
# ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
# entire agreement between Licensee and Licensor as to the matter set forth
# herein and supersedes any previous agreements, understandings, and
# arrangements between the parties relating hereto.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#
#


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
