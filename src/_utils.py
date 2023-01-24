import json
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics
import numpy as np
from tensorflow import keras
from sklearn import metrics

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'), 
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

def read_json_f(input_file,kmer = None, sample_id = False):
    seq_list = []
    with open(input_file,"r",encoding='utf-8') as reader:
        seq_list = [json.loads(line) for line in reader]
        
    sentences = []
    labels = []
    sampleIds = []
    ret = []

    for idx, entry in enumerate(seq_list):
        labels.append(int(entry['label'])+1)
        
        if sample_id:
            sampleIds.append(entry['sample_id'])
            
        if kmer is not None:
            sentences.append(seq2kmer(entry['sentence'],kmer))
        else:
            sentences.append(entry['sentence'])
            
    if sample_id:
        ret = [sentences,labels,sampleIds]
    else:
        ret = [sentences,labels]
        
    return ret

def seq2kmer(seq, k):
    kmer_li = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = ' '.join(kmer_li)
    return kmers

def get_elapsed_time(start_t, end_t):
    hh,rem = divmod(end_t-start_t,(60*60))
    mm,ss = divmod(rem,60)
    time_str = '%d:%d:%0.2f'%(hh,mm,ss)
    return time_str

def plot_cm(labels, predictions, p_threshold=.5,cmap='Reds',verbose=1):
    cm = metrics.confusion_matrix(labels, predictions > p_threshold)
    plt.figure(figsize=(4,4),dpi=120)
    sns.heatmap(cm, annot=True, fmt="d",cmap='Reds')
    plt.title('Confusion matrix @{:.2f}'.format(p_threshold))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    if verbose:
        print('Normal events correctly labeled (True Negatives): ', cm[0][0])
        print('Normal events incorrectly labeled as cancerous (False Positives): ', cm[0][1])
        print('Missed cancerous events (False Negatives): ', cm[1][0])
        print('Detected cancerous events (True Positives): ', cm[1][1])
        print('Total Fraudulent Transactions: ', np.sum(cm[1]))
        