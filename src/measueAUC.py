import json
from tqdm import tqdm
import sys

# path
#rel_file = 'log/DeepSAD/BGL_Conv_graph_MLP_mean'
#rel_file = 'log/DeepSAD/tbird_Conv_graph_MLP_mean_128_maxpool'

rel_file = sys.argv[1]
rel_file = rel_file + '/results.json'
print(rel_file)

res = ''
with open(rel_file,'r') as f:
    res = f.read()
f.close()

rel_dic = json.loads(res)


y_pred = []
y_sores = []
y_true = []
for pre in rel_dic['test_scores']:
    idx = pre[0]
    plabel = pre[1]
    scores = pre[2]
    y_sores.append(scores)
    y_true.append(plabel)


print(len(y_true))
print(len(y_sores))


from sklearn.metrics import confusion_matrix


def mesure(TP,FP,FN,TN):
    acc = (TP+TN)/(TP+FP+FN+TN)
    precision = (TP)/(TP+FP)
    recall = (TP)/(TP+FN)
    f1 = 2*precision*recall/(precision+recall)
    # print('======')
    # print('TP, FN', TP,FN)
    # print('FP, TN', FP,TN)
    # print('acc',acc)
    # print('precision',precision)
    # print('recall',recall)
    # print('f1',f1)
    return acc,precision,recall,f1



import matplotlib.pyplot as plt
def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()


from sklearn.metrics import roc_auc_score,roc_curve
print(roc_auc_score(y_true, y_sores))


fper, tper, thresholds = roc_curve(y_true, y_sores)
plot_roc_curve(fper, tper)

macc = -9999999
max_f1 = -999999
m_precision = -999999
m_recell = -999999
threshod = -999999

mTN =0
mFP =0
mFN =0
mTP =0

for i in tqdm(thresholds):
    y_pred = []
    y_sores = []
    for pre in rel_dic['test_scores']:
        plabel = pre[1]
        scores = pre[2]
        y_sores.append(scores)
        if scores> i:
            y_pred.append(1)
        else:
            y_pred.append(0)

    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    # print('thresholds',i)
    acc,precision,recall,f1 = mesure(TP,FP,FN,TN)
    if f1 > max_f1:
        macc = acc
        max_f1 = f1
        m_precision=precision
        m_recell= recall
        threshod = i
        mTN = TN
        mFP = FP
        mFN = FN
        mTP = TP
    
    #print('max_f1', f1)
    #print('m_precision', precision)
    #print('m_recell', recall)
    #print('threshod', i)
    #print('acc', acc)


print('max_f1', max_f1)
print('m_precision', m_precision)
print('m_recell', m_recell)
print('threshod', threshod)
print('acc', macc)
print('TN,FP,FN,TP=',mTN,mFP,mFN,mTP)
