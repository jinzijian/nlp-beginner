from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

def evaluate(all_pred, all_label):
    acc = 0
    for i in range(len(all_pred)):
        if(all_pred[i] == all_label[i]):
            acc += 1
        else:
            continue
    acc = acc / len(all_pred)
    all_label = all_label.cpu()
    all_pred = all_pred.cpu()
    # p_micro = precision_score(all_label, all_pred, average='binary', pos_label=1)
    # r_micro = recall_score(all_label, all_pred, average='binary', pos_label=1)
    # f1 = (2*p_micro*r_micro)/(p_micro + r_micro)
    return acc