roles = ['subject','object','predict','time','location']
if __name__ == '__main__':
    golds = [gold.split() for gold in open("resource/golds_tgt.txt",'r').read().strip().split('\n')]
    preds = [pred.split() for pred in open("resource/predict_tgt.txt",'r').read().strip().split('\n')]
    case_recall = 0
    case_precision = 0
    case_true = 0

    for index,gold in enumerate(golds):
        for i, label in enumerate(gold):
            if label in roles:
                case_recall += 1
                if label == preds[index][i]:
                    case_true += 1

    for index,pred in enumerate(preds):
        for label in pred:
            if label in roles:
                case_precision += 1
    recall = 1.0*case_true / case_recall
    precision = 1.0*case_true/case_precision
    f1 = 2.0*recall*precision/(recall+precision)
    print(recall,precision,f1)