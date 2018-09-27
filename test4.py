roles_list =['агенс','пациенс','тема','субъект психологического состояния','субъект перемещения','причина','говорящий',
        'место ','содержание действия','содержание мысли'] #选择出现次数最高的十种角色
if __name__ == '__main__':
    #计算准确率和召回率，这个地方一定要这么写，因为预测的结果长度肯定比原来的短，被我截取了
    golds = [gold.split(',') for gold in open("resource/golds2.txt",'r').read().strip().split('\n')]
    preds = [pred.split(',') for pred in open("resource/predict_result.txt",'r').read().strip().split('\n')]
    # print(golds)
    case_recall = 0
    case_precision = 0
    case_true = 0
    for index,gold in enumerate(golds):
        for label in gold:
                if label in roles_list:
                    case_recall += 1
    for index,pred in enumerate(preds):
        for i ,label in enumerate(pred):
            if label in roles_list:
                case_precision += 1
                if label == golds[index][i]:
                    case_true += 1
    recall = 1.0*case_true / case_recall
    precision = 1.0*case_true/case_precision
    f1 = 2.0*recall*precision/(recall+precision)

    print(recall,precision,f1)
