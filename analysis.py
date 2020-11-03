# _*_ coding: utf-8 _*_
# @Time : 2020/10/10 下午2:48 
# @Author : yanqiuxia
# @Version：V 0.1
# @File : analysis.py
import re
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer


def safe_divide(x, y):
    if y == 0.0:
        return 0.0
    return x / y


def calc_f1(y_trues, y_preds, id2label, calc_report=False):

    classes = list(id2label.keys())
    mlb = MultiLabelBinarizer(classes=classes)

    y_trues = mlb.fit_transform(y_trues)
    y_preds = mlb.transform(y_preds)
    f1_micro = f1_score(y_trues, y_preds, average='micro')
    f1_macro = f1_score(y_trues, y_preds, average='macro')
    avg_f1 = (f1_micro + f1_macro)/2

    classify_report = ''
    if calc_report:

        classify_report = classification_report(y_trues, y_preds)
        classify_report = format_classify_report(classify_report, id2label)

    return f1_micro, f1_macro, avg_f1, classify_report


def calc_acc(correct_label_num, pred_label_num, true_label_num):

    if type(correct_label_num) != np.ndarray:
        correct_label_num = np.asarray(correct_label_num)

    acc_matrix = np.zeros([4, correct_label_num.shape[0]], dtype=np.float)
    acc_matrix[0][:] = correct_label_num / pred_label_num
    acc_matrix[1][:] = correct_label_num / true_label_num
    acc_matrix[2][:] = 2*acc_matrix[0][:]*acc_matrix[1][:]/(acc_matrix[0][:]+acc_matrix[1][:]+1e-16)
    acc_matrix[3][:] = true_label_num
    acc_matrix = np.transpose(acc_matrix)
    return acc_matrix


def calc_mul_f1(y_trues, y_preds, id2label,calc_report=False):
    if isinstance(y_trues, np.ndarray):
        y_trues = np.asarray(y_trues)

    if isinstance(y_preds, np.ndarray):
        y_preds = np.asarray(y_preds)
    # y_trues/y_preds [batch_size,class_num]

    f1_micro = f1_score(y_trues, y_preds, average='micro')
    f1_macro = f1_score(y_trues, y_preds, average='macro')

    avg_f1 = (f1_micro + f1_macro) / 2

    classify_report = ''
    if calc_report:

        classify_report = classification_report(y_trues, y_preds)

        classify_report = format_classify_report(classify_report, id2label)

    return f1_micro, f1_macro, avg_f1, classify_report


def format_classify_report(classify_report, id2label):
    splits = re.split('\n', classify_report)
    new_classify_report = 'classify_report: \n'
    for i, var in enumerate(splits):

        if (i >= 2 and i < 2 + len(id2label)):
            temps = var.split(' ')
            id_ = int(temps[11])
            label = id2label.get(id_)
            temps[11] = label
            var2 = ' '.join(temps)
            new_classify_report += var2
            new_classify_report += '\n'
        else:
            new_classify_report += var
            new_classify_report += '\n'
    return new_classify_report


def calc_f_beta(y_trues, y_preds, id2label, calc_report=False, focus_beta=2, nonfocus_beta=0.5):
    if isinstance(y_trues, np.ndarray):
        y_trues = np.asarray(y_trues)

    if isinstance(y_preds, np.ndarray):
        y_preds = np.asarray(y_preds)

    classify_report = ''
    if calc_report:
        # 单标签
        category = list(id2label.values())
        y_true_labels = []
        for y_true in y_trues:
            y_true_label = id2label[y_true[0]]
            y_true_labels.append(y_true_label)

        y_pred_labels = []
        for y_pred in y_preds:
            y_pred = id2label[y_pred[0]]
            y_pred_labels.append(y_pred)

        classify_report = classification_report(y_true_labels, y_pred_labels, labels=category)

    classes = list(id2label.keys())
    mlb = MultiLabelBinarizer(classes=classes)

    y_trues = mlb.fit_transform(y_trues)
    y_preds = mlb.transform(y_preds)

    # 用于计算宏平均
    true_label_num = np.sum(y_trues, axis=0)
    pred_label_num = np.sum(y_preds, axis=0)
    correct_label_num = np.sum(np.multiply(y_trues, y_preds), axis=0)

    if type(correct_label_num) != np.ndarray:
        correct_label_num = np.asarray(correct_label_num)

    nonfocus_P = safe_divide(correct_label_num[0], pred_label_num[0])
    nonfocus_R = safe_divide(correct_label_num[0], true_label_num[0])
    nonfocus_F05 = safe_divide((1 + nonfocus_beta ** 2) * nonfocus_P * nonfocus_R,
                                      nonfocus_beta ** 2 * nonfocus_P + nonfocus_R)

    focus_P = safe_divide(correct_label_num[1], pred_label_num[1])
    focus_R = safe_divide(correct_label_num[1], true_label_num[1])
    focus_F2 = safe_divide((1 + focus_beta ** 2) * focus_P* focus_R,
                                    focus_beta ** 2 * focus_P + focus_R)

    avg_f = (nonfocus_F05 + focus_F2) / 2

    return [focus_P, focus_R, focus_F2], [nonfocus_P, nonfocus_R, nonfocus_F05], avg_f, classify_report
