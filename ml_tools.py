# !/usr/bin/env python
# -*-encoding: utf-8-*-
# author:LiYanwei
# version:0.1


from __future__ import division, print_function
import pandas as pd
from sklearn import preprocessing
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


def perform_machine_learning(data_filepath, numeric_cols, category_cols, label_col):
    """
        数据集处理及模型学习

        参数
        ======
        data_filepath:  数据集路径
        numeric_cols:   数值类型列
        category_cols:  类别类型列
        label_col:      标签列

        返回值
        ======
        None
    """
    data = pd.read_csv(data_filepath)
    numeric_feat = data[numeric_cols].values
    category_val = data[category_cols].values[:, 0]  # 如果有多列，每次处理一列

    # 处理类别数据
    # label encoder
    label_enc = preprocessing.LabelEncoder()
    label_val = label_enc.fit_transform(category_val)
    label_val = label_val.reshape(-1, 1)

    # one-hot encoder
    onehot_enc = preprocessing.OneHotEncoder()
    category_feat = onehot_enc.fit_transform(label_val)
    category_feat = category_feat.toarray()

    # 生成最终特征和标签用于模型的训练
    X = np.hstack((numeric_feat, category_feat))
    y = data[label_col].values

    # 数据集信息
    n_sample = y.shape[0]
    n_pos_sample = y[y == 1].shape[0]
    n_neg_sample = y[y == 0].shape[0]
    print('样本个数：{}; 正样本占{:.2%}; 负样本占{:.2%}'.format(n_sample,
                                                   n_pos_sample / n_sample,
                                                   n_neg_sample / n_sample))
    print('特征维数：', X.shape[1])

    # 处理不平衡数据
    sm = SMOTE(random_state=42)
    X, y = sm.fit_sample(X, y)
    print('通过SMOTE方法平衡正负样本后')
    n_sample = y.shape[0]
    n_pos_sample = y[y == 1].shape[0]
    n_neg_sample = y[y == 0].shape[0]
    print('样本个数：{}; 正样本占{:.2%}; 负样本占{:.2%}'.format(n_sample,
                                                   n_pos_sample / n_sample,
                                                   n_neg_sample / n_sample))

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

    # 课后作业：C为超参数，尝试使用交叉验证选取最优的C值
    lr_model = LogisticRegression(C=1.0)
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)

    accuracy = metrics.accuracy_score(y_pred, y_test)
    precision = metrics.precision_score(y_pred, y_test, pos_label=1)
    recall = metrics.recall_score(y_pred, y_test, pos_label=1)

    print(accuracy)
    print(precision)
    print(recall)


if __name__ == '__main__':
    # 用于单元测试

    numeric_cols = ['int_rate', 'grade_feat', 'loan_amnt', 'installment', 'annual_inc', 'dti',
                    'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util',
                    'total_acc', 'collections_12_mths_ex_med', 'acc_now_delinq', 'term_feat',
                    'installment_feat', 'emp_length_feat']

    category_cols = ['home_ownership']

    label_col = ['label']
    data_filepath = './output/proc_data.csv'

    perform_machine_learning(data_filepath, numeric_cols, category_cols, label_col)
