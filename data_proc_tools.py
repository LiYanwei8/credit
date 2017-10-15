# !/usr/bin/env python
# -*-encoding: utf-8-*-
# author:LiYanwei
# version:0.1

from __future__ import division, print_function


def insepct_data(df_data):
    """
        审查数据集

        参数
        ======
        df_data:    dataframe类型的数据

        返回值
        ======
        None
    """
    # 查看数据集信息
    print('\n数据预览：')
    print(df_data.head())

    print('\n数据统计信息：')
    print(df_data.describe())

    print('\n数据集基本信息：')
    print(df_data.info())
