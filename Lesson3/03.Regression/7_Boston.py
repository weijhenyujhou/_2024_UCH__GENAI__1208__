#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ = "柯博文老師 Powen Ko, www.powenko.com"

# 匯入必要的模組
import tensorflow as tf  # TensorFlow：深度學習框架
from sklearn.model_selection import train_test_split  # sklearn 用於拆分資料集
import numpy as np        # NumPy：用於數值計算
import matplotlib.pyplot as plt  # Matplotlib：用於繪圖
import pandas as pd         # Pandas：用於資料處理與分析
# 從 TensorFlow 中的 keras 模組匯入波士頓房價數據集
from tensorflow.keras.datasets import boston_housing

# 加載波士頓房價數據集，分別為訓練集和測試集
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# 打印訓練數據的形狀（樣本數量和特徵數量）
print(x_train.shape)
# 打印訓練標籤的形狀（樣本數量）
print(y_train.shape)

# 定義特徵名稱列表，對應波士頓房價數據集中的各項特徵
classes = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
print(classes)  # 打印特徵名稱

# 將訓練數據轉換為 DataFrame 形式，列名使用特徵名稱列表
data = pd.DataFrame(x_train, columns=classes)
print(data.head())  # 打印前五行數據

# 將房價標籤添加為新列 'MEDV'
data['MEDV'] = pd.Series(data=y_train)
print(data.head())  # 再次打印前五行數據，查看 'MEDV' 列的添加情況
print(data.describe())  # 打印數據的描述性統計資訊

# 將數據保存為 CSV 文件，使用 Tab 鍵作為分隔符
data.to_csv("boston.csv", sep='\t')

# 使用 xlsxwriter 引擎將數據保存為 Excel 文件 需要安裝 xlsxwriter 模組 pip install xlsxwriter
writer = pd.ExcelWriter('boston.xlsx', engine='xlsxwriter')
data.to_excel(writer, sheet_name='Sheet1')  # 將數據寫入 Excel 文件中的 Sheet1 表
writer.close()  # 使用 close() 方法保存並關閉 Excel 文件#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ = "柯博文老師 Powen Ko, www.powenko.com"
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print(x_train.shape)
print(y_train.shape)

classes = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
print(classes)

data = pd.DataFrame(x_train, columns=classes)
print(data.head())


data['MEDV'] = pd.Series(data=y_train)
print(data.head())
print(data.describe())


data.to_csv("boston.csv", sep='\t')
writer = pd.ExcelWriter('boston.xlsx', engine='xlsxwriter')
data.to_excel(writer, sheet_name='Sheet1')
writer.close()


import seaborn as sns
sns.pairplot(data[["MEDV", "CRIM","AGE","DIS","TAX"]], diag_kind="kde")
plt.show()


g = sns.PairGrid(data[["MEDV", "CRIM","AGE","DIS","TAX"]])
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=6)
plt.show()