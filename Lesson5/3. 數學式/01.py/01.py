"""
這是一個簡單的神經網路模型，使用 Python 語言實現，主要分為以下幾個部分。

1.匯入必要的模組
"""
import numpy as np

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime
from losses import mse, mse_prime,mae,mae_prime
import debug
"""
導入了 numpy、自定義的神經網路、全連接層、激活層、激活函數、損失函數和自定義的除錯工具等模組。
2.訓練數據準備
"""
# 0 -> 0 , 1-> 1
x_train = np.array([[[0]], [[1]] ])
y_train = np.array([[[0]], [[1]] ])
"""
x_train 和 y_train 分別為輸入和輸出的訓練數據，是一個包含多個樣本的 numpy 陣列。在此範例中，x_train 和 y_train 都只包含兩個樣本，每個樣本只有一個輸入和一個輸出。
3.構建神經網路
"""
net = Network()
net.add(FCLayer(1, 1))
# net.add(ActivationLayer(tanh, tanh_prime))

# 創建一個神經網路對象 net，然後向 net 中添加一個全連接層。此全連接層的輸入維度為 1，輸出維度為 1，即該層只有一個神經元。
#  4. 設置損失函數、優化器和超參數

net.use(mae, mae_prime)
#.將損失函數設置為 mae（平均絕對誤差），優化器則是隨機梯度下降法。訓練過程中的超參數有兩個：epochs 表示訓練輪數，learning_rate 表示學習率。
#. 5. 訓練模型

net.fit(x_train, y_train, epochs=2, learning_rate=0.1)


#. 使用訓練數據 x_train 和 y_train 訓練神經網路，總共訓練 2 輪，學習率為 0.1。
#. 6. 測試模型


debug.debugPrint("預測----")
out = net.predict(x_train)
print(out)