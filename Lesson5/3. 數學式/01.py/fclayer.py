#.這是一個 Python 程式，用來實現全連接層（Fully Connected Layer）的前向傳播和反向傳播功能。以下是程式碼的說明和註解：
# import Layer class from layer.py
from layer import Layer
import numpy as np
import debug

# 繼承 Layer 類別
class FCLayer(Layer):
    # 建構函式，初始化全連接層的權重和偏差
    # input_size: 輸入層神經元數量
    # output_size: 輸出層神經元數量
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5   #產生 array[input_size,output_size] 的二維陣列，並將每個值減 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # 前向傳播函式，計算輸入的輸出
    # input_data: 輸入數據
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        debug.debugPrint("input="+str(self.input)+"　乘　w="+str(self.weights)+" 加　bias="+str(self.bias)+"　等於　output="+str(self.output))
        return self.output

    # 反向傳播函式，計算損失對權重和偏差的梯度，並更新權重和偏差 #用答案回去調整計算式的權重
    # output_error: 輸出層損失對輸出的偏差
    # learning_rate: 學習率
    def backward_propagation(self, output_error, learning_rate):
        # 計算輸入層的損失對輸入的偏差，以便傳遞給上一層
        input_error = np.dot(output_error, self.weights.T)
        # 計算損失對權重的偏差
        weights_error = np.dot(self.input.T, output_error)
        debug.debugPrint("向後傳播 input_error   等於--->　 output_error=" + str(output_error) + "　乘self.weights.T=" + str(self.weights.T) + "　等於　input_error=" + str(input_error))
        debug.debugPrint("向後傳播 weights_error 等於--->　  self.input.T=" + str(self.input.T) + "　乘　output_error=" + str(output_error) + "　等於　weights_error=" + str(weights_error))

        # 更新權重和偏差
        self.weights2 =self.weights-( learning_rate * weights_error)
        self.bias2 = self.bias-( learning_rate * output_error)
        debug.debugPrint("更新後　self.weights2 = self.weights -( learning_rate * weights_error)  ---->"+str(self.weights2 )+"=" +str(self.weights)+"-( "+ str(learning_rate)+" * "+str(weights_error)+")")
        debug.debugPrint("更新後　self.bias2    = self.bias    -( learning_rate * output_error )  ---->" + str(
            self.bias2) + "=" + str(self.bias) + "-( " + str(learning_rate) + " * " + str(output_error) + ")")

        # 更新權重和偏差
        self.weights = self.weights2
        self.bias = self.bias2
        # 返回輸入層的損失對輸入的偏差
        return input_error