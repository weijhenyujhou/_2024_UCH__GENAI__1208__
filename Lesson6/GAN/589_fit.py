# -*- coding: utf-8 -*-
# 資料來源： https://medium.com/datadriveninvestor/generative-adversarial-network-gan-using-keras-ce1c05cfdfd3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm       # 顯示進度   50%|█████     | 5/10 [00:21<00:22,  4.55s/it]


for i in tqdm(range(10)):
    print(i)

###############################################################################
# 讀取資料，並且標準化
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() # 讀取手寫資料
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5    # 標準化 -1~1

    # convert shape of x_train from (60000, 28, 28) to (60000, 784)
    # 784 columns per row
    print("x_train.shape:",x_train.shape)
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2]*1) #60000,784)
    return (x_train, y_train, x_test, y_test)


(X_train, y_train, X_test, y_test) = load_data()     # 讀取資料，並且標準化 -1~1
print("X_train.shape:",X_train.shape)



###############################################################################
"""
 discriminator 鑑別器 
 754 的輸入MLP
 1  個輸出
"""
def create_discriminator():
    discriminator = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=1024, input_dim=784,
                              activation=tf.keras.layers.LeakyReLU(0.2) ),
        tf.keras.layers.Dropout(rate=0.3),
        tf.keras.layers.Dense(units=512, activation=tf.keras.layers.LeakyReLU(0.2)),
        tf.keras.layers.Dropout(rate=0.3),
        tf.keras.layers.Dense(units=256, activation=tf.keras.layers.LeakyReLU(0.2)),
        tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)   # 0 或 1  的答案
    ])
    discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))
    return discriminator


print("=================discriminator 鑑別器 =================")
d = create_discriminator()
d.summary()
# pip install graphviz
# pip install pydotplus
# pip install pydot
# tf.keras.utils.plot_model(d, to_file='create_discriminator.png')
###############################################################################
"""
 generator 產生器
 輸入 100 的 MLP
 輸出 784 個 數字
"""

def create_generator():
    generator = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=256, input_dim=100,activation=tf.keras.layers.LeakyReLU(0.2)  ),
        tf.keras.layers.Dense(units=512, activation=tf.keras.layers.LeakyReLU(0.2)),
        tf.keras.layers.Dense(units=1024, activation=tf.keras.layers.LeakyReLU(0.2)),
        tf.keras.layers.Dense(units=784, activation=tf.nn.tanh)
    ])
    generator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))
    return generator

print("=================generator 產生器=================")
g = create_generator()
g.summary()
# tf.keras.utils.plot_model(g, to_file='create_generator.png')

###############################################################################
"""
 gan (Generative adversarial network) 生成對抗網絡
 輸入 100  的 MLP
 輸出 1  個 
"""
def create_gan(discriminator, generator):
    discriminator.trainable=False  # 默認情況下， tf.Keras 模型是可訓練的 - 
                                   # 您有兩種方法可以凍結所有權重

    # 建立Functional API model的程式片段如下：
    # 可以參考： https://ithelp.ithome.com.tw/articles/10234389
    gan_input = tf.keras.layers.Input(shape=(100,))       # 設定輸入 100 個 特徵值
    x = generator(gan_input)    
    gan_output= discriminator(x)
    gan= tf.keras.models.Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')

    return gan

gan = create_gan(d,g)  # d=鑑別器 模型  g=產生器 模型
gan.summary()
# tf.keras.utils.plot_model(gan, to_file='create_gan.png')


################################################################
"""
generator 預測答案
透過plt 顯示，並存成 png
"""
def plot_generated_images(epoch, generator, examples=100, dim=(10,10), figsize=(10,10)):
    noise= np.random.normal(loc=0, scale=1, size=[examples, 100]) # 產生100筆的 100個亂數
    generated_images = generator.predict(noise)                   # generator 預測
    # 畫出目前generator的預測
    generated_images = generated_images.reshape(100,28,28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)                           # 畫面切割
        plt.imshow(generated_images[i], interpolation='nearest')   # 顯示 目前得預測圖
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image %d.png' %epoch)               # 儲存

################################################################
""" 
"""
def training(epochs=1, batch_size=128):


    # Loading the data
    (X_train, y_train, X_test, y_test) = load_data()
    batch_count = X_train.shape[0] / batch_size  # 60000/128=468.75

    # Creating GAN
    generator = create_generator()
    discriminator = create_discriminator()
    gan = create_gan(discriminator, generator)

    ######
    """
    # 保存模型
#  tf.keras.models.save('model.h5')              # 這行為備註，原本用於保存模型
model.save('model.h5')                           # 保存完整模型到檔案 model.h5

##### 載入模型
model = tf.keras.models.load_model('model.h5')   # 載入已保存的模型

# 保存模型權重
model.save_weights("model.weights.h5")           # 保存模型的權重到檔案 model.weights.h5

# 讀取模型權重
model.load_weights("model.weights.h5")           # 從 model.weights.h5 檔案讀取模型權重
    """
    # 讀取權重
    try:

        # 保存模型架構
        discriminator.save_weights("discriminator.weights.h5")
        generator.save_weights("generator.weights.h5")
        gan.save_weights("gan.weights.h5") 
        
        # 保存模型權重
        discriminator.save_weights("discriminator.weights.h5")           # 保存模型的權重到檔案 model.weights.h5
        generator.save_weights("generator.weights.h5")           # 保存模型的權重到檔案 model.weights.h5
        gan.save_weights("gan.weights.h5")           # 保存模型的權重到檔案 model.weights.h5


        """
        with open("discriminator.json", "w") as json_file:
            json_file.write(discriminator.to_json())
        # 保存模型架構
        with open("generator.json", "w") as json_file:
            json_file.write(generator.to_json())
        # 保存模型架構
        with open("gan.json", "w") as json_file:
            json_file.write(gan.to_json())
        
        with open('discriminator.h5', 'r') as load_weights:
            # 讀取權重
            d.load_weights("discriminator.h5")
        with open('gan.h5', 'r') as load_weights:
            # 讀取權重
            gan.load_weights("gan.h5")
        with open('generator.h5', 'r') as load_weights:
            # 讀取權重
            gan.load_weights("generator.h5")
        """
    except IOError:
        print("File not accessible")

    ######
    for e in range(1, epochs + 1):
        print("/n Epoch",e,"/",epochs)
        for _ in tqdm(range(batch_size)):
            # generate  random noise as an input  to  initialize the  generator
            # 生成隨機噪聲作為輸入以初始化生成器
            noise = np.random.normal(0, 1, [batch_size, 100])  # 產生亂數資料

            # Generate fake MNIST images from noised input
            # 從噪聲輸入生成假 MNIST 圖像
            generated_images = generator.predict(noise)

            # Get a random set of  real images
            # 獲取一組隨機的真實圖像
            image_batch = X_train[np.random.randint(low=0, high=X_train.shape[0], size=batch_size)]

            # Construct different batches of  real and fake data
            # 構造不同批次的真假數據 一半假的資料，一半真的 MNIST
            X = np.concatenate([image_batch, generated_images])   # 放在後面

            # Labels for generated and real data
            # 生成數據和真實數據的標籤
            y_dis = np.zeros(2 * batch_size)
            y_dis[:batch_size] = 0.9               # 前面一半的Y資料0.9 後面一半為0

            # Pre train discriminator on  fake and real data  before starting the gan.
            # 在開始 GAN 之前，對假數據和真實數據進行預訓練鑑別器。
            discriminator.trainable = True          #  設定可以訓練
            discriminator.train_on_batch(X, y_dis)  #  訓練

            # Tricking the noised input of the Generator as real data
            # 將生成器的噪聲輸入欺騙為真實數據
            noise = np.random.normal(0, 1, [batch_size, 100]) # 產生0～1 亂數資料 [128,100]
            y_gen = np.ones(batch_size)             # 答案為 1

            # During the training of gan,
            # the weights of discriminator should be fixed.
            # We can enforce that by setting the trainable flag
            # 在 gan 的訓練過程中，
            # 鑑別器的權重應該是固定的。
            # 我們可以通過設置可訓練標誌來強制執行
            discriminator.trainable = False

            # training  the GAN by alternating the training of the Discriminator
            # and training the chained GAN model with Discriminator’s weights freezed.
            # 通過交替訓練鑑別器來訓練 GAN
            # 並在凍結判別器權重的情況下訓練鍊式 GAN 模型。
            gan.train_on_batch(noise, y_gen)

        if e == 1 or e % 20 == 0: # 每個一次echo 存一次照片
            plot_generated_images(e, generator) #  預測儲存圖片
            # 保存模型權重
            discriminator.save_weights("discriminator.weights.h5")           # 保存模型的權重到檔案 model.weights.h5
            generator.save_weights("generator.weights.h5")           # 保存模型的權重到檔案 model.weights.h5
            gan.save_weights("gan.weights.h5")           # 保存模型的權重到檔案 model.weights.h5




training(epochs=400, batch_size=128)   # 訓練GAN
