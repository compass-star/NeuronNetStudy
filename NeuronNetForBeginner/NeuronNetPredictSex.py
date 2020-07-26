#! /usr/bin/python
# -*- coding:utf-8 -*-

'''
@Time    :   2020-07-21
@Version :   V1.0
@Remarks :   神经网络的训练预测简单实现，并预测性别
'''
import numpy as np


class NeuronNetWork(object):
    def __init__(self, learn_rate=0.1, epochs=1000):
        ### 初始化模型参数
        self.learn_rate = learn_rate
        self.epochs = epochs

        ### 初始化w参数
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        ### 初始化截距参数b
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    ### sigmoid求导
    def deriv_sigmoid(self, x):
        fx = self.sigmoid(x)
        return fx * (1 - fx)

    ### mse
    def mse_loss(self, y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean()

    ### 前向传播 即预测
    def feedforward(self, x):
        h1 = self.sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = self.sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = self.sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        '''
        - data is a (n x 2) numpy array, n = # of samples in the dataset.
        - all_y_trues is a numpy array with n elements.
          Elements in all_y_trues correspond to those in data.
        '''
        for epoch in range(self.epochs):
            for x, y_true in zip(data, all_y_trues):
                ### h1层的结果
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = self.sigmoid(sum_h1)

                ### h2层的结果
                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = self.sigmoid(sum_h2)

                ### o1层的结果
                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = self.sigmoid(sum_o1)
                y_pred = o1

                # --- Calculate partical derivatives
                # --- Naming: d_L_d_w1 represents "partial L / partial w1"
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Neuron o1
                d_L_d_w5 = d_L_d_ypred * h1 * self.deriv_sigmoid(sum_o1)
                d_L_d_w6 = d_L_d_ypred * h2 * self.deriv_sigmoid(sum_o1)
                d_L_d_b3 = d_L_d_ypred * self.deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * self.deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * self.deriv_sigmoid(sum_o1)

                # Neuron h1
                d_L_d_w1 = d_L_d_ypred * d_ypred_d_h1 * x[0] * self.deriv_sigmoid(sum_h1)
                d_L_d_w2 = d_L_d_ypred * d_ypred_d_h1 * x[1] * self.deriv_sigmoid(sum_h1)
                d_L_d_b1 = d_L_d_ypred * d_ypred_d_h1 * self.deriv_sigmoid(sum_h1)

                # Neuron h2
                d_L_d_w3 = d_L_d_ypred * d_ypred_d_h2 * x[0] * self.deriv_sigmoid(sum_h2)
                d_L_d_w4 = d_L_d_ypred * d_ypred_d_h2 * x[1] * self.deriv_sigmoid(sum_h2)
                d_L_d_b2 = d_L_d_ypred * d_ypred_d_h2 * self.deriv_sigmoid(sum_h2)

                ### -- update weights and bias
                self.w1 -= self.learn_rate * d_L_d_w1
                self.w2 -= self.learn_rate * d_L_d_w2
                self.w3 -= self.learn_rate * d_L_d_w3
                self.w4 -= self.learn_rate * d_L_d_w4
                self.w5 -= self.learn_rate * d_L_d_w5
                self.w6 -= self.learn_rate * d_L_d_w6

                self.b3 -= self.learn_rate * d_L_d_b3
                self.b2 -= self.learn_rate * d_L_d_b2
                self.b1 -= self.learn_rate * d_L_d_b1

            if epoch % 10 == 0:
                y_pred = np.apply_along_axis(self.feedforward, 1, data)
                loss = self.mse_loss(all_y_trues, y_pred)
                print("Epoch %d loss: %.3f" % (epoch, loss))

if __name__ == "__main__":
   # Define dataset
   data = np.array([
       [-2, -1],  # Alice
       [25, 6],   # Bob
       [17, 4],   # Charlie
       [-15, -6], # Diana
   ])
   all_y_trues = np.array([
       1,  # Alice
       0,  # Bob
       0,  # Charlie
       1,  # Diana
   ])

   # Train our neural network!
   network = NeuronNetWork()
   network.train(data, all_y_trues)

   # Predict
   # Make some predictions
   emily = np.array([-7, -3])  # 128 pounds, 63 inches
   frank = np.array([20, 2])  # 155 pounds, 68 inches
   print("Emily: %.3f" % network.feedforward(emily))  # 0.951 - F
   print("Frank: %.3f" % network.feedforward(frank))  # 0.039 - M