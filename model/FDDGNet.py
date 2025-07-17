import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


# 定义LSTM编码器
class LSTM_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,latent_size):
        super(LSTM_Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        # 定义LSTM层
        self.LSTM = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)

    def forward(self, x):
        output, (hidden_state, _) = self.LSTM(x)  # LSTM的前向传播，返回LSTM的隐藏状态
        hidden_state0 = hidden_state.squeeze(0)  # 去掉多余维度。输出维度为(num_layers, batch_size, hidden_size)

        return output, hidden_state0

class LSTM_Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, device,latent_size):
        super(LSTM_Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.device = device

        self.LSTMs111 = nn.LSTM(input_size=self.hidden_size,
                             hidden_size=self.hidden_size,
                             num_layers=self.num_layers,
                             batch_first=True)
        self.fc_out111 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, z, output,seq_len):

        z = z.unsqueeze(0)
        c0 = torch.zeros(z.shape).to(self.device)
        output = torch.flip(output, dims=[1])
        outputx, _ = self.LSTMs111(output, (z, c0))
        outputx = torch.flip(outputx, dims=[1])
        outputx = self.fc_out111(outputx)

        return outputx

# 定义矩阵分解模块
class MatrixDecomposer(nn.Module):
    def __init__(self, hidden_size):
        super(MatrixDecomposer, self).__init__()
        self.hidden_size = hidden_size
        self.P2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False) # bias=False表示不使用偏置项

        # 定义一个可学习的权重向量，大小为hidden_size
        self.weight = nn.Parameter(torch.randn(self.hidden_size))  # 初始化权重
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函数用于将权重限制在 [0, 1] 范围

    def forward(self, Z):
        # Sigmoid限制权重在[0, 1]范围
        weight = self.sigmoid(self.weight)
        # 域特定情绪相关特征 Z_prime
        Z_star = weight * Z
        Z_prime = Z - Z_star
        return Z_prime, Z_star


# 定义分类器
class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.fc_reduction = nn.Linear(2 * self.input_size, self.input_size)
        if self.num_classes < int(self.input_size/4):
            layer5 = nn.Sequential()
            layer5.add_module('fc1', nn.Linear(self.input_size, int(self.input_size/2)))
            layer5.add_module('fc_nb', nn.BatchNorm1d(int(self.input_size/2)))
            layer5.add_module('fc_relu1', nn.ReLU(True))
            layer5.add_module('fc2', nn.Linear(int(self.input_size/2), int(self.input_size/4)))
            layer5.add_module('c_drop1', nn.Dropout())
            layer5.add_module('fc_nb1', nn.BatchNorm1d(int(self.input_size/4)))
            layer5.add_module('fc_relu2', nn.ReLU(True))
            layer5.add_module('fc3', nn.Linear(int(self.input_size/4), self.num_classes))
            layer5.add_module('c_softmax', nn.LogSoftmax(dim=1))
        elif self.num_classes < int(self.input_size/2):
            layer5 = nn.Sequential()
            layer5.add_module("fc1_2", nn.Linear(self.input_size, int(self.input_size/2)))  # 128*5*5=3200
            layer5.add_module("fc_nb_2", nn.BatchNorm1d(int(self.input_size/2)))
            layer5.add_module("fc_relu1_2", nn.ReLU(True))
            layer5.add_module("fc2_2", nn.Linear(int(self.input_size/2), self.num_classes))
            layer5.add_module('c_softmax_2', nn.LogSoftmax(dim=1))
        else:
            layer5 = nn.Sequential()
            layer5.add_module("fc1_3", nn.Linear(self.input_size, self.num_classes))
            layer5.add_module('c_softmax_3', nn.LogSoftmax(dim=1))
        self.layer5 = layer5

    def forward(self, x):
        if x.shape[1] == 2 * self.input_size:
            # 如果是 2 * input_size，先通过降维层
            x = self.fc_reduction(x)
        fc_input = x.view(x.size(0), -1)  # 高维数据 ‘压’成 低维数据
        if self.num_classes < int(self.input_size / 4):
            # 逐层执行直到fc_relu2
            x = self.layer5.fc1(fc_input)
            x = self.layer5.fc_nb(x)
            x = self.layer5.fc_relu1(x)
            x = self.layer5.fc2(x)
            x = self.layer5.c_drop1(x)
            x = self.layer5.fc_nb1(x)
            x = self.layer5.fc_relu2(x)
            features = x  # 获取目标输出

            # 继续后续层
            x = self.layer5.fc3(x)
            probs = self.layer5.c_softmax(x)
        elif self.num_classes < int(self.input_size / 2):
            # 逐层执行直到fc_relu2
            x = self.layer5.fc1_2(fc_input)
            x = self.layer5.fc_nb_2(x)
            x = self.layer5.fc_relu1_2(x)
            features = x  # 获取目标输出

            # 继续后续层
            x = self.layer5.fc2_2(x)
            probs = self.layer5.c_softmax_2(x)
        else:
            features = fc_input
            x = self.layer5.fc1_3(fc_input)
            probs = self.layer5.c_softmax_3(x)

        return probs, features

class FDDGNet(nn.Module):
    """
    DGER，由以上各个模块组成。
    """
    def __init__(self, input_size, hidden_size,seq_len, classNumber, domainNumber,device):

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.classNumber = classNumber
        self.domainNumber = domainNumber
        self.device = device
        self.latent_size = 256

        self.emotion_related_Encoder = LSTM_Encoder(self.input_size, self.hidden_size, num_layers=1,latent_size=self.latent_size)
        self.matrix_decomposer = MatrixDecomposer(self.hidden_size)
        self.decoder = LSTM_Decoder(self.hidden_size, self.input_size, num_layers=1, device=self.device,latent_size=self.latent_size)
        self.emotionClassifier = Classifier(self.hidden_size, self.classNumber)
        self.domainClassifier = Classifier(self.hidden_size, self.domainNumber)


    def forward(self, EEG):
        # EEG类型转换
        EEG = EEG.to(torch.float32)

        # 提取特征
        output_Z, Z = self.emotion_related_Encoder(EEG)

        Z = self.my_normlized(Z)

        # # Z分解为Z_prime, Z_star
        Z_prime,Z_star = self.matrix_decomposer(Z)

        # 拼接特征
        mix_feature_domain = Z

        # Z_prime的域标签分类
        domain_Z_prime, features_Z_prime = self.domainClassifier(Z_prime)

        # 混合特征分类
        mix_domain, features_mix = self.domainClassifier(mix_feature_domain)

        # 解码器重建EEG
        rebuild_eeg = self.decoder(Z, output_Z, self.seq_len)

        # 最终预测情绪
        predict_emotion, features_Z_star = self.emotionClassifier(Z_star)

        return predict_emotion, rebuild_eeg, domain_Z_prime, Z, Z_prime, Z_star, mix_domain, features_Z_prime, features_Z_star

    def my_normlized(self,Z):
        # Min-Max 归一化到 [0, 1]
        Z_min = Z.min(dim=0, keepdim=True)[0]  # 计算每列的最小值
        Z_max = Z.max(dim=0, keepdim=True)[0]  # 计算每列的最大值
        fenmu = torch.maximum(Z_max - Z_min, torch.tensor(1e-8))
        Z_normalized = (Z - Z_min) / fenmu  # 归一化到 [0, 1]
        Z = Z_normalized
        return Z
