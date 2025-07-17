import torch
import torch.nn as nn


class LSTM_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM_Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.LSTM = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)

    def forward(self, x):
        output, (hidden_state, _) = self.LSTM(x)
        hidden_state0 = hidden_state.squeeze(0)  #(num_layers, batch_size, hidden_size)

        return output, hidden_state0

class LSTM_Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, device):
        super(LSTM_Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.device = device

        self.LSTM = nn.LSTM(input_size=self.hidden_size,
                             hidden_size=self.hidden_size,
                             num_layers=self.num_layers,
                             batch_first=True)
        self.fc_out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, z, output):

        z = z.unsqueeze(0)
        c0 = torch.zeros(z.shape).to(self.device)
        output = torch.flip(output, dims=[1])

        outputx, _ = self.LSTM(output, (z, c0))
        outputx = torch.flip(outputx, dims=[1])
        outputx = self.fc_out(outputx)

        return outputx

class MatrixDecomposer(nn.Module):
    def __init__(self, hidden_size):
        super(MatrixDecomposer, self).__init__()
        self.hidden_size = hidden_size

        self.weight = nn.Parameter(torch.randn(self.hidden_size))
        self.sigmoid = nn.Sigmoid()

    def forward(self, Z):
        weight = self.sigmoid(self.weight)

        Z_star = weight * Z
        Z_prime = Z - Z_star
        return Z_prime, Z_star


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
            x = self.fc_reduction(x)
        fc_input = x.view(x.size(0), -1)

        if self.num_classes < int(self.input_size / 4):
            x = self.layer5.fc1(fc_input)
            x = self.layer5.fc_nb(x)
            x = self.layer5.fc_relu1(x)
            x = self.layer5.fc2(x)
            x = self.layer5.c_drop1(x)
            x = self.layer5.fc_nb1(x)
            x = self.layer5.fc_relu2(x)
            features = x

            x = self.layer5.fc3(x)
            probs = self.layer5.c_softmax(x)

        elif self.num_classes < int(self.input_size / 2):
            x = self.layer5.fc1_2(fc_input)
            x = self.layer5.fc_nb_2(x)
            x = self.layer5.fc_relu1_2(x)
            features = x

            x = self.layer5.fc2_2(x)
            probs = self.layer5.c_softmax_2(x)

        else:
            features = fc_input
            x = self.layer5.fc1_3(fc_input)
            probs = self.layer5.c_softmax_3(x)

        return probs, features

class FDDGNet(nn.Module):
    def __init__(self, input_size, hidden_size,seq_len, classNumber, domainNumber,device):

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.classNumber = classNumber
        self.domainNumber = domainNumber
        self.device = device

        self.encoder = LSTM_Encoder(self.input_size, self.hidden_size, num_layers=1)
        self.matrix_decomposer = MatrixDecomposer(self.hidden_size)
        self.decoder = LSTM_Decoder(self.hidden_size, self.input_size, num_layers=1, device=self.device)
        self.emotionClassifier = Classifier(self.hidden_size, self.classNumber)
        self.domainClassifier = Classifier(self.hidden_size, self.domainNumber)


    def forward(self, EEG):
        EEG = EEG.to(torch.float32)

        output_Z, Z = self.encoder(EEG)

        Z = self.my_normlized(Z)

        Z_prime,Z_star = self.matrix_decomposer(Z)

        mix_feature_domain = Z

        domain_Z_prime, features_Z_prime = self.domainClassifier(Z_prime)

        domain_Z, features_z = self.domainClassifier(mix_feature_domain)

        rebuild_eeg = self.decoder(Z, output_Z)

        predict_emotion, features_Z_star = self.emotionClassifier(Z_star)

        return predict_emotion, rebuild_eeg, domain_Z_prime, Z, Z_prime, Z_star, domain_Z, features_Z_prime, features_Z_star

    def my_normlized(self,Z):
        Z_min = Z.min(dim=0, keepdim=True)[0]
        Z_max = Z.max(dim=0, keepdim=True)[0]
        denominator = torch.maximum(Z_max - Z_min, torch.tensor(1e-8))
        Z_normalized = (Z - Z_min) / denominator
        Z = Z_normalized
        return Z
