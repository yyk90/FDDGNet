"""
模型训练文件。定义模型的训练（train），测试（evaluate）方法，在外部直接调用。
"""
import csv

import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
from Path import Path
from model.FDDGNet import FDDGNet
from torch.utils.data import DataLoader
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt



class FDDGNetTrainer(object):
    """
    Information_Theory_DG模型训练类。定义模型训练过程。
    """

    def __init__(self, input_size, hidden_size,seq_len, classNumber, domainNumber,datasets,subject_id):
        """
        模型导入及参数设置。
        """
        # 导入模型
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

        # 设置随机数种子
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # 导入模型
        self.model = FDDGNet(input_size=input_size, hidden_size=hidden_size, seq_len=seq_len, classNumber=classNumber, domainNumber=domainNumber,device=self.device)

        self.model = self.model.to(torch.float32)
        self.model = self.model.to(self.device)

        self.classNumber = classNumber
        self.datasets = datasets
        self.subject_id = subject_id
        self.seq_len = seq_len

        self.epsilon = 1e-8

        # 学习率
        self.learning_rate = 1e-3
        # 批次大小(以秒为单位，每秒采样率128)
        self.batch_size = 512
        # 训练轮数
        self.num_epochs = 10001
        # 训练集数据加载器
        self.train_loader = None
        # 测试集数据加载器
        self.test_loader = None
        # 模型优化器，针对不同模块设置不同学习率
        self.optimizer = optim.Adam([{"params": self.model.emotion_related_Encoder.parameters(), "lr": 0.0001},
                                     {"params": self.model.matrix_decomposer.parameters(), "lr": 0.0001},
                                     {"params": self.model.decoder.parameters(), "lr": 0.0001},
                                     {"params": self.model.domainClassifier.parameters(), "lr": 0.0001},
                                     {"params": self.model.emotionClassifier.parameters(), "lr": 0.0001}])
        self.delta = 0.05
        self.alpha = 3
        self.gamma = 5
        self.epsilon = 5
        self.bate = 0.004

        self.save_level = 0.65

        if datasets == 'amigos&dreamer':
            self.save_level = 0.50

        if datasets == 'dreamer' or (datasets == 'amigos&dreamer' and subject_id == 2):
            self.optimizer = optim.Adam([{"params": self.model.emotion_related_Encoder.parameters(), "lr": 0.001},
                                         {"params": self.model.matrix_decomposer.parameters(), "lr": 0.0005},
                                         {"params": self.model.decoder.parameters(), "lr": 0.001},
                                         {"params": self.model.domainClassifier.parameters(), "lr": 0.0005},
                                         {"params": self.model.emotionClassifier.parameters(), "lr": 0.0005}])
            self.delta = 0.5

        # EEG重建损失
        self.recon_loss = nn.MSELoss()
        # 情绪分类损失
        self.emotion_loss = nn.CrossEntropyLoss()
        # 域分类损失
        self.domain_loss = nn.CrossEntropyLoss()

        # 项目路径********************************************************************************
        self.path = Path.PROJECT_DIR


        self.save_path = "/loss_result/FDDGNet/" + self.datasets

        # # 载入预训练好的模型
        # checkpoint = torch.load(self.path + "/checkpoint/AMIGOS_LSTM_VAE_domain_512_230.pth")
        # #checkpoint = torch.load(self.path + "/checkpoint/AMIGOS_VAE_emotion_1v450.pth")
        # self.model.load_state_dict(checkpoint['net'])


    def train(self, train_dataset, test_dataset):
        """
        模型训练方法。
        """
        if not os.path.isdir(self.path + self.save_path + "/loss_excel"):
            os.makedirs(self.path + self.save_path + "/loss_excel")

        if not os.path.isfile(self.path + self.save_path + '/loss_excel/result_test_'+self.datasets+'_'+str(self.subject_id)+'.csv'):
            with open(self.path + self.save_path + '/loss_excel/result_test_'+self.datasets+'_'+str(self.subject_id)+'.csv', 'w', newline='') as file:
                flags = 1
                writer = csv.writer(file)
                writer.writerow(['epoch','Z_star_emotion_loss', 'Z_prime_domain_loss','orthogonal_loss', 'kl_loss', 'sufficiency_loss','recon_loss', 'train_accuracy_emotion', 'train_accuracy_domain', 'accuracy', 'precision_score', 'recall_score', 'roc_auc', 'f1_score', 'specificity'])
        else:
            flags = 0

        # 训练集数据加载器
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        # 测试集数据加载器
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        if flags == 1:

            for epoch in range(self.num_epochs+1):

                # 开启模型训练
                self.model.train()

                times = 0
                train_accuracy_domain = 0
                train_accuracy_Z_star = 0
                total_recon_loss = 0.0
                total_Z_prime_domain_loss = 0
                total_orthogonal_loss = 0
                total_sufficiency_loss = 0
                total_kl_loss = 0
                total_Z_star_emotion_loss = 0

                for eeg, label, subject_id, trail_id, _, _ in self.train_loader:

                    eeg, label, subject_id, trail_id = eeg.to(self.device), label.to(self.device), subject_id.to(self.device),trail_id.to(self.device)   # 将数据移到 GPU

                    # 梯度置零
                    self.optimizer.zero_grad()

                    # 前向传播
                    eeg = eeg.to(torch.float32)

                    predict_emotion, rebuild_eeg, domain_Z_prime, Z, Z_prime, Z_star, mix_domain, features_Z_prime, features_mix  = self.model(eeg)
                    predict_emotion, rebuild_eeg, domain_Z_prime, Z, Z_prime, Z_star, mix_domain = predict_emotion.to(self.device), rebuild_eeg.to(self.device), domain_Z_prime.to(self.device), Z.to(self.device), Z_prime.to(self.device), Z_star.to(self.device), mix_domain.to(self.device)

                    Z_star_emotion_loss = self.emotion_loss(predict_emotion, label)

                    recon_loss = self.recon_loss(rebuild_eeg, eeg)

                    Z_prime_domain_loss = self.domain_loss(domain_Z_prime, subject_id)

                    orthogonal_loss = torch.norm(torch.mm(Z_prime.T, Z_star), p='fro') ** 2   # 正交性约束

                    # 直接归一化为概率分布
                    mix_domain_2 = mix_domain / mix_domain.sum(dim=1, keepdim=True)
                    domain_Z_prime_2 = domain_Z_prime / domain_Z_prime.sum(dim=1, keepdim=True)

                    # 避免数值问题
                    epsilon = 1e-8
                    mix_domain_2 = torch.clamp(mix_domain_2, min=epsilon)
                    domain_Z_prime_2 = torch.clamp(domain_Z_prime_2, min=epsilon)

                    # 计算 KL 散度
                    sufficiency_loss = torch.nn.functional.kl_div(mix_domain_2.log(), domain_Z_prime_2, reduction='batchmean')


                    kl_loss = self.kl_divergence_with_standard_normal(Z_prime)            # KL散度逼近正态分布

                    l_recon = recon_loss
                    l_pro = orthogonal_loss
                    l_ece = Z_star_emotion_loss
                    l_sib = (Z_prime_domain_loss + self.bate * kl_loss) + sufficiency_loss

                    loss = self.alpha * l_pro + self.gamma * l_ece + self.epsilon * l_sib + self.delta * l_recon

                    loss = loss.to(self.device)
                    # 反向传播
                    loss = loss.float()
                    loss.backward()
                    # 更新模型参数
                    self.optimizer.step()

                    # 统计各类损失
                    total_Z_star_emotion_loss += Z_star_emotion_loss
                    total_recon_loss += recon_loss
                    total_Z_prime_domain_loss += Z_prime_domain_loss
                    total_orthogonal_loss += orthogonal_loss
                    total_sufficiency_loss += sufficiency_loss
                    total_kl_loss += kl_loss

                    # 计算训练集上的准确率
                    times += label.size(0)

                    predict_domain = torch.argmax(domain_Z_prime, dim=1)
                    train_accuracy_domain += (predict_domain == subject_id).sum().item()

                    predict_emotionss = torch.argmax(predict_emotion, dim=1)
                    train_accuracy_Z_star += (predict_emotionss == label).sum().item()

                # 计算平均损失
                avg_Z_star_emotion_loss = total_Z_star_emotion_loss / len(self.train_loader)
                avg_recon_loss = total_recon_loss / len(self.train_loader) / self.seq_len
                avg_Z_prime_domain_loss = total_Z_prime_domain_loss / len(self.train_loader)
                avg_orthogonal_loss = total_orthogonal_loss / len(self.train_loader)
                avg_sufficiency_loss = total_sufficiency_loss / len(self.train_loader)
                avg_kl_loss = total_kl_loss / len(self.train_loader)

                train_accuracy_domain = train_accuracy_domain / times
                train_accuracy_Z_star = train_accuracy_Z_star / times

                # accuracy = 0
                accuracy, precision_score, recall_score, roc_auc, f1_score, specificity = self.evaluate(epoch,self.classNumber)

                # 输出模型损失
                print("**************************************************************************************")
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], EEG重建损失: {avg_recon_loss:.4f}")
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], Z_star情绪分类损失: {avg_Z_star_emotion_loss:.4f}")
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], Z_prime域分类损失: {avg_Z_prime_domain_loss:.4f}")
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], Z_prime,Z_star正交损失: {avg_orthogonal_loss:.4f}")
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], Z_star与域标签无关损失: {avg_sufficiency_loss:.4f}")
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], Z_prime稀疏性约束损失: {avg_kl_loss:.4f}")
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], Z_star在训练集上的情绪准确率: {train_accuracy_Z_star:.4f}")
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], Z_prime在训练集上的域准确率: {train_accuracy_domain:.4f}")
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], 在测试集上的准确率: {accuracy:.4f}")
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], 在测试集上的精确率分数: {precision_score:.4f}")
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], 在测试集上的召回率分数: {recall_score:.4f}")
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], 在测试集上的roc_auc: {roc_auc:.4f}")
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], 在测试集上的F1分数: {f1_score:.4f}")
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], 在测试集上的特异性分数: {specificity:.4f}")

                with open(self.path + self.save_path + '/loss_excel/result_test_'+self.datasets+'_'+str(self.subject_id)+'.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        f"{epoch:.4f}",
                        f"{avg_Z_star_emotion_loss:.4f}",
                        f"{avg_Z_prime_domain_loss:.4f}",
                        f"{avg_orthogonal_loss:.4f}",
                        f"{avg_kl_loss:.4f}",
                        f"{avg_sufficiency_loss:.4f}",
                        f"{avg_recon_loss:.4f}",
                        f"{train_accuracy_Z_star:.4f}",
                        f"{train_accuracy_domain:.4f}",
                        f"{accuracy:.4f}",  # 假设 accuracy 是标量
                        f"{precision_score:.4f}",
                        f"{recall_score:.4f}",
                        f"{roc_auc:.4f}",
                        f"{f1_score:.4f}",
                        f"{specificity:.4f}"
                    ])

                # 定期保存模型数据
                if accuracy > self.save_level or epoch % 100 == 0 and epoch != 0:
                    if not os.path.isdir(self.path + self.save_path + "/checkpoint/subject_" + str(self.subject_id)):
                        os.makedirs(self.path + self.save_path + "/checkpoint/subject_" + str(self.subject_id))
                    checkpoint = {
                        "net": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "epoch": epoch
                    }
                    torch.save(checkpoint, self.path + self.save_path + "/checkpoint/subject_" + str(self.subject_id) + "/checkpoint_plan_A_%s.pth" % (str(epoch)))

                # 清理缓存
                torch.cuda.empty_cache()

    def evaluate(self, epoch, classNumber):
        """
        测试模型，所有计算在GPU上进行。
        :return: 准确率、精确率、召回率、AUC、F1、特异度。
        """
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for eeg, label, *_ in self.test_loader:
                eeg, label = eeg.to(self.device), label.to(self.device)
                output = self.model(eeg)[0]  # logits

                probs = torch.softmax(output, dim=1)  # (batch, num_classes)
                preds = torch.argmax(probs, dim=1)

                all_preds.append(preds)
                all_labels.append(label)
                all_probs.append(probs[:, 1])  # 假设二分类任务，取正类概率

        # 拼接结果（保持在GPU上）
        y_pred = torch.cat(all_preds, dim=0)
        y_true = torch.cat(all_labels, dim=0)
        y_score = torch.cat(all_probs, dim=0)

        # 准确率
        accuracy = (y_pred == y_true).float().mean()

        # 精确率、召回率、F1（macro）
        precisions = []
        recalls = []
        f1s = []
        for cls in range(classNumber):
            tp = ((y_pred == cls) & (y_true == cls)).sum().float()
            fp = ((y_pred == cls) & (y_true != cls)).sum().float()
            fn = ((y_pred != cls) & (y_true == cls)).sum().float()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        precision = torch.stack(precisions).mean()
        recall = torch.stack(recalls).mean()
        f1 = torch.stack(f1s).mean()

        # AUC（仅限二分类）
        if classNumber == 2:
            y_true_float = y_true.float()
            # 排序法计算AUC
            desc_score_indices = torch.argsort(y_score, descending=True)
            y_true_sorted = y_true_float[desc_score_indices]
            cum_pos = torch.cumsum(y_true_sorted == 1, dim=0)
            cum_neg = torch.cumsum(y_true_sorted == 0, dim=0)
            total_pos = (y_true == 1).sum().float()
            total_neg = (y_true == 0).sum().float()

            auc = (cum_pos[y_true_sorted == 0].sum()) / (total_pos * total_neg + 1e-8)
        else:
            auc = torch.tensor(0.0, device=self.device)

        # 特异度（binary only）
        if classNumber == 2:
            tn = ((y_pred == 0) & (y_true == 0)).sum().float()
            fp = ((y_pred == 1) & (y_true == 0)).sum().float()
            specificity = tn / (tn + fp + 1e-8)
        else:
            specificity = torch.tensor(0.0, device=self.device)

        return accuracy.item(), precision.item(), recall.item(), auc.item(), f1.item(), specificity.item()

    def kl_divergence_with_standard_normal(self, x):
        """
        计算特征矩阵 x 和标准正态分布之间的 KL 散度。
        x 的形状为 (batch_size, feature_size)，每一行是一个样本的特征向量。
        """
        # 计算每个样本的均值和标准差
        mu = torch.mean(x, dim=1, keepdim=True)  # 均值 (batch_size, 1)
        sigma = torch.std(x, dim=1, keepdim=True)  # 标准差 (batch_size, 1)

        # 防止 sigma 为 0，加入一个小的 epsilon 防止除0错误
        mu = torch.maximum(mu, torch.tensor(1e-8).to(mu.device))
        sigma = torch.maximum(sigma, torch.tensor(1e-8).to(sigma.device))


        # 计算 KL 散度
        kl_div = 0.5 * (mu ** 2 + sigma ** 2 - 1 - torch.log(sigma ** 2))

        # 返回每个样本的 KL 散度
        return torch.mean(kl_div)
