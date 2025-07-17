
import csv

import torch
import os
import torch.nn as nn
import torch.optim as optim
from Path import Path
from model.FDDGNet import FDDGNet
from torch.utils.data import DataLoader



class FDDGNetTrainer(object):

    def __init__(self, input_size, hidden_size,seq_len, classNumber, domainNumber,datasets,subject_id):

        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        self.model = FDDGNet(input_size=input_size, hidden_size=hidden_size, seq_len=seq_len, classNumber=classNumber, domainNumber=domainNumber,device=self.device)

        self.model = self.model.to(torch.float32)
        self.model = self.model.to(self.device)

        self.classNumber = classNumber
        self.datasets = datasets
        self.subject_id = subject_id
        self.seq_len = seq_len

        self.epsilon = 1e-8

        self.learning_rate = 1e-3

        self.batch_size = 512

        self.num_epochs = 10001

        self.train_loader = None

        self.test_loader = None

        self.optimizer = optim.Adam([{"params": self.model.encoder.parameters(), "lr": 0.0001},
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

        self.recon_loss = nn.MSELoss()

        self.emotion_loss = nn.CrossEntropyLoss()

        self.domain_loss = nn.CrossEntropyLoss()

        self.path = Path.PROJECT_DIR


        self.save_path = "/loss_result/FDDGNet/" + self.datasets


    def train(self, train_dataset, test_dataset):
        if not os.path.isdir(self.path + self.save_path + "/loss_excel"):
            os.makedirs(self.path + self.save_path + "/loss_excel")

        if not os.path.isfile(self.path + self.save_path + '/loss_excel/result_test_'+self.datasets+'_'+str(self.subject_id)+'.csv'):
            with open(self.path + self.save_path + '/loss_excel/result_test_'+self.datasets+'_'+str(self.subject_id)+'.csv', 'w', newline='') as file:
                flags = 1
                writer = csv.writer(file)
                writer.writerow(['epoch','Z_star_emotion_loss', 'Z_prime_domain_loss','orthogonal_loss', 'kl_loss', 'sufficiency_loss','recon_loss', 'train_accuracy_emotion', 'train_accuracy_Z_prime', 'accuracy', 'precision_score', 'recall_score', 'roc_auc', 'f1_score', 'specificity'])
        else:
            flags = 0

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        if flags == 1:

            for epoch in range(self.num_epochs+1):

                self.model.train()

                times = 0
                train_accuracy_Z_prime = 0
                train_accuracy_Z_star = 0
                total_recon_loss = 0.0
                total_Z_prime_domain_loss = 0
                total_orthogonal_loss = 0
                total_sufficiency_loss = 0
                total_kl_loss = 0
                total_Z_star_emotion_loss = 0

                for eeg, label, subject_id, trail_id, _, _ in self.train_loader:

                    eeg, label, subject_id, trail_id = eeg.to(self.device), label.to(self.device), subject_id.to(self.device),trail_id.to(self.device)   # 将数据移到 GPU

                    self.optimizer.zero_grad()

                    eeg = eeg.to(torch.float32)

                    predict_emotion, rebuild_eeg, domain_Z_prime, Z, Z_prime, Z_star, domain_Z, features_Z_prime, features_mix  = self.model(eeg)
                    predict_emotion, rebuild_eeg, domain_Z_prime, Z, Z_prime, Z_star, domain_Z = predict_emotion.to(self.device), rebuild_eeg.to(self.device), domain_Z_prime.to(self.device), Z.to(self.device), Z_prime.to(self.device), Z_star.to(self.device), domain_Z.to(self.device)

                    Z_star_emotion_loss = self.emotion_loss(predict_emotion, label)

                    recon_loss = self.recon_loss(rebuild_eeg, eeg)

                    Z_prime_domain_loss = self.domain_loss(domain_Z_prime, subject_id)

                    orthogonal_loss = torch.norm(torch.mm(Z_prime.T, Z_star), p='fro') ** 2

                    domain_Z_new = domain_Z / domain_Z.sum(dim=1, keepdim=True)
                    domain_Z_prime_new = domain_Z_prime / domain_Z_prime.sum(dim=1, keepdim=True)

                    epsilon = 1e-8
                    domain_Z_new = torch.clamp(domain_Z_new, min=epsilon)
                    domain_Z_prime_new = torch.clamp(domain_Z_prime_new, min=epsilon)

                    sufficiency_loss = torch.nn.functional.kl_div(domain_Z_new.log(), domain_Z_prime_new, reduction='batchmean')

                    kl_loss = self.kl_divergence_with_standard_normal(Z_prime)

                    l_recon = recon_loss
                    l_pro = orthogonal_loss
                    l_ece = Z_star_emotion_loss
                    l_sib = (Z_prime_domain_loss + self.bate * kl_loss) + sufficiency_loss

                    loss = self.alpha * l_pro + self.gamma * l_ece + self.epsilon * l_sib + self.delta * l_recon

                    loss = loss.to(self.device)
                    loss = loss.float()
                    loss.backward()
                    self.optimizer.step()

                    total_Z_star_emotion_loss += Z_star_emotion_loss
                    total_recon_loss += recon_loss
                    total_Z_prime_domain_loss += Z_prime_domain_loss
                    total_orthogonal_loss += orthogonal_loss
                    total_sufficiency_loss += sufficiency_loss
                    total_kl_loss += kl_loss

                    times += label.size(0)

                    predict_domain = torch.argmax(domain_Z_prime, dim=1)
                    train_accuracy_Z_prime += (predict_domain == subject_id).sum().item()

                    predict_emotionss = torch.argmax(predict_emotion, dim=1)
                    train_accuracy_Z_star += (predict_emotionss == label).sum().item()

                avg_Z_star_emotion_loss = total_Z_star_emotion_loss / len(self.train_loader)
                avg_recon_loss = total_recon_loss / len(self.train_loader) / self.seq_len
                avg_Z_prime_domain_loss = total_Z_prime_domain_loss / len(self.train_loader)
                avg_orthogonal_loss = total_orthogonal_loss / len(self.train_loader)
                avg_sufficiency_loss = total_sufficiency_loss / len(self.train_loader)
                avg_kl_loss = total_kl_loss / len(self.train_loader)

                train_accuracy_Z_prime = train_accuracy_Z_prime / times
                train_accuracy_Z_star = train_accuracy_Z_star / times

                accuracy, precision_score, recall_score, roc_auc, f1_score, specificity = self.evaluate(self.classNumber)

                print("**************************************************************************************")
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], EEG recon loss: {avg_recon_loss:.4f}")
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], Z_star emotion CrossEntropyLoss: {avg_Z_star_emotion_loss:.4f}")
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], Z_prime domain CrossEntropyLoss: {avg_Z_prime_domain_loss:.4f}")
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], Orthogonal loss: {avg_orthogonal_loss:.4f}")
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], Sufficiency loss: {avg_sufficiency_loss:.4f}")
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], Kl_loss: {avg_kl_loss:.4f}")
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], Emotion train accuracy: {train_accuracy_Z_star:.4f}")
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], Domain train accuracy: {train_accuracy_Z_prime:.4f}")
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], Emotion test accuracy: {accuracy:.4f}")
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], precision_score: {precision_score:.4f}")
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], recall_score: {recall_score:.4f}")
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], roc_auc: {roc_auc:.4f}")
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], f1_score: {f1_score:.4f}")
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], specificity: {specificity:.4f}")

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
                        f"{train_accuracy_Z_prime:.4f}",
                        f"{accuracy:.4f}",
                        f"{precision_score:.4f}",
                        f"{recall_score:.4f}",
                        f"{roc_auc:.4f}",
                        f"{f1_score:.4f}",
                        f"{specificity:.4f}"
                    ])


                if accuracy > self.save_level or epoch % 100 == 0 and epoch != 0:
                    if not os.path.isdir(self.path + self.save_path + "/checkpoint/subject_" + str(self.subject_id)):
                        os.makedirs(self.path + self.save_path + "/checkpoint/subject_" + str(self.subject_id))
                    checkpoint = {
                        "net": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "epoch": epoch
                    }
                    torch.save(checkpoint, self.path + self.save_path + "/checkpoint/subject_" + str(self.subject_id) + "/checkpoint_plan_A_%s.pth" % (str(epoch)))

                torch.cuda.empty_cache()

    def evaluate(self, classNumber):

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
                all_probs.append(probs[:, 1])

        y_pred = torch.cat(all_preds, dim=0)
        y_true = torch.cat(all_labels, dim=0)
        y_score = torch.cat(all_probs, dim=0)

        accuracy = (y_pred == y_true).float().mean()

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

        if classNumber == 2:
            y_true_float = y_true.float()
            desc_score_indices = torch.argsort(y_score, descending=True)
            y_true_sorted = y_true_float[desc_score_indices]
            cum_pos = torch.cumsum(y_true_sorted == 1, dim=0)
            cum_neg = torch.cumsum(y_true_sorted == 0, dim=0)
            total_pos = (y_true == 1).sum().float()
            total_neg = (y_true == 0).sum().float()

            auc = (cum_pos[y_true_sorted == 0].sum()) / (total_pos * total_neg + 1e-8)
        else:
            auc = torch.tensor(0.0, device=self.device)

        if classNumber == 2:
            tn = ((y_pred == 0) & (y_true == 0)).sum().float()
            fp = ((y_pred == 1) & (y_true == 0)).sum().float()
            specificity = tn / (tn + fp + 1e-8)
        else:
            specificity = torch.tensor(0.0, device=self.device)

        return accuracy.item(), precision.item(), recall.item(), auc.item(), f1.item(), specificity.item()

    def kl_divergence_with_standard_normal(self, x):
        mu = torch.mean(x, dim=1, keepdim=True)
        sigma = torch.std(x, dim=1, keepdim=True)


        mu = torch.maximum(mu, torch.tensor(1e-8).to(mu.device))
        sigma = torch.maximum(sigma, torch.tensor(1e-8).to(sigma.device))

        kl_div = 0.5 * (mu ** 2 + sigma ** 2 - 1 - torch.log(sigma ** 2))

        return torch.mean(kl_div)
