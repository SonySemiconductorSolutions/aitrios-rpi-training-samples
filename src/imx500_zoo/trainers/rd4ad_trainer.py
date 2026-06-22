import torch
from third_party.rd4ad.test import evaluation
import os
import csv
import matplotlib.pyplot as plt
import torch.nn.functional as F


class RD4ADTrainer:
    def __init__(self, config):
        self.config = config

    def fit(self, model, dataloader_train, dataloader_valid):
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.encoder.eval()

        self.lr = float(self.config["TRAINER"]["LEARNING_RATE"])
        self.num_epochs = int(self.config["TRAINER"]["NUM_EPOCHS"])
        self.optimizer = torch.optim.Adam(
            list(self.model.bn.parameters()) +
            list(self.model.decoder.parameters()),
            lr=self.lr,
            betas=(0.5, 0.999)
        )

        self.dataloader_train = dataloader_train
        self.dataloader_valid = dataloader_valid

        print("Training started on", self.device)

        log_path = os.path.join(
            self.config["PATH"]["MODEL"], "training_log.csv")
        with open(log_path, mode='w', newline='') as log_file:
            writer = csv.writer(log_file)
            writer.writerow(["Epoch", "Loss_train", "Loss_valid",
                            "Seg-AUROC", "Det-AUROC", "AUPRO"])

        for epoch in range(self.num_epochs):
            loss_train = self._training_step()
            loss_valid, auroc_px, auroc_sp, aupro_px = self._validation_step()
            print(
                f"Epoch: {epoch+1}/{self.num_epochs}, "
                f"Loss_train: {loss_train:.4f}, "
                f"Loss_valid: {loss_valid:.4f}, "
                f"Seg-AUROC: {auroc_px * 100:.2f}%, "
                f"Det-AUROC: {auroc_sp * 100:.2f}%, "
                f"AUPRO: {aupro_px * 100:.2f}%"
            )

            with open(log_path, mode='a', newline='') as log_file:
                writer = csv.writer(log_file)
                writer.writerow([
                    epoch + 1,
                    f"{loss_train:.4f}",
                    f"{loss_valid:.4f}",
                    f"{auroc_px * 100:.2f}",
                    f"{auroc_sp * 100:.2f}",
                    f"{aupro_px * 100:.2f}"
                ])

        self.pytorch_path = self.config["PATH"]["MODEL"] + \
            self.config["SOLUTION"]["NAME"] + ".pth"
        torch.save(self.model.state_dict(), self.pytorch_path)
        print(f"Model saved to: {self.pytorch_path}")

        epochs, loss_train, loss_valid = [], [], []
        seg_auroc, det_auroc, aupro = [], [], []

        with open(log_path, 'r') as log_file:
            reader = csv.DictReader(log_file)
            for row in reader:
                epochs.append(int(row["Epoch"]))
                loss_train.append(float(row["Loss_train"]))
                loss_valid.append(float(row["Loss_valid"]))
                seg_auroc.append(float(row["Seg-AUROC"]))
                det_auroc.append(float(row["Det-AUROC"]))
                aupro.append(float(row["AUPRO"]))

        fig, ax1 = plt.subplots()

        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Accuracy [%]", color="tab:blue")
        ax1.plot(epochs, seg_auroc, label="Seg-AUROC",
                 color="tab:blue", linestyle='-')
        ax1.plot(epochs, det_auroc, label="Det-AUROC",
                 color="tab:cyan", linestyle='--')
        ax1.plot(epochs, aupro, label="AUPRO",
                 color="tab:purple", linestyle='-.')
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Loss", color="tab:red")
        ax2.plot(epochs, loss_train, label="Loss_train",
                 color="tab:red", linestyle='-')
        ax2.plot(epochs, loss_valid, label="Loss_valid",
                 color="tab:orange", linestyle='--')
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax2.legend(loc="upper right")

        graph_path = os.path.join(
            self.config["PATH"]["MODEL"], "training_log_graph.png")
        plt.title("Training Progress")
        fig.tight_layout()
        plt.savefig(graph_path)
        plt.close(fig)
        print(f"Training progress graph saved to {graph_path}")

        if self.device == "cuda":
            torch.cuda.empty_cache()

        return self.model

    def _training_step(self):
        self.model.bn.train()
        self.model.decoder.train()

        total_loss = 0
        for images, _ in self.dataloader_train:
            images = images.to(self.device)

            features = self.model.encoder(images)
            normalized_features = self.model.bn(features)
            reconstructed = self.model.decoder(normalized_features)

            loss = self._loss_function(features, reconstructed)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.dataloader_train)

    def _validation_step(self):
        self.model.bn.eval()
        self.model.decoder.eval()

        total_loss = 0
        with torch.no_grad():
            for images, _, _, _ in self.dataloader_valid:
                images = images.to(self.device)

                features = self.model.encoder(images)
                normalized_features = self.model.bn(features)
                reconstructed = self.model.decoder(normalized_features)

                loss = self._loss_function(features, reconstructed)
                total_loss += loss.item()

        auroc_px, auroc_sp, aupro_px = evaluation(
            self.model, self.dataloader_valid, self.device)

        return (
            total_loss / len(self.dataloader_valid),
            auroc_px,
            auroc_sp,
            aupro_px
        )

    def _loss_function(self, features, reconstructed):
        cos_loss = torch.nn.CosineSimilarity()
        loss = 0
        for f, r in zip(features, reconstructed):
            if f.size() != r.size():
                r = F.interpolate(
                    r, size=f.size()[2:], mode="bilinear", align_corners=False)
            loss += torch.mean(1 - cos_loss(f.view(f.size(0), -1),
                               r.view(r.size(0), -1)))
        return loss
