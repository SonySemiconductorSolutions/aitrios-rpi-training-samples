import torch


class ClassificationTrainer:
    def __init__(self, config):
        self.config = config

    def fit(self, model, dataloader_train, dataloader_valid):
        # model
        self.model = model
        self.torch_model = self.model.get()

        # Configrations
        self.num_classes = int(self.config["MODEL"]["NUM_CLASSES"])
        self.lr = float(self.config["TRAINER"]["LEARNING_RATE"])
        self.num_epochs = int(self.config["TRAINER"]["NUM_EPOCHS"])

        # Preperations
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.torch_model.parameters(), lr=self.lr
        )
        self.dataloader_train = dataloader_train
        self.dataloader_valid = dataloader_valid

        print("Training started in", self.device)

        # Training Loop
        self.torch_model.to(self.device)
        for epoch in range(self.num_epochs):
            loss_train = self._training_step()
            loss_valid, acc_valid = self._validation_step()
            print(
                f"Epoch: {epoch+1}/{self.num_epochs}, Loss_train: {loss_train}, Loss_valid: {loss_valid}, Acc_valid: {acc_valid}"
            )

        # clear GPU memory
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return self.model.get_trained_model()

    def _training_step(self):
        loss_train_sum = 0
        # Train
        self.torch_model.train()  # switch to training mode
        for inputs_train, labels_train in self.dataloader_train:
            # Inference
            inputs_train = inputs_train.to(self.device)
            labels_train = labels_train.to(self.device)
            self.optimizer.zero_grad()
            outputs_train = self.torch_model(inputs_train)
            # Loss
            loss_train = self.criterion(outputs_train, labels_train)
            loss_train.backward()
            # Log
            loss_train_sum += loss_train.item()
            # BP
            self.optimizer.step()

        loss_train_average = loss_train_sum / len(self.dataloader_train)
        return loss_train_average

    def _validation_step(self):
        loss_valid_sum = 0
        correct_valid_sum = 0
        total_instances_valid = 0
        self.torch_model.eval()  # switch to evaluation mode
        with torch.no_grad():
            for inputs_valid, labels_valid in self.dataloader_valid:
                # Inference
                inputs_valid = inputs_valid.to(self.device)
                labels_valid = labels_valid.to(self.device)
                outputs_valid = self.torch_model(inputs_valid)
                # Loss
                loss_valid = self.criterion(outputs_valid, labels_valid)
                # Accuracy
                classifications = torch.argmax(outputs_valid, dim=1)
                correct_predictions = sum(
                    classifications == labels_valid
                ).item()
                correct_valid_sum += correct_predictions
                # Log
                loss_valid_sum += loss_valid.item()
                total_instances_valid += len(inputs_valid)

        loss_valid_average = loss_valid_sum / len(self.dataloader_valid)
        acc_valid_average = correct_valid_sum / total_instances_valid

        return loss_valid_average, acc_valid_average
