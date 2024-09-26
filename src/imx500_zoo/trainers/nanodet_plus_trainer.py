import third_party.nanodet.nanodet_plus_trainer as NT


class NanodetPlusTrainer:
    def __init__(self, config):
        # call from imx500_zoo.py/Solution.setup_trainer(), setup_model()/setup_data() are executed
        self.config = config
        self.t = NT.NanodetPlusTrainer(config)

    def fit(self, model, dataloader_train, dataloader_valid):
        self.t.fit(model, dataloader_train, dataloader_valid)
