import os
from keras.optimizers import Adam
from keras.callbacks import (
    TensorBoard,
    ModelCheckpoint,
    EarlyStopping,
    LearningRateScheduler,
)
import multiprocessing
import gc
import copy

class DeeplabV3pTrainer:
    def __init__(self, config):
        self.ini = config

    def fit(self, model, dataloader_train, dataloader_valid):
        self.model_gen = model
        self.dataloader_train = dataloader_train
        self.dataloader_valid = dataloader_valid
        self.run()


    def build_callbacks(self, config, tf_board=False):
        """
        Build and return a list of Keras callbacks for model training.

        This function creates a set of callbacks to be used during the training
        of a deep learning model. The callbacks include model checkpointing,
        early stopping, learning rate reduction, and optional TensorBoard logging.
        The callbacks are configured based on the parameters provided and are
        intended to help in monitoring and managing the training process.

        Parameters:
        -----------
        config : class
            data of parsed yaml file
            - weights_dir: Directory path where model weights will be saved.
            - log_dir: Directory path where TensorBoard logs will be saved.
            - model_name: Name of the model to include in the file names.

        tf_board : bool, optional, default=False
            If True, includes TensorBoard callback in the list of callbacks.
            If False, TensorBoard callback is excluded.

        Returns:
        --------
        list
            A list of Keras callbacks including ModelCheckpoint, EarlyStopping,
            ReduceLROnPlateau, and optionally TensorBoard.
        """
        monitor = "miou"
        mode = "max"

        d_weights = os.path.join(config.weights_dir, "checkpoint")
        os.makedirs(d_weights, exist_ok=True)
        # Define the output path for saving weights and TensorBoard logs
        weights_path = os.path.join(  # https://keras.io/guides/serialization_and_saving/
            d_weights,
            "mobilenetv2_"
            + config.model_name
            + "_{epoch:04d}_{val_loss:.2f}.weights.h5"  # noqa: E501
        )

        logs_path = os.path.join(config.log_dir, config.model_name)
        os.makedirs(logs_path, exist_ok=True)

        # Create the callbacks
        tensorboard = TensorBoard(
            log_dir=logs_path,
            histogram_freq=0,
            write_graph=False,
            write_images=False,
        )
        checkpointer = ModelCheckpoint(
            weights_path,
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            monitor="val_{}".format(monitor),
            mode=mode,
        )
        stop_train = EarlyStopping(
            monitor="val_{}".format(monitor),
            patience=200,
            verbose=1,
            mode=mode,
        )

        def lr_scheduler(epoch, lr):
            lr = config.learn_rate
            if epoch < 100:
                return lr
            elif 100 <= epoch < 200:
                return 0.0005
            elif 200 <= epoch < 250:
                return 0.000025
            else:
                return 0.000125

        lr_schedule = LearningRateScheduler(lr_scheduler, verbose=1)
        if tf_board:
            callbacks = [checkpointer, stop_train, tensorboard, lr_schedule]
        else:
            callbacks = [checkpointer, stop_train, lr_schedule]

        return callbacks


    def run(self):
        dlab = self.ini.deeplab
        config = dlab.config

        losses = dlab.categorical_crossentropy
        metrics = {"pred_mask": [dlab.miou, dlab.accuracy]}

        # Load the model
        model = self.model_gen.model

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=7e-4, epsilon=1e-8, weight_decay=1e-6),
            sample_weight_mode="temporal",
            loss=losses,
            metrics=metrics,
        )

        # Define the train and valid data generator.
        train_generator = copy.deepcopy(self.dataloader_train)
        valid_generator = copy.deepcopy(self.dataloader_valid)

        # fine-tune model (train only last conv layers)
        if config.load_pretrained_weights:
            flag = 0
            for k, layer in enumerate(model.layers):
                layer.trainable = False
                if layer.name == "concat_projection":
                    flag = 1
                if flag:
                    layer.trainable = True

        callbacks = self.build_callbacks(config, tf_board=True)
        workers = multiprocessing.cpu_count() // 2
        history = model.fit(
            train_generator,
            validation_data=valid_generator,
            verbose=1,
            batch_size=config.batch_size,
            epochs=config.epochs,
            callbacks=callbacks,
            max_queue_size=10,
            workers=workers,
            use_multiprocessing=True,
        )
        print(history.history)

        return history

