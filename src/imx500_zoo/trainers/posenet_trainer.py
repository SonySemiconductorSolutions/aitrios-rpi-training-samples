import tensorflow as tf

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam, schedules

from imx500_zoo.utilities.posenet.utils.model import get_personlab_model
from third_party.keraspersonlab.loss import (
    kp_map_loss_fn,
    short_offset_loss_fn,
    mid_offset_loss_fn,
)
from imx500_zoo.utilities.posenet.data_generator import DataGenerator
from third_party.tensorflow.mobilenet import get_mobilenetv1_base
from imx500_zoo.utilities.posenet.utils.metrics import (
    BinaryAccuracy,
    identity_metric,
)
import multiprocessing
import os

workers = None
multiprocessing = None
BATCH_SIZE = None
nb_epochs = None
RETRAIN_FLAG = None
img_shape = None
final_model_path = None

config = None


def set_config(config_i):
    global config
    global workers
    global multiprocessing
    global BATCH_SIZE
    global nb_epochs
    global RETRAIN_FLAG
    global img_shape
    global final_model_path

    config = config_i
    workers = config.WORKERS
    multiprocessing = config.MULTIPROCESSING_FLAG
    BATCH_SIZE = config.BATCH_SIZE
    nb_epochs = config.NUM_EPOCHS
    RETRAIN_FLAG = config.RETRAIN_FLAG
    img_shape = [config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]]
    final_model_path = config.train_model_path()


def build_callbacks(
    tf_board=False,
    train_steps=None,
    val_steps=None,
    train_gen_return_kp=None,
    val_gen_return_kp=None,
):
    checkpoint_path = config.SAVE_MODEL_PATH + "/best_model_{epoch:04d}.h5"
    log_path = config.LOGS_PATH
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    monitor = "loss"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)
    earlystop_callback = EarlyStopping(
        monitor="val_{}".format(monitor), patience=config.EARLYSTOP_PATIENCE, verbose=1, mode="min"
    )
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,
        monitor="val_{}".format(monitor),
        mode="min",
        save_best_only=True,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_{}".format(monitor), factor=0.5, patience=10, min_lr=1e-6
    )
    # PCK_FULL_callback=Keypoint_callback(None,val_gen_return_kp,train_steps,val_steps,"PCK_FULL")
    # OKS_FULL_callback=Keypoint_callback(None,val_gen_return_kp,train_steps,val_steps,"OKS_FULL")

    if tf_board:
        # callbacks = [model_checkpoint_callback, earlystop_callback, reduce_lr,PCK_FULL_callback,OKS_FULL_callback,tensorboard_callback]
        callbacks = [
            model_checkpoint_callback,
            earlystop_callback,
            reduce_lr,
            tensorboard_callback,
        ]
    else:
        callbacks = [model_checkpoint_callback, reduce_lr, earlystop_callback]
    return callbacks


class PosenetTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.dataloader_train = None
        self.dataloader_valid = None

    def fit(self, model, dataloader_train, dataloader_valid):
        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_valid = dataloader_valid
        self.run()

    def run(self):
        config = self.config.posenet

        ##select GPU explicitly for different experiments
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            # Restrict TensorFlow to only use the first GPU
            try:
                tf.config.set_visible_devices(gpus[config.GPU_ID], "GPU")
            except Exception as e:
                print(f"GPU unavailable : {e}")

        # initialize generator object
        train_generator = (
            DataGenerator(BATCH_SIZE, "train", 0.8, seed=0.5)
            if self.dataloader_train is None
            else self.dataloader_train
        )
        val_generator = (
            DataGenerator(BATCH_SIZE, "val", 0.8, seed=0.5)
            if self.dataloader_valid is None
            else self.dataloader_valid
        )

        # generator for OSK metric
        train_generator_return_kp = DataGenerator(
            BATCH_SIZE, "train", 0.8, seed=0.5, return_kp=True
        )
        val_generator_return_kp = DataGenerator(
            BATCH_SIZE, "val", 0.8, seed=0.5, return_kp=True
        )

        train_ds = tf.data.Dataset.from_generator(
            train_generator,
            output_types=(tf.float32, (tf.float32, tf.float32, tf.float32)),
            output_shapes=(
                (None, img_shape[0], img_shape[1], 3),
                (
                    (None, img_shape[0], img_shape[1], config.NUM_KP),
                    (None, img_shape[0], img_shape[1], config.NUM_KP * 2),
                    (None, img_shape[0], img_shape[1], config.NUM_EDGES * 4),
                ),
            ),
        )

        validation_ds = tf.data.Dataset.from_generator(
            val_generator,
            output_types=(tf.float32, (tf.float32, tf.float32, tf.float32)),
            output_shapes=(
                (None, img_shape[0], img_shape[1], 3),
                (
                    (None, img_shape[0], img_shape[1], config.NUM_KP),
                    (None, img_shape[0], img_shape[1], config.NUM_KP * 2),
                    (None, img_shape[0], img_shape[1], config.NUM_EDGES * 4),
                ),
            ),
        )

        print("train samples", train_generator.train_len)
        print("val samples", val_generator.val_len)

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.prefetch(buffer_size=AUTOTUNE).repeat()
        validation_ds = validation_ds.prefetch(buffer_size=AUTOTUNE).repeat()

        # model
        is_model_none = self.model is None
        if not is_model_none:
            model = self.model.get()
        else:
            if RETRAIN_FLAG:
                custom_objects = {
                    "kp_map_loss_fn": kp_map_loss_fn,
                    "short_offset_loss_fn": short_offset_loss_fn,
                    "mid_offset_loss_fn": mid_offset_loss_fn,
                    "identity_metric": identity_metric,
                }
                model = tf.keras.models.load_model(
                    config.RETRAIN_MODEL_PATH,
                    custom_objects=custom_objects,
                    compile=False,
                )
            else:
                model = get_personlab_model(get_mobilenetv1_base)
        # print(model.summary())

        # callbacks
        callbacks = build_callbacks(
            tf_board=True,
            train_steps=len(train_generator),
            val_steps=len(val_generator),
            train_gen_return_kp=train_generator_return_kp,
            val_gen_return_kp=val_generator_return_kp,
        )

        losses = {
            "kp_maps_head": kp_map_loss_fn,
            "short_offsets_head": short_offset_loss_fn,
            "mid_offsets_head": mid_offset_loss_fn,
        }
        metrics = {
            "kp_maps_head": BinaryAccuracy(),
            "short_offsets_head": identity_metric,
            "mid_offsets_head": identity_metric,
        }

        # compile model
        lr_schedule = schedules.ExponentialDecay(
            initial_learning_rate=config.LEARNING_RATE,
            decay_steps=config.LR_DECAY_STEPS,
            decay_rate=config.LR_DECAY_RATE)
        model.compile(loss=losses, optimizer=Adam(learning_rate=lr_schedule), metrics=metrics)

        # fit model
        model.fit(
            train_ds,
            steps_per_epoch=len(train_generator),
            validation_data=validation_ds,
            validation_steps=len(val_generator),
            epochs=nb_epochs,
            verbose=1,
            callbacks=callbacks,
            max_queue_size=10,
            workers=workers,
            use_multiprocessing=multiprocessing,
        )

        # save final model
        if not is_model_none:
            final_model_path = config.TRAIN_MODEL_H5

        print(f"saved model : {final_model_path}")
        model.save(final_model_path)
