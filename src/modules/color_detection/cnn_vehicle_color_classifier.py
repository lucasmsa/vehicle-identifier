import os
import numpy as np
from os.path import exists
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adamax
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization
from keras.applications import EfficientNetB3
from keras.utils import load_img, img_to_array
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.preprocessing.image import ImageDataGenerator


class CnnColorClassifier:
    BASE_DATASET_DIRECTORY_PATH = "./src/modules/color_detection/vehicle-color-dataset"
    WEIGHTS_FILE_PATH = "./src/weights/color"
    BATCH_SIZE = 32
    IMAGE_SIZE = (224, 224)

    def __init__(self):
        self.classes = self.fetch_classes()
        self.model = self.fetch_model()

    def preprocess_data(self):
        train_data_generator = ImageDataGenerator(
            rescale=1/255.,
            brightness_range=None,
            width_shift_range=0.5,
            rotation_range=False,
            horizontal_flip=True,
            vertical_flip=False
        )
        validation_data_generator = ImageDataGenerator(
            rescale=1./255)

        self.training_set = train_data_generator.flow_from_directory(
            f"{self.BASE_DATASET_DIRECTORY_PATH}/train", target_size=self.IMAGE_SIZE, batch_size=self.BATCH_SIZE, subset="training", shuffle=True, class_mode="categorical")

        self.validation_set = validation_data_generator.flow_from_directory(
            f"{self.BASE_DATASET_DIRECTORY_PATH}/val", target_size=(self.IMAGE_SIZE), batch_size=self.BATCH_SIZE, subset="training", shuffle=True, class_mode="categorical")

    def run_training(self):
        self.preprocess_data()
        self.train_cnn_model()

    def run_test(self, image_path: str):
        test_image = load_img(image_path, target_size=(self.IMAGE_SIZE))
        test_image = img_to_array(test_image)
        ndata = np.expand_dims(test_image, axis=0)
        y_prob = self.model.predict(ndata/255)
        y_prob.argmax(axis=-1)
        predicted_classes = [sorted(self.classes)[i]
                             for i in np.where(np.ravel(y_prob) > 0.1)[0]]

        return predicted_classes

    def fetch_model(self):
        model_path = f"{self.WEIGHTS_FILE_PATH}/EFN-model.best.h5"
        file_exists = exists(model_path)
        if file_exists:
            return load_model(model_path)
        else:
            return self.create_model()

    def fetch_classes(self):
        color_directories_list = os.listdir(
            f"{self.BASE_DATASET_DIRECTORY_PATH}/train")
        return list(filter(lambda directory: (directory != ".DS_Store"), color_directories_list))

    def create_model(self):
        base_model = EfficientNetB3(include_top=False, weights='imagenet', input_shape=(
            self.IMAGE_SIZE[0], self.IMAGE_SIZE[1], 3), pooling='max')
        x = base_model.output
        x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(rate=.45, seed=123)(x)
        output = Dense(len(self.classes), activation='softmax')(x)
        return Model(inputs=base_model.input, outputs=output)

    def train_cnn_model(self):
        self.model = self.create_model()
        self.model.compile(Adamax(learning_rate=.0001),
                           loss='categorical_crossentropy', metrics=['accuracy'])

        steps_per_epoch = self.training_set.samples // self.BATCH_SIZE
        val_steps = self.validation_set.samples // self.BATCH_SIZE
        n_epochs = 100

        csv_logger = CSVLogger('training.log', separator=',', append=False)

        checkpointer = ModelCheckpoint(filepath=f"{self.WEIGHTS_FILE_PATH}/EFN-model.best.h5",
                                       verbose=1,
                                       save_best_only=True)

        early_stop = EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   restore_best_weights=True,
                                   mode='min')

        self.model.fit(self.training_set,
                       epochs=n_epochs,
                       steps_per_epoch=steps_per_epoch,
                       validation_data=self.validation_set,
                       validation_steps=val_steps,
                       callbacks=[early_stop,
                                  checkpointer, csv_logger],
                       verbose=True,
                       shuffle=True,
                       workers=4)
        self.model.save_weights(f"{self.WEIGHTS_FILE_PATH}/model_weights.h5")


cnn_color_classifier = CnnColorClassifier()
cnn_color_classifier.run_test("./assets/bright_yellow_car.jpg")
