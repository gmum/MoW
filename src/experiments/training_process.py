import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from evaluation.model_evaluator import ModelEvaluator
from experiments.output_path_builder import OutputPathBuilder
from autoencoder_model import AutoEncoderModel
from dataset_loaders import Dataset
from experiments.canvas_fillers import RgbCanvasFiller


class TrainingProcess:

    def __init__(self, dataset: Dataset, model: AutoEncoderModel, model_evaluator: ModelEvaluator,
                 path_builder: OutputPathBuilder, optimizer: tf.train.Optimizer, canvas_filler: RgbCanvasFiller):
        self.__model = model
        self.__model_evaluator = model_evaluator
        self.__dataset = dataset
        self.__path_builder = path_builder
        self.__optimizer = optimizer
        self.__canvas_filler = canvas_filler

    def get_model(self) -> AutoEncoderModel:
        return self.__model

    def start(self, tf_session: tf.Session, extra_latent_shape):
        print('Starting training process')
        self.__tf_session = tf_session
        self.__extra_latent_shape = extra_latent_shape
        self.__extra_latent_size = self.__extra_latent_shape[0]

        print('Building cost tensor')
        if extra_latent_shape[0] != 0:
            print(f'Training with extra latent: {self.__extra_latent_shape[0]}')
            self.__tensor_extra_latent = tf.placeholder(tf.float32, shape=extra_latent_shape)
        else:
            print('Not using extra latent')
            self.__tensor_extra_latent = None
        self.__tensor_cost = self.__model.build_cost_tensor(self.__tensor_extra_latent)

        print('Building training ops')
        self.__tensor_train_ops = self.__optimizer.minimize(self.__tensor_cost)

        print('Initializing global vars')
        self.__tf_session.run(tf.global_variables_initializer())
        print('Training setup complete')

    def next_epoch(self, epoch_number: int):
        index = 0

        if self.__tensor_extra_latent is not None:
            extra_latent = np.random.normal(loc=0.0, scale=1.0, size=self.__extra_latent_shape)

        starting_index = 0
        try:
            while True:
                index = index + 1

                if self.__tensor_extra_latent is not None:
                    _, z = self.__tf_session.run([self.__tensor_train_ops, self.__model.tensor_z],
                                                 feed_dict={self.__tensor_extra_latent: extra_latent})
                else:
                    _ = self.__tf_session.run(self.__tensor_train_ops)

                if self.__tensor_extra_latent is not None:
                    for i in range(len(z)):
                        extra_latent[starting_index] = z[i]
                        starting_index += 1
                        if starting_index >= self.__extra_latent_size:
                            starting_index = 0

        except tf.errors.OutOfRangeError:
            pass
        self.__path_builder.set_prefix(epoch_number)

    def end(self):
        print('Ending training process')
        with open(self.__path_builder.get_path('report.json'), 'w') as json_file:
            json_file.write(self.__model_evaluator.get_json())
        print('Results available in directory:', self.__path_builder.get_base_dir())

    def evaluate(self, epoch_number: int):
        res, z = self.__model_evaluator.evaluate(self.__tf_session,
                                                 self.__dataset.validation_images[0:], epoch_number)
        print(res)
        return res, z

    def save_image(self, name, images, size=(1, 1)):
        path = self.__path_builder.get_path(name)
        if size == (1, 1):
            canvas = self.__canvas_filler.build_canvas([images], (1, 1), self.__dataset.x_dim)
        else:
            canvas = self.__canvas_filler.build_canvas(images, size, self.__dataset.x_dim)

        if self.__canvas_filler.cmap is None:
            plt.imsave(path, canvas)
        else:
            plt.imsave(path, canvas, cmap=self.__canvas_filler.cmap)
