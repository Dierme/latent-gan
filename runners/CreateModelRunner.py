import pickle
from models.Discriminator import Discriminator
from models.Generator import Generator
import os
import numpy as np
import json


class CreateModelRunner:
    def __init__(self, input_data_path, output_model_folder, latent_dim=512):
        self.input_data_path = input_data_path
        self.output_model_folder = output_model_folder
        self.latent_dim = latent_dim

    def run(self):
        # get data
        latent_vector_file = open(self.input_data_path, "r")
        latent_space_mols = np.array(json.load(latent_vector_file))
        shape = latent_space_mols.shape     # expecting tuple (set_size, dim_1, dim_2)

        data_shape = tuple([shape[1], shape[2]])
        # create Discriminator
        D = Discriminator(data_shape)

        # save Discriminator
        discriminator_path = os.path.join(self.output_model_folder, 'discriminator.txt')
        D.save(discriminator_path)

        # create Generator
        G = Generator(data_shape, latent_dim=self.latent_dim)

        # save generator
        generator_path = os.path.join(self.output_model_folder, 'generator.txt')
        G.save(generator_path)

        return True
