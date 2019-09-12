import argparse
from encode import encode
import os
from runners.CreateModelRunner import CreateModelRunner
from runners.TrainModelRunner import TrainModelRunner


class RunRunner:

    def __init__(self, smiles_file="data/EGFR_training.smi", storage_path="storage/example/",
                 latent_file="encoded_smiles.latent", decoded_smiles="decoded_smiles.csv", n_epochs=2000,sample_n=30000, encoder=None):


        # init params
        self.storage_path=storage_path
        self.smiles_file=smiles_file
        self.output_latent=os.path.join(self.storage_path,latent_file)
        self.decoded_smiles=os.path.join(self.storage_path,decoded_smiles)
        self.n_epochs=n_epochs
        self.encoder=encoder
        self.sample_size=sample_n


    def run(self):
        print("Model LatentGAN running, encoding training set")
        encode(smiles_file=self.smiles_file, output_smiles_file_path=self.output_latent,encoder=self.encoder)
        print("Encoding finished finished. Creating model files")
        C = CreateModelRunner(input_data_path=self.output_latent, output_model_folder=self.storage_path)
        C.run()
        print("Model Created. Training model")
        T= TrainModelRunner(input_data_path=self.output_latent, output_model_folder=self.storage_path,
                         decode_mols_save_path=self.decoded_smiles,n_epochs=self.n_epochs,sample_after_training=self.sample_size)
        T.run()
        print("Model finished.")




        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and train a model")

    parser.add_argument("--smiles-file", "-sf", help="The path to a data file.", type=str)
    parser.add_argument("--storage-path", "-st", help="The path to all outputs", type=str)
    parser.add_argument("--latent-file", "-lf", help="Name of latent vector file", type=str)
    parser.add_argument("--decoded-smiles", "-ds", help="Name of output generated smiles file", type=str)
    parser.add_argument("--n-epochs", type=int, help="number of epochs of training")
    parser.add_argument("--sample-n", type=int, help="Number of molecules to sample after training")
    parser.add_argument("--encoder", help="The data set the pre-trained heteroencoder has been trained on [chembl|moses] DEFAULT:chembl", type=str)
    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
    runner=RunRunner(**args)
    runner.run()
