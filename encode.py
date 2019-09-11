from rdkit import Chem
import json
import argparse
from autoencoder import autoencoder
import os

def encode(smiles_file, output_smiles_file_path=None, encoder=None):
    """
    Encoding smile strings into latent vectors that are used to train GAN
    :param smiles_file:
    :param output_smiles_file_path:
    :return:
    """
    model = autoencoder.load_model(model_version=encoder)

    # Input SMILES
    smiles_in = []
    with open(smiles_file, "r") as file:
        line = file.readline()
        while line:
            smiles_in.append(line.strip('\n'))
            line = file.readline()

    # MUST convert SMILES to binary mols for the model to accept them (it re-converts them to SMILES internally)
    mols_in = [Chem.rdchem.Mol.ToBinary(Chem.MolFromSmiles(smiles)) for smiles in smiles_in]
    latent = model.transform(model.vectorize(mols_in))

    # Writing JSON data
    os.makedirs(os.path.dirname(output_smiles_file_path), exist_ok=True)
    with open(output_smiles_file_path, 'w') as f:
        json.dump(latent.tolist(), f)

    print('Success!')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and train a model")

    parser.add_argument("--smiles-file", "-sf", help="The path to a data file.", type=str, required=True)
    parser.add_argument("--output_smiles_file_path", "-o", help="Path to output smiles.", type=str)
    parser.add_argument("--encoder",
                        help="The data set the pre-trained heteroencoder has been trained on [chembl|moses] DEFAULT:chembl",
                        type=str)
    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
    encode(**args)
