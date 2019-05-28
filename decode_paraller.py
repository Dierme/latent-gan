import numpy as np
from rdkit import Chem
import json
import argparse
import os, sys
from autoencoder import autoencoder
from multiprocessing import Pool
from functools import partial


def latent_to_smile_string(latent, model, file_path):
    """
    NOTE! Script is not stable. More testing required. Use decode.py
    Decode latent vectors in parallel
    :param latent:
    :param model:
    :param file_path:
    :return:
    """
    print("NEW THREAD")
    sys.stdout.flush()
    with open(file_path, 'a+') as smiles_file:
        for indx, lat in enumerate(latent):
            if indx % 1000:
                print("[%d/%d]" % (indx, len(latent)))
            sys.stdout.flush()
            smile, _ = model.predict(np.reshape(lat, (1, 128)))
            mol = Chem.MolFromSmiles(smile)
            if mol:
                smile_string = Chem.MolToSmiles(mol)
                smiles_file.write(smile_string + '\n')
            smiles_file.flush()


def decode(latent_mols_file, output_smiles_file_path=None):
    print("BEGIN")
    sys.stdout.flush()
    model = autoencoder.load_model()

    if output_smiles_file_path is None:
        output_smiles_file_path = os.path.join(os.path.dirname(latent_mols_file), 'decoded_smiles.smi')

    with open(latent_mols_file, 'r') as f:
        latent = json.load(f)

    # Convert back to SMILES
    to_smile_partial = partial(latent_to_smile_string, model=model, file_path=output_smiles_file_path)

    open(output_smiles_file_path, 'w')
    n = 2000
    chunke_latent = [latent[i:i + n] for i in range(0, len(latent), n)]
    with Pool(2) as pool:
        pool.map(to_smile_partial, chunke_latent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and train a model")

    parser.add_argument("--latent_mols_file", "-l", help="The path to a data file.", type=str, required=True)
    parser.add_argument("--output_smiles_file_path", "-o", help="Prefix to the folder to save output smiles.", type=str)

    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
    decode(**args)
