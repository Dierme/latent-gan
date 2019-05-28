import argparse
import csv
from rdkit import Chem


def filter_smile():
    parser = argparse.ArgumentParser(description="Load and train a model")

    parser.add_argument("--input-smile-file", "-isf", help="The path to a data file.", type=str, required=True)
    parser.add_argument("--output-smiles-path", "-o", help="Prefix to the folder to save output model.", type=str)

    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
    smiles = []
    with open(args['input_smile_file'], newline='') as csvfile:
        smilesreader = csv.reader(csvfile, delimiter='	')
        for row in smilesreader:
            filtered_smile = Chem.MolToSmiles(Chem.MolFromSmiles(row[0]), canonical=True, isomericSmiles=False)
            if filtered_smile not in smiles:
                smiles.append(filtered_smile)

    with open(args['output_smiles_path'], 'w') as writefile:
        for item in smiles:
            writefile.write("%s\n" % item)


if __name__ == "__main__":
    filter_smile()
