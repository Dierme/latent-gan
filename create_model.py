import argparse
from runners.CreateModelRunner import CreateModelRunner


def create_model():
    parser = argparse.ArgumentParser(description="Create a model with the vocabulary extracted from a SMILES file.")

    parser.add_argument("--input-data-path", "-i", help="The path to a data file.", type=str, required=True)
    parser.add_argument("--output-model-folder", "-o", help="Prefix to the folder to save output model.", type=str)
    parser.add_argument("--latent_dim", "-ld", help="dimensionality of the noise", type=int)
    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}

    runner = CreateModelRunner(**args)
    runner.run()


if __name__ == "__main__":
    create_model()
