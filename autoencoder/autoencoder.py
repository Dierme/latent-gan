from ddc_pub import ddc_v3 as ddc
import os

DEFAULT_MODEL_VERSION = 'chembl'


def load_model(model_version=None):
    # Import model
    if model_version == 'chembl':
        model_name = 'chembl_pretrained'
    elif model_version == 'moses':
        model_name = 'moses_pretrained'
    else:
        model_name = DEFAULT_MODEL_VERSION


    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_name)
    model = ddc.DDC(model_name=path)

    return model

model = load_model()
