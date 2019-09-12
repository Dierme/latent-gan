from ddc_pub import ddc_v3 as ddc
import os

DEFAULT_MODEL_VERSION = 'new_chembl_model'


def load_model(model_version=None):
    # Import model
    if model_version == 'chembl':
        model_name = 'new_chembl_model'
    elif model_version == 'moses':
        model_name = '16888509--1000--0.0927--0.0000010'
    else:
        model_name = DEFAULT_MODEL_VERSION


    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_name)
    model = ddc.DDC(model_name=path)

    return model
