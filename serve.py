from model.ner_model import NERModel
from model.config import Config
from model.utils import align_data


def get_model_api():
    """Returns lambda function for api"""
    # 1. initialize model once and for all and reload weights
    config = Config()
    model  = NERModel(config)
    model.build()
    model.restore_session("/data/")

    def model_api(input_data):

        return "output_data"

    return model_api