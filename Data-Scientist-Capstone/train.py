import argparse
from pprint import pprint

import orjson

from src.trainer import ChestXRayClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-D", type=str)
    parser.add_argument("--model_path", "-M", type=str)
    args = parser.parse_args()

    CONFIG_PATH = "./config/train_config.json"
    params = orjson.loads(open(CONFIG_PATH, "rb").read())
    params["data_dir"] = args.data_dir
    params["model_path"] = args.model_path
    pprint(params)

    model = ChestXRayClassifier(**params)
    print('START TRAINING')
    model.train_model()