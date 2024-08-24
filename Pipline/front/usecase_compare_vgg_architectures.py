
from train_val_test_clean import TrainValModels
from predictor import PredictModels


def train_val_vgg_model(backbone_name):
    tvmodels = TrainValModels()
    tvmodels.train_val_model(backbone_name)


def test_vgg_model(backbone_name, lr_val):
    tvmodels = TrainValModels()
    tvmodels.test_model(backbone_name, lr_val)

def predict_vgg_model(backbone_name, lr_val, amount_of_data):
    predmodels = PredictModels()
    predmodels.predict_data(backbone_name, lr_val, amount_of_data)


def main():
    pass

    # train_val_vgg_model(backbone_name = 'VGG16')

    # test_vgg_model(backbone_name = 'VGG16', lr_val = 0.0003)

    # predict_vgg_model(backbone_name = 'VGG16', lr_val = 0.0003, amount_of_data=8)





if __name__ == "__main__":
    main()