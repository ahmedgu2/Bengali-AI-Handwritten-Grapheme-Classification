import model.models as models

MODEL_DISPATCHER = {
    'resnet34' : models.ResNet34,
    'se_resnext50' : models.SeResNext
}
