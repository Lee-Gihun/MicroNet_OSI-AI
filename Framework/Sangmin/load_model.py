from alexnet import AlexNet
from mobilenetv2 import MobileNetV2

NEURALNET_MAP = {'alexnet': AlexNet,
                 'mobilenetv2': MobileNetV2
                }

def load_model(opt, device):
    model = NEURALNET_MAP[opt['model']['algo']](opt).to(device)
    
    return model