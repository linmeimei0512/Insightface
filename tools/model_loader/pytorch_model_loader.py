import sys
import torch
from torchsummary import summary
import pkg_resources
import time

sys.path.append('../../')
try:
    from utils.exception_printer import exception_printer
except ImportError:
    sys.path.append('../../../')
    from Insightface.utils.exception_printer import exception_printer

from backbones import get_model
import backbones

'''
########################################
PyTorch Model Loader
########################################
'''
class PyTorchModelLoader:
    def __init__(self):
        self.init_device()

        print('\n*********** PyTorch Model Loader ***********')
        print('torch version: ', torch.__version__)
        print('torch cuda is available: ', torch.cuda.is_available())
        print('torch device: ', self.device)
        print('torchsummary version: ', pkg_resources.get_distribution('torchsummary').version)
        print('*******************************************')


    # ================================================
    # Initialize device
    # ================================================
    def init_device(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # ================================================
    # Load PyTorch model
    #
    # :param pytorch_model_path
    # :param pytorch_weight_path
    # :param train
    #
    # return pytorch model
    # ================================================
    def load_pytorch_model(self, pytorch_model_path, pytorch_weight_path, input_shape, train):
        print('\nStarting load pytorch model \'' + str(pytorch_model_path) + '\'...')
        start_time = time.time()
        self.input_shape = input_shape

        # Load pytorch model
        try:
            self.pytorch_model = torch.load(pytorch_model_path)

        except Exception as ex:
            exception_printer('Load pytorch model failed.')
            return None

        # Load pytorch weight
        if pytorch_weight_path is not None:
            print('\nStarting load pytorch weight ', pytorch_weight_path, '...')

            try:
                self.pytorch_weight = torch.load(pytorch_weight_path)
                self.pytorch_model.load_state_dict(self.pytorch_weight)

            except Exception as ex:
                exception_printer('Load pytorch weight failed.')
                return None

        self.pytorch_model.to(device=self.device)
        self.pytorch_model.train(train)

        summary(self.pytorch_model, input_size=input_shape)

        print('Load pytorch model success. Cost time: ' + str(time.time() - start_time) + 's.')
        return self.pytorch_model


    def load_insightface_pytorch_model(self, model_name=None, pytorch_model_path=None, pytorch_weight_path=None, input_shape=(3, 112, 112), train=False):
        start_time = time.time()

        if pytorch_model_path is not None:
            print('\nStarting load insightface pytorch model \'' + str(pytorch_model_path) + '\'...')

            try:
                self.pytorch_model = torch.load(pytorch_model_path)

            except Exception as ex:
                exception_printer('Load pytorch model failed.')
                return None

        elif model_name is not None and pytorch_weight_path is not None:
            print('\nStarting load insightface pytorch model name: ' + str(model_name) + ', weight: \'' + str(pytorch_weight_path) + '\'...')

            try:
                self.pytorch_model = get_model(name=model_name)
                self.pytorch_model.load_state_dict(torch.load(pytorch_weight_path))

            except Exception as ex:
                exception_printer('Load pytorch weight failed.')
                return None

        self.pytorch_model.to(device=self.device)
        self.pytorch_model.train(train)

        summary(self.pytorch_model, input_size=input_shape)

        print('Load pytorch model success. Cost time: ' + str(time.time() - start_time) + 's.')
        return self.pytorch_model
