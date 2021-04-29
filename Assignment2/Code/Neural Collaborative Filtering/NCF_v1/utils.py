"""
    Some handy functions for Pytorch model training ...
"""
import torch


def save_checkpoint(model, model_dir):
    """
    Function to save model checkpoints
    :param model: model used
    :param model_dir: directory to be saved
    """
    torch.save(model.state_dict(), model_dir)


# def resume_checkpoint(model, model_dir, device_id):
#     state_dict = torch.load(model_dir, map_location = lambda storage, loc: storage.cuda(device = device_id))
#     model.load_state_dict(state_dict)

def resume_checkpoint(model, model_dir):
    """
    Function to resume model checkpoint
    :param model: model used
    :param model_dir: directory to be saved
    """
    state_dict = torch.load(model_dir)
    model.load_state_dict(state_dict)


# def use_cuda(enabled, device_id=0):
#     if enabled:
#         assert torch.cuda.is_available(), 'CUDA is not available'
#         torch.cuda.set_device(device_id)

def use_optimizer(network, params):
    """
    Function to perform optimization
    :param network: network used
    :param params: parameters used
    :return: choice of optimizer
    """
    if params['optimizer'] == 'sgd':
        # Stochastic Gradient Descent optimizer
        optimizer = torch.optim.SGD(network.parameters(), lr=params['sgd_lr'], momentum=params['sgd_momentum'],
                                    weight_decay=params['l2_regularization'])

    elif params['optimizer'] == 'adam':
        # Adam optimizer
        optimizer = torch.optim.Adam(network.parameters(), lr=params['adam_lr'],
                                     weight_decay=params['l2_regularization'])

    elif params['optimizer'] == 'rmsprop':
        # RMSprop optimizer
        optimizer = torch.optim.RMSprop(network.parameters(), lr=params['rmsprop_lr'], alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])

    return optimizer
