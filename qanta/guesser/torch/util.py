import torch


def create_save_model(model):
    def save_model(path):
        torch.save(model, path)
    return save_model