import copy
import datetime
import os
import random

import numpy as np
import torch
from matplotlib import pyplot as plt


def plot_progress(progress, title=None, loss="loss", save=True):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    epochs = np.arange(1, len(progress["loss"]) + 1)

    axs[0].plot(epochs, progress["hit_ratio@10"])
    axs[0].set_xlabel('epochs')
    axs[0].set_ylabel('HR@10')

    axs[1].plot(epochs, progress["loss"])
    axs[1].set_xlabel('epochs')
    axs[1].set_ylabel(loss)

    axs[2].plot(epochs, progress["ndcg@10"])
    axs[2].set_xlabel('epochs')
    axs[2].set_ylabel('ndcg@10')

    if title:
        fig.suptitle(title, fontsize=30)
    if save:
        plt.savefig(f'{datetime.datetime.now()}.png')

    plt.show()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class Utils:
    def __init__(self, num_clients, local_path="./models/local_items/", server_path="./models/central/"):
        self.epoch = 0
        self.num_clients = num_clients
        self.local_path = local_path
        self.server_path = server_path

    @staticmethod
    def load_pytorch_client_model(path):
        # Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
        return torch.jit.load(path)

    def get_user_models(self, loader):
        # get clients models from "./models/local_items/"
        models = []
        for client_id in range(self.num_clients):
            models.append({'model': loader(self.local_path + "dp" + str(client_id) + ".pt")})
        return models

    def get_previous_federated_model(self):
        self.epoch += 1
        return torch.jit.load(self.server_path + "server" + str(self.epoch - 1) + ".pt")

    def save_federated_model(self, model):
        torch.jit.save(model, self.server_path + "server" + str(self.epoch) + ".pt")

    # after each epoch the framework saves the result in "models/server"

    def federate(self, client_nums):

        all_data = sum(client_nums)
        client_models = self.get_user_models(self.load_pytorch_client_model)
        server_model = self.get_previous_federated_model()  # get the last model saved by the last epoch
        if len(client_models) == 0:
            self.save_federated_model(server_model)  # if there's no models for clients then save the last readed model
            return
        server_new_dict = copy.deepcopy(client_models[0]['model'].state_dict())
        for i in range(0, len(client_models)):
            client_dict = client_models[i]['model'].state_dict()
            for k in client_dict.keys():
                server_new_dict[k] += client_dict[k] * client_nums[i]
        for k in server_new_dict.keys():
            server_new_dict[k] = server_new_dict[k] / all_data
        server_model.load_state_dict(server_new_dict)
        self.save_federated_model(server_model)

    def simple_avg(self):
        client_models = self.get_user_models(self.load_pytorch_client_model)
        server_model = self.get_previous_federated_model()  # get the last model saved by the last epoch
        if len(client_models) == 0:
            self.save_federated_model(server_model)  # if there's no models for clients then save the last readed model
            return
        n = len(client_models)

        server_new_dict = copy.deepcopy(client_models[0]['model'].state_dict())
        for i in range(1, len(client_models)):
            client_dict = client_models[i]['model'].state_dict()
            for k in client_dict.keys():
                server_new_dict[k] += client_dict[k]
        for k in server_new_dict.keys():
            server_new_dict[k] = server_new_dict[k] / n
        server_model.load_state_dict(server_new_dict)
        self.save_federated_model(server_model)


    def fedfast(self, client_nums,model):
        all_data = sum(client_nums)
        client_models = self.get_user_models(self.load_pytorch_client_model)
        server_model = self.get_previous_federated_model()
        if len(client_models) == 0:
            self.save_federated_model(server_model)
            return
        n = len(client_models)

        if model == "ncf":
            server_item_embeddings = server_model.state_dict()['mlp_item_embeddings.weight']
            client_item_embeddings_updates = [torch.zeros_like(server_item_embeddings) for _ in range(n)]

            for i in range(n):
                client_item_embeddings = client_models[i]['model'].state_dict()['mlp_item_embeddings.weight']
                client_item_embeddings_updates[i] = torch.abs(client_item_embeddings - server_item_embeddings)

            server_new_dict = copy.deepcopy(server_model.state_dict())
            for item_idx in range(server_item_embeddings.shape[0]):
                item_weights = torch.tensor([update[item_idx].sum() for update in client_item_embeddings_updates])
                item_weights_sum = item_weights.sum()
                if item_weights_sum > 0:
                    item_embeddings = torch.stack(
                        [model['model'].state_dict()['mlp_item_embeddings.weight'][item_idx] for model in
                         client_models])
                    server_new_dict['mlp_item_embeddings.weight'][item_idx] = torch.sum(
                        item_weights.view(-1, 1) * item_embeddings, dim=0) / item_weights_sum

            for key in [k for k in server_new_dict.keys() if 'mlp_item_embeddings' not in k]:
                server_new_dict[key] = torch.sum(
                    torch.stack(
                        [model['model'].state_dict()[key] * num for model, num in zip(client_models, client_nums)]),
                    dim=0) / all_data

            server_model.load_state_dict(server_new_dict)
            self.save_federated_model(server_model)

        elif model == "deepfm":
            server_item_embeddings = server_model.state_dict()['item_embeddings.weight']
            client_item_embeddings_updates = [torch.zeros_like(server_item_embeddings) for _ in range(n)]

            for i in range(n):
                client_item_embeddings = client_models[i]['model'].state_dict()['item_embeddings.weight']
                client_item_embeddings_updates[i] = torch.abs(client_item_embeddings - server_item_embeddings)

            server_new_dict = copy.deepcopy(server_model.state_dict())
            for item_idx in range(server_item_embeddings.shape[0]):
                item_weights = torch.tensor([update[item_idx].sum() for update in client_item_embeddings_updates])
                item_weights_sum = item_weights.sum()
                if item_weights_sum > 0:
                    item_embeddings = torch.stack(
                        [model['model'].state_dict()['item_embeddings.weight'][item_idx] for model in
                         client_models])
                    server_new_dict['item_embeddings.weight'][item_idx] = torch.sum(
                        item_weights.view(-1, 1) * item_embeddings, dim=0) / item_weights_sum

            for key in [k for k in server_new_dict.keys() if 'item_embeddings' not in k]:
                server_new_dict[key] = torch.sum(
                    torch.stack(
                        [model['model'].state_dict()[key] * num for model, num in zip(client_models, client_nums)]),
                    dim=0) / all_data

            server_model.load_state_dict(server_new_dict)
            self.save_federated_model(server_model)

        elif model == "svdpp":
            server_item_embeddings = server_model.state_dict()['gmf_item_embeddings.weight']
            client_item_embeddings_updates = [torch.zeros_like(server_item_embeddings) for _ in range(n)]

            for i in range(n):
                client_item_embeddings = client_models[i]['model'].state_dict()['gmf_item_embeddings.weight']
                client_item_embeddings_updates[i] = torch.abs(client_item_embeddings - server_item_embeddings)

            server_new_dict = copy.deepcopy(server_model.state_dict())
            for item_idx in range(server_item_embeddings.shape[0]):
                item_weights = torch.tensor([update[item_idx].sum() for update in client_item_embeddings_updates])
                item_weights_sum = item_weights.sum()
                if item_weights_sum > 0:
                    item_embeddings = torch.stack(
                        [model['model'].state_dict()['gmf_item_embeddings.weight'][item_idx] for model in
                         client_models])
                    server_new_dict['gmf_item_embeddings.weight'][item_idx] = torch.sum(
                        item_weights.view(-1, 1) * item_embeddings, dim=0) / item_weights_sum

            for key in [k for k in server_new_dict.keys() if 'gmf_item_embeddings' not in k]:
                server_new_dict[key] = torch.sum(
                    torch.stack(
                        [model['model'].state_dict()[key] * num for model, num in zip(client_models, client_nums)]),
                    dim=0) / all_data

            server_model.load_state_dict(server_new_dict)
            self.save_federated_model(server_model)
        else:
            raise ValueError('can not find a model')
