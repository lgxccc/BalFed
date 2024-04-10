import torch
import torch.nn as nn
import torch.nn.functional as F


class ServerDeepFM(nn.Module):
    def __init__(self, num_items, embedding_size):

        super(ServerDeepFM, self).__init__()

        self.item_embeddings = nn.Embedding(num_items, embedding_size)
        self.bias = nn.Parameter(torch.randn(1))

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * embedding_size, 64), torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32), torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 8), torch.nn.ReLU(),
            torch.nn.Linear(8, 1)
        )
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.xavier_uniform_(self.item_embeddings.weight)
        for module in self.mlp:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.01)

    def forward(self):
        return torch.tensor(0.0)

    def layer_setter(self, model, model_copy):
        model_state_dict = model.state_dict()
        model_copy.load_state_dict(model_state_dict)

    def set_weights(self, server_model):
        self.layer_setter(server_model.item_embeddings, self.item_embeddings)
        self.bias.data.copy_(server_model.bias.data)
        self.layer_setter(server_model.mlp, self.mlp)