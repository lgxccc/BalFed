import torch
import torch.nn as nn
class DeepFM(nn.Module):
    def __init__(self, num_users, num_items, embedding_size):
        super(DeepFM, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_size)
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
        torch.nn.init.xavier_uniform_(self.user_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.item_embeddings.weight)
        for module in self.mlp:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.01)

    def forward(self, x):
        user_ids, item_ids = x[:, 0], x[:, 1]
        user_embeds = self.user_embeddings(user_ids)
        item_embeds = self.item_embeddings(item_ids)

        fm_part = torch.mul(user_embeds, item_embeds)
        fm_part = torch.sum(fm_part, dim=1, keepdim=True)

        dnn_input = torch.cat([user_embeds, item_embeds], dim=1)
        dnn_output = self.mlp(dnn_input)

        output = torch.sigmoid(fm_part + dnn_output + self.bias)
        return output.view(-1)

    def layer_setter(self, model, model_copy):
        model_state_dict = model.state_dict()
        model_copy.load_state_dict(model_state_dict)

    def load_server_weights(self, server_model):
        self.layer_setter(server_model.item_embeddings, self.item_embeddings)
        self.bias.data.copy_(server_model.bias.data)
        self.layer_setter(server_model.mlp, self.mlp)