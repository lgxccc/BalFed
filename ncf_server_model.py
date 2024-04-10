import torch


class ServerNeuralCollaborativeFiltering(torch.nn.Module):
    def __init__(self, item_num, predictive_factor=12):
        """
        Initializes the layers of the model.

        Args:
            item_num (int): The number of items in the dataset.
            predictive_factor (int, optional): The latent dimension of the model. Default is 12.
        """
        super(ServerNeuralCollaborativeFiltering, self).__init__()
        self.mlp_item_embeddings = torch.nn.Embedding(num_embeddings=item_num, embedding_dim=predictive_factor)
        self.gmf_item_embeddings = torch.nn.Embedding(num_embeddings=item_num, embedding_dim=predictive_factor)
        self.gmf_out = torch.nn.Linear(predictive_factor, 1)
        self.gmf_out.weight = torch.nn.Parameter(torch.ones(1, predictive_factor))

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * predictive_factor, 64), torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.LayerNorm(64),
            # torch.nn.BatchNorm1d(64),
            torch.nn.Linear(64, 32), torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.LayerNorm(32),
            # torch.nn.BatchNorm1d(32),
            torch.nn.Linear(32, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 8), torch.nn.ReLU()
        )
        # Linear layer of MLP
        self.mlp_out = torch.nn.Linear(8, 1)

        self.output_logits = torch.nn.Linear(predictive_factor, 1)
        self.model_blending = 0.5  # alpha parameter, equation 13 in the paper
        self.initialize_weights()
        self.join_output_weights()

    def initialize_weights(self):
        torch.nn.init.xavier_uniform_(self.mlp_item_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.gmf_item_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.gmf_out.weight)
        torch.nn.init.xavier_uniform_(self.mlp_out.weight)
        torch.nn.init.xavier_uniform_(self.output_logits.weight)

    def layer_setter(self, model, model_copy):
        model_state_dict = model.state_dict()
        model_copy.load_state_dict(model_state_dict)

    def set_weights(self, model):
        self.layer_setter(model.mlp_item_embeddings, self.mlp_item_embeddings)
        self.layer_setter(model.gmf_item_embeddings, self.gmf_item_embeddings)
        self.layer_setter(model.mlp, self.mlp)
        self.layer_setter(model.gmf_out, self.gmf_out)
        self.layer_setter(model.mlp_out, self.mlp_out)
        self.layer_setter(model.output_logits, self.output_logits)

    def forward(self):
        return torch.tensor(0.0)

    def join_output_weights(self):
        W = torch.nn.Parameter(
            torch.cat((self.model_blending * self.gmf_out.weight, (1 - self.model_blending) * self.mlp_out.weight),
                      dim=1))
        self.output_logits.weight = W


if __name__ == '__main__':
    ncf = ServerNeuralCollaborativeFiltering(100, 64)
    print(ncf)
