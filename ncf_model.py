import torch


class NeuralCollaborativeFiltering(torch.nn.Module):
    def __init__(self, user_num, item_num, predictive_factor=12):
        """
        Initializes the layers of the model.

        Parameters:
            user_num (int): The number of users in the dataset.
            item_num (int): The number of items in the dataset.
            predictive_factor (int, optional): The latent dimension of the model. Default is 12.
        """
        super(NeuralCollaborativeFiltering, self).__init__()
        # Initialize user and item latent vectors for MLP and GMF
        self.mlp_user_embeddings = torch.nn.Embedding(num_embeddings=user_num, embedding_dim=predictive_factor)
        self.mlp_item_embeddings = torch.nn.Embedding(num_embeddings=item_num, embedding_dim=predictive_factor)
        self.gmf_user_embeddings = torch.nn.Embedding(num_embeddings=user_num, embedding_dim=predictive_factor)
        self.gmf_item_embeddings = torch.nn.Embedding(num_embeddings=item_num, embedding_dim=predictive_factor)

        # Linear layer of GMF that we will fe/ed with the mul(user_emb, item_emb),
        # it will output the predicted scores from GMF
        self.gmf_out = torch.nn.Linear(predictive_factor, 1)
        self.gmf_out.weight = torch.nn.Parameter(torch.ones(1, predictive_factor))

        #self.mlp_out = torch.nn.Linear(6, 1)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * predictive_factor, 64), torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32), torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 8), torch.nn.ReLU()
        )
        # Linear layer of MLP
        self.mlp_out = torch.nn.Linear(8, 1)
        # Contains the output of the GMF concatenated with MLP
        self.output_logits = torch.nn.Linear(predictive_factor, 1)
        # Percentage of each model in the final output
        self.model_blending = 0.5  # alpha parameter, equation 13 in the paper

        self.initialize_weights()
        self.join_output_weights()

    def initialize_weights(self):
        """Initializes the weight parameters using Xavier initialization."""
        torch.nn.init.xavier_uniform_(self.mlp_user_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.mlp_item_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.gmf_user_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.gmf_item_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.gmf_out.weight)
        torch.nn.init.xavier_uniform_(self.mlp_out.weight)
        torch.nn.init.xavier_uniform_(self.output_logits.weight)

    def forward(self, x):
        user_id, item_id = x[:, 0], x[:, 1]
        gmf_product = self.gmf_forward(user_id, item_id)
        mlp_output = self.mlp_forward(user_id, item_id)
        concat = torch.cat([gmf_product, mlp_output], dim=1)
        output_logits = self.output_logits(concat)
        output_scores = torch.sigmoid(output_logits)
        return output_scores.view(-1)

    def gmf_forward(self, user_id, item_id):
        # with torch.no_grad():
        user_emb = self.gmf_user_embeddings(user_id)
        item_emb = self.gmf_item_embeddings(item_id)
        return torch.mul(user_emb, item_emb)

    def mlp_forward(self, user_id, item_id):
        # with torch.no_grad():
        user_emb = self.mlp_user_embeddings(user_id)
        item_emb = self.mlp_item_embeddings(item_id)
        return self.mlp(torch.cat([user_emb, item_emb], dim=1))

    def join_output_weights(self):
        W = torch.nn.Parameter(
            torch.cat((self.model_blending * self.gmf_out.weight, (1 - self.model_blending) * self.mlp_out.weight),
                      dim=1))
        self.output_logits.weight = W

    def layer_setter(self, model, model_copy):
        model_state_dict = model.state_dict()
        model_copy.load_state_dict(model_state_dict)

    def load_server_weights(self, server_model):
        self.layer_setter(server_model.mlp_item_embeddings, self.mlp_item_embeddings)
        self.layer_setter(server_model.gmf_item_embeddings, self.gmf_item_embeddings)
        # self.layer_setter(server_model.item_bias, self.item_bias)###
        self.layer_setter(server_model.mlp, self.mlp)
        self.layer_setter(server_model.gmf_out, self.gmf_out)
        self.layer_setter(server_model.mlp_out, self.mlp_out)
        self.layer_setter(server_model.output_logits, self.output_logits)


if __name__ == '__main__':
    ncf = NeuralCollaborativeFiltering(100, 100, 64)
    print(ncf)
