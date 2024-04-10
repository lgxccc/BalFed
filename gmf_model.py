import torch


class GMF(torch.nn.Module):
    def __init__(self, user_num, item_num, predictive_factor=12):
        """
        Initializes the layers of the model.

        Parameters:
            user_num (int): The number of users in the dataset.
            item_num (int): The number of items in the dataset.
            predictive_factor (int, optional): The latent dimension of the model. Default is 12.
        """
        super(GMF, self).__init__()
        self.gmf_user_embeddings = torch.nn.Embedding(num_embeddings=user_num, embedding_dim=predictive_factor)
        self.gmf_item_embeddings = torch.nn.Embedding(num_embeddings=item_num, embedding_dim=predictive_factor)
        self.gmf_out = torch.nn.Linear(predictive_factor, 1)
        self.output_logits = torch.nn.Linear(predictive_factor, 1)
        self.initialize_weights()

    def initialize_weights(self):
        """Initializes the weight parameters using Xavier initialization."""
        torch.nn.init.xavier_uniform_(self.gmf_user_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.gmf_item_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.gmf_out.weight)
        torch.nn.init.xavier_uniform_(self.output_logits.weight)

    def forward(self, x):
        user_id, item_id = x[:, 0], x[:, 1]
        gmf_product = self.gmf_forward(user_id, item_id)
        output_logits = self.output_logits(gmf_product)
        output_scores = torch.sigmoid(output_logits)
        return output_scores.view(-1)

    def gmf_forward(self, user_id, item_id):
        user_emb = self.gmf_user_embeddings(user_id)
        item_emb = self.gmf_item_embeddings(item_id)
        return torch.mul(user_emb, item_emb)

    def layer_setter(self, model, model_copy):
        model_state_dict = model.state_dict()
        model_copy.load_state_dict(model_state_dict)

    def load_server_weights(self, server_model):
        self.layer_setter(server_model.gmf_item_embeddings, self.gmf_item_embeddings)
        self.layer_setter(server_model.gmf_out, self.gmf_out)
        self.layer_setter(server_model.output_logits, self.output_logits)
