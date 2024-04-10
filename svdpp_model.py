import torch


class SVDPP(torch.nn.Module):
    def __init__(self, user_num, item_num, model, predictive_factor=12):
        """
        Initializes the layers of the model.

        Parameters:
            user_num (int): The number of users in the dataset.
            item_num (int): The number of items in the dataset.
            predictive_factor (int, optional): The latent dimension of the model. Default is 12.
        """
        super(SVDPP, self).__init__()
        self.gmf_user_embeddings = torch.nn.Embedding(num_embeddings=user_num, embedding_dim=predictive_factor)
        self.gmf_item_embeddings = torch.nn.Embedding(num_embeddings=item_num, embedding_dim=predictive_factor)
        self.user_bias = torch.nn.Embedding(user_num, 1)
        self.item_bias = torch.nn.Embedding(item_num, 1)
        self.initialize_weights()

    def initialize_weights(self):
        """Initializes the weight parameters using Xavier initialization."""
        torch.nn.init.xavier_uniform_(self.gmf_user_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.gmf_item_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.user_bias.weight)
        torch.nn.init.xavier_uniform_(self.item_bias.weight)

    def forward(self, x):
        user_id, item_id = x[:, 0], x[:, 1]
        svdpp_output = self.svdpp_forward(user_id, item_id)
        return svdpp_output

    def gmf_forward(self, user_id, item_id):
        user_emb = self.gmf_user_embeddings(user_id)
        item_emb = self.gmf_item_embeddings(item_id)
        return torch.mul(user_emb, item_emb)

    def svdpp_forward(self, user_id, item_id):
        user_emb = self.gmf_user_embeddings(user_id)
        item_emb = self.gmf_item_embeddings(item_id)
        user_bias = self.user_bias(user_id)
        item_bias = self.item_bias(item_id)
        user_avg = user_emb.mean(dim=1)
        user_preference = torch.bmm(user_emb.unsqueeze(1), item_emb.unsqueeze(-1)).squeeze()
        user_preference += item_bias.squeeze() + user_bias.squeeze() + user_avg.squeeze()
        pred = torch.sigmoid(user_preference)
        return pred.view(-1)

    def layer_setter(self, model, model_copy):
        model_state_dict = model.state_dict()
        model_copy.load_state_dict(model_state_dict)

    def load_server_weights(self, server_model):
        self.layer_setter(server_model.gmf_item_embeddings, self.gmf_item_embeddings)
        self.layer_setter(server_model.item_bias, self.item_bias)###