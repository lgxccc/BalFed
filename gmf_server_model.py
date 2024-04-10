import torch


class ServerGMF(torch.nn.Module):
    def __init__(self, item_num, predictive_factor=12):
        """
        Initializes the layers of the model.

        Args:
            item_num (int): The number of items in the dataset.
            predictive_factor (int, optional): The latent dimension of the model. Default is 12.
        """
        super(ServerGMF, self).__init__()
        self.gmf_item_embeddings = torch.nn.Embedding(num_embeddings=item_num, embedding_dim=predictive_factor)
        self.gmf_out = torch.nn.Linear(predictive_factor, 1)
        self.output_logits = torch.nn.Linear(predictive_factor, 1)
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.xavier_uniform_(self.gmf_item_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.gmf_out.weight)
        torch.nn.init.xavier_uniform_(self.output_logits.weight)

    def layer_setter(self, model, model_copy):
        model_state_dict = model.state_dict()
        model_copy.load_state_dict(model_state_dict)

    def set_weights(self, model):
        self.layer_setter(model.gmf_item_embeddings, self.gmf_item_embeddings)
        self.layer_setter(model.gmf_out, self.gmf_out)
        self.layer_setter(model.output_logits, self.output_logits)

    def forward(self):
        return torch.tensor(0.0)