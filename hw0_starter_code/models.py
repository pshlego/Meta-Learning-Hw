"""
Classes defining user and item latent representations in
factorization models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to zero.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    sparse: boolean, optional
        Use sparse gradients.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.

    """

    def __init__(self, num_users, num_items, embedding_dim=32, layer_sizes=[96, 64],
                 sparse=False, embedding_sharing=True):

        super().__init__()

        self.embedding_dim = embedding_dim

        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************

        self.embedding_sharing = embedding_sharing

        self.embedding_users = ScaledEmbedding(num_users, embedding_dim)
        self.embedding_items = ScaledEmbedding(num_items, embedding_dim)

        if not embedding_sharing:
            self.embedding_users_score = ScaledEmbedding(num_users, embedding_dim)
            self.embedding_items_score = ScaledEmbedding(num_items, embedding_dim)
        
        self.A = ZeroEmbedding(num_users, 1)
        self.B = ZeroEmbedding(num_items, 1)

        self.layers = []
        for i in range(0, len(layer_sizes)):
            if i != len(layer_sizes)-1:
                self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            else:
                self.layers.append(nn.Linear(layer_sizes[i], 1))

        #********************************************************
        #********************************************************
        #********************************************************

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of shape (batch,)
        score: tensor
            Tensor of user-item score predictions of shape (batch,)
        """
        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************
        predictions = torch.matmul(torch.transpose(self.embedding_users(user_ids).unsqueeze(2), 1,2),self.embedding_items(item_ids).unsqueeze(2)).squeeze(2) + self.A(user_ids) + self.B(item_ids)
        predictions = predictions.squeeze()
        
        if self.embedding_sharing:
            mlp_input = torch.cat([
                self.embedding_users(user_ids), 
                self.embedding_items(item_ids), 
                torch.mul(self.embedding_users(user_ids), self.embedding_items(item_ids))]
            , 1)
        else:
            mlp_input = torch.cat([
                self.embedding_users_score(user_ids), 
                self.embedding_items_score(item_ids),
                torch.mul(self.embedding_users_score(user_ids), self.embedding_items_score(item_ids))], 
            1)
        
        for layer_id, layer in enumerate(self.layers):
            if layer_id ==0:
                mlp_output = layer(mlp_input)
            else:
                mlp_output = layer(mlp_output)
            mlp_output = F.relu(mlp_output)
        
        score = mlp_output.squeeze()
        #********************************************************
        #********************************************************
        #********************************************************
        ## Make sure you return predictions and scores of shape (batch,)
        if (len(predictions.shape) > 1) or (len(score.shape) > 1):
            raise ValueError("Check your shapes!")
        
        return predictions, score