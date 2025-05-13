import torch.nn as nn
from layers.snippets import SigmoidRange

class PretrainHead(nn.Module):
    def __init__(self, d_model, n_output):
        super(PretrainHead, self).__init__()
        self.head = nn.Linear(d_model, n_output)

    def forward(self, x):
        """
        x: tensor [batch_size x seq_len x d_model]
        output: tensor [batch_size x seq_len x n_output]
        """
        return self.head(x)


class ForecastHead(nn.Module):
    def __init__(self, d_model, c_in, red_factor=4):
        super(ForecastHead, self).__init__()
        self.d_model = d_model
        self.c_in = c_in
        self.red_factor = red_factor # default is 4 since conv-subsampling reduce sequence lengh to 1/4

        self.token_projection = nn.Linear(self.d_model, self.d_model)
        self.time_projection = nn.Linear(self.d_model, red_factor*self.c_in)

    def forward(self, hidden_states):
        """
        hidden_states: tensor of shape [batch_size, seq_len, d_model]
        """
        x = hidden_states.mean(dim=1)  # Average pool over the hidden states
        next_token = self.token_projection(x)
        compressed_next_timesteps = self.time_projection(next_token)
        # Reshaping the compressed next timesteps into the required form [red_factor, c_in]
        next_timesteps = compressed_next_timesteps.view(-1, self.red_factor, self.c_in)

        return next_token, next_timesteps


class ClfHead(nn.Module):
    def __init__(self, d_model, n_output):
        super(ClfHead, self).__init__()
        self.head = nn.Linear(d_model, n_output)

    def forward(self, x):
        """
        x: tensor [batch_size x seq_len x d_model]
        output: tensor [batch_size x num_classes]
        """
        # x = x[:, 0, :]     # Only use the [sos] token
        x = x.mean(dim=1)    # Average pool over the hidden states
        return self.head(x)


class RegHead(nn.Module):
    def __init__(self, d_model, reg_dim=1, y_range=None):
        super(RegHead, self).__init__()
        self.y_range = y_range
        self.regr_layer = nn.Linear(d_model, reg_dim)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x output_dim]
        """
        x = x.mean(dim=1)         # Average pool over the sequence dimension
        y = self.regr_layer(x)
        if self.y_range: y = SigmoidRange(*self.y_range)(y)

        return y