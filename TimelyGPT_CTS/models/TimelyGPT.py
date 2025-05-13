import gc
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
# from layers.Retention_layers import RetNetBlock
from layers.RevIN import RevIN
from layers.Retention_layers import RetNetBlock
from layers.heads import PretrainHead, ForecastHead, ClfHead, RegHead
from layers.Embed import TokenEmbeddingFixed
from layers.Conv_layers import Conv1dSubampling, Conv1dUpsampling
from layers.snippets import get_gpu_memory_usage, SigmoidRange
from sklearn.metrics import roc_auc_score, average_precision_score


class TimelyGPT(nn.Module):
    '''
    TimelyGPT leverages recurrent attention (Retention) architecture for continuous time-series data with multiple variates
    '''
    def __init__(self, configs, head_type='pretrain'):
        super(TimelyGPT, self).__init__()

        # load parameters
        self.num_layers = configs.num_layers
        self.num_heads = configs.num_heads
        self.d_model = configs.d_model
        self.qk_dim = configs.qk_dim
        self.v_dim = configs.v_dim if configs.v_dim else self.qk_dim
        self.dropout = configs.dropout

        # the start token for shifted right
        self.sos = torch.nn.Parameter(torch.zeros(self.d_model))
        nn.init.normal_(self.sos)

        # Initialize number of variates
        self.c_in = configs.c_in
        self.revin_layer = RevIN(self.c_in)

        # Integration of ConvSubampling for embedding
        self.conv_subsampling = Conv1dSubampling(in_channels=self.c_in, out_channels=self.d_model, reduce_time_layers=2)
        # Add ConvUpampling for upsampling, note that the argument intermediate_channels is not used
        self.conv_upsampling = Conv1dUpsampling(hidden_dim=self.d_model, reduce_time_layers=2)
        self.input_projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Dropout(p=self.dropout)
        )

        # the stacked decoder layer
        self.blocks = nn.ModuleList([RetNetBlock(configs) for _ in range(self.num_layers)])

        # output layer
        self.ln_f = nn.LayerNorm(self.d_model)  # Layer Normalization
        self.head_type = head_type
        if self.head_type == "pretrain":
            self.head = PretrainHead(self.d_model, self.c_in) # the token is [batch_size x seq_len x c_in]
        elif self.head_type == "forecast":
            self.head = ForecastHead(self.d_model, self.c_in)
        elif self.head_type == "clf":
            self.n_class = configs.n_class
            self.head = ClfHead(self.d_model, self.n_class)
        elif self.head_type == "reg":
            self.reg_dim = configs.reg_dim
            self.head = RegHead(self.d_model, self.reg_dim)
        else:
            raise ValueError("Invalid head_type provided.")
        self.gradient_checkpointing = configs.use_grad_ckp

    def forward(self,
                X, y,
                retention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                forward_impl: Optional[str] = 'parallel',
                chunk_size: Optional[int] = None,
                sequence_offset: Optional[int] = 0,
                output_retentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                ):
        # Use ConvSubsampling as tokenizer, input_project as embedding layer
        X, X_tokens = self.conv_subsampling(X)
        hidden_states = self.input_projection(X)
        batch_size, seq_len, dim = X.shape

        # Add the SOS token to the input sequence
        sos_token = self.sos.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape [batch_size, 1, d_model]
        # Add the SOS token for shifted right,  so the sequence length will be seq_len + 1
        hidden_states = torch.cat([sos_token, hidden_states], dim=1)

        if retention_mask is None: # what is the usage of rentention mask
            # not sure whether we need to mask the first token (SOS token)
            retention_mask = torch.ones((batch_size, seq_len+1), dtype=torch.bool, device=X.device) # batch_size x token_num

        all_hidden_states = () if output_hidden_states else None
        all_retentions = () if output_retentions else None
        present_key_values = ()  # To store current key-value pairs
        for i, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training: # Use gradient checkpointing for the forward pass
                def custom_forward(*inputs):
                    return block(*inputs, sequence_offset, chunk_size, output_retentions)

                block_outputs = torch.utils.checkpoint.checkpoint(
                                custom_forward,
                                hidden_states,
                                retention_mask,
                                forward_impl,
                                past_key_value,
                                )
            else:
                block_outputs = block(hidden_states,
                                      retention_mask=retention_mask,
                                      forward_impl=forward_impl,
                                      past_key_value=past_key_value,
                                      sequence_offset=sequence_offset,
                                      chunk_size=chunk_size,
                                      output_retentions=output_retentions)

            # outputs two variables if output_retentions is False: output hidden states (self.proj(out)), present key values (curr_kv)
            hidden_states = block_outputs[0]
            present_key_values += (block_outputs[1],)

            torch.cuda.empty_cache()
            gc.collect()
            # calculate memory usage after processing the current layer
            # gpu_mem_usage = get_gpu_memory_usage()
            # print("GPU memory usage after %d th layer:" % i)
            # print("Total GPU Memory: {} MiB".format(gpu_mem_usage['total']))
            # print("Used GPU Memory: {} MiB".format(gpu_mem_usage['used']))
            # print("Free GPU Memory: {} MiB".format(gpu_mem_usage['free']))

            # if output_retentions is True, we have an extra variable in block_outputs: retentions (visualization analysis)
            if output_retentions:
                all_retentions += (block_outputs[2],)

        # add hidden states from the last layer
        if output_hidden_states:
            all_hidden_states += (hidden_states)

        # Apply the custom head on the hidden states for output
        outputs = self.ln_f(hidden_states)
        logits = self.head(outputs)
        if self.head_type == 'pretrain':
            return self.compute_pretrain_loss(logits, X_tokens) # return pre-trained loss
        elif self.head_type == 'forecast':
            return self.compute_forecast_loss(logits, y) # return forecasting loss
        elif self.head_type == 'clf':
            return self.compute_cls_loss(logits, y) # return classification loss
        elif self.head_type == 'reg':
            return self.compute_reg_loss(logits, y) # return regression loss

    def compute_pretrain_loss(self, logits, targets):
        """
        Compute the loss of the pre-training task (next token prediction)
        """
        self.mse_loss = nn.MSELoss()
        # Drop the last prediction (since there's no next token beyond it)
        preds = logits[:, :-1, :]  # (B, 750, d_model)
        return self.mse_loss(preds, targets)

    def compute_forecast_loss(self, logits, targets):
        """
        Compute the MSE loss for the forecasting head
        """
        mse = nn.MSELoss()
        return mse(logits, targets)

    def compute_cls_loss(self, logits, targets):
        """
        Compute the loss of classification task
        """
        # Apply sigmoid to logits to get probabilities
        probs = torch.sigmoid(logits)
        # Apply a threshold of 0.5 to get binary predictions
        predicted = (probs > 0.5).float()
        # Compute accuracy
        correct = (predicted == targets).float()
        accuracy = correct.mean() * 100.0
        return accuracy

    def compute_regr_loss(self, regr_predictions, regr_targets):
        """
        Compute the loss of the regression task
        """
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

        # Ensure regr_targets is a float tensor and has the same shape as regr_predictions
        regr_targets = regr_targets.float().view_as(regr_predictions)

        regr_loss = self.mse_loss(regr_predictions, regr_targets)
        mae = F.l1_loss(regr_predictions, regr_targets)  # L1 loss is equivalent to MAE
        return regr_loss, mae



