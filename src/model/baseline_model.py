from torch import nn
from torch.nn import Sequential
import torch
import torch.nn.functional as F
import torchaudio.transforms as T

class BaselineModel(nn.Module):

    def __init__(self, n_feats, n_tokens, fc_hidden=512, conv_out_channels=32, rnn_hidden_size=512, n_rnn_layers=2, dropout_p=0.1):
        """
        Args:
            n_feats (int): Number of input features (the height of input spectrogram).
            n_tokens (int): Number of tokens in the vocabulary.
            fc_hidden (int): Number of features for the fully connected layer.
            conv_out_channels (int): Number of output channels for the convolutional layer.
            rnn_hidden_size (int): Number of hidden units in RNN layers.
            n_rnn_layers (int): Number of RNN layers.
            dropout_p (float): Dropout probability.
        """
        super().__init__()
        
        # Convolutional front-end
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=conv_out_channels, kernel_size=(11, 11), stride=(2, 2), padding=(5, 5)),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU()
        )
        
        self.rnn_input_size = (n_feats // 2) * conv_out_channels  # Adjusted based on convolution output
        
        # RNN layers
        self.rnn_layers = nn.ModuleList([
            nn.GRU(
                input_size=self.rnn_input_size if idx == 0 else 2 * rnn_hidden_size,
                hidden_size=rnn_hidden_size,
                num_layers=1,
                batch_first=True,
                dropout=dropout_p,
                bidirectional=True
            ) for idx in range(n_rnn_layers)
        ])
        
        # Fully connected layers
        self.fc_layers = Sequential(
            nn.Linear(in_features=2 * rnn_hidden_size, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=n_tokens)
        )
        
        # Data augmentation
        self.time_mask = T.TimeMasking(time_mask_param=30)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=15)

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): Input spectrogram (B, T, F).
            spectrogram_length (Tensor): Original lengths of spectrograms.

        Returns:
            output (dict): Output containing log_probs and transformed lengths.
        """
        # Data augmentation
        spectrogram = self.time_mask(spectrogram)
        spectrogram = self.freq_mask(spectrogram)
        
        # Convolutional layer
        x = self.conv(spectrogram.unsqueeze(1))  # Adding channel dimension
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, f * c)

        # RNN layers
        for rnn_layer in self.rnn_layers:
            x, _ = rnn_layer(x)

        # Fully connected layer
        x = self.fc_layers(x)
        log_probs = F.log_softmax(x, dim=-1)
        
        log_probs_length = self.transform_input_lengths(spectrogram_length)
        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        Calculate the temporal dimension after convolution down-sampling.

        Args:
            input_lengths (Tensor): Old input lengths.

        Returns:
            output_lengths (Tensor): New temporal lengths.
        """
        return input_lengths // 2

    def __str__(self):
        """
        Print model with the number of parameters.
        """
        all_parameters = sum(p.numel() for p in self.parameters())
        trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

        result_info = super().__str__()
        result_info += f"\nAll parameters: {all_parameters}"
        result_info += f"\nTrainable parameters: {trainable_parameters}"

        return result_info
