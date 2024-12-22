import torch
import torch.nn.functional as F
from torch import nn
from src.model.baseline_model import BaselineModel


class BlockRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, rnn_type : nn.GRU, dropout = 0.1, bidirectional_ = True, batch_first_ = True, bias_ = True):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size, bias=bias_, dropout=dropout, bidirectional=bidirectional_, batch_first=batch_first_)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        inputs = F.relu(self.bn(inputs.transpose(1, 2))).transpose(1, 2)
        outputs, _ = self.rnn(nn.utils.rnn.pack_padded_sequence(inputs, input_lengths.cpu(), batch_first=True, enforce_sorted=False))
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        return outputs


class maskCNN(nn.Module):
    def __init__(self, sequential_: nn.Sequential):
        super().__init__()
        self.sequential = sequential_

    def forward(self, inputs: torch.Tensor, seq_lengths: torch.Tensor):
        
        output = None
        for module in self.sequential:
            output = module(inputs)
            mask = torch.BoolTensor(output.size()).fill_(0)
            if output.is_cuda:
                mask = mask.cuda()

            seq_lengths = self._get_sequence_lengths(module, seq_lengths, dim=1)
            for idx, length in enumerate(seq_lengths):
                length = length.item()
                if (mask[idx].size(2) - length) > 0:
                    mask[idx].narrow(dim=2, start=length, length=mask[idx].size(2) - length).fill_(1)

            output = output.masked_fill(mask, 0)
            inputs = output

        return output, seq_lengths
    def get_output_size(self, input_size: int):
        size = torch.Tensor([input_size]).int()
        for module in self.sequential:
            size = self._get_sequence_lengths(module, size, dim=0)
        return size.item()

    def transform_input_lengths(self, input_size: torch.Tensor):
        for module in self.sequential:
            input_size = self._get_sequence_lengths(module, input_size, dim=1)
        return input_size

    def _get_sequence_lengths(self, module: nn.Module, seq_lengths: torch.Tensor, dim = 1) -> torch.Tensor:
        if isinstance(module, nn.Conv2d):
            numerator = seq_lengths + 2 * module.padding[dim] - module.dilation[dim] * (module.kernel_size[dim] - 1) - 1
            seq_lengths = numerator.float() / float(module.stride[dim])
            seq_lengths = seq_lengths.int() + 1

        elif isinstance(module, nn.MaxPool2d):
            seq_lengths >>= 1

        return seq_lengths.int()


class ConvolutionsModule(nn.Module):
    def __init__(self, n_feats: int, in_channels: int, out_channels: int, activation = nn.ReLU) -> None:
        super().__init__()
        self.activation = activation()
        self.mask_conv = maskCNN(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5), bias=False),
                nn.BatchNorm2d(out_channels),
                self.activation,
                nn.Conv2d(out_channels, out_channels, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5), bias=False),
                nn.BatchNorm2d(out_channels),
                self.activation,
            )
        )
        self.output_size = self.mask_conv.get_output_size(n_feats)

    def forward(self, spectrogram: torch.Tensor, spectrogram_length: torch.Tensor):
        outputs, output_lengths = self.mask_conv(spectrogram.unsqueeze(1), spectrogram_length)
        batch_size, channels, features, time = outputs.size()
        outputs = outputs.permute(0, 3, 1, 2)
        outputs = outputs.view(batch_size, time, channels * features)
        return outputs, output_lengths


class DeepSpeech2(BaselineModel):
    def __init__(
            self,
            n_feats: int,
            n_tokens: int,
            rnn_type=nn.GRU,
            n_rnn_layers: int = 2,
            conv_out_channels: int = 32,
            rnn_hidden_size: int = 512,
            dropout_p: float = 0.1,
            activation = nn.ReLU
    ):
        super().__init__(n_feats=n_feats, n_tokens=n_tokens)
        self.conv = ConvolutionsModule(n_feats=n_feats, in_channels=1, out_channels=conv_out_channels, activation=activation)

        rnn_output_size = rnn_hidden_size * 2
        self.rnn_layers = nn.ModuleList([
            BlockRNN(
                input_size=self.conv.mask_conv.get_output_size(n_feats) * conv_out_channels if idx == 0 else rnn_output_size,
                hidden_size=rnn_hidden_size,
                rnn_type=rnn_type,
                dropout=dropout_p
            ) for idx in range(n_rnn_layers)
        ])
        self.batch_norm = nn.BatchNorm1d(rnn_output_size)
        self.fc = nn.Linear(rnn_output_size, n_tokens, bias=False)

    def forward(self, spectrogram: torch.Tensor, spectrogram_length: torch.Tensor, **batch):
        outputs, output_lengths = self.conv(spectrogram, spectrogram_length)
        outputs = outputs.permute(1, 0, 2).contiguous()
        for rnn_layer in self.rnn_layers:
            outputs = rnn_layer(outputs.transpose(0, 1), output_lengths)

        outputs = self.batch_norm(outputs.permute(1, 2, 0))

        outputs = self.fc(outputs.transpose(1, 2))
        return {"log_probs": outputs, "log_probs_length": output_lengths}

    def transform_input_lengths(self, input_lengths):
        return self.conv.mask_conv.transform_input_lengths(input_lengths)
