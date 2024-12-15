import logging
from typing import List
import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    result_batch = {}
    result_batch["audio"] = pad_sequence([item["audio"].squeeze(0)
                                                for item in dataset_items], batch_first=True)
    result_batch["spectrogram_length"] = torch.tensor([item["spectrogram"].shape[2] 
                                                        for item in dataset_items])
    result_batch["spectrogram"] = pad_sequence([item["spectrogram"].squeeze(0).permute(1, 0)
                                    for item in dataset_items], batch_first=True).permute(0, 2, 1)
    result_batch["text_encoded_length"] = torch.tensor([item["text_encoded"].shape[1] 
                                                        for item in dataset_items])
    result_batch["text_encoded"] = pad_sequence([item["text_encoded"].squeeze(0) 
                                                 for item in dataset_items], batch_first=True)
    result_batch["text"] = [item["text"] for item in dataset_items]
    result_batch["audio_path"] = [item["audio_path"] for item in dataset_items]
    return result_batch
# TODO
# DONE
