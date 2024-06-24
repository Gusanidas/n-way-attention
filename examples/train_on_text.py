import tiktoken
from nway_attention.cfgs import Config
from nway_attention.utils_misc import get_device
from nway_attention.modules.transformer_models import TriformerMixed
from nway_attention.utils_misc import get_log_probs
from datasets import load_dataset
from transformers import AutoTokenizer
import time
import torch
from torch.utils.data import IterableDataset, DataLoader, Dataset
import gc

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token