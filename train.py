import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import yaml
from pathlib import Path
from model import Transformer
from dataset import get_dataloaders

from tqdm import tqdm

# Load configuration from YAML file
def load_config(config_path='config.yml'):
    print(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config()

# Training hyperparameters
N_EPOCHS = int(config['training']['n_epochs'])
BATCH_SIZE = int(config['training']['batch_size'])
LR = float(config['training']['lr'])
WARMUP = float(config['training']['warmup'])

# Model hyperparameters
SEQ_LEN = config['model']['seq_len']
D_MODEL = config['model']['d_model']
D_FF = config['model']['d_ff']
NUM_HEADS = config['model']['num_heads']
VOCAB_SIZE = config['model']['vocab_size']
NUM_LAYERS = config['model']['num_layers']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Getting dataloaders...")
train_dl, val_dl, en_tokenizer, _ = get_dataloaders(
    SEQ_LEN, BATCH_SIZE, VOCAB_SIZE, 
    config['data']['src_tokenizer_path'], 
    config['data']['tgt_tokenizer_path'],
    config['data']['test_size']
)

model = Transformer(VOCAB_SIZE, VOCAB_SIZE, 
                    SEQ_LEN, SEQ_LEN, en_tokenizer.token_to_id("[PAD]"),
                    NUM_LAYERS, D_MODEL, D_FF,
                    NUM_HEADS)
model = model.to(device)

loss_fn = nn.CrossEntropyLoss(ignore_index=en_tokenizer.token_to_id("[PAD]"), label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# lr_lambda = lambda step: math.sqrt(D_MODEL) * min(math.sqrt(step), step*WARMUP*math.sqrt(WARMUP))
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, )
batch_iterator = tqdm(train_dl, desc="Training", unit="batch")

print("Starting training...")
for epoch in range(N_EPOCHS):
    for batch in batch_iterator:
        
        optimizer.zero_grad()
        
        input_ids = batch["src_ids"].to(device)
        decoder_input = batch["tgt_ids"].to(device)
        target_labels = batch["tgt_ids"].to(device)
        
        # right shift
        # decoder_input = batch["tgt_ids"][:, :-1].to(device) # Input: [SOS, I, Am, EOS]
        decoder_input[:, 0] = en_tokenizer.token_to_id("[PAD]")  # make last token as PAD to maintain same seq len
        # target_labels = batch["tgt_ids"][:, 1:].to(device)  # Truth: [I, Am, EOS, PAD], shape: (batch, seq_len)
        target_labels[:, -1] = en_tokenizer.token_to_id("[PAD]")  # make first token as PAD to maintain same seq len
        
        logits = model(input_ids, decoder_input) # shape: (batch, seq_len, vocab_size)
        
        loss = loss_fn(logits.view(-1, VOCAB_SIZE),
                       target_labels.view(-1)) # flattens the target_labels, as NLLLoss expects the target to be indices of tgt class (NOT a 1-hot vector)
        batch_iterator.set_postfix(epoch=epoch, loss=f"{loss.item():.4f}")
        
        loss.backward()
        optimizer.step()
                