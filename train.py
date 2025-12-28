import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
import argparse
from model import Transformer
from dataset import get_dataloaders
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split


try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None

# Load configuration from YAML file
def load_config(config_path='config.yml'):
    print(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def _load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: torch.optim.Optimizer | None, device: torch.device):
    checkpoint_path = str(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Resuming from checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' not in ckpt:
        raise KeyError("Checkpoint missing 'model_state_dict'")
    model.load_state_dict(ckpt['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in ckpt and ckpt['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    start_epoch = int(ckpt.get('epoch', 0))
    global_step = int(ckpt.get('global_step', 0))
    print(f"Loaded checkpoint (epoch={start_epoch}, global_step={global_step})")
    return start_epoch, global_step

def train(config):
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
    
    # test dl
    # from torch.utils.data import Dataset, DataLoader

    # class DummyTranslationDataset(Dataset):
    #     def __init__(self, n_samples, seq_len, vocab_size):
    #         self.src = torch.randint(1, vocab_size, (n_samples, seq_len))
    #         self.tgt = torch.randint(1, vocab_size, (n_samples, seq_len))
    
    #     def __len__(self):
    #         return self.src.size(0)
    
    #     def __getitem__(self, idx):
    #         return {
    #             "src_ids": self.src[idx],
    #             "tgt_ids": self.tgt[idx],
    #         }
    
    # SEQ_LEN = 32
    # VOCAB_SIZE = 300
    # dummy_ds = DummyTranslationDataset(n_samples=64, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)
    # train_dl = DataLoader(dummy_ds, batch_size=2, shuffle=False)
    
    model = Transformer(VOCAB_SIZE, VOCAB_SIZE, 
                        SEQ_LEN, SEQ_LEN, en_tokenizer.token_to_id("[PAD]"),
                        NUM_LAYERS, D_MODEL, D_FF,
                        NUM_HEADS)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=en_tokenizer.token_to_id("[PAD]"), label_smoothing=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Noam Scheduler
    lr_lambda = lambda step: (D_MODEL ** -0.5) * min(max(1, step) ** -0.5, step*(WARMUP ** -1.5))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda )

    start_epoch = 0
    global_step = 0
    resume_path = config['training'].get('resume_path', None)
    if resume_path:
        start_epoch, global_step = _load_checkpoint(resume_path, model, optimizer, device)
        if start_epoch >= N_EPOCHS:
            print(f"Checkpoint epoch ({start_epoch}) >= n_epochs ({N_EPOCHS}); nothing to train.")
            return

    writer = None
    if SummaryWriter is None:
        print("TensorBoard logging disabled (could not import torch.utils.tensorboard).")
        print("If you want TensorBoard logs, install it with: pip install tensorboard")
    else:
        writer = SummaryWriter()
        print(f"TensorBoard logging ")

    print("Starting training...")
    for epoch in range(start_epoch, N_EPOCHS):
        batch_iterator = tqdm(train_dl, desc=f"Training (epoch {epoch+1}/{N_EPOCHS})", unit="batch")
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        
        for batch in batch_iterator:

            optimizer.zero_grad()

            input_ids = batch["src_ids"].to(device)

            # right shift
            decoder_input = batch["tgt_ids"][:, :-1].to(device) # Input: [SOS, I, Am, EOS]
            target_labels = batch["tgt_ids"][:, 1:].to(device)  # Truth: [I, Am, EOS, PAD], shape: (batch, seq_len)

            logits = model(input_ids, decoder_input) # shape: (batch, seq_len, vocab_size)

            loss = loss_fn(logits.view(-1, VOCAB_SIZE),
                           target_labels.view(-1)) # flattens the target_labels, as NLLLoss expects the target to be indices of tgt class (NOT a 1-hot vector)
            global_step += 1
            epoch_loss_sum += float(loss.item())
            epoch_loss_count += 1
            batch_iterator.set_postfix(epoch=epoch, step=global_step, loss=f"{loss.item():.4f}")
                        
            if writer is not None:
                writer.add_scalar('train/loss', float(loss.item()), global_step)
                writer.add_scalar('train/lr', float(optimizer.param_groups[0]['lr']), global_step)

            loss.backward()
            
            # compute the L2 norm of the gradient, and also clip to 1.0
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # log this as a scaler
            if global_step % 10 == 0:
                writer.add_scalar("gradients/global_norm", grad_norm, global_step)
                writer.add_scalar('train/loss_epoch', epoch_loss_sum / epoch_loss_count, epoch + 1)
            
            # Log Histograms 
            if global_step % 500 == 0:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        # tag example: "encoder.layers.0.self_attn.in_proj_weight"
                        writer.add_histogram(f"grad/{name}", param.grad, global_step)
                        # You can also log weights to see if they are changing
                        # writer.add_histogram(f"weight/{name}", param, global_step)
            
            optimizer.step()           
            scheduler.step()

        # save the model after each epoch. Save under folder 'checkpoints/'
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch+1,
            'global_step': global_step,
            'config' : config
        }, f'checkpoints/transformer_noam_epoch_{epoch+1}.pt')

    if writer is not None:
        writer.flush()
        writer.close()
    
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Train Transformer")
    # parser.add_argument('--config', type=str, default='config.yml', help='Path to YAML config')
    # args = parser.parse_args()

    config = load_config()
    # config = load_config(args.config)
    train(config)