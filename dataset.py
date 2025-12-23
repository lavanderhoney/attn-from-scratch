#%%
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
dataset = load_dataset("Helsinki-NLP/opus_books", "en-fr", split="train")

def get_training_corpus(lang:str):
    for item in dataset:
        yield item["translation"][lang] # yield a sentence of specified language
        
            
# train a BPE for en and fr separately
def get_tokenizer(lang, vocab_size:int = 32_000, tokenizer_path=None ) -> Tokenizer:
    if tokenizer_path:
        print("Loading tokenizer from: ", tokenizer_path)
        return Tokenizer.from_file(tokenizer_path)

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFKC(), normalizers.Replace(r"\s+", " "), normalizers.Strip()]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens, min_frequency=2)
    tokenizer.train_from_iterator(get_training_corpus(lang), trainer=trainer)
    
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    if not os.path.exists("tokenizers/"):
        os.makedirs("tokenizers/")
    saved_path =f"tokenizers/bpe_tokenizer_opus_{lang}.json"
    tokenizer.save(saved_path)
    print("tokenizer saved to: " ,saved_path)
    return tokenizer

class BilingualDataset(Dataset):
    """
    Returns variable-length token IDs only.
    """
    def __init__(self, dataset, seq_len:int, src_tokenizer: Tokenizer, tgt_tokenizer: Tokenizer, src_lang="en", tgt_lang="fr"):
        super().__init__()
        self.dataset = dataset
        self.seq_len = seq_len
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        self.sos_id = src_tokenizer.token_to_id("[SOS]")
        self.eos_id = src_tokenizer.token_to_id("[EOS]")
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        ex = self.dataset[index]["translation"]
        src_ids = self.src_tokenizer.encode(ex[self.src_lang]).ids
        tgt_ids = self.tgt_tokenizer.encode(ex[self.tgt_lang]).ids
    
        src_ids = [self.sos_id] + src_ids + [self.eos_id]
        tgt_ids = [self.sos_id] + tgt_ids + [self.eos_id]
        return {
            "src_ids" : torch.tensor(src_ids, dtype=torch.long), # NOTE: nn.Embedding expects the input seq to have Long or Int64 dtype, NOT float
            "tgt_ids" : torch.tensor(tgt_ids, dtype=torch.long)
        }
    
def collate_fn(batch, pad_id, seq_len):
    B = len(batch)
    
    # Pre-allocate memory for the collated batch. Helps with memory efficiency and performance
    src_batch = torch.full((B, seq_len), pad_id)
    tgt_batch = torch.full((B, seq_len), pad_id)
    
    for i, item in enumerate(batch):
        src = item["src_ids"]
        tgt = item["tgt_ids"]
        
        src_len = min(len(src), seq_len)
        tgt_len = min(len(tgt), seq_len)
        
        # 3. Assign directly (No need to wrap in torch.tensor() again)
        # This copies the data into the pre-allocated batch tensor
        src_batch[i, :src_len] = src[:src_len]
        tgt_batch[i, :tgt_len] = tgt[:tgt_len]
        
        # NOTE: we abruptly truncate a longer sequence, i.e, it won't end with EOS.
        # forcing a longer sentence to end with EOS will teach false grammer to the transformer, and it will hallucinate more.
    
    return {
        "src_ids" : src_batch,
        "tgt_ids" : tgt_batch
    }

def get_dataloaders(seq_len, batch_size, vocab_size, src_tokenizer_path, tgt_tokenizer_path, test_size):
    
    en_tokenizer = get_tokenizer("en", vocab_size=vocab_size, tokenizer_path=src_tokenizer_path)
    fr_tokenizer = get_tokenizer("fr", vocab_size=vocab_size, tokenizer_path=tgt_tokenizer_path)
    
    splits = dataset.train_test_split(test_size=test_size, seed=42)
    
    train_ds = BilingualDataset(splits["train"], seq_len, en_tokenizer, fr_tokenizer)
    val_ds   = BilingualDataset(splits["test"],  seq_len, en_tokenizer, fr_tokenizer)

    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, en_tokenizer.token_to_id("[PAD]"), seq_len))
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, en_tokenizer.token_to_id("[PAD]"), seq_len))
    
    return train_dl, val_dl, en_tokenizer, fr_tokenizer

#%%
if __name__ == "__main__":
    train_dl, val_dl, en_tokenizer, fr_tokenizer = get_dataloaders(
        seq_len=32, 
        batch_size=16, 
        vocab_size=32_000, 
        src_tokenizer_path="tokenizers/bpe_tokenizer_opus_en.json", 
        tgt_tokenizer_path="tokenizers/bpe_tokenizer_opus_fr.json",
        test_size=0.1
    )
    
    for batch in train_dl:
        print(batch["src_ids"].shape)  # should be (batch_size, seq_len)
        print(batch["tgt_ids"].shape)  # should be (batch_size, seq_len)
        print(batch["src_ids"][0])    # print first sample's src_ids
        print(batch["tgt_ids"][0])    # print first sample's tgt_ids
        break
#%%