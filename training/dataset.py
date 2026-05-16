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
        # print("Loading tokenizer from: ", tokenizer_path)
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
    def __init__(self, seq_len, dataset, src_tokenizer: Tokenizer, tgt_tokenizer: Tokenizer, src_lang="en", tgt_lang="fr"):
        super().__init__()
        self.max_seq_len = seq_len
        self.dataset = dataset
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        self.sos_id_src = src_tokenizer.token_to_id("[SOS]")
        self.eos_id_src = src_tokenizer.token_to_id("[EOS]")
        self.sos_id_tgt = tgt_tokenizer.token_to_id("[SOS]")
        self.eos_id_tgt = tgt_tokenizer.token_to_id("[EOS]")
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        ex = self.dataset[index]["translation"]
        src_ids = self.src_tokenizer.encode(ex[self.src_lang]).ids
        tgt_ids = self.tgt_tokenizer.encode(ex[self.tgt_lang]).ids
    
        max_content_len = self.max_seq_len - 2

        src_ids = src_ids[:max_content_len]
        tgt_ids = tgt_ids[:max_content_len]

        src_ids = [self.sos_id_src] + src_ids + [self.eos_id_src]
        tgt_ids = [self.sos_id_tgt] + tgt_ids + [self.eos_id_tgt]

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long)
        }
    
def collate_fn(batch, pad_id_src, pad_id_tgt):
    """
    batch: list of dicts containing tokenized sequences.
    Dynamic padding
    """
    src_ids = [item["src_ids"] for item in batch]
    tgt_ids = [item["tgt_ids"] for item in batch]
    
    src_padded = nn.utils.rnn.pad_sequence(
        src_ids,
        batch_first=True,
        padding_value=pad_id_src
    ) # (B, L_src_max)
    
    tgt_padded = nn.utils.rnn.pad_sequence(
        tgt_ids,
        batch_first=True,
        padding_value=pad_id_tgt
    )
    
    return {
        "src_ids" : src_padded,
        "tgt_ids" : tgt_padded  
    }

def get_dataloaders(seq_len, batch_size, vocab_size, src_tokenizer_path, tgt_tokenizer_path, test_size):
    
    en_tokenizer = get_tokenizer("en", vocab_size=vocab_size, tokenizer_path=src_tokenizer_path)
    fr_tokenizer = get_tokenizer("fr", vocab_size=vocab_size, tokenizer_path=tgt_tokenizer_path)
    
    splits = dataset.train_test_split(test_size=test_size, seed=42)
    
    train_ds = BilingualDataset(seq_len, splits["train"], en_tokenizer, fr_tokenizer)
    val_ds   = BilingualDataset(seq_len, splits["test"], en_tokenizer, fr_tokenizer)

    # print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, en_tokenizer.token_to_id("[PAD]"), fr_tokenizer.token_to_id("[PAD]")))
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, en_tokenizer.token_to_id("[PAD]"), fr_tokenizer.token_to_id("[PAD]")))
    
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
        print(batch["src_ids"].shape)  # should be (batch_size)
        print(batch["tgt_ids"].shape)  # should be (batch_size)
        print(batch["src_ids"][0])    # print first sample's src_ids
        print(batch["tgt_ids"][0])    # print first sample's tgt_ids
        break
#%%