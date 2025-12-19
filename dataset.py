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

def get_training_corpus(lang:str, batch_size):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["translation"][lang] # yield a sentence of specified language
        
            
# train a BPE for en and fr separately
def get_tokenizer(lang, batch_size:int = 100000, vocab_size:int = 32000, tokenizer_path=None ) -> Tokenizer:
    if tokenizer_path:
        return Tokenizer.from_file(tokenizer_path)

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFKC(), normalizers.Replace(r"\s+", " "), normalizers.Strip()]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens, min_frequency=2)
    tokenizer.train_from_iterator(get_training_corpus(lang, batch_size), trainer=trainer)
    
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    saved_path =f"bpe_tokenizer_opus_{lang}.json"
    tokenizer.save(saved_path)
    print("tokenizer saved to: " ,saved_path)
    return tokenizer

class BilingualDataset(Dataset):
    """
    The dataset returns (seq_len) tensor token_ids of both the languages for the given corpus.
    It also handles the padding.
    """
    def __init__(self, dataset, seq_len:int, src_tokenizer: Tokenizer, tgt_tokenizer: Tokenizer, src_lang="en", tgt_lang="fr"):
        super().__init__()
        self.dataset = dataset
        self.seq_len = seq_len
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
    def __len__(self):
        return len(self.dataset)
    
    def pad_sequence(self, tokenizer, enc):
        # src input is padded with SOS, and EOS.
        # In the output, our sentence tokens will have size seq_len-2.
        if len(enc) < self.seq_len-2:
           padded_tensor = torch.tensor([
               tokenizer.token_to_id("[SOS]"),
               *enc,
               *[tokenizer.token_to_id("[PAD]") for _ in range((self.seq_len - 2) - len(enc))],
               tokenizer.token_to_id("[EOS]")
           ])
        else:
            padded_tensor = torch.tensor([
                tokenizer.token_to_id("[SOS]"),
                enc[:self.seq_len-2],
                tokenizer.token_to_id("[EOS]")
            ])
        return padded_tensor
    def __getitem__(self, index):
        ex = self.dataset[index]["translation"]
        src_enc = self.src_tokenizer.encode(ex[self.src_lang]).ids
        tgt_enc = self.tgt_tokenizer.encode(ex[self.tgt_lang]).ids
    
        src_tensor = self.pad_sequence(self.src_tokenizer, src_enc)
        tgt_tensor = self.pad_sequence(self.tgt_tokenizer, tgt_enc)
        return (src_tensor, tgt_tensor)   
       
en_tokenizer = get_tokenizer("en")
fr_tokenizer = get_tokenizer("fr")
splits = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = BilingualDataset(splits["train"], en_tokenizer, fr_tokenizer)
val_ds   = BilingualDataset(splits["test"],  en_tokenizer, fr_tokenizer)


train_dl = DataLoader(train_ds, batch_size=32)
val_dl = DataLoader(val_ds, batch_size=32)