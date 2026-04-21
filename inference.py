import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Transformer
from dataset import get_dataloaders, get_tokenizer
from typing import List
from tokenizers import Tokenizer
import argparse

def load_model(model_path: str):
    ckpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    config = ckpoint['config']
    SEQ_LEN = config['model']['seq_len']
    D_MODEL = config['model']['d_model']
    D_FF = config['model']['d_ff']
    NUM_HEADS = config['model']['num_heads']
    VOCAB_SIZE = config['model']['vocab_size']
    NUM_LAYERS = config['model']['num_layers']
    en_tokenizer = get_tokenizer(lang="en", tokenizer_path=config['data']['src_tokenizer_path'])

    model = Transformer(VOCAB_SIZE, VOCAB_SIZE, 
                        SEQ_LEN, SEQ_LEN, en_tokenizer.token_to_id("[PAD]"),
                        NUM_LAYERS, D_MODEL, D_FF,
                        NUM_HEADS)
    model.load_state_dict(ckpoint['model_state_dict'])
    
    return model, config

def tokenize_user_input(user_input:str, tokenizer:Tokenizer, seq_len:int):
    ids = tokenizer.encode(user_input).ids
    ids = ids[:seq_len-2] # account for SOS and EOS
    sos_id = tokenizer.token_to_id("[SOS]")
    pad_id = tokenizer.token_to_id("[PAD]")
    eos_id = tokenizer.token_to_id("[EOS]")
    ids = [sos_id] + ids + [eos_id]
    if len(ids) < seq_len:
        ids += [pad_id] * (seq_len - len(ids))
    ids_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0) # (
    return ids_tensor

def generate_target(model: Transformer, config, n_examples:int, user_exs:str=None) -> List[str]:
    en_tokenizer = get_tokenizer(lang="en", tokenizer_path=config['data']['src_tokenizer_path'])
    fr_tokenizer = get_tokenizer(lang="fr", tokenizer_path=config['data']['tgt_tokenizer_path'])
    sos_id = fr_tokenizer.token_to_id("[SOS]")
    eos_id = fr_tokenizer.token_to_id("[EOS]")
    tgt_sentence = []
    
    if user_exs is None:
        # sample n_examples sentences from the dataset and return those
        _, val_dl, _, _ = get_dataloaders(
            config['model']['seq_len'],
            n_examples,
            config['model']['vocab_size'],
            config['data']['src_tokenizer_path'], 
            config['data']['tgt_tokenizer_path'],
            config['data']['test_size']
        ) 
        for item in val_dl:
            examples = item["src_ids"]
            tgt_sentence = item["tgt_ids"].detach().cpu().tolist()
            break
    else:
        examples = tokenize_user_input(user_exs, en_tokenizer, config['model']['seq_len'])
        
    model.eval()
    with torch.no_grad():
        gen_ids = model.greedy_decode(examples, sos_id, eos_id).detach().cpu().tolist()
    
    gen_sentences = fr_tokenizer.decode_batch(gen_ids)
    
    src_sentence = en_tokenizer.decode_batch(examples.tolist())
    print(f"\n##################### ENGLISH SETNECE  #####################")
    print(src_sentence[0])
    
    print(f"\n##################### TRANSLATION  #####################")
    print( gen_sentences[0])
    
    if tgt_sentence:
        print(f"\n##################### TARGET SENTENCE  #####################")
        tgt_sentence = fr_tokenizer.decode_batch(tgt_sentence)
        print(tgt_sentence[0])

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, help="Path to the model checkpoint")
    parser.add_argument("--n-examples", type=int, default=1, help="Number of examples to generate")
    parser.add_argument("--user-exs", type=str, default=None, help="User provided examples to translate")
    args = parser.parse_args()
    
    ckpoint_path = args.model_path
    model, config = load_model(ckpoint_path)
    generate_target(model, config, args.n_examples, user_exs=args.user_exs)