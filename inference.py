import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Transformer
from dataset import get_dataloaders, get_tokenizer
from typing import List


def load_model(model_path: str):
    ckpoint = torch.load(model_path)
    
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

def generate_target(model: Transformer, config, n_examples:int, user_exs:str=None) -> List[str]:
    en_tokenizer = get_tokenizer(lang="en", tokenizer_path=config['data']['src_tokenizer_path'])
    fr_tokenizer = get_tokenizer(lang="fr", tokenizer_path=config['data']['tgt_tokenizer_path'])
    sos_id = fr_tokenizer.token_to_id("[SOS]")
    eos_id = fr_tokenizer.token_to_id("[EOS]")
    tgt_sentence = []
    
    # _, val_dl, _, _ = get_dataloaders(
    #         config['model']['seq_len'],
    #         16,
    #         config['model']['vocab_size'],
    #         config['data']['src_tokenizer_path'], 
    #         config['data']['tgt_tokenizer_path'],
    #         config['data']['test_size']
    #     ) 
    # model.eval()
    
    # with torch.no_grad():
    #     for batch in val_dl:
    #         input_ids = batch["src_ids"]
    #         decoder_input = batch["tgt_ids"][:, :-1]
    #         logits = model(input_ids, decoder_input)
    #         out = F.softmax(logits, dim=-1)
    #         next_tokens = out.argmax(dim=-1)
    #         break
    
    # src_sentence = en_tokenizer.decode_batch(input_ids.tolist())
    # print("##################### ENGLISH SETNECE  #####################")
    # print(src_sentence[0])
    # gen_sentences = fr_tokenizer.decode_batch(next_tokens.tolist())
    # print("##################### TRANSLATION  #####################")
    # print( gen_sentences[0])
    
    # print("##################### ACTUAL  #####################")
    # tgt_sentence = fr_tokenizer.decode_batch(decoder_input.tolist())
    # print(tgt_sentence[0])
    
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
            tgt = item["tgt_ids"]
            tgt_sentence.append(tgt)
            break
    else:
        examples = [user_exs]
        examples = torch.tensor(examples).to(next(model.parameters()).device)
        
    model.eval()
    with torch.no_grad():
        gen_ids = model.greedy_decode(examples, sos_id, eos_id).detach().cpu().tolist()
    
    gen_sentences = fr_tokenizer.decode_batch(gen_ids)
    
    src_sentence = en_tokenizer.decode_batch(examples.tolist())
    print("##################### ENGLISH SETNECE  #####################")
    print(src_sentence[0])
    
    print("##################### TRANSLATION  #####################")
    print( gen_sentences[0])
    
    if tgt_sentence:
        print("Target sentces: ", tgt_sentence)

if __name__ == "__main__":
    ckpoint_path = "checkpoints/transformer_epoch_5.pt"
    model, config = load_model(ckpoint_path)
    generate_target(model, config, 1)