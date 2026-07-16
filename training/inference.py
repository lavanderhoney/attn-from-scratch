import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model import Transformer
from training.dataset import get_dataloaders, get_tokenizer
from typing import List
from tokenizers import Tokenizer
import argparse
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except Exception:  # pragma: no cover
    hf_hub_download = None
    
def _resolve_checkpoint_path(
    checkpoint_path: str | None = None,
    src_tokenizer_path: str = "tokenizers/bpe_tokenizer_opus_en.json", # for now, keep it as tokenizer/bpe_tokenizer_opus_en.json
    tgt_tokenizer_path: str = "tokenizers/bpe_tokenizer_opus_fr.json", # for now, keep it as tokenizer/bpe_tokenizer_opus_fr.json
    hf_repo_id: str | None = None,
    hf_filename: str | None = None,
    hf_revision: str | None = None,
) -> str:
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        return str(checkpoint_path), str(src_tokenizer_path), str(tgt_tokenizer_path)

    if hf_repo_id:
        model_cache_path = hf_hub_download(repo_id=hf_repo_id, filename=hf_filename, revision=hf_revision)
        src_tokenizer_path = hf_hub_download(repo_id=hf_repo_id, filename=src_tokenizer_path, revision=hf_revision)
        tgt_tokenizer_path = hf_hub_download(repo_id=hf_repo_id, filename=tgt_tokenizer_path, revision=hf_revision)
        return model_cache_path, src_tokenizer_path, tgt_tokenizer_path

    if checkpoint_path is not None:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    raise ValueError("Provide either a local checkpoint_path or a Hugging Face repo reference")

def load_model_from_source(
    checkpoint_path: str | None = None,
    src_tokenizer_path: str  = "tokenizers/bpe_tokenizer_opus_en.json", # for now, keep it as tokenizer/bpe_tokenizer_opus_en.json
    tgt_tokenizer_path: str  = "tokenizers/bpe_tokenizer_opus_fr.json", # for now, keep it as tokenizer/bpe_tokenizer_opus_fr.json
    hf_repo_id: str | None = None,
    hf_filename: str | None = None,
    hf_revision: str | None = None,
):
    model_path, src_tokenizer_path, tgt_tokenizer_path = _resolve_checkpoint_path(
        checkpoint_path=checkpoint_path,
        src_tokenizer_path=src_tokenizer_path,
        tgt_tokenizer_path=tgt_tokenizer_path,
        hf_repo_id=hf_repo_id,
        hf_filename=hf_filename,
        hf_revision=hf_revision,
    )
    model, config = load_model(model_path, src_tokenizer_path, tgt_tokenizer_path)
    return (model, config, src_tokenizer_path, tgt_tokenizer_path)

def load_model(model_path: str, 
               src_tokenizer_path: str = "tokenizers/bpe_tokenizer_opus_en.json",
               tgt_tokenizer_path: str = "tokenizers/bpe_tokenizer_opus_fr.json"):
    ckpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    print(f"tokenizer path: {src_tokenizer_path}, {tgt_tokenizer_path}")
    
    config = ckpoint['config']
    SEQ_LEN = config['model']['seq_len']
    D_MODEL = config['model']['d_model']
    D_FF = config['model']['d_ff']
    NUM_HEADS = config['model']['num_heads']
    VOCAB_SIZE = config['model']['vocab_size']
    NUM_LAYERS = config['model']['num_layers']
    en_tokenizer = get_tokenizer(lang="en", tokenizer_path=src_tokenizer_path)
    fr_tokenizer = get_tokenizer(lang="fr", tokenizer_path=tgt_tokenizer_path)

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

def generate_target(
    model: Transformer,
    config,
    n_examples: int,
    src_tokenizer_path: str = "tokenizers/bpe_tokenizer_opus_en.json",
    tgt_tokenizer_path: str = "tokenizers/bpe_tokenizer_opus_fr.json",
    user_exs: str = None,
    decoding_method: str = "greedy",
    beam_width: int = 3,
) -> List[str]:
    
    print(f"Generating target sentences with decoding method: {decoding_method}, beam_width: {beam_width}, user_exs: {user_exs}")
    print(f"src_tokenizer_path: {src_tokenizer_path}, tgt_tokenizer_path: {tgt_tokenizer_path}")
    
    en_tokenizer = get_tokenizer(lang="en", tokenizer_path=src_tokenizer_path)
    fr_tokenizer = get_tokenizer(lang="fr", tokenizer_path=tgt_tokenizer_path)
    sos_id = fr_tokenizer.token_to_id("[SOS]")
    eos_id = fr_tokenizer.token_to_id("[EOS]")
    tgt_sentence = []
    
    if user_exs is None:
        # sample n_examples sentences from the dataset and return those
        _, val_dl, _, _ = get_dataloaders(
            config['model']['seq_len'],
            n_examples,
            config['model']['vocab_size'],
            src_tokenizer_path, 
            tgt_tokenizer_path,
            config['data']['test_size']
        ) 
        for item in val_dl:
            examples = item["src_ids"]
            tgt_sentence = item["tgt_ids"].detach().cpu().tolist()
            break
    else:
        examples = tokenize_user_input(user_exs, en_tokenizer, config['model']['seq_len'])

    if decoding_method not in {"greedy", "beam"}:
        raise ValueError(f"Unsupported decoding method: {decoding_method}")

    decode_examples = examples if decoding_method == "greedy" else examples[:1] # beam-search supports single sentence only, not batch

    model.eval()
    with torch.no_grad():
        if decoding_method == "greedy":
            gen_ids = model.greedy_decode(decode_examples, sos_id, eos_id).detach().cpu().tolist()
        else:
            gen_ids = model.beam_search_decode(decode_examples, sos_id, eos_id, beam_width).detach().cpu().tolist()

    gen_sentences = fr_tokenizer.decode_batch(gen_ids)

    src_sentence = en_tokenizer.decode_batch(examples.tolist())
    print(f"\n##################### ENGLISH SETNECE  #####################")
    print(src_sentence[0])
    
    print(f"\n##################### {decoding_method.upper()} DECODING  #####################")
    print(gen_sentences[0])

    if tgt_sentence:
        print(f"\n##################### TARGET SENTENCE  #####################")
        tgt_sentence = fr_tokenizer.decode_batch(tgt_sentence)
        print(tgt_sentence[0])

    return gen_sentences

if __name__ == "__main__":
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, help="Path to the model checkpoint")
    # parser.add_argument("--n-examples", type=int, default=1, help="Number of examples to generate")
    # parser.add_argument("--user-exs", type=str, default=None, help="User provided examples to translate")
    # args = parser.parse_args()
    
    # ckpoint_path = args.model_path
    ckpoint_path = "/teamspace/studios/this_studio/attn-from-scratch/checkpoints/transformer_noam_v2_epoch_30.pt"
    model, config = load_model_from_source(checkpoint_path=ckpoint_path)
    generate_target(model, config, 1, user_exs=None)