#!/usr/bin/env python3
"""Simple terminal interface for English -> Telugu translation (60M, Beam-4)."""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import sentencepiece as spm
import torch
import torch.nn as nn


VOCAB_SIZE = 32000
MAX_SEQ_LEN = 1024
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

DEFAULT_60M_CONFIG: Dict[str, float] = {
    "d_model": 640,
    "nhead": 8,
    "num_enc_layers": 6,
    "num_dec_layers": 6,
    "d_ff": 2560,
    "dropout": 0.12,
}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class EncoderStack(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ff: int, num_layers: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


class DecoderStack(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ff: int, num_layers: int, dropout: float):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )


class NMTTrans(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_enc_layers: int,
        num_dec_layers: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.scale = math.sqrt(d_model)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.position = PositionalEncoding(d_model, MAX_SEQ_LEN, dropout)
        self.encoder = EncoderStack(d_model, nhead, d_ff, num_enc_layers, dropout)
        self.decoder = DecoderStack(d_model, nhead, d_ff, num_dec_layers, dropout)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight


def load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model(checkpoint_path: Path, device: torch.device) -> NMTTrans:
    ckpt = load_checkpoint(checkpoint_path, device)
    config = ckpt.get("config", DEFAULT_60M_CONFIG) if isinstance(ckpt, dict) else DEFAULT_60M_CONFIG
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    model = NMTTrans(vocab_size=VOCAB_SIZE, **config).to(device)
    # When MAX_SEQ_LEN is increased, the checkpoint's saved positional buffer
    # (e.g., [1,128,d_model]) can mismatch current model shape (e.g., [1,1024,d_model]).
    # Remove it so the model keeps the freshly initialized longer buffer.
    if isinstance(state, dict) and "position.pe" in state:
        state = dict(state)
        state.pop("position.pe", None)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def beam_search_decode(
    model: NMTTrans,
    sp: spm.SentencePieceProcessor,
    src_text: str,
    device: torch.device,
    beam_size: int = 4,
    max_len: int = 128,
    length_penalty: float = 0.6,
) -> str:
    src_ids = [BOS_ID] + sp.encode(src_text, out_type=int) + [EOS_ID]
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_pad = src_tensor.eq(PAD_ID)

    src_emb = model.position(model.embedding(src_tensor) * model.scale)
    memory = model.encoder(src_emb, src_key_padding_mask=src_pad)

    beams: List[Tuple[List[int], float, bool]] = [([BOS_ID], 0.0, False)]

    for _ in range(max_len):
        candidates: List[Tuple[List[int], float, bool]] = []
        all_ended = True

        for token_ids, score, ended in beams:
            if ended:
                candidates.append((token_ids, score, True))
                continue

            all_ended = False
            tgt = torch.tensor([token_ids], dtype=torch.long, device=device)
            tgt_pad = tgt.eq(PAD_ID)
            # tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=device)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)
            tgt_emb = model.position(model.embedding(tgt) * model.scale)

            dec = model.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_pad,
                memory_key_padding_mask=src_pad,
            )
            logits = model.lm_head(dec)[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            topk_vals, topk_ids = torch.topk(log_probs, k=beam_size, dim=-1)

            for lp, idx in zip(topk_vals[0].tolist(), topk_ids[0].tolist()):
                new_tokens = token_ids + [int(idx)]
                new_score = score + float(lp)
                new_ended = int(idx) == EOS_ID
                candidates.append((new_tokens, new_score, new_ended))

        if all_ended:
            break

        def rank_key(item: Tuple[List[int], float, bool]) -> float:
            toks, sc, _ = item
            length = max(1, len(toks) - 1)
            return sc / (length ** length_penalty)

        candidates.sort(key=rank_key, reverse=True)
        beams = candidates[:beam_size]

    best = max(
        beams,
        key=lambda item: item[1] / ((max(1, len(item[0]) - 1)) ** length_penalty),
    )
    best_ids = best[0]

    if best_ids and best_ids[0] == BOS_ID:
        best_ids = best_ids[1:]
    if EOS_ID in best_ids:
        best_ids = best_ids[: best_ids.index(EOS_ID)]

    return sp.decode(best_ids)


def main():
    parser = argparse.ArgumentParser(description="English -> Telugu translator using 60M model + Beam-4.")
    SCRIPT_DIR = Path(__file__).parent
    parser.add_argument("--checkpoint", type=str, default=str(SCRIPT_DIR.parent / "final_60m_model.pt"))
    parser.add_argument("--sp-model", type=str, default=str(SCRIPT_DIR / "cache/nmt_unigram_v1_32000.model"))
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cuda" if args.device == "cuda" else "cpu")
    )

    checkpoint_path = Path(args.checkpoint)
    sp_model_path = Path(args.sp_model)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not sp_model_path.exists():
        raise FileNotFoundError(f"SentencePiece model not found: {sp_model_path}")

    print(f"Loading model from: {checkpoint_path}")
    print(f"Loading tokenizer from: {sp_model_path}")
    print(f"Device: {device} | Beam size: {args.beam_size}")

    sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
    model = build_model(checkpoint_path, device)

    print("\nType an English sentence and press Enter.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        src = input("English> ").strip()
        if src.lower() in {"exit", "quit"}:
            print("Exiting.")
            break
        if not src:
            continue

        telugu = beam_search_decode(
            model=model,
            sp=sp,
            src_text=src,
            device=device,
            beam_size=args.beam_size,
            max_len=args.max_len,
            length_penalty=args.length_penalty,
        )
        print(f"Telugu> {telugu}\n")
        
        # Save to a file so it can be viewed in the VS Code editor with proper font rendering
        with open("Submission/translations.txt", "a", encoding="utf-8") as f:
            f.write(f"English: {src}\nTelugu:  {telugu}\n{'-'*40}\n")


if __name__ == "__main__":
    main()
