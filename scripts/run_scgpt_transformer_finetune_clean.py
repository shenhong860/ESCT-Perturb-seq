"""Full scGPT transformer fine-tuning on the clean Norman cat0 protocol.

All 12 transformer encoder layers are updated for each gene split.  The
pretrained gene-token embedding table is frozen to keep the large vocabulary
table fixed; the response head and transformer backbone are trained only on
training-split observed combinations.  Test means are used only after fitting.
"""

import copy
import json
import os
import sys

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr

from paths import dataset_paths, env_path, experiments_dir, scgpt_weights_dir

DATA = dataset_paths("norman")[0]
WEIGHTS = scgpt_weights_dir()
OUT = env_path("ESCT_SCGPT_FT_OUT",
               experiments_dir() / "scgpt_transformer_finetune_clean.json")
N_SPLITS = 12
TRAIN_FRAC = 0.8
DEV_FRAC = 0.1
SEQ_LEN = 256
EPOCHS = 10
BATCH_SIZE = 16
LR = 5e-5
WEIGHT_DECAY = 1e-4
TOP_K = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return str(o)


def sparse_mean(X, idx):
    return np.asarray(X[idx].mean(0)).ravel()


def split_terms(label):
    return sorted(set(x for x in str(label).split("+") if x and x != "control"))


def norman_labels(a):
    guides = a.obs["guide_identity"].astype(str).values

    def genes_from_guide(g):
        parts = g.replace("__", "_").split("_")
        return sorted({x for x in parts if not x.startswith("NegCtrl")})

    pert = np.array(["+".join(genes_from_guide(g)) if genes_from_guide(g)
                     else "control" for g in guides])
    return pert


def pcc(pred, true):
    de = np.argsort(-np.abs(true))[:TOP_K]
    if np.std(pred[de]) < 1e-12 or np.std(true[de]) < 1e-12:
        return 0.0
    return float(pearsonr(pred[de], true[de])[0])


def dir_acc(pred, true):
    de = np.argsort(-np.abs(true))[:TOP_K]
    return float(np.mean(np.sign(pred[de]) == np.sign(true[de])))


def load_model_and_vocab():
    from collections import OrderedDict
    from scgpt.model import TransformerModel
    try:
        from torchtext.vocab import Vocab
    except ImportError:
        from stubs.torchtext.vocab import Vocab

    with open(os.path.join(WEIGHTS, "vocab.json"), encoding="utf-8") as f:
        raw_vocab = json.load(f)
    vocab = Vocab(OrderedDict(raw_vocab))
    args = json.load(open(os.path.join(WEIGHTS, "args.json")))
    model = TransformerModel(
        ntoken=len(vocab), d_model=args["embsize"], nhead=args["nheads"],
        d_hid=args["d_hid"], nlayers=args["nlayers"],
        nlayers_cls=args.get("n_layers_cls", 3), n_cls=args.get("n_cls", 1),
        vocab=vocab, dropout=args.get("dropout", 0.2),
        pad_token=args.get("pad_token", "<pad>"),
        pad_value=args.get("pad_value", -2), do_mvc=False, do_dab=False,
        use_batch_labels=False, input_emb_style="continuous",
        cell_emb_style="cls", use_fast_transformer=False,
        pre_norm=args.get("pre_norm", False),
    )
    ckpt = torch.load(os.path.join(WEIGHTS, "best_model.pt"),
                      map_location="cpu", weights_only=False)
    state = model.state_dict()
    matched = {k: v for k, v in ckpt.items()
               if k in state and v.shape == state[k].shape}
    # The checkpoint stores FlashMHA's fused projection; remap it to PyTorch
    # MultiheadAttention when loading with use_fast_transformer=False.
    for key, value in ckpt.items():
        if ".self_attn.Wqkv." in key and key not in matched:
            target = key.replace(".self_attn.Wqkv.", ".self_attn.in_proj_")
            if target in state and value.shape == state[target].shape:
                matched[target] = value
    state.update(matched)
    model.load_state_dict(state)
    return model, vocab, matched, len(state)


class ResponseHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 768), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(768, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def run_one_split(seed, model0, X, pert, all_genes, combos, singles, pert_idx,
                  ctrl_mean, token_ids, values, pair_token_ids, output_dim):
    from scgpt.model import TransformerModel

    rng = np.random.RandomState(seed)
    perm = rng.permutation(all_genes).tolist()
    n_tr = int(len(all_genes) * TRAIN_FRAC)
    n_dev = int(len(all_genes) * DEV_FRAC)
    train_set = set(perm[:n_tr])
    dev_set = set(perm[n_tr:n_tr + n_dev])
    test_set = set(perm[n_tr + n_dev:])

    train_combos = [c for c in combos
                    if all(g in train_set for g in split_terms(c))
                    and all(g in pair_token_ids for g in split_terms(c))]
    dev_combos = [c for c in combos
                  if any(g in dev_set for g in split_terms(c))
                  and not any(g in test_set for g in split_terms(c))
                  and all(g in pair_token_ids for g in split_terms(c))]
    cat0 = [c for c in combos
            if any(g in test_set for g in split_terms(c))
            and all(g not in train_set for g in split_terms(c))
            and all(g in pair_token_ids for g in split_terms(c))]
    if not train_combos or not cat0:
        return {"seed": seed, "n_cat0": len(cat0), "skipped": True}

    model = TransformerModel(
        ntoken=model0.config_ntoken, d_model=model0.d_model,
        nhead=model0.transformer_encoder.layers[0].self_attn.num_heads,
        d_hid=model0.transformer_encoder.layers[0].linear1.out_features,
        nlayers=len(model0.transformer_encoder.layers),
        nlayers_cls=model0.nlayers_cls, n_cls=model0.n_cls,
        vocab=model0.vocab, dropout=model0.dropout.p,
        pad_token=model0.pad_token, pad_value=model0.pad_value,
        input_emb_style="continuous", cell_emb_style="cls",
        use_fast_transformer=False, pre_norm=model0.norm_scheme == "pre",
    ) if False else copy.deepcopy(model0)
    model.load_state_dict(model0.state_dict())
    model = model.to(DEVICE)

    head = ResponseHead(4 * model.d_model, output_dim).to(DEVICE)
    trainable = (list(model.transformer_encoder.parameters()) +
                 list(model.value_encoder.parameters()) +
                 list(head.parameters()))
    optimizer = torch.optim.AdamW(trainable, lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS)
    loss_fn = nn.MSELoss()

    ctrl_values = torch.as_tensor(values, dtype=torch.float32,
                                  device=DEVICE)[None].repeat(len(train_combos), 1)
    pair_tokens = torch.as_tensor(
        [[pair_token_ids[g] for g in split_terms(c)] for c in train_combos],
        dtype=torch.long, device=DEVICE)
    targets = torch.as_tensor(
        np.stack([sparse_mean(X, pert_idx[c]) - ctrl_mean
                  for c in train_combos]),
        dtype=torch.float32, device=DEVICE)
    pair_embedder = model.encoder

    model.train()
    head.train()
    n = len(train_combos)
    for epoch in range(EPOCHS):
        order = np.random.RandomState(seed * 100 + epoch).permutation(n)
        losses = []
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            batch_pair = pair_tokens[idx]
            tokens = torch.cat([
                torch.as_tensor(token_ids, dtype=torch.long,
                                device=DEVICE)[None].repeat(len(idx), 1),
                batch_pair], dim=1)
            values_in = torch.cat([
                ctrl_values[idx],
                torch.zeros(len(idx), 2, device=DEVICE)], dim=1)
            mask = torch.zeros(tokens.shape, dtype=torch.bool, device=DEVICE)
            hidden = model._encode(tokens, values_in, mask)
            pair_emb = pair_embedder(batch_pair).flatten(1)
            cell_part = model._get_cell_emb_from_layer(hidden)
            token_part = hidden.mean(dim=1)
            pooled = torch.cat((cell_part, token_part, pair_emb), dim=1)
            pred = head(pooled)
            loss = loss_fn(pred, targets[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"  seed {seed} epoch {epoch + 1}/{EPOCHS} "
                  f"loss={np.mean(losses):.4f}", flush=True)

    model.eval()
    head.eval()

    def evaluate(combos):
        rows = []
        with torch.no_grad():
            for c in combos:
                pair = torch.as_tensor([[pair_token_ids[g] for g in split_terms(c)]],
                                       dtype=torch.long, device=DEVICE)
                tokens = torch.cat([
                    torch.as_tensor(token_ids, dtype=torch.long,
                                    device=DEVICE)[None], pair], dim=1)
                values_in = torch.cat([
                    torch.as_tensor(values, dtype=torch.float32,
                                    device=DEVICE)[None],
                    torch.zeros(1, 2, device=DEVICE)], dim=1)
                mask = torch.zeros(tokens.shape, dtype=torch.bool,
                                   device=DEVICE)
                hidden = model._encode(tokens, values_in, mask)
                pair_emb = pair_embedder(pair).flatten(1)
                pooled = torch.cat([model._get_cell_emb_from_layer(hidden),
                                    hidden.mean(dim=1), pair_emb], dim=1)
                pred = head(pooled).cpu().numpy()[0]
                true = sparse_mean(X, pert_idx[c]) - ctrl_mean
                rows.append({"combo": c, "pcc": pcc(pred, true),
                             "dir": dir_acc(pred, true),
                             "l2": float(np.sqrt(np.sum((pred - true) ** 2)))})
        return rows

    train_rows = evaluate(train_combos)
    dev_rows = evaluate(dev_combos)
    rows = evaluate(cat0)
    return {"seed": seed, "n_train": len(train_combos),
            "n_dev": len(dev_combos), "n_cat0": len(cat0),
            "skipped": False,
            "train_pcc": float(np.mean([r["pcc"] for r in train_rows])),
            "train_dir": float(np.mean([r["dir"] for r in train_rows])),
            "train_l2": float(np.mean([r["l2"] for r in train_rows])),
            "dev_pcc": float(np.mean([r["pcc"] for r in dev_rows]))
                       if dev_rows else None,
            "dev_dir": float(np.mean([r["dir"] for r in dev_rows]))
                       if dev_rows else None,
            "dev_l2": float(np.mean([r["l2"] for r in dev_rows]))
                      if dev_rows else None,
            "pcc": float(np.mean([r["pcc"] for r in rows])),
            "dir": float(np.mean([r["dir"] for r in rows])),
            "l2": float(np.mean([r["l2"] for r in rows])), "rows": rows}


def main():
    torch.backends.cudnn.benchmark = True
    model0, vocab, matched, n_state = load_model_and_vocab()
    stoi = vocab.get_stoi()
    model0 = model0.to(DEVICE)
    print(f"scGPT loaded: matched={len(matched)}/{n_state}, device={DEVICE}",
          flush=True)

    a = ad.read_h5ad(DATA)
    X = a.X
    pert = norman_labels(a)
    labels = sorted(set(pert) - {"control"})
    singles = [x for x in labels if len(split_terms(x)) == 1]
    combos = [x for x in labels if len(split_terms(x)) == 2]
    all_genes = sorted({g for x in singles + combos for g in split_terms(x)
                        if g in stoi})
    in_vocab = set(all_genes)
    all_genes = [g for g in all_genes]
    pert_idx = {x: np.where(pert == x)[0] for x in labels}
    ctrl_idx = np.where(pert == "control")[0]
    ctrl_mean = sparse_mean(X, ctrl_idx)

    # Select informative control-expressed genes represented in scGPT vocab.
    candidate = np.array([i for i, g in enumerate(a.var_names) if g in stoi])
    ctrl_block = X[ctrl_idx]
    ctrl_first = np.asarray(ctrl_block.mean(0)).ravel()
    ctrl_second = np.asarray(ctrl_block.power(2).mean(0)).ravel()
    ctrl_var = np.maximum(ctrl_second - ctrl_first ** 2, 0)
    selected = candidate[np.argsort(-ctrl_var[candidate])[:SEQ_LEN]]
    selected = np.sort(selected)
    token_ids = [stoi[a.var_names[i]] for i in selected]
    # Last two sequence positions carry the pair-identity tokens.
    values = ctrl_mean[selected]
    pair_token_ids = {g: stoi[g] for g in all_genes}
    print(f"genes={len(all_genes)}, train_pairs={len(combos)}, "
          f"sequence={len(token_ids) + 2}", flush=True)

    # Attach immutable attributes used only to reconstruct clean copies.
    model0.config_ntoken = len(vocab)
    model0.nlayers_cls = 3
    model0.n_cls = 1
    model0.vocab = vocab
    model0.dropout = model0.transformer_encoder.layers[0].dropout
    model0.pad_token = "<pad>"
    model0.pad_value = -2

    n_splits = int(sys.argv[1]) if len(sys.argv) > 1 else N_SPLITS
    if len(sys.argv) > 2:
        global OUT
        OUT = sys.argv[2]
    results = []
    for seed in range(1, n_splits + 1):
        r = run_one_split(seed, model0, X, pert, all_genes, combos, singles,
                          pert_idx, ctrl_mean, token_ids, values,
                          pair_token_ids, X.shape[1])
        results.append(r)
        if not r["skipped"]:
            print(f"seed {seed}: n={r['n_cat0']} PCC={r['pcc']:.4f} "
                  f"dir={r['dir']:.4f} L2={r['l2']:.4f}", flush=True)

    valid = [r for r in results if not r["skipped"]]
    summary = {}
    for metric in ["pcc", "dir", "l2"]:
        vals = np.array([r[metric] for r in valid])
        summary[metric] = {"mean": float(vals.mean()),
                           "std": float(vals.std()), "n_splits": len(vals)}
    output = {"n": int(sum(r["n_cat0"] for r in valid)),
              "n_splits_requested": n_splits,
              "summary": summary,
              "matched_weights": len(matched), "total_state": n_state,
              "trainable": "all 12 transformer layers + value encoder + head",
              "per_seed": results, "protocol": "same clean all-gene splits"}
    with open(OUT, "w") as f:
        json.dump(output, f, indent=2, default=json_default)
    print(json.dumps(summary, indent=2))
    print("written:", OUT)


if __name__ == "__main__":
    main()
