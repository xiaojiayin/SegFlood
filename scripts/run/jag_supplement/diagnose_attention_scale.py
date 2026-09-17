"""Diagnose the magnitude of MA-XAttn attention logits and the sharpness of the resulting
attention maps (CPU, a few test tiles). Answers "should QK be scaled by 1/sqrt(d)?" with data
instead of a single test number.

For every SpatialAttBlock / ChannelAttBlock in the fusion strategy, capture the block inputs with
a forward pre-hook, recompute logits exactly as in blocks.py, and report per level:
  logit std, |logit| p99, normalised entropy of att_opt / att_sar / att_cross
  (1 = uniform, 0 = one-hot), effective number of attended tokens exp(H), max weight.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.environ.get("PROJECT_ROOT", os.getcwd()))
os.chdir(os.environ.get("PROJECT_ROOT", os.getcwd()))

import torch  # noqa: E402

from scripts.run.jag_supplement.analyze_modality_gap import load_model  # noqa: E402
from src.models.fusion.blocks import ChannelAttBlock, SpatialAttBlock  # noqa: E402


def entropy_stats(att):
    # att: [B, N, N] rows sum to 1
    n = att.shape[-1]
    h = -(att.clamp_min(1e-12) * att.clamp_min(1e-12).log()).sum(-1)  # [B,N]
    am = att.argmax(-1)
    diag = (am == torch.arange(n, device=att.device)).float().mean()
    # mean |Δposition| of the routed token (spatial only meaningful); in tokens
    n_uniq = sum(am[b].unique().numel() for b in range(am.shape[0])) / am.shape[0]
    top_col = float(sum(torch.bincount(am[b], minlength=n).max().item() / n for b in range(am.shape[0])) / am.shape[0])
    return {"H_norm": float((h / math.log(n)).mean()), "eff_tokens": float(h.exp().mean()),
            "max_w": float(att.max(-1).values.mean()), "N": n, "diag_frac": float(diag),
            "n_uniq_cols": n_uniq, "top_col_share": top_col,
            "diag_w": float(att.diagonal(dim1=-2, dim2=-1).mean())}


def analyse_block(blk, x_opt, x_sar):
    n, _, h, w = x_opt.shape
    if isinstance(blk, SpatialAttBlock):
        q_opt = blk.q_opt(x_opt).reshape(n, -1, h * w).permute(0, 2, 1)
        k_opt = blk.k_opt(x_opt).reshape(n, -1, h * w)
        q_sar = blk.q_sar(x_sar).reshape(n, -1, h * w).permute(0, 2, 1)
        k_sar = blk.k_sar(x_sar).reshape(n, -1, h * w)
        d = blk.att_channels
    else:
        q_opt = blk.q_opt(x_opt).reshape(n, -1, h * w)
        k_opt = blk.k_opt(x_opt).reshape(n, -1, h * w).permute(0, 2, 1)
        q_sar = blk.q_sar(x_sar).reshape(n, -1, h * w)
        k_sar = blk.k_sar(x_sar).reshape(n, -1, h * w).permute(0, 2, 1)
        d = h * w
    lo, ls = torch.bmm(q_opt, k_opt), torch.bmm(q_sar, k_sar)  # raw (un-normalised) logits for reference
    sc = 1.0 / math.sqrt(d) if blk.qk_scale else 1.0
    if getattr(blk, "attention_norm", "none") == "cosine":
        if isinstance(blk, SpatialAttBlock):
            q_opt, q_sar = torch.nn.functional.normalize(q_opt, dim=-1), torch.nn.functional.normalize(q_sar, dim=-1)
            k_opt, k_sar = torch.nn.functional.normalize(k_opt, dim=1), torch.nn.functional.normalize(k_sar, dim=1)
        else:
            q_opt, q_sar = torch.nn.functional.normalize(q_opt, dim=-1), torch.nn.functional.normalize(q_sar, dim=-1)
            k_opt, k_sar = torch.nn.functional.normalize(k_opt, dim=1), torch.nn.functional.normalize(k_sar, dim=1)
        sc = float(blk.logit_scale.clamp(max=math.log(100.0)).exp())
        lo, ls = torch.bmm(q_opt, k_opt), torch.bmm(q_sar, k_sar)
    ao, as_ = torch.softmax(lo * sc, -1), torch.softmax(ls * sc, -1)
    ac = torch.bmm(ao, as_) if blk.affinity_order == "opt_sar" else torch.bmm(as_, ao)
    return {
        "type": type(blk).__name__, "d": d, "qk_scale": (f"cos×{sc:.1f}" if getattr(blk, "attention_norm", "none") == "cosine" else blk.qk_scale),
        "logit_std_raw": float(torch.stack([lo.std(), ls.std()]).mean()),
        "logit_p99_raw": float(torch.quantile(torch.cat([lo.flatten(), ls.flatten()]).abs()[:200000], 0.99)),
        "logit_std_used": float(torch.stack([lo.std(), ls.std()]).mean() * sc),
        "att_opt": entropy_stats(ao), "att_sar": entropy_stats(as_), "att_cross": entropy_stats(ac),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--tag", action="append", required=True)
    ap.add_argument("--n-tiles", type=int, default=16)
    args = ap.parse_args()
    device = torch.device("cpu")
    torch.set_num_threads(8)
    loader = None
    for run, tag in zip(args.run, args.tag):
        model, data_cfg, _ = load_model(run, device)
        if loader is None:
            import hydra
            data_cfg.batch_size = 8
            data_cfg.num_workers = 0
            dm = hydra.utils.instantiate(data_cfg)
            dm.setup("test")
            loader = dm.test_dataloader()
            batches = []
            for b in loader:
                if (b["mask"] == 1).flatten(1).float().mean(1).max() >= 0.05:
                    batches.append(b)
                if sum(x["mask"].shape[0] for x in batches) >= args.n_tiles:
                    break
        captured = []

        def pre_hook(mod, inp):
            captured.append((mod, inp[0].detach(), inp[1].detach()))

        hooks = [m.register_forward_pre_hook(pre_hook) for m in model.fusion.modules()
                 if isinstance(m, (SpatialAttBlock, ChannelAttBlock))]
        with torch.no_grad():
            for b in batches:
                model(b)
        for hk in hooks:
            hk.remove()
        # aggregate per module
        agg = {}
        for mod, xo, xs in captured:
            r = analyse_block(mod, xo, xs)
            key = (id(mod), r["type"], tuple(xo.shape[1:]))
            agg.setdefault(key, []).append(r)
        print(f"\n=== {tag} ===")
        print(f"{'block':<16}{'C,h,w':<14}{'d':>6}{'scale':>7}{'std_raw':>9}{'p99_raw':>9}{'std_used':>9} | "
              f"{'Hn_opt':>7}{'eff_opt':>8}{'max_opt':>8}{'diag_opt':>9} | {'Hn_cross':>9}{'eff_x':>7}{'max_x':>7}{'diag_x':>8}{'uniq_x':>7}{'topcol_x':>9}")
        for (i, t, shp), rs in agg.items():
            m = lambda f: sum(f(r) for r in rs) / len(rs)  # noqa: E731
            print(f"{t:<16}{str(shp):<14}{rs[0]['d']:>6}{str(rs[0]['qk_scale']):>9}{m(lambda r: r['logit_std_raw']):>9.2f}"
                  f"{m(lambda r: r['logit_p99_raw']):>9.2f}{m(lambda r: r['logit_std_used']):>9.2f} | "
                  f"{m(lambda r: r['att_opt']['H_norm']):>7.3f}{m(lambda r: r['att_opt']['eff_tokens']):>8.1f}{m(lambda r: r['att_opt']['max_w']):>8.3f}{m(lambda r: r['att_opt']['diag_frac']):>9.3f} | "
                  f"{m(lambda r: r['att_cross']['H_norm']):>9.3f}{m(lambda r: r['att_cross']['eff_tokens']):>7.1f}{m(lambda r: r['att_cross']['max_w']):>7.3f}{m(lambda r: r['att_cross']['diag_frac']):>8.3f}{m(lambda r: r['att_cross']['n_uniq_cols']):>7.1f}{m(lambda r: r['att_cross']['top_col_share']):>9.3f}"
                  f"   N={rs[0]['att_opt']['N']}")


if __name__ == "__main__":
    main()
