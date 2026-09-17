"""Representation-level evidence for the agreement regulariser (Supplement SF4).

For each checkpoint, run N test tiles through the encoders + the fusion projections
(project_opt / project_sar) at the alignment levels [-1, -2], then compute:
  * same-location cosine between optical and SAR tokens (mean / median)
  * modality gap (Liang et al. 2022): ||mean(z_opt) - mean(z_sar)|| on L2-normalised tokens
  * cross-modal retrieval top-1 within image (does optical token i retrieve SAR token i?)
  * linear CKA between the two token sets
and a t-SNE of a token subsample coloured by modality and by class (water / background).

Submit with sbatch (GPU). Usage:
  python analyze_modality_gap.py --run <run_dir> --tag M0 --run <run_dir2> --tag MA-XAttn-R --n-tiles 64
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.environ.get("PROJECT_ROOT", os.getcwd()))
os.chdir(os.environ.get("PROJECT_ROOT", os.getcwd()))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402



def load_model(run_dir, device):
    # "run_dir@last" selects last.ckpt (full-schedule models); default = best-validation checkpoint
    use_last = run_dir.endswith("@last")
    run_dir = run_dir[:-5] if use_last else run_dir
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    if use_last:
        ckpt = "last.ckpt"
    else:
        ckpt = sorted([f for f in os.listdir(os.path.join(run_dir, "checkpoints")) if f.startswith("water_iou_")])[-1]
    ckpt_path = os.path.join(run_dir, "checkpoints", ckpt)
    with initialize_config_dir(config_dir=os.path.join(os.environ.get("PROJECT_ROOT", os.getcwd()), "configs"), version_base="1.3"):
        base = compose(config_name="train.yaml", overrides=["experiment=resnet50_resnet50_s1s2water", "data=s1s2_water"], return_hydra_config=True)
    HydraConfig.instance().set_config(base)
    os.environ.setdefault("PROJECT_ROOT", os.environ.get("PROJECT_ROOT", os.getcwd()))
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    model_cfg = cfg.model
    model_cfg.encoder.optical_pretrained = False
    model_cfg.encoder.sar_pretrained = False
    model = hydra.utils.instantiate(model_cfg, _recursive_=False)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    model.on_load_checkpoint({"state_dict": sd})
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[{run_dir}] loaded {ckpt}; missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    data_cfg = cfg.data
    data_cfg.batch_size = 16
    data_cfg.num_workers = 4
    return model.to(device).eval(), data_cfg, ckpt_path


def linear_cka(x, y):
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    hsic = (x.T @ y).norm() ** 2
    return float(hsic / ((x.T @ x).norm() * (y.T @ y).norm() + 1e-12))


@torch.no_grad()
def collect(model, loader, n_tiles, device, levels=(-1, -2), min_water_frac=0.05):
    tok_o, tok_s, lab, img_id = {l: [] for l in levels}, {l: [] for l in levels}, {l: [] for l in levels}, {l: [] for l in levels}
    seen = 0
    for batch in loader:
        # keep only tiles with a meaningful water fraction so both classes appear in the token sets
        m = batch["mask"]
        frac = (m == 1).flatten(1).float().mean(1)
        keep = (frac >= min_water_frac).nonzero().flatten()
        if keep.numel() == 0:
            continue
        batch = {k: (v[keep] if torch.is_tensor(v) and v.shape[0] == m.shape[0] else v) for k, v in batch.items()}
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        feats = model.encoder(batch)
        fo_all, fs_all = feats["optical"], feats["sar"]
        strat = model.fusion.strategy
        masks = batch["mask"]
        for l in levels:
            i = l % len(fo_all)
            fo, fs = fo_all[i], fs_all[i]
            if fo.shape[2:] != fs.shape[2:]:
                fs = F.interpolate(fs, size=fo.shape[2:], mode="bilinear", align_corners=False)
            zo, zs = strat.project_opt[i](fo), strat.project_sar[i](fs)
            b, c, h, w = zo.shape
            y = F.interpolate(masks.float().unsqueeze(1), size=(h, w), mode="nearest").squeeze(1).long()
            tok_o[l].append(zo.permute(0, 2, 3, 1).reshape(-1, c).cpu())
            tok_s[l].append(zs.permute(0, 2, 3, 1).reshape(-1, c).cpu())
            lab[l].append(y.reshape(-1).cpu())
            img_id[l].append(torch.arange(seen, seen + b).view(b, 1).expand(b, h * w).reshape(-1))
        seen += masks.shape[0]
        if seen >= n_tiles:
            break
    return {l: (torch.cat(tok_o[l]), torch.cat(tok_s[l]), torch.cat(lab[l]), torch.cat(img_id[l])) for l in levels}


def metrics(zo, zs, lab, img):
    zo_n, zs_n = F.normalize(zo, dim=1), F.normalize(zs, dim=1)
    cos = (zo_n * zs_n).sum(1)
    gap = float((zo_n.mean(0) - zs_n.mean(0)).norm())
    # within-image retrieval top-1
    hits, total = 0, 0
    for k in img.unique().tolist():
        idx = (img == k).nonzero().flatten()
        if idx.numel() < 2:
            continue
        sim = zo_n[idx] @ zs_n[idx].T
        hits += int((sim.argmax(1) == torch.arange(idx.numel())).sum())
        total += idx.numel()
    valid = lab >= 0
    return {
        "same_loc_cos_mean": float(cos[valid].mean()), "same_loc_cos_median": float(cos[valid].median()),
        "same_loc_cos_water": float(cos[valid & (lab == 1)].mean()) if (valid & (lab == 1)).any() else None,
        "same_loc_cos_bg": float(cos[valid & (lab == 0)].mean()),
        "modality_gap": gap, "retrieval_top1_within_image": hits / max(1, total),
        "linear_cka": linear_cka(zo_n[valid][:20000].double(), zs_n[valid][:20000].double()),
        "n_tokens": int(valid.sum()),
    }


def tsne_plot(results, level, out_png, tags=None, n_per=1500, seed=0, names=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    tags = tags or list(results)
    names = names or {}
    fig, axes = plt.subplots(1, len(tags), figsize=(4.2 * len(tags), 4.4), squeeze=False)
    rng = np.random.default_rng(seed)
    for ax, tag in zip(axes[0], tags):
        res = results[tag]
        zo, zs, lab, _ = res[level]
        valid = (lab >= 0).nonzero().flatten().numpy()
        sel = rng.choice(valid, size=min(n_per, valid.size), replace=False)
        z = torch.cat([F.normalize(zo[sel], dim=1), F.normalize(zs[sel], dim=1)]).numpy()
        emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=seed).fit_transform(z)
        n = sel.size
        l = lab[sel].numpy()
        for mod, sl, mk in (("optical", slice(0, n), "o"), ("SAR", slice(n, 2 * n), "^")):
            e = emb[sl]
            for cls, col in ((0, "tab:gray"), (1, "tab:blue")):
                m = l == cls
                ax.scatter(e[m, 0], e[m, 1], s=6, marker=mk, c=col, alpha=0.55, label=f"{mod}, {'water' if cls else 'background'}")
        mt = results[tag]['metrics'][level]
        ax.set_title(f"{names.get(tag, tag)}\nsame-location cos = {mt['same_loc_cos_mean']:.2f}, modality gap = {mt['modality_gap']:.2f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0][0].legend(fontsize=7, loc="lower left", frameon=False)
    for ax in axes[0]:
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_linewidth(0.5); ax.spines[side].set_color("0.6")
    fig.tight_layout(); fig.savefig(out_png, dpi=200); fig.savefig(os.path.splitext(out_png)[0] + ".pdf"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--tag", action="append", required=True)
    ap.add_argument("--n-tiles", type=int, default=64)
    ap.add_argument("--out", default="paper/03_实验结果与计划/modality_gap")
    ap.add_argument("--tsne-tags", default="", help="comma-separated subset of tags to draw")
    ap.add_argument("--tsne-names", default="", help="comma-separated display names aligned with --tsne-tags")
    args = ap.parse_args()
    OUT_DIR = args.out
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results, summary = {}, {}
    loader = None
    for run, tag in zip(args.run, args.tag):
        model, data_cfg, ckpt_path = load_model(run, device)
        if loader is None:
            dm = hydra.utils.instantiate(data_cfg)
            dm.setup("test")
            loader = dm.test_dataloader()
        res = collect(model, loader, args.n_tiles, device)
        m = {l: metrics(*res[l]) for l in res}
        results[tag] = dict(res); results[tag]["metrics"] = m
        summary[tag] = {"checkpoint": ckpt_path, "metrics": {str(k): v for k, v in m.items()}}
        print(tag, json.dumps(m, indent=1), flush=True)
    for level in (-1, -2):
        _tags = [t for t in args.tsne_tags.split(",") if t] or None
        _names = dict(zip(_tags or [], [n for n in args.tsne_names.split(",") if n])) if args.tsne_names else None
        tsne_plot(results, level, os.path.join(OUT_DIR, f"tsne_level{level}.png"), tags=_tags, names=_names)
    json.dump(summary, open(os.path.join(OUT_DIR, "modality_gap_summary.json"), "w"), indent=2, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, "modality_gap_summary.tsv"), "w") as f:
        f.write("tag\tlevel\tcos_mean\tcos_water\tcos_bg\tgap\tretrieval_top1\tcka\n")
        for tag, d in summary.items():
            for lv, m in d["metrics"].items():
                f.write(f"{tag}\t{lv}\t{m['same_loc_cos_mean']:.3f}\t{(m['same_loc_cos_water'] or float('nan')):.3f}\t{m['same_loc_cos_bg']:.3f}\t{m['modality_gap']:.3f}\t{m['retrieval_top1_within_image']:.4f}\t{m['linear_cka']:.3f}\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
