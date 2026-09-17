"""Gradient-level loss budget for MA-XAttn-R training (which loss term actually steers which module?).

Loss *magnitude* shares are misleading (focal ~1e-3, Dice ~3e-2, agreement ~5e-5), so for a given
checkpoint and a few training batches (with the training-time pixel dropout applied) we back-propagate
each weighted loss term separately and report, per parameter group (optical encoder / SAR encoder /
fusion / decoder / aux heads / alignment projections):
  * gradient L2 norm of each term,
  * its share of the total gradient norm (sum over terms),
  * pairwise cosine between term gradients on shared parameters (conflict < 0, synergy > 0).

sbatch (GPU). Usage:
  python analyze_grad_share.py --run <run_dir> --tag R_s42 [--ckpt last|best] [--epoch 40] [--n-batches 6]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.environ.get("PROJECT_ROOT", os.getcwd()))
os.chdir(os.environ.get("PROJECT_ROOT", os.getcwd()))

import hydra  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

OUT_DIR = "paper/03_实验结果与计划/grad_share"


def load(run_dir, which, device):
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    ckdir = os.path.join(run_dir, "checkpoints")
    ckpt = "last.ckpt" if which == "last" else sorted(f for f in os.listdir(ckdir) if f.startswith("water_iou_"))[-1]
    with initialize_config_dir(config_dir=os.path.join(os.environ.get("PROJECT_ROOT", os.getcwd()), "configs"), version_base="1.3"):
        base = compose(config_name="train.yaml", overrides=["experiment=resnet50_resnet50_s1s2water", "data=s1s2_water"], return_hydra_config=True)
    HydraConfig.instance().set_config(base)
    os.environ.setdefault("PROJECT_ROOT", os.environ.get("PROJECT_ROOT", os.getcwd()))
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.model.encoder.optical_pretrained = False
    cfg.model.encoder.sar_pretrained = False
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    sd = torch.load(os.path.join(ckdir, ckpt), map_location="cpu", weights_only=False)
    model.on_load_checkpoint(sd)
    model.load_state_dict(sd["state_dict"], strict=False)
    cfg.data.batch_size = 16
    cfg.data.num_workers = 4
    return model.to(device), cfg, ckpt, int(sd.get("epoch", 0))


def groups(model):
    g = {"enc_optical": [], "enc_sar": [], "fusion": [], "decoder": [], "aux_heads": [], "align_proj": [], "other": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("encoder.") and "optical" in n:
            g["enc_optical"].append(p)
        elif n.startswith("encoder.") and "sar" in n:
            g["enc_sar"].append(p)
        elif n.startswith("fusion."):
            g["fusion"].append(p)
        elif n.startswith("decoder.") or n.startswith("segmentation_head") or n.startswith("head"):
            g["decoder"].append(p)
        elif n.startswith("aux_head"):
            g["aux_heads"].append(p)
        elif "align" in n or "proj" in n:
            g["align_proj"].append(p)
        else:
            g["other"].append(p)
    return {k: v for k, v in g.items() if v}


def flat_grad(params):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in params])


def terms_for_batch(model, batch):
    """Return dict name -> weighted loss tensor, mirroring _shared_step(train)."""
    model._apply_pixel_optical_dropout(batch)
    out = model(batch)
    masks = batch["mask"]
    t = {}
    t["main_focal"] = model.loss_fn(model._resize_to_target(out["main_logits"], masks), masks)
    if model.aux_loss_weight > 0 and model.aux_loss_fn:
        fused_last = out["fused_features"][-1]
        losses = []
        for h in ("aux_head_1", "aux_head_2"):
            if hasattr(model, h):
                lg = torch.nn.functional.interpolate(getattr(model, h)(fused_last), size=masks.shape[1:], mode="bilinear", align_corners=False)
                losses.append(model.aux_loss_fn(lg, masks))
        t["aux_dice"] = model.aux_loss_weight * sum(losses) / len(losses)
    if getattr(model, "sar_only_head_weight", 0) > 0 and out.get("sar_only_logits") is not None:
        t["sar_only"] = model.sar_only_head_weight * model.loss_fn(model._resize_to_target(out["sar_only_logits"], masks), masks)
    if getattr(model, "optical_only_head_weight", 0) > 0 and out.get("optical_only_logits") is not None:
        t["optical_only"] = model.optical_only_head_weight * model.loss_fn(model._resize_to_target(out["optical_only_logits"], masks), masks)
    if model.alignment_enabled:
        al, _ = model._compute_alignment_loss(out["optical_features"], out["sar_features"], target=masks)
        t["agreement"] = model._get_alignment_schedule() * al
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--epoch", type=int, default=None, help="epoch used for dropout / lambda schedules (default: checkpoint epoch)")
    ap.add_argument("--n-batches", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, ckpt, ck_epoch = load(a.run, a.ckpt, dev)
    epoch = a.epoch if a.epoch is not None else ck_epoch
    # emulate trainer state used by the schedules
    model.trainer = None
    model.__dict__["_current_epoch_override"] = epoch
    type(model).current_epoch = property(lambda self: self.__dict__.get("_current_epoch_override", 0))
    model.train()
    for m in model.modules():  # keep BN statistics frozen; dropout heads still in train mode
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup("fit")
    loader = dm.train_dataloader()
    G = groups(model)
    print("param groups:", {k: sum(p.numel() for p in v) for k, v in G.items()}, flush=True)
    acc_norm, acc_loss, acc_cos, n = {}, {}, {}, 0
    for bi, batch in enumerate(loader):
        if bi >= a.n_batches:
            break
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        terms = terms_for_batch(model, batch)
        grads = {}
        for name, loss in terms.items():
            model.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            grads[name] = {g: flat_grad(ps).detach().clone() for g, ps in G.items()}
            acc_loss[name] = acc_loss.get(name, 0.0) + float(loss)
        for name in terms:
            for g in G:
                acc_norm.setdefault(name, {}).setdefault(g, 0.0)
                acc_norm[name][g] += float(grads[name][g].norm())
        names = list(terms)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                for g in ("enc_optical", "enc_sar", "fusion", "decoder"):
                    if g not in G:
                        continue
                    x, y = grads[names[i]][g], grads[names[j]][g]
                    c = float(torch.dot(x, y) / (x.norm() * y.norm() + 1e-12))
                    acc_cos.setdefault(f"{names[i]}~{names[j]}", {}).setdefault(g, 0.0)
                    acc_cos[f"{names[i]}~{names[j]}"][g] += c
        n += 1
        model.zero_grad(set_to_none=True)
    res = {"tag": a.tag, "checkpoint": ckpt, "schedule_epoch": epoch, "n_batches": n,
           "weighted_loss_mean": {k: v / n for k, v in acc_loss.items()},
           "grad_norm_mean": {k: {g: v / n for g, v in d.items()} for k, d in acc_norm.items()},
           "grad_cosine_mean": {k: {g: v / n for g, v in d.items()} for k, d in acc_cos.items()}}
    # shares per group
    share = {}
    for g in G:
        tot = sum(res["grad_norm_mean"][t][g] for t in res["grad_norm_mean"])
        share[g] = {t: (res["grad_norm_mean"][t][g] / tot if tot > 0 else 0.0) for t in res["grad_norm_mean"]}
    res["grad_share_per_group"] = share
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(os.path.join(OUT_DIR, f"grad_share_{a.tag}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
