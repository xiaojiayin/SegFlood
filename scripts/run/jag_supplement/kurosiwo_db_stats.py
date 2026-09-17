"""Recompute per-polarization dB mean/SD of Kuro Siwo GRD (VV, VH) per split, to trace the provenance of the
fixed constants in configs/data/kurosiwo.yaml (mean -11.8709/-19.0754, SD 6.8735/7.2449)."""
import sys, json, torch, hydra
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
sys.path.insert(0, os.environ.get("PROJECT_ROOT", os.getcwd()))
with initialize_config_dir(config_dir=os.path.join(os.environ.get("PROJECT_ROOT", os.getcwd()), "configs"), version_base="1.3"):
    cfg = compose(config_name="train.yaml", overrides=["experiment=resnet50_kurosiwo", "data=kurosiwo", "data.root=$PROJECT_ROOT/data/KuroSiwoGRD",
                  "data.data_mean=null", "data.data_std=null", "data.batch_size=64", "data.num_workers=16"])
import os; os.environ.setdefault("PROJECT_ROOT", os.environ.get("PROJECT_ROOT", os.getcwd()))
dm = hydra.utils.instantiate(cfg.data)
out = {}
for split, setup_stage, getter in (("train", "fit", "train_dataloader"), ("val", "fit", "val_dataloader"), ("test", "test", "test_dataloader")):
    dm.setup(setup_stage)
    dl = getattr(dm, getter)()
    n = torch.zeros(2, dtype=torch.float64); s = torch.zeros(2, dtype=torch.float64); ss = torch.zeros(2, dtype=torch.float64)
    n_valid = torch.zeros(2, dtype=torch.float64); s_valid = torch.zeros(2, dtype=torch.float64); ss_valid = torch.zeros(2, dtype=torch.float64)
    ntiles = 0
    for b in dl:
        x = b["image"][:, :2].double(); m = b["mask"]
        ntiles += x.shape[0]
        n += x[:, 0].numel(); s += x.sum(dim=(0, 2, 3)); ss += (x ** 2).sum(dim=(0, 2, 3))
        valid = (m >= 0).unsqueeze(1).expand_as(x)
        n_valid += valid[:, 0].sum(); s_valid += (x * valid).sum(dim=(0, 2, 3)); ss_valid += ((x ** 2) * valid).sum(dim=(0, 2, 3))
    mean = s / n; std = (ss / n - mean ** 2).sqrt()
    mean_v = s_valid / n_valid; std_v = (ss_valid / n_valid - mean_v ** 2).sqrt()
    out[split] = {"tiles": ntiles, "mean_all": mean.tolist(), "std_all": std.tolist(), "mean_valid": mean_v.tolist(), "std_valid": std_v.tolist(),
                  "sums": {"n": n.tolist(), "s": s.tolist(), "ss": ss.tolist()}}
    print(split, ntiles, "all:", [round(v, 4) for v in mean.tolist()], [round(v, 4) for v in std.tolist()],
          "valid:", [round(v, 4) for v in mean_v.tolist()], [round(v, 4) for v in std_v.tolist()], flush=True)
# pooled over all splits
N = sum(torch.tensor(o["sums"]["n"]) for o in out.values()); S = sum(torch.tensor(o["sums"]["s"]) for o in out.values()); SS = sum(torch.tensor(o["sums"]["ss"]) for o in out.values())
mean = S / N; std = (SS / N - mean ** 2).sqrt()
out["all_splits"] = {"mean_all": mean.tolist(), "std_all": std.tolist()}
print("all splits pooled:", [round(v, 4) for v in mean.tolist()], [round(v, 4) for v in std.tolist()])
print("config constants: mean [-11.8709, -19.0754] std [6.8735, 7.2449]")
json.dump(out, open(os.path.join(os.environ.get("PROJECT_ROOT", os.getcwd()), "paper/03_实验结果与计划/kurosiwo_db_stats_recomputed.json"), "w"), indent=1)
