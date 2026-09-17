"""Inference latency / throughput of the six S1S2-Water fusion strategies (Table 5),
measured in ONE session on ONE GPU so that rows are comparable.
fp32, eval mode, 256x256 tiles; batch 1 (latency, ms/tile) and batch 32 (throughput, tiles/s);
20 warm-up + 100 timed iterations, CUDA-synchronised. Submit with sbatch (GPU required)."""
import sys, os, time, statistics
sys.path.insert(0, os.environ.get("PROJECT_ROOT", os.getcwd())); os.chdir(os.environ.get("PROJECT_ROOT", os.getcwd()))
import importlib.util
spec = importlib.util.spec_from_file_location("pm", "scripts/run/jag_supplement/profile_model.py")
pm = importlib.util.module_from_spec(spec); spec.loader.exec_module(pm)
import hydra, torch
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

BASE = ("data.add_dem=false data.add_slope=false model.aux_loss_weight=0.0 model.alignment_enabled=false "
        "model.alignment_target_weight=0.0")
DUAL = " model.encoder.optical_channels=4 model.encoder.sar_channels=2"
CASES = {
    "Input concatenation": BASE + " +data.output_image_key=true model.encoder.optical_channels=6 model.encoder.sar_channels=0 model.encoder.sar_pretrained=false model.fusion.fusion_type=identity",
    "Feature addition": BASE + DUAL + " model.fusion.fusion_type=canonical_add",
    "Feature concatenation": BASE + DUAL + " model.fusion.fusion_type=concat",
    "Cross-attention (bidirectional)": BASE + DUAL + " model.fusion.fusion_type=canonical_cross",
    "Gated fusion": BASE + DUAL + " model.fusion.fusion_type=canonical_gated",
    "MA-XAttn": BASE + DUAL + " model.fusion.fusion_type=xattn +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_attention_type=ma +model.fusion.xattn_affinity_order=opt_sar +model.fusion.xattn_component_fusion=gated model.fusion.xattn_align_bias=false",
}
EXP = "resnet50_resnet50_s1s2water"
dev = torch.device("cuda")
torch.backends.cudnn.benchmark = True
gpu = torch.cuda.get_device_name(0)
out_path = "scripts/run/jag_supplement/profile_results/latency_s1s2.tsv"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
out = open(out_path, "w")
out.write(f"# GPU: {gpu}; torch {torch.__version__}; fp32; 256x256; warmup 20, timed 100\n")
out.write("method\tbatch1_ms_median\tbatch1_ms_iqr\tbatch32_tiles_per_s\n")

def timed(w, args, n_warm=20, n=100):
    with torch.no_grad():
        for _ in range(n_warm):
            w(*args)
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter(); w(*args); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    return ts

for name, ov in CASES.items():
    with initialize_config_dir(config_dir=os.path.join(os.environ.get("PROJECT_ROOT", os.getcwd()), "configs"), version_base="1.3"):
        cfg = compose(config_name="train.yaml", overrides=[f"experiment={EXP}", f"data={pm.infer_dataset_key(EXP)}", *pm.split_overrides(ov)], return_hydra_config=True)
    HydraConfig.instance().set_config(cfg)
    mc = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    for k in ("optical_pretrained", "sar_pretrained"):
        if k in mc.encoder:
            mc.encoder[k] = False
    m = hydra.utils.instantiate(mc, _recursive_=False).to(dev).eval()
    b1 = pm.make_batch(cfg, 1, 256, dev); keys = [k for k in b1 if k != "mask"]
    w = pm._ProfileWrapper(m, keys)
    ts1 = timed(w, [b1[k] for k in keys])
    med = statistics.median(ts1); q = statistics.quantiles(ts1, n=4); iqr = q[2] - q[0]
    b32 = pm.make_batch(cfg, 32, 256, dev)
    ts32 = timed(w, [b32[k] for k in keys], n_warm=10, n=50)
    thr = 32 / (statistics.median(ts32) / 1e3)
    line = f"{name}\t{med:.2f}\t{iqr:.2f}\t{thr:.1f}"
    print(line, flush=True); out.write(line + "\n"); out.flush()
    del m, w, b1, b32; torch.cuda.empty_cache()
print("GPU:", gpu)
