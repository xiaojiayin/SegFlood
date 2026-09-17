import sys, os
sys.path.insert(0, os.environ.get("PROJECT_ROOT", os.getcwd())); os.chdir(os.environ.get("PROJECT_ROOT", os.getcwd()))
import importlib.util, torch
spec = importlib.util.spec_from_file_location("pm", "scripts/run/jag_supplement/profile_model.py"); pm = importlib.util.module_from_spec(spec); spec.loader.exec_module(pm)
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torch.utils.flop_counter import FlopCounterMode
C = "data.add_dem=false data.add_slope=false model.encoder.optical_channels=4 model.encoder.sar_channels=2 model.fusion.fusion_type=concat model.aux_loss_weight=0.0 model.alignment_enabled=false model.alignment_target_weight=0.0"
CASES = {"concat_mobilenet": ("efficientnetb4_mobilenetv3_s1s2water", C + " model.encoder.optical_model_name=mobilenetv3_large_100 model.encoder.sar_model_name=mobilenetv3_large_100"),
         "concat_efficientnet": ("efficientnetb4_mobilenetv3_s1s2water", C + " model.encoder.optical_model_name=efficientnet_b4 model.encoder.sar_model_name=efficientnet_b4"),
         "concat_sam2": ("sam2_sam2_s1s2water", C), "concat_dinov3": ("dinov3_dinov3_s1s2water", C)}
out = open("scripts/run/jag_supplement/profile_results/aten_flops.tsv", "a")
for name, (exp, ov) in CASES.items():
    with initialize_config_dir(config_dir=os.path.join(os.environ.get("PROJECT_ROOT", os.getcwd()), "configs"), version_base="1.3"):
        cfg = compose(config_name="train.yaml", overrides=[f"experiment={exp}", f"data={pm.infer_dataset_key(exp)}", *pm.split_overrides(ov)], return_hydra_config=True)
    HydraConfig.instance().set_config(cfg)
    mc = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    for k in ("optical_pretrained", "sar_pretrained"):
        if k in mc.encoder: mc.encoder[k] = False
    m = hydra.utils.instantiate(mc, _recursive_=False).eval()
    params = sum(p.numel() for p in m.parameters()) / 1e6
    b = pm.make_batch(cfg, 1, 256, torch.device("cpu")); keys = [k for k in b if k != "mask"]; w = pm._ProfileWrapper(m, keys)
    with torch.no_grad():
        fc = FlopCounterMode(display=False)
        with fc: w(*[b[k] for k in keys])
    line = f"{name}\t{params:.2f}\t{fc.get_total_flops()/1e9:.2f}"; print(line, flush=True); out.write(line + "\n"); out.flush()
