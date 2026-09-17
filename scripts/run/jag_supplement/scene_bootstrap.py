#!/usr/bin/env python
"""Scene-level paired bootstrap for S1S2-Water Table 5: resample the 10 test SCENES with replacement
(instead of the 14,394 tiles), pool confusion counts, average the Water-IoU difference over the three
positionally paired seeds. Also reports the tile-level interval for reference."""
import glob, os, re, numpy as np, csv, json
ROOT=os.environ.get("PROJECT_ROOT", os.getcwd())
files=sorted(glob.glob(f"{ROOT}/data/S1S2-Water/test/images/*.tif"))
scene=np.array([re.sub(r"_tile.*","",os.path.basename(f)) for f in files]); scenes=sorted(set(scene)); sid=np.array([scenes.index(s) for s in scene])
def load(run):
    p=sorted(glob.glob(f"{ROOT}/logs/tensorboard/{run}/version_*/event_metrics/test_per_image_confusion.csv"))[-1]
    a=np.loadtxt(p,delimiter=",",skiprows=1,dtype=np.int64); assert a.shape[0]==len(files),(run,a.shape); return a[:,2:6]
def wiou(c): tp,fp,fn,tn=c.sum(0); return tp/max(tp+fp+fn,1)
seeds=["42","123","2026"]
MA={s:load(f"jag26_t9test_s1s2_gated_s{s}") for s in seeds}
base={"Input concatenation":"inputconcat","Feature addition":"dualadd","Feature concatenation":"dualconcat","Cross-attention":"stdcross","Gated fusion":"gated"}
rng=np.random.default_rng(0); B=5000; ns=len(scenes)
# per-scene aggregated confusions
def per_scene(c): return np.stack([c[sid==k].sum(0) for k in range(ns)])
out={}
print(f"test tiles={len(files)} scenes={ns} ->", {s:int((sid==k).sum()) for k,s in enumerate(scenes)})
for name,tag in base.items():
    Bs={s:load(f"jag26_s1s2test_{tag}_s{s}") for s in seeds}
    point=np.mean([wiou(MA[s])-wiou(Bs[s]) for s in seeds])*100
    MAsc={s:per_scene(MA[s]) for s in seeds}; Bsc={s:per_scene(Bs[s]) for s in seeds}
    d_scene=[]; d_tile=[]
    for _ in range(B):
        idx=rng.integers(0,ns,ns)
        d_scene.append(np.mean([wiou(MAsc[s][idx])-wiou(Bsc[s][idx]) for s in seeds])*100)
        tidx=rng.integers(0,len(files),len(files))
        d_tile.append(np.mean([wiou(MA[s][tidx])-wiou(Bs[s][tidx]) for s in seeds])*100)
    lo,hi=np.percentile(d_scene,[2.5,97.5]); tlo,thi=np.percentile(d_tile,[2.5,97.5])
    per=[ (wiou(MAsc["42"][k:k+1])-wiou(Bsc["42"][k:k+1]))*100 for k in range(ns)]
    out[name]=dict(point=point,scene_ci=[lo,hi],tile_ci=[tlo,thi],per_scene_seed42=per)
    print(f"{name:24s} delta={point:+.2f}  scene-CI=[{lo:+.2f},{hi:+.2f}]  tile-CI=[{tlo:+.2f},{thi:+.2f}]  per-scene(s42)={np.round(per,2).tolist()}")
json.dump(out,open(f"{ROOT}/paper/03_实验结果与计划/scene_bootstrap_s1s2.json","w"),indent=1)
