#!/usr/bin/env python
"""Prepare tiled GeoTIFFs for S1S2-Water.

Key features:
- Split sources:
  - Read official catalog.json (STAC Items contain `properties.split`)
  - Or provide a custom split JSON (train/val/test -> [id])
- Sensor selection: --sensor s1 | s2 | dual
- Optional extra channels: --include-slope / --include-elevation
- Nodata filtering: --exclude-nodata (uses *_valid.tif), configurable via --valid-threshold
- Resampling: use a reference grid (S2 if available else S1) and reproject other rasters
- Scaling:
  - S1: Int16 dB*100 -> /100 to dB
  - S2: UInt16 TOA*10000 -> /10000 to [0,1]
- Writes a metadata JSON describing channel order and scaling.

Note: this script does not perform normalization; do it in training using computed stats.
"""
from __future__ import annotations
import argparse, os, json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window
from affine import Affine
import csv, json as jsonlib, time
import warnings
from typing import Dict, List, Optional


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--root', required=True, help='Dataset root (contains 1/, 5/, catalog.json)')
    p.add_argument('--out-root', required=True, help='Output root directory')
    group_split = p.add_mutually_exclusive_group(required=True)
    group_split.add_argument('--catalog', help='Official catalog.json path (STAC)')
    group_split.add_argument('--split-json', help='Custom split JSON (train/val/test)')
    p.add_argument('--sensor', choices=['s1','s2','dual'], default='dual', help='Process single or dual sensors')
    p.add_argument('--include-slope', action='store_true', help='Append slope channel (SLOPE)')
    p.add_argument('--include-elevation', action='store_true', help='Append elevation channel (optional)')
    p.add_argument('--exclude-nodata', action='store_true', help='Skip tiles with invalid pixels (use valid mask)')
    p.add_argument('--valid-threshold', type=float, default=1.0, help='Valid pixel ratio threshold (0-1)')
    p.add_argument('--tile-size', type=int, default=256, help='Tile size (square)')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing tiles')
    p.add_argument('--write-valid-mask', action='store_true', help='Write union valid mask under valid/')
    p.add_argument('--metadata-name', default='s1s2_water_metadata.json', help='Metadata JSON filename')
    return p.parse_args()


def load_sample(root: Path, sid: int, need_s1: bool, need_s2: bool, need_elev: bool, need_slope: bool) -> Dict[str, Path]:
    d = root / str(sid)
    if not d.exists():
        raise FileNotFoundError(f'Sample directory missing: {d}')
    def one(pattern: str) -> Path:
        p = next(d.glob(pattern), None)
        if not p:
            raise FileNotFoundError(f'Missing {pattern} in {d}')
        return p
    out = {}
    if need_s2:
        out['s2_img'] = one(f'sentinel12_s2_{sid}_img.tif')
        out['s2_msk'] = one(f'sentinel12_s2_{sid}_msk.tif')
        out['s2_valid'] = one(f'sentinel12_s2_{sid}_valid.tif')
    if need_s1:
        out['s1_img'] = one(f'sentinel12_s1_{sid}_img.tif')
        out['s1_msk'] = one(f'sentinel12_s1_{sid}_msk.tif')
        out['s1_valid'] = one(f'sentinel12_s1_{sid}_valid.tif')
    if need_elev:
        out['elev'] = one(f'sentinel12_copdem30_{sid}_elevation.tif')
    if need_slope:
        out['slope'] = one(f'sentinel12_copdem30_{sid}_slope.tif')
    return out


def center_crop_dims(h: int, w: int, tile: int):
    new_h = (h // tile) * tile
    new_w = (w // tile) * tile
    off_h = (h - new_h)//2
    off_w = (w - new_w)//2
    return off_h, off_w, new_h, new_w


def read_raster(path: Path):
    ds = rasterio.open(path)
    arr = ds.read()  # [bands, H, W]
    return ds, arr


def resample_to_match(ds_src, ref_ds, resampling: Resampling = Resampling.bilinear) -> np.ndarray:
    """Resample ds_src to match ref_ds grid exactly (alignment-safe)."""
    if ds_src.crs is None or ref_ds.crs is None:
        raise ValueError("Source or reference dataset CRS is None; cannot reproject. Check input files.")
    bands = ds_src.count
    dst = np.zeros((bands, ref_ds.height, ref_ds.width), dtype=np.float32)
    # Read source once (original resolution)
    src_data = ds_src.read(out_dtype='float32')  # shape [bands, H, W]
    for b in range(bands):
        reproject(
            source=src_data[b],
            destination=dst[b],
            src_transform=ds_src.transform,
            src_crs=ds_src.crs,
            dst_transform=ref_ds.transform,
            dst_crs=ref_ds.crs,
            resampling=resampling,
            num_threads=2,
            src_nodata=getattr(ds_src, 'nodata', None),
            dst_nodata=getattr(ds_src, 'nodata', None),
        )
    return dst

def resample_array_to(ref_ds, arr: np.ndarray, src_transform, src_crs, resampling: Resampling, src_crs_obj) -> np.ndarray:
    """Generic array (single-band) resample helper given target ref_ds grid."""
    dst = np.zeros((ref_ds.height, ref_ds.width), dtype=arr.dtype)
    reproject(
        source=arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs_obj,
        dst_transform=ref_ds.transform,
        dst_crs=ref_ds.crs,
        resampling=resampling,
        num_threads=2,
        src_nodata=None,
        dst_nodata=None,
    )
    return dst


def write_geotiff(path: Path, data: np.ndarray, ref_profile: dict, transform):
    path.parent.mkdir(parents=True, exist_ok=True)
    count, h, w = data.shape
    profile = ref_profile.copy()
    predictor = 2 if np.issubdtype(data.dtype, np.floating) else 1
    profile.update({
        'height': h,
        'width': w,
        'count': count,
        'dtype': str(data.dtype),
        'transform': transform,
        'compress': 'deflate',
        'predictor': predictor,
        'tiled': True,
        'blockxsize': min(256, w),
        'blockysize': min(256, h),
        'interleave': 'band'
    })
    # Remove colormap if present in source (not applicable)
    profile.pop('colormap', None)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data)


def process_sample(root: Path,
                   out_root: Path,
                   sid: int,
                   split: str,
                   tile: int,
                   overwrite: bool,
                   index_writer,
                   band_order: List[str],
                   sensor: str,
                   include_elev: bool,
                   include_slope: bool,
                   exclude_nodata: bool,
                   valid_threshold: float,
                   write_valid_mask: bool):
    need_s1 = sensor in ('s1','dual')
    need_s2 = sensor in ('s2','dual')
    files = load_sample(root, sid, need_s1, need_s2, include_elev, include_slope)

    # Choose reference grid (prefer S2).
    ref_key = 's2_img' if 's2_img' in files else 's1_img'
    with rasterio.open(files[ref_key]) as ds_ref:
        # Open other rasters
        ctx = {}
        for k, v in files.items():
            if k == ref_key:
                ctx[k] = ds_ref
            else:
                ctx[k] = rasterio.open(v)
        ds_s2 = ctx.get('s2_img')
        ds_s1 = ctx.get('s1_img')
        ds_s2_m = ctx.get('s2_msk')
        ds_s1_m = ctx.get('s1_msk')
        ds_s2_v = ctx.get('s2_valid')
        ds_s1_v = ctx.get('s1_valid')
        ds_elev = ctx.get('elev')
        ds_slope = ctx.get('slope')

        # Read & scale
        if ds_s2 is not None:
            s2 = ds_s2.read(out_dtype='float32') / 10000.0
            mask_s2 = ds_s2_m.read(1).astype(np.uint8) if ds_s2_m is not None else None
            valid_s2 = ds_s2_v.read(1).astype(np.uint8) if ds_s2_v is not None else None
        else:
            s2 = None
            mask_s2 = None
            valid_s2 = None
        if ds_s1 is not None:
            s1_lin = ds_s1.read(out_dtype='float32') / 100.0  # -> dB
            mask_s1 = ds_s1_m.read(1).astype(np.uint8) if ds_s1_m is not None else None
            valid_s1 = ds_s1_v.read(1).astype(np.uint8) if ds_s1_v is not None else None
        else:
            s1_lin = None
            mask_s1 = None
            valid_s1 = None

        # Resample if dual and sizes differ
        if s1_lin is not None and s2 is not None and ds_s1 is not None and ds_s2 is not None:
            if (ds_s1.height != ds_s2.height) or (ds_s1.width != ds_s2.width):
                warnings.warn(f"[WARN] S1 shape {ds_s1.height}x{ds_s1.width} != S2 {ds_s2.height}x{ds_s2.width}, resampling S1 & its mask/valid")
                rs_s1 = np.zeros((ds_s1.count, ds_s2.height, ds_s2.width), dtype=np.float32)
                for b in range(ds_s1.count):
                    rs_s1[b] = resample_array_to(ds_s2, s1_lin[b], ds_s1.transform, ds_s1.crs, Resampling.bilinear, ds_s1.crs)
                s1_lin = rs_s1
                if mask_s1 is not None and ds_s1_m is not None:
                    mask_s1 = resample_array_to(ds_s2, mask_s1, ds_s1_m.transform, ds_s1_m.crs, Resampling.nearest, ds_s1_m.crs).astype(np.uint8)
                if valid_s1 is not None and ds_s1_v is not None:
                    valid_s1 = resample_array_to(ds_s2, valid_s1, ds_s1_v.transform, ds_s1_v.crs, Resampling.nearest, ds_s1_v.crs).astype(np.uint8)

        ref_ds = ds_s2 if ds_s2 is not None else ds_s1

        elev_res = None
        if include_elev and ds_elev is not None and ref_ds is not None:
            elev_res = resample_to_match(ds_elev, ref_ds, Resampling.bilinear)
        slope_res = None
        if include_slope and ds_slope is not None and ref_ds is not None:
            slope_res = resample_to_match(ds_slope, ref_ds, Resampling.bilinear)

        if ref_ds is None:
            warnings.warn(f"[WARN] sample {sid} missing reference dataset, skip")
            return

        height, width = ref_ds.height, ref_ds.width
        off_h, off_w, new_h, new_w = center_crop_dims(height, width, tile)

        def crop_any(a):
            if a is None:
                return None
            if a.ndim == 2:
                return a[off_h:off_h+new_h, off_w:off_w+new_w]
            return a[:, off_h:off_h+new_h, off_w:off_w+new_w]

        s2_c = crop_any(s2)
        s1_c = crop_any(s1_lin)
        elev_c = crop_any(elev_res)
        slope_c = crop_any(slope_res)
        m_s1_c = crop_any(mask_s1)
        m_s2_c = crop_any(mask_s2)
        v_s1_c = crop_any(valid_s1)
        v_s2_c = crop_any(valid_s2)

        # Assemble image channels
        stacks = []
        if s2_c is not None:
            stacks.append(s2_c)
        if s1_c is not None:
            stacks.append(s1_c)
        if elev_c is not None:
            stacks.append(elev_c)
        if slope_c is not None:
            stacks.append(slope_c)
        image_stack = np.concatenate(stacks, axis=0)

        # Assemble masks
        mask_list = []
        if m_s1_c is not None:
            mask_list.append(m_s1_c)
        if m_s2_c is not None:
            mask_list.append(m_s2_c)
        masks_stack = np.stack(mask_list, axis=0)

        # Union valid mask
        union_valid = None
        if exclude_nodata or write_valid_mask:
            val_list = []
            if v_s1_c is not None:
                val_list.append(v_s1_c)
            if v_s2_c is not None:
                val_list.append(v_s2_c)
            if val_list:
                union_valid = val_list[0].copy()
                for vv in val_list[1:]:
                    union_valid = union_valid * vv
            else:
                union_valid = np.ones((new_h, new_w), dtype=np.uint8)

        rows = new_h // tile
        cols = new_w // tile
        base_transform = ref_ds.window_transform(Window(off_w, off_h, new_w, new_h))
        ref_profile = ref_ds.profile

        for r in range(rows):
            for c in range(cols):
                y0, x0 = r * tile, c * tile
                img_tile = image_stack[:, y0:y0 + tile, x0:x0 + tile]
                m_tile = masks_stack[:, y0:y0 + tile, x0:x0 + tile]
                valid_tile = None
                if union_valid is not None:
                    valid_tile = union_valid[y0:y0 + tile, x0:x0 + tile]
                    if exclude_nodata:
                        ratio = valid_tile.mean()
                        if ratio < valid_threshold:
                            continue

                tile_transform = base_transform * Affine.translation(c * tile, r * tile)
                img_name = f"sample{sid}_tile_r{r:02d}_c{c:02d}.tif"
                img_out = out_root / split / 'images' / img_name
                m_out = out_root / split / 'masks' / img_name
                v_out = out_root / split / 'valid' / img_name if write_valid_mask and valid_tile is not None else None

                if (img_out.exists() or m_out.exists() or (v_out and v_out.exists())) and overwrite:
                    for p in [img_out, m_out, v_out]:
                        if p and p.exists():
                            p.unlink()
                elif img_out.exists() and m_out.exists() and (v_out is None or v_out.exists()) and not overwrite:
                    index_writer.writerow([split, sid, r, c, str(img_out), str(m_out), str(v_out) if v_out else ''])
                    continue

                write_geotiff(img_out, img_tile, ref_profile, tile_transform)
                write_geotiff(m_out, m_tile, ref_profile, tile_transform)
                if v_out and valid_tile is not None:
                    write_geotiff(v_out, valid_tile[None].astype(np.uint8), ref_profile, tile_transform)
                index_writer.writerow([split, sid, r, c, str(img_out), str(m_out), str(v_out) if v_out else ''])
                yield img_tile, (valid_tile if union_valid is not None else None)


def load_splits(args) -> Dict[str, List[int]]:
    if args.catalog:
        cat_path = Path(args.catalog)
        data = json.loads(cat_path.read_text())
        # STAC catalog or collection; items can be nested. The official release provides catalog.json.
        # Simple strategy: iterate features/items; read properties.split and infer sample id from assets.
        items = []
        if 'features' in data:
            items = data['features']
        elif 'items' in data:
            items = data['items']
        else:
            # fallback: assume list of items
            if isinstance(data, list):
                items = data
            else:
                raise ValueError('Unrecognized catalog.json structure')
        splits = {'train': [], 'val': [], 'test': []}
        for it in items:
            props = it.get('properties', {})
            sp = props.get('split')
            # Infer sample id from asset paths
            assets = it.get('assets', {})
            sid = None
            for a in assets.values():
                href = a.get('href','')
                # sentinel12_s2_<id>_img.tif
                import re
                m = re.search(r'sentinel12_[a-z0-9]+_(\d+)_img', href)
                if m:
                    sid = int(m.group(1))
                    break
            if sp in splits and sid is not None:
                splits[sp].append(sid)
        return splits
    else:
        with open(args.split_json, 'r') as f:
            return json.load(f)

def build_band_order(args) -> List[str]:
    order = []
    if args.sensor in ('s2','dual'):
        order.extend([f'S2_B{i}' for i in range(6)])
    if args.sensor in ('s1','dual'):
        order.extend(['S1_VV','S1_VH'])
    if args.include_elevation:
        order.append('DEM_ELEVATION')
    if args.include_slope:
        order.append('DEM_SLOPE')
    return order

def write_metadata(out_root: Path, args, band_order: List[str]):
    meta_path = out_root / args.metadata_name
    meta = {
        'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'sensor_mode': args.sensor,
        'band_order': band_order,
        'scales': {
            'S1': 'dB_scaled(/100)',
            'S2': 'TOA(/10000)',
            'DEM_ELEVATION': 'meters',
            'DEM_SLOPE': 'degrees'
        },
        'include_elevation': args.include_elevation,
        'include_slope': args.include_slope,
        'exclude_nodata': args.exclude_nodata,
        'valid_threshold': args.valid_threshold,
        'description': 'Extended preparation combining official logic with dual-sensor & DEM support.'
    }
    meta_path.write_text(jsonlib.dumps(meta, indent=2))
    return meta_path

def main():
    args = parse_args()
    root = Path(args.root)
    out_root = Path(args.out_root)
    splits = load_splits(args)
    band_order = build_band_order(args)
    meta_path = write_metadata(out_root, args, band_order)
    print(f'[META] Wrote metadata -> {meta_path}')

    index_csv = out_root / 'tiles_index.csv'
    new_index = not index_csv.exists() or args.overwrite
    sum_vec = np.zeros(len(band_order), dtype=np.float64)
    sumsq_vec = np.zeros(len(band_order), dtype=np.float64)
    min_vec = np.full(len(band_order), np.inf, dtype=np.float64)
    max_vec = np.full(len(band_order), -np.inf, dtype=np.float64)
    pixel_count = 0

    with open(index_csv, 'a', newline='') as fcsv:
        writer = csv.writer(fcsv)
        if new_index:
            writer.writerow(['split','sample_id','row','col','image_path','mask_path','valid_path'])
        for split_name in ['train','val','test']:
            if split_name not in splits:
                continue
            ids = splits[split_name]
            for sid in ids:
                print(f'[PROC] sample {sid} split={split_name}')
                for tile_arr, valid_tile in process_sample(
                        root, out_root, sid, split_name, args.tile_size, args.overwrite,
                        writer, band_order, args.sensor, args.include_elevation, args.include_slope,
                        args.exclude_nodata, args.valid_threshold, args.write_valid_mask):
                    c, h, w = tile_arr.shape
                    reshaped = tile_arr.reshape(c, -1)
                    if args.exclude_nodata and valid_tile is not None:
                        mask_flat = valid_tile.reshape(-1).astype(bool)
                        if mask_flat.sum() == 0:
                            continue
                        sum_vec += (reshaped[:, mask_flat]).sum(axis=1)
                        sumsq_vec += (reshaped[:, mask_flat] ** 2).sum(axis=1)
                        min_vec = np.minimum(min_vec, (reshaped[:, mask_flat]).min(axis=1))
                        max_vec = np.maximum(max_vec, (reshaped[:, mask_flat]).max(axis=1))
                        pixel_count += mask_flat.sum()
                    else:
                        sum_vec += reshaped.sum(axis=1)
                        sumsq_vec += (reshaped ** 2).sum(axis=1)
                        min_vec = np.minimum(min_vec, reshaped.min(axis=1))
                        max_vec = np.maximum(max_vec, reshaped.max(axis=1))
                        pixel_count += h * w

    means = (sum_vec / pixel_count).tolist()
    vars_ = (sumsq_vec / pixel_count - (sum_vec / pixel_count) ** 2)
    vars_ = np.clip(vars_, 0, None)
    stds = np.sqrt(vars_).tolist()
    stats = {
        'band_order': band_order,
        'pixel_count_used': int(pixel_count),
        'mean': means,
        'std': stds,
        'min': min_vec.tolist(),
        'max': max_vec.tolist(),
        'exclude_nodata': args.exclude_nodata
    }
    stats_path = out_root / 'channel_stats.json'
    with open(stats_path, 'w') as fjs:
        jsonlib.dump(stats, fjs, indent=2)
    print(f'[STATS] Wrote stats -> {stats_path}')

if __name__ == '__main__':
    main()
