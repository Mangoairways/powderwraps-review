#!/usr/bin/env python3
"""Apply one per-scene V7 gear grading preset to a V7 composite.

GEAR LAYER ONLY: pixels are changed only where the gear alpha mask > 0 (blended by alpha).
Background, shadows, geometry, bindings' shape and print masters are never touched.

usage: apply_preset.py PRESET.json COMPOSITE.png GEAR_ALPHA.png OUT.png [--seed N]
  GEAR_ALPHA.png = 8-bit mask (same size as the composite) of the composited gear pixels.
"""
import json, sys
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)

def apply(preset, comp, alpha, seed=None):
    P = preset; img = comp.astype(np.float32); a = alpha.astype(np.float32)/255.0
    m = a > 0; core = a >= 0.999
    if not m.any(): return comp.copy(), {}
    L = img @ LUMA
    # (a) levels: gear 1st/99th percentile luminance -> pulled toward the scene black/white points
    g1, g99 = np.percentile(L[core if core.any() else m], [1, 99])
    lv = P['levels']; s = lv['strength']
    t1 = g1 + s*(lv['scene_black'] - g1); t99 = g99 + s*(lv['scene_white'] - g99)
    t1 = float(np.clip(t1, g1 - lv['max_shift'], g1 + lv['max_shift'])); t99 = float(np.clip(t99, g99 - lv['max_shift'], g99 + lv['max_shift']))
    gain = (t99 - t1)/max(1.0, g99 - g1); Ln = (L - g1)*gain + t1
    out = img + (Ln - L)[..., None]                                         # luminance-only map: chroma offsets kept, no hue change
    # (b) tint toward the scene light (per-channel gains, capped)
    out *= np.array(P['tint']['gains'], np.float32)
    # (c) soft brightness gradient across the gear, brighter toward the light side
    ys, xs = np.nonzero(m); x0, x1 = xs.min(), xs.max(); y0, y1 = ys.min(), ys.max()
    gr = P['gradient']; H, W = L.shape
    xx = (np.arange(W, dtype=np.float32) - (x0 + x1)/2)/max(1.0, (x1 - x0)/2)
    yy = (np.arange(H, dtype=np.float32) - (y0 + y1)/2)/max(1.0, (y1 - y0)/2)
    f = 1.0 + gr['amp_x']*np.clip(xx, -1, 1)[None, :] + gr['amp_y']*np.clip(yy, -1, 1)[:, None]
    out *= f[..., None]
    # (d) light luminance grain matching the photo noise
    rng = np.random.default_rng(P['grain']['seed'] if seed is None else seed)
    n = rng.standard_normal((H, W)).astype(np.float32)
    if P['grain'].get('blur', 0) > 0: n = gaussian_filter(n, P['grain']['blur']); n /= max(1e-6, n.std())
    out += (P['grain']['sigma']*n)[..., None]
    out = np.clip(out, 0, 255)
    res = img*(1 - a[..., None]) + out*a[..., None]
    res8 = np.clip(np.round(res), 0, 255).astype(np.uint8)
    res8[~m] = comp[~m]                                                     # hard guarantee: nothing outside the gear mask changes
    info = {'gear_p1': round(float(g1), 2), 'gear_p99': round(float(g99), 2), 'target_p1': round(t1, 2), 'target_p99': round(t99, 2), 'level_gain': round(float(gain), 4),
            'px_changed_outside_gear_mask': int((res8 != comp).any(2)[~m].sum())}
    return res8, info

if __name__ == '__main__':
    args = [x for x in sys.argv[1:] if not x.startswith('--')]; seed = None
    if '--seed' in sys.argv: seed = int(sys.argv[sys.argv.index('--seed') + 1]); args = [x for x in args if x != str(seed)]
    P = json.load(open(args[0])); comp = np.asarray(Image.open(args[1]).convert('RGB')); al = np.asarray(Image.open(args[2]).convert('L'))
    r, info = apply(P, comp, al, seed); Image.fromarray(r).save(args[3]); print(json.dumps(info))
