#!/usr/bin/env python3
"""Derive one light gear-grading preset per V7 background scene (19 scenes; B08 dropped).
Inputs: only /workspace/v7-backgrounds/{ski,board}/*-bg-NN.png and the 20261115f placements (items.json)."""
import json, hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
R = Path(__file__).resolve().parent; BG = Path('/workspace/v7-backgrounds'); IT = Path('/workspace/pw-c04-bgtest/items.json')
TV = Path('/workspace/pw-c04-master/approved-template-v7-copy')
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
SCENES = ['S%02d' % i for i in range(1, 11)] + ['B%02d' % i for i in range(1, 11) if i != 8]
DESC = {'S01': 'lodge wood wall', 'S02': 'fireplace corner', 'S03': 'deep lodge room, back wall', 'S04': 'outdoor rack in snow', 'S05': 'panelled boot room',
        'S06': 'timber apres bar (dark)', 'S07': 'gear shop back wall', 'S08': 'cabin porch log wall', 'S09': 'outdoor rack in snow', 'S10': 'wood panel between snow windows',
        'B01': 'barn-wood wall', 'B02': 'fireplace / stone chimney wall', 'B03': 'deep boot room back wall', 'B04': 'outdoor rack in snow', 'B05': 'dark mudroom',
        'B06': 'apres bar plank wall', 'B07': 'repair shop white block wall', 'B09': 'outdoor rack in snow', 'B10': 'apres lounge, wood panel between snow windows'}
def g(p):
    a = np.asarray(Image.open(p)); return a[..., 0] if a.ndim == 3 else a
def gear_aspect(v):
    d = TV/v; matte = (g(d/'editable-art-mask.png') > 0) | (g(d/'binding-opaque-core.png') > 0) | (np.asarray(Image.open(d/'fixed-source-hardware.png'))[..., 3] > 0) | (g(d/'surface-trace.png') > 0)
    ys, xs = np.nonzero(matte); return (xs.max()+1-xs.min())/(ys.max()+1-ys.min())
ASP = {v: gear_aspect(v) for v in ['matching', 'spread', 'board']}
items = {i['id']: i for i in json.loads(IT.read_text())}
CAP_LEVEL_SHIFT = 22.0; LEVEL_STRENGTH = 0.5; TINT_K = 0.12; TINT_K_LIGHT = 0.5; TINT_CAP = 0.05; GRAD_K = 0.6; GRAD_CAP = 0.07; GRAIN_CAP = 3.0
out = {}
for sid in SCENES:
    it = items[sid]; p = BG/it['bg']; img = np.asarray(Image.open(p).convert('RGB')).astype(np.float32); H, W = img.shape[:2]
    h = it['h']; w = round(h*ASP[it['layout']]); x0, x1 = int(it['cx']-w/2), int(it['cx']+w/2); y0, y1 = it['bottom']-h, it['bottom']
    # region around the gear = gear box grown by 60% of the gear width each side and 10% of height top/bottom
    mx, my = int(0.6*w) + 40, int(0.1*h); rx0, rx1, ry0, ry1 = max(0, x0-mx), min(W, x1+mx), max(0, y0-my), min(H, y1+my)
    reg = img[ry0:ry1, rx0:rx1]; L = reg @ LUMA
    b1, w99 = np.percentile(L, [1, 99])
    # tint: scene mean colour relative to its own grey (chromatic only), damped and capped
    mean = reg.reshape(-1, 3).mean(0); chrom = mean/max(1e-3, mean.mean())
    # key-light colour = mean of the brightest 3% of the whole photo (fire glow / window / snow); ambient = local wall colour
    Lall = img @ LUMA; hl = img[Lall >= np.percentile(Lall, 97)].mean(0); hchrom = hl/max(1e-3, hl.mean())
    warm = (hl[0]-hl[2])/max(1, hl.mean())
    k_amb = TINT_K if warm > 0.05 else 0.03          # wall bounce only adds warmth when the key light itself is warm (fire/lamps); daylight/snow scenes stay neutral-cool
    gains = np.clip(1 + TINT_K_LIGHT*(hchrom-1) + k_amb*(chrom-1), 1-TINT_CAP, 1+TINT_CAP); gains /= (gains @ LUMA)   # keep luminance neutral
    light = 'warm (lodge/fire)' if warm > 0.05 else ('cool (snow/window daylight)' if warm < -0.005 else 'neutral white')
    # gradient: compare background luminance in side strips left/right of the gear (and above/below) across the gear's span
    Lf = img @ LUMA
    left = Lf[y0:y1, max(0, x0-mx):x0].mean() if x0 > 0 else Lf[y0:y1, :max(1, x0+1)].mean()
    right = Lf[y0:y1, x1:min(W, x1+mx)].mean()
    top = Lf[max(0, y0-my*3):y0+h//4, rx0:rx1].mean(); bot = Lf[y1-h//4:min(H, y1+my), rx0:rx1].mean()
    amp_x = float(np.clip(GRAD_K*(right-left)/max(1, right+left), -GRAD_CAP, GRAD_CAP))
    amp_y = float(np.clip(0.5*GRAD_K*(bot-top)/max(1, bot+top), -GRAD_CAP/2, GRAD_CAP/2))
    # grain: robust std of the photo's high-pass luminance in flat-ish parts of the region (MAD), capped
    hp = (L - gaussian_filter(L, 1.2)); flat = np.abs(gaussian_filter(L, 3) - gaussian_filter(L, 1.2)) < 3
    sig = float(1.4826*np.median(np.abs(hp[flat] - np.median(hp[flat])))) if flat.sum() > 500 else float(hp.std())
    sig = round(min(GRAIN_CAP, sig*0.9), 3)
    P = {'scene': sid, 'description': DESC[sid], 'background': str(p), 'background_sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
         'derived_from': {'gear_box_xyxy': [x0, y0, x1, y1], 'region_xyxy': [rx0, ry0, rx1, ry1], 'scene_mean_rgb': [round(float(c), 1) for c in mean], 'key_light_rgb_top3pct': [round(float(c), 1) for c in hl],
                          'side_luma_left_right': [round(float(left), 1), round(float(right), 1)], 'luma_above_below': [round(float(top), 1), round(float(bot), 1)], 'key_light_warmth_index': round(float(warm), 3)},
         'levels': {'scene_black': round(float(b1), 2), 'scene_white': round(float(w99), 2), 'strength': LEVEL_STRENGTH, 'max_shift': CAP_LEVEL_SHIFT,
                    'note': 'gear 1st/99th pct luminance (measured at apply time) moved halfway toward the scene black/white, max 22 levels; applied to luminance only (RGB offsets added), so chroma and hue are kept'},
         'tint': {'light': light, 'gains': [round(float(x), 4) for x in gains], 'cap': TINT_CAP, 'note': 'per-channel gains = 0.5 x key-light chroma + local wall chroma (0.12 if the key light is warm, else 0.03), luminance-neutral, capped +/-5%'},
         'gradient': {'amp_x': round(amp_x, 4), 'amp_y': round(amp_y, 4), 'brighter_side': ('right' if amp_x > 0 else 'left') if abs(amp_x) > 0.005 else 'even',
                      'cap': GRAD_CAP, 'note': 'multiplicative 1+amp_x*x+amp_y*y across the gear box, x,y in [-1,1]'},
         'grain': {'sigma': sig, 'blur': 0.6, 'seed': int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16), 'cap': GRAIN_CAP}}
    (R/f'{sid}.json').write_text(json.dumps(P, indent=1)); out[sid] = P
    print(sid, light, 'lv', round(b1), round(w99), 'tint', P['tint']['gains'], 'grad', P['gradient']['amp_x'], P['gradient']['amp_y'], 'grain', sig)
(R/'index.json').write_text(json.dumps({'scenes': SCENES, 'dropped': {'B08': 'angled 3/4 porch background (Nov 15 9:39 AM MT); files kept'},
    'apply': 'python3 apply_preset.py <scene>.json <V7 composite.png> <gear alpha mask.png> <out.png>'}, indent=1))
