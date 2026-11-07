#!/usr/bin/env python3
"""Generate an 8-second 1080x1920 Reels video: animated window scene + synthesized
piano music + elegant text. Everything is original (no third-party media).

Usage:
  python3 tools/reel.py --out /tmp/x.mp4 --palette dusk --seed 7 \
      --lines "Quiet isn't empty." "It's where the small things" "finally get loud."

Palettes: dusk, dawn, night, mist.  Seed changes the melody and dust/cloud layout.
Needs only numpy, opencv, Pillow and ffmpeg (all preinstalled).
Prints one JSON line with the QA numbers; exit code != 0 on failure.
"""
import argparse, json, math, os, re, subprocess, sys, tempfile, wave
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS, DUR = 1080, 1920, 30, 8.0
N = int(FPS * DUR)
SR = 44100

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/freefont/FreeSerifItalic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
]

PALETTES = {
    "dusk": dict(top=(196, 178, 205), mid=(244, 196, 170), bot=(250, 224, 196),
                 glow=(46, 36, 12), glow_xy=(650, 880), glow_r=330,
                 hills=[((186, 150, 170), 150), ((150, 118, 146), 190), ((112, 88, 118), 230)],
                 frame=(46, 34, 48), sill=(62, 46, 60), mug=(30, 22, 34),
                 ray=(255, 226, 176), ray_a=0.24, cloud=(255, 238, 228), cloud_a=0.38,
                 steam=(255, 242, 228), mote=(1.0, 0.92, 0.78), band=0.30),
    "dawn": dict(top=(226, 186, 202), mid=(252, 212, 182), bot=(255, 236, 204),
                 glow=(52, 40, 12), glow_xy=(430, 900), glow_r=360,
                 hills=[((228, 176, 176), 150), ((196, 146, 156), 190), ((150, 110, 132), 230)],
                 frame=(58, 40, 52), sill=(76, 54, 66), mug=(40, 26, 38),
                 ray=(255, 230, 190), ray_a=0.26, cloud=(255, 240, 232), cloud_a=0.42,
                 steam=(255, 246, 236), mote=(1.0, 0.94, 0.82), band=0.40),
    "night": dict(top=(22, 26, 58), mid=(58, 58, 108), bot=(126, 108, 150),
                  glow=(40, 52, 96), glow_xy=(700, 700), glow_r=260,
                  hills=[((70, 70, 112), 150), ((46, 48, 88), 200), ((28, 30, 60), 235)],
                  frame=(12, 10, 22), sill=(28, 24, 42), mug=(10, 8, 18),
                  ray=(170, 190, 255), ray_a=0.16, cloud=(190, 200, 235), cloud_a=0.22,
                  steam=(220, 230, 255), mote=(0.80, 0.88, 1.0), band=0.16),
    "mist": dict(top=(172, 188, 206), mid=(204, 212, 222), bot=(232, 230, 226),
                 glow=(30, 30, 24), glow_xy=(560, 860), glow_r=420,
                 hills=[((176, 186, 202), 150), ((144, 156, 176), 190), ((108, 120, 146), 230)],
                 frame=(40, 44, 56), sill=(56, 60, 74), mug=(26, 28, 38),
                 ray=(255, 250, 236), ray_a=0.22, cloud=(246, 247, 250), cloud_a=0.5,
                 steam=(255, 255, 255), mote=(1.0, 1.0, 0.95), band=0.52),
}
CREAM = (255, 245, 232)


# --------------------------------------------------------------------------- scene
class Scene:
    def __init__(self, pal, seed):
        p = self.p = PALETTES[pal]
        rng = self.rng = np.random.default_rng(seed)
        self.wx0, self.wy0, self.wx1, self.wy1 = 120, 300, 960, 1360
        wx0, wy0, wx1, wy1 = self.wx0, self.wy0, self.wx1, self.wy1
        self.MX, self.MY = 700, wy1
        yy = np.arange(H, dtype=np.float32)[:, None]
        xx = np.arange(W, dtype=np.float32)[None, :]
        top, mid, bot = (np.array(p[k], np.float32) for k in ("top", "mid", "bot"))
        tv = np.clip((yy - wy0) / (wy1 - wy0), 0, 1)
        self.sky = np.zeros((H, W, 3), np.float32)
        for c in range(3):
            g = np.where(tv < 0.55, top[c] + (mid[c] - top[c]) * (tv / 0.55),
                         mid[c] + (bot[c] - mid[c]) * ((tv - 0.55) / 0.45))
            self.sky[..., c] = np.broadcast_to(g, (H, W))
        gh, gw = H // 4, W // 4
        gy = (np.arange(gh, dtype=np.float32) * 4)[:, None]
        gx = (np.arange(gw, dtype=np.float32) * 4)[None, :]
        gxp, gyp = p["glow_xy"]
        self.glow_lr = np.exp(-(((gx - gxp) ** 2) + ((gy - gyp) ** 2) * 1.2) / (p["glow_r"] ** 2)).astype(np.float32)
        self.glow_col = np.array(p["glow"], np.float32)
        # clouds
        CW = int(W * 1.4)
        cl = Image.new("L", (CW, H), 0)
        dc = ImageDraw.Draw(cl)
        for _ in range(14):
            cx = rng.integers(0, CW); cy = rng.integers(360, 880)
            rx = rng.integers(120, 260); ry = rng.integers(22, 48)
            dc.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=int(rng.integers(120, 220)))
        cl = cl.filter(ImageFilter.GaussianBlur(38))
        self.cloud_a = (np.asarray(cl, np.float32) / 255.0 * p["cloud_a"])[..., None]
        self.cloud_col = np.array(p["cloud"], np.float32)
        # hills
        self.HX, self.HY0, self.HY1 = W + 320, 880, 1400
        phase0 = float(rng.uniform(0, 2))
        self.hills = [
            (self._ridge(950, 30, 5.5, 0.5 + phase0, p["hills"][0]), 6.0),
            (self._ridge(1030, 24, 4.2, 2.0 + phase0, p["hills"][1]), 11.0),
            (self._ridge(1110, 20, 3.6, 4.0 + phase0, p["hills"][2]), 18.0),
        ]
        # frame, window bars, sill, mug
        dark = tuple(p["frame"]) + (255,)
        fr = Image.new("RGBA", (W, H), dark)
        df = ImageDraw.Draw(fr)
        df.rounded_rectangle([wx0, wy0, wx1, wy1], radius=20, fill=(0, 0, 0, 0))
        mxm = (wx0 + wx1) // 2
        df.rectangle([mxm - 10, wy0, mxm + 10, wy1], fill=dark)
        by = wy0 + 440
        df.rectangle([wx0, by, wx1, by + 20], fill=dark)
        df.rectangle([wx0 - 44, wy1, wx1 + 44, wy1 + 34], fill=tuple(p["sill"]) + (255,))
        mug = tuple(p["mug"]) + (255,)
        MX, MY = self.MX, self.MY
        df.rounded_rectangle([MX, MY - 110, MX + 100, MY], radius=16, fill=mug)
        df.arc([MX + 78, MY - 92, MX + 140, MY - 28], -90, 90, fill=mug, width=12)
        fr = fr.filter(ImageFilter.GaussianBlur(1.1))
        a = np.asarray(fr, np.float32)
        self.frame_rgb, self.frame_a = a[..., :3], a[..., 3:4] / 255.0
        # floor light rays
        rm = Image.new("L", (W, H), 0)
        dr = ImageDraw.Draw(rm)
        sy = wy1 + 34
        for off in (-300, -70, 160, 380):
            dr.polygon([(wx0 + 80 + off * 0.25, sy), (wx0 + 80 + off * 0.25 + 190, sy),
                        (wx0 + 80 + off + 360, H), (wx0 + 80 + off - 140, H)], fill=255)
        rm = rm.filter(ImageFilter.GaussianBlur(34))
        self.rays = np.asarray(rm, np.float32)[..., None] / 255.0
        self.ray_col = np.array(p["ray"], np.float32)
        # steam
        self.SW, self.SH = 240, 400
        self.steam_fade = (np.clip(1 - np.arange(self.SH, dtype=np.float32) / (self.SH * 0.92), 0, 1) ** 1.3)[::-1][:, None]
        self.steam_col = np.array(p["steam"], np.float32)
        self.sx0, self.sy0 = MX - 50, MY - 110 - self.SH + 20
        # dust motes
        self.NP = 30
        self.px0 = rng.uniform(150, 930, self.NP); self.py0 = rng.uniform(420, 1330, self.NP)
        self.pvy = rng.uniform(6, 16, self.NP); self.pph = rng.uniform(0, 6.28, self.NP)
        self.prad = rng.uniform(1.6, 3.4, self.NP)
        self.sr_ = 9
        sg = np.arange(-self.sr_, self.sr_ + 1, dtype=np.float32)
        self.stamp = np.exp(-(sg[:, None] ** 2 + sg[None, :] ** 2) / 8.0)
        self.mote = np.array(p["mote"], np.float32)
        self.vig = (1 - 0.30 * np.clip(((xx / W - 0.5) ** 2 + (yy / H - 0.5) ** 2) / 0.5, 0, 1))[..., None].astype(np.float32)
        self.grain_rng = np.random.default_rng(seed + 1000)

    def _ridge(self, y0, amp, freq, phase, spec):
        color, alpha = spec
        im = Image.new("RGBA", (self.HX, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        pts = [(0, H)]
        for px in range(0, self.HX + 1, 6):
            py = y0 + amp * math.sin(px / W * freq + phase) + amp * 0.4 * math.sin(px / W * freq * 2.3 + phase * 1.7)
            pts.append((px, py))
        pts.append((self.HX, H))
        d.polygon(pts, fill=tuple(color) + (alpha,))
        im = im.filter(ImageFilter.GaussianBlur(4))
        a = np.asarray(im, np.float32)[self.HY0:self.HY1]
        return a[..., :3], a[..., 3:4] / 255.0

    def steam_alpha(self, t):
        SW, SH = self.SW, self.SH
        m = np.zeros((SH, SW), np.uint8)
        for i, bx in enumerate((70, 112, 150)):
            pts = []
            for k in range(0, 330, 4):
                x = bx + 16 * (k / 330.0 + 0.3) * math.sin(k * 0.042 - 2 * math.pi * t / 2.6 + i * 1.9)
                pts.append([x, SH - 8 - k])
            cv2.polylines(m, [np.array(pts, np.int32)], False, 255, 7, cv2.LINE_AA)
        m = cv2.GaussianBlur(m, (0, 0), 4).astype(np.float32) / 255.0
        return (m * self.steam_fade * 0.55)[..., None]

    def render(self, t):
        p = self.p
        img = self.sky.copy()
        amp = 0.92 + 0.08 * math.sin(2 * math.pi * t / 4.0)
        glow = cv2.resize(self.glow_lr * amp, (W, H), interpolation=cv2.INTER_LINEAR)
        img += glow[..., None] * self.glow_col
        off = int(11 * t)
        a = self.cloud_a[:, off:off + W]
        img = img * (1 - a) + self.cloud_col * a
        for (rgb, al), spd in self.hills:
            o = int(spd * t)
            sl = img[self.HY0:self.HY1]
            sl[:] = sl * (1 - al[:, o:o + W]) + rgb[:, o:o + W] * al[:, o:o + W]
        sa = self.steam_alpha(t)
        reg = img[self.sy0:self.sy0 + self.SH, self.sx0:self.sx0 + self.SW]
        reg[:] = reg * (1 - sa) + self.steam_col * sa
        r = self.sr_
        for i in range(self.NP):
            y = 420 + (self.py0[i] - 420 - self.pvy[i] * t) % 910
            x = self.px0[i] + 14 * math.sin(0.7 * t + self.pph[i])
            tw = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(1.6 * t + self.pph[i] * 2))
            xi, yi = int(x), int(y)
            img[yi - r:yi + r + 1, xi - r:xi + r + 1] += (self.stamp * tw * (self.prad[i] / 3.4) * 60)[..., None] * self.mote
        img = img * (1 - self.frame_a) + self.frame_rgb * self.frame_a
        sh = int(14 * math.sin(2 * math.pi * t / DUR * 2))
        ra = np.roll(self.rays, sh, axis=1) * p["ray_a"] * (0.82 + 0.18 * math.sin(2 * math.pi * t / 3.2))
        img = img * (1 - ra) + self.ray_col * ra
        small = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 7)
        img = img * 0.80 + cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR) * 0.20
        e = 0.5 - 0.5 * math.cos(math.pi * t / DUR)
        s = 1.0 + 0.065 * e
        cx, cy = W / 2 + 10 * math.sin(2 * math.pi * t / DUR), H * 0.46
        M = np.array([[s, 0, (1 - s) * cx], [0, s, (1 - s) * cy]], np.float32)
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        img *= self.vig
        img *= 1.0 + 0.012 * math.sin(2 * math.pi * t * 3.1)
        img += cv2.resize(self.grain_rng.normal(0, 4.2, (H // 2, W // 2)).astype(np.float32), (W, H))[..., None]
        f = min(1.0, t / 0.6, (DUR - t) / 0.6)
        img *= f * f * (3 - 2 * f)
        return img


# --------------------------------------------------------------------------- text
def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


class TextOverlay:
    def __init__(self, lines, band=0.26):
        self.band_strength = band
        font_path = next(f for f in FONT_CANDIDATES if os.path.exists(f))
        n = len(lines)
        specs = []
        for i, txt in enumerate(lines):
            if n == 1:
                size, yc = 104, 765
            elif i == 0:
                size, yc = 104, 640
            else:
                size, yc = 66, 800 + 84 * (i - 1)
            start = 0.7 if i == 0 else 2.2 + 0.7 * (i - 1)
            specs.append((txt, size, yc, start, start + 1.2))
        self.specs = specs
        self.layers = [self._layer(font_path, *s[:3]) for s in specs]
        yy = np.arange(H, dtype=np.float32)[:, None]
        cy = float(np.mean([s[2] for s in specs]))
        self.band = np.exp(-((yy - cy) / 330.0) ** 2)[..., None].astype(np.float32)

    def _layer(self, font_path, text, size, yc):
        max_w = 900
        draw0 = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
        font = ImageFont.truetype(font_path, size)
        while draw0.textlength(text, font=font) > max_w and size > 36:
            size -= 2
            font = ImageFont.truetype(font_path, size)
        h = int(size * 2.4)
        lay = Image.new("RGBA", (W, h), (0, 0, 0, 0))
        tw = draw0.textlength(text, font=font)
        x, y = (W - tw) / 2, h / 2 - size * 0.62
        sh = Image.new("RGBA", (W, h), (0, 0, 0, 0))
        ImageDraw.Draw(sh).text((x, y + 4), text, font=font, fill=(40, 24, 44, 230))
        lay = Image.alpha_composite(lay, sh.filter(ImageFilter.GaussianBlur(size * 0.16)))
        lay = Image.alpha_composite(lay, sh.filter(ImageFilter.GaussianBlur(size * 0.04)))
        ImageDraw.Draw(lay).text((x, y), text, font=font, fill=CREAM + (255,))
        a = np.asarray(lay, np.float32)
        alpha = a[..., 3:4] / 255.0
        return a[..., :3] * alpha, alpha, int(yc - h / 2)

    def apply(self, img, t):
        out_f = 1.0 - ease((t - 6.9) / 0.6)
        img *= 1 - self.band_strength * self.band * (ease((t - self.specs[0][3]) / 1.2) * out_f)
        for (prem, alpha, y0), (_, _, _, st, en) in zip(self.layers, self.specs):
            pr_ = ease((t - st) / (en - st))
            f = pr_ * out_f
            if f <= 0.003:
                continue
            dy = (1 - pr_) * 18.0 + (t - st) * 1.2
            h = prem.shape[0]
            M = np.float32([[1, 0, 0], [0, 1, -dy]])
            pr = cv2.warpAffine(prem, M, (W, h), flags=cv2.INTER_LINEAR)
            al = cv2.warpAffine(alpha[..., 0], M, (W, h), flags=cv2.INTER_LINEAR)[..., None]
            ya, yb = max(0, y0), min(H, y0 + h)
            reg = img[ya:yb]
            reg[:] = reg * (1 - al[ya - y0:yb - y0] * f) + pr[ya - y0:yb - y0] * f
        return np.clip(img, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- music
CHORD_IV = {"m9": [0, 3, 7, 10, 14], "maj7": [0, 4, 7, 11, 14], "maj9": [0, 4, 7, 11, 14], "m7": [0, 3, 7, 10, 14]}
PROGRESSIONS = [
    [(45, "m9"), (41, "maj7")],
    [(48, "maj9"), (45, "m7")],
    [(50, "m9"), (46, "maj7")],
    [(52, "m7"), (48, "maj9")],
]


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def make_music(seed, path):
    rng = np.random.default_rng(seed + 500)
    n_all = int(SR * DUR)
    L = np.zeros(n_all); R = np.zeros(n_all)

    def piano(m, dur, vel):
        n = int(SR * dur)
        t = np.arange(n) / SR
        f = hz(m)
        sig = np.zeros(n)
        for k, (a, d) in enumerate([(1.0, 1.0), (0.45, 1.3), (0.22, 1.8), (0.12, 2.4), (0.06, 3.0), (0.03, 3.8)], start=1):
            fk = f * k * (1 + 0.0004 * k * k)
            sig += a * np.sin(2 * np.pi * fk * t + rng.uniform(0, 6.28)) * np.exp(-t * d * (1.0 + (m - 60) / 60.0))
        sig += 0.25 * np.sin(2 * np.pi * f * 1.003 * t) * np.exp(-t * 1.0)
        return sig * np.minimum(1, t / 0.006) * vel

    def add(sig, start, pan):
        i = int(SR * start)
        j = min(n_all, i + len(sig))
        if j <= i:
            return
        seg = sig[: j - i]
        L[i:j] += seg * (1 - pan) * 0.7071
        R[i:j] += seg * (1 + pan) * 0.7071

    prog = PROGRESSIONS[seed % len(PROGRESSIONS)]
    prev = None
    for ci, (root, q) in enumerate(prog):
        t0 = ci * 4.0
        iv = CHORD_IV[q]
        add(piano(root, 4.0, 0.45), t0, -0.1)
        arp = [root + 12 + x for x in iv[:4]]
        for i in range(8):
            add(piano(arp[i % 4], 2.5, 0.28), t0 + i * 0.5, -0.25) if t0 + i * 0.5 < t0 + 4.0 else None
        pool = [root + 24 + x for x in iv] + [root + 36 + x for x in iv]
        pool = sorted({m for m in pool if 69 <= m <= 81})
        times = [0.3, 1.4, 2.2, 3.0] if ci == 0 else [0.3, 1.2, 2.0, 2.9]
        for tt in times:
            if prev is None:
                m = int(rng.choice(pool))
            else:
                w = np.array([math.exp(-abs(x - prev) / 3.0) for x in pool])
                m = int(rng.choice(pool, p=w / w.sum()))
            prev = m
            add(piano(m, 3.5, 0.55), t0 + tt + float(rng.uniform(-0.05, 0.08)), 0.15)
        # airy pad
        dur = 4.4
        n = int(SR * dur)
        tt_ = np.arange(n) / SR
        pad = np.zeros(n)
        for x in [root + 12 + i for i in iv[:4]]:
            pad += np.sin(2 * np.pi * hz(x) * tt_) + 0.3 * np.sin(2 * np.pi * hz(x) * 2.001 * tt_)
        pad *= np.minimum(1, tt_ / 1.2) * np.minimum(1, (dur - tt_) / 1.4) * 0.035
        i0 = int(SR * max(0, t0 - 0.2)); j0 = min(n_all, i0 + n)
        L[i0:j0] += pad[: j0 - i0]; R[i0:j0] += pad[: j0 - i0]

    def make_ir(s):
        r = np.random.default_rng(s)
        nn = int(SR * 2.4)
        t = np.arange(nn) / SR
        ir = r.normal(0, 1, nn) * np.exp(-t * 2.3)
        k = np.exp(-np.arange(200) / 40.0); k /= k.sum()
        ir = np.convolve(ir, k, mode="same")
        return ir / np.sqrt(np.sum(ir ** 2))

    def conv(x, ir):
        nn = 1
        while nn < len(x) + len(ir):
            nn *= 2
        return np.fft.irfft(np.fft.rfft(x, nn) * np.fft.rfft(ir, nn), nn)[: len(x)]

    L2 = L * 0.62 + conv(L, make_ir(1)) * 0.456
    R2 = R * 0.62 + conv(R, make_ir(2)) * 0.456
    t = np.arange(n_all) / SR
    fade = np.minimum(1, t / 0.7) * np.minimum(1, (DUR - t) / 1.6)
    L2 *= fade; R2 *= fade
    pk = max(np.abs(L2).max(), np.abs(R2).max())
    g = 10 ** (-3 / 20) / pk
    L2 *= g; R2 *= g
    pcm = (np.clip(np.stack([L2, R2], 1), -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--palette", default="dusk", choices=sorted(PALETTES))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lines", nargs="+", required=True, help="1 title line + up to 3 smaller lines")
    a = ap.parse_args()
    if len(a.lines) > 4:
        sys.exit("max 4 lines")
    tmp = tempfile.mkdtemp(prefix="reel_")
    wav = os.path.join(tmp, "music.wav")
    make_music(a.seed, wav)
    scene = Scene(a.palette, a.seed)
    text = TextOverlay(a.lines, band=PALETTES[a.palette]["band"])
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", wav, "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-shortest", "-movflags", "+faststart", a.out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(N):
        t = i / FPS
        p.stdin.write(text.apply(scene.render(t), t).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        sys.exit("ffmpeg failed")
    # QA
    base = os.path.join(tmp, "preview")
    for s in (1.2, 5.0):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(s), "-i", a.out, "-frames:v", "1", f"{base}_t{s}.jpg"], check=True)
    pr = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration,size", "-of", "json", a.out],
                        capture_output=True, text=True, check=True)
    fmt = json.loads(pr.stdout)["format"]
    vd = subprocess.run(["ffmpeg", "-hide_banner", "-i", a.out, "-af", "volumedetect", "-vn", "-f", "null", "-"],
                        capture_output=True, text=True).stderr
    mean = re.search(r"mean_volume: (-?[\d.]+)", vd); mx = re.search(r"max_volume: (-?[\d.]+)", vd)
    im = cv2.imread(f"{base}_t5.0.jpg")
    res = dict(out=a.out, preview=f"{base}_t5.0.jpg", duration=round(float(fmt["duration"]), 2), size_mb=round(int(fmt["size"]) / 1e6, 2),
               mean_db=float(mean.group(1)) if mean else None, max_db=float(mx.group(1)) if mx else None,
               preview_mean_brightness=round(float(im.mean()), 1))
    ok = (7.9 <= res["duration"] <= 8.1 and res["size_mb"] < 50 and res["max_db"] is not None and res["max_db"] <= -1.0
          and res["mean_db"] is not None and res["mean_db"] > -35 and 25 < res["preview_mean_brightness"] < 230)
    res["qa_pass"] = bool(ok)
    print(json.dumps(res))
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
