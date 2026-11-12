"""Repaint the underside of the ChocChip plush to match the photo input/down.jpeg.

Seen from below, the real plush has:
  * a white belly that runs all the way down between the legs,
  * all-white legs and foot soles,
  * a brown bottom behind the legs, joined to the brown tail.
The generated texture has a brown strip between the legs and white patches on the
brown bottom. This script paints the down-facing body triangles in the texture:
white in front of the line behind the legs, brown behind it. Fur detail is kept by
adding the original texel's fine light/dark variation on top of the new colour.
Only the texture image is replaced; geometry, UVs and skin are copied as-is.

usage: python fix_bottom_color.py IN.glb OUT.glb [preview.png]
"""
import io
import json
import struct
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

src, dst = sys.argv[1], sys.argv[2]
data = open(src, 'rb').read()
magic, version, _ = struct.unpack_from('<III', data, 0)
jlen = struct.unpack_from('<I', data, 12)[0]
g = json.loads(data[20:20 + jlen])
boff = 20 + jlen
blen = struct.unpack_from('<I', data, boff)[0]
buf = bytes(data[boff + 8:boff + 8 + blen])
CT = {5126: np.float32, 5123: np.uint16, 5125: np.uint32, 5121: np.uint8}
NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


def acc(i):
    a = g['accessors'][i]
    bv = g['bufferViews'][a['bufferView']]
    off = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
    return np.frombuffer(buf, CT[a['componentType']], a['count'] * NC[a['type']], off).reshape(a['count'], -1)


prim = g['meshes'][0]['primitives'][0]
at = prim['attributes']
P, N, UV = acc(at['POSITION']), acc(at['NORMAL']), acc(at['TEXCOORD_0'])
JN, WT = acc(at['JOINTS_0']), acc(at['WEIGHTS_0'])
tri = acc(prim['indices']).reshape(-1, 3)
names = [g['nodes'][j]['name'] for j in g['skins'][0]['joints']]

img_i = g['textures'][g['materials'][prim['material']]['pbrMetallicRoughness']['baseColorTexture']['index']]['source']
ibv = g['bufferViews'][g['images'][img_i]['bufferView']]
io0 = ibv.get('byteOffset', 0)
tex = Image.open(io.BytesIO(buf[io0:io0 + ibv['byteLength']])).convert('RGB')
W, H = tex.size
T = np.asarray(tex).astype(np.float32)

# dominant bone per vertex
dom = np.array(names)[JN[np.arange(len(JN)), WT.argmax(1)].astype(int)]
body = ~np.char.startswith(dom, 'leg') & ~np.char.startswith(dom, 'tail') & ~np.char.startswith(dom, 'arm')

# ---- which triangles to paint, and with what ----
c = P[tri].mean(1)
n = N[tri].mean(1)
n /= np.linalg.norm(n, axis=1, keepdims=True)
tb = body[tri].all(1)
LOW = -0.36                       # only the lower body
REAR = -0.10                      # line just behind the back edge of the feet
paint = tb & (c[:, 1] < LOW) & (n[:, 1] < -0.15)
white = paint & (c[:, 2] > REAR)
brown = paint & ~white

# the legs are all white on the real plush: paint every triangle on a leg
# (owned by a leg bone, or inside the leg tube below the body) white
ibm = acc(g['skins'][0]['inverseBindMatrices']).reshape(-1, 4, 4)
legs = np.zeros(len(tri), bool)
for nm in ('leg.L', 'leg.R'):
    m = np.linalg.inv(ibm[names.index(nm)].T)
    p, d = m[:3, 3], m[:3, 1] / np.linalg.norm(m[:3, 1])
    v = c - p
    s = v @ d
    r = np.linalg.norm(v - np.outer(s, d), axis=1)
    legs |= (s > 0.2) & (r < 0.22)
legs |= np.char.startswith(dom, 'leg')[tri].all(1)
# the crotch and inner thighs, whichever way their faces point (they hide in creases):
# everything low and in front of the brown bottom, except the tail
legs |= tb & (c[:, 1] < -0.55) & (np.abs(c[:, 0] + 0.035) < 0.22) & (c[:, 2] > REAR)
legs |= (c[:, 1] < -0.5) & (c[:, 2] > REAR) & ~np.char.startswith(dom, 'tail')[tri].any(1)
white |= legs
brown &= ~legs
print('leg triangles painted white:', int(legs.sum()))

# the arms are all white too: everything inside the arm tube, out from the shoulder
arms = np.zeros(len(tri), bool)
for nm in ('arm.L', 'arm.R'):
    m = np.linalg.inv(ibm[names.index(nm)].T)
    p, d = m[:3, 3], m[:3, 1] / np.linalg.norm(m[:3, 1])
    v = c - p
    s = v @ d
    r = np.linalg.norm(v - np.outer(s, d), axis=1)
    arms |= (s > 0.06) & (r < 0.24)
white |= arms
brown &= ~arms
print('arm triangles painted white:', int(arms.sum()))

# near-black specks on the body below the head (tiny triangles whose UVs land on the
# black padding of the texture atlas) take the colour of the fur around them
from scipy.spatial import cKDTree
uu = np.clip((UV[:, 0] * W).astype(int), 0, W - 1)
vv = np.clip((UV[:, 1] * H).astype(int), 0, H - 1)
vlum = T[vv, uu].mean(1)
dark = (vlum[tri].min(1) < 35) & (c[:, 1] < -0.1)
ok = np.where(vlum >= 35)[0]
_, nb = cKDTree(P[ok]).query(c[dark], k=40)
around_white = np.median(vlum[ok][nb], axis=1) > 150
dw = np.zeros(len(tri), bool)
dw[np.where(dark)[0][around_white]] = True
dw |= dark & (legs | arms)


white |= dw
brown |= dark & ~dw
print('dark specks fixed:', int(dark.sum()))
print('paint triangles: white', int(white.sum()), 'brown', int(brown.sum()))


def mean_colour(mask_tris):
    v = np.unique(tri[mask_tris])
    u = np.clip((UV[v, 0] * W).astype(int), 0, W - 1)
    w = np.clip((UV[v, 1] * H).astype(int), 0, H - 1)
    return np.median(T[w, u], axis=0)


# reference colours taken from parts of the texture that are already right:
# the front of the belly (white) and the back of the body (brown)
ref_white = mean_colour(tb & (n[:, 2] > 0.6) & (c[:, 1] < -0.2) & (c[:, 1] > -0.35) & (np.abs(c[:, 0]) < 0.12))
ref_brown = mean_colour(tb & (n[:, 2] < -0.6) & (c[:, 1] < -0.2) & (c[:, 1] > -0.35) & (np.abs(c[:, 0]) < 0.12))
print('white', ref_white.round(), 'brown', ref_brown.round())

# masks in texture space (polygons grown a little so UV seams don't show a line)
masks = {}
for key, sel in (('w', white), ('b', brown)):
    m = Image.new('L', (W, H), 0)
    d = ImageDraw.Draw(m)
    for t in tri[sel]:
        pts = [(float(UV[k, 0] * W), float(UV[k, 1] * H)) for k in t]
        d.polygon(pts, fill=255, outline=255)
    # white grows 4 px so no dark texels are left along the UV seams of the legs
    masks[key] = np.asarray(m.filter(ImageFilter.MaxFilter(9 if key == 'w' else 5))) > 0
# where both grew into each other, the triangle's own colour wins
masks['w'] &= ~(masks['b'] & ~np.asarray(Image.fromarray(masks['w'].astype(np.uint8) * 255).filter(ImageFilter.MinFilter(9))).astype(bool))

lum = T.mean(2)
blur = np.asarray(Image.fromarray(lum.astype(np.uint8)).filter(ImageFilter.GaussianBlur(6))).astype(np.float32)
detail = np.clip(lum - blur, -40, 40)[..., None]   # fine fur variation of the original texels

out = T.copy()
# white keeps only faint fur detail, so the edges of old brown patches don't show as dark lines
for key, ref, k, lim in (('w', ref_white, 0.5, 8), ('b', ref_brown, 0.8, 40)):
    m = masks[key]
    out[m] = np.clip(ref + np.clip(detail[m] * k, -lim, lim), 0, 255)

# soften the white/brown border a touch inside the painted area
res = Image.fromarray(out.astype(np.uint8))
both = masks['w'] | masks['b']
soft = np.asarray(res.filter(ImageFilter.GaussianBlur(1.2))).astype(np.float32)
edge = np.asarray(Image.fromarray((masks['w'] * 255).astype(np.uint8)).filter(ImageFilter.FIND_EDGES)) > 0
edge = np.asarray(Image.fromarray((edge * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5))) > 0
sel = edge & both
out[sel] = soft[sel]
res = Image.fromarray(out.astype(np.uint8))
if len(sys.argv) > 3:
    res.save(sys.argv[3])

png = io.BytesIO()
res.save(png, 'PNG', optimize=True)
png = png.getvalue()

# rebuild the binary buffer with the new image in place of the old one
old = buf[io0:io0 + ibv['byteLength']]
pad = (-len(png)) % 4
newbuf = buf[:io0] + png + b'\0' * pad + buf[io0 + ibv['byteLength']:]
shift = len(png) + pad - ibv['byteLength']
for bv in g['bufferViews']:
    if bv is not ibv and bv.get('byteOffset', 0) > io0:
        bv['byteOffset'] += shift
ibv['byteLength'] = len(png)
g['buffers'][0]['byteLength'] = len(newbuf)

jb = json.dumps(g, separators=(',', ':')).encode()
jb += b' ' * (-len(jb) % 4)
bb = newbuf + b'\0' * (-len(newbuf) % 4)
o = struct.pack('<III', magic, version, 12 + 8 + len(jb) + 8 + len(bb))
o += struct.pack('<II', len(jb), 0x4E4F534A) + jb + struct.pack('<II', len(bb), 0x004E4942) + bb
open(dst, 'wb').write(o)
print('wrote', dst, len(o), 'bytes')
