"""Round off the square block under the chin of the ChocChip plush.

In the scan, the top of the chest just under the big head is a flat-fronted block
with sharp corners, so a square lump shows under the chin. This script reshapes
only that small area (the front of the chest top, between the arms, below the
head), in cylinder coordinates around the body's vertical axis: the outer skin
is moved onto a smooth surface averaged over a band of angle and height, which
rounds the corners and fills the sharp crease against the head. The change fades
out towards the head, the arms, the sides and further down the chest, so the
rest of the body keeps its original shape. Seam duplicates are welded while
reshaping so no cracks open. Only POSITION and NORMAL bytes (and the POSITION
min/max) are rewritten.

usage: python round_chin.py IN.glb OUT.glb
"""
import json
import struct
import sys

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

src, dst = sys.argv[1], sys.argv[2]
data = open(src, 'rb').read()
magic, version, _ = struct.unpack_from('<III', data, 0)
jlen = struct.unpack_from('<I', data, 12)[0]
g = json.loads(data[20:20 + jlen])
boff = 20 + jlen
blen = struct.unpack_from('<I', data, boff)[0]
buf = bytearray(data[boff + 8:boff + 8 + blen])
CT = {5126: np.float32, 5123: np.uint16, 5125: np.uint32, 5121: np.uint8}
NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


def acc(i):
    a = g['accessors'][i]
    bv = g['bufferViews'][a['bufferView']]
    off = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
    arr = np.frombuffer(bytes(buf), CT[a['componentType']], a['count'] * NC[a['type']], off).reshape(a['count'], -1)
    return arr, off


def ss(a, b, x):
    u = np.clip((x - a) / (b - a), 0, 1)
    return u * u * (3 - 2 * u)


prim = g['meshes'][0]['primitives'][0]
at = prim['attributes']
P, poff = acc(at['POSITION'])
N, noff = acc(at['NORMAL'])
tri = acc(prim['indices'])[0].reshape(-1, 3).astype(np.int64)
P = P.astype(np.float64)
names = [g['nodes'][j]['name'] for j in g['skins'][0]['joints']]
ibm = acc(g['skins'][0]['inverseBindMatrices'])[0].reshape(-1, 4, 4)

# weld seam duplicates
key = np.round(P / 1e-5).astype(np.int64)
_, uid, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
inv = inv.ravel()
U = P[uid].copy()
T = inv[tri]

# how much of each point the head bone owns: the underside of the head must not be
# pulled into the chest (that folds a rim along the crease)
Jw, Ww = acc(at['JOINTS_0'])[0], acc(at['WEIGHTS_0'])[0].astype(float)
headw = (Ww * (Jw == names.index('head'))).sum(1)[uid]
nohead = 1 - ss(0.3, 0.6, headw)

x, y, z = U[:, 0], U[:, 1], U[:, 2]
ZC = -0.03
th = np.arctan2(x, z - ZC)          # 0 = straight ahead
rho = np.hypot(x, z - ZC)


def limb(nm, s0, r0, soft):
    """0..1: how much a point belongs to the root/tube of an arm."""
    m = np.linalg.inv(ibm[names.index(nm)].T)
    p, d = m[:3, 3], m[:3, 1] / np.linalg.norm(m[:3, 1])
    v = U - p
    s = v @ d
    r = np.linalg.norm(v - np.outer(s, d), axis=1)
    return ss(s0 - soft, s0, s) * (1 - ss(r0, r0 + soft, r))


# where points may move: the front of the chest top, from a little below the block
# up to the crease against the head, fading out to the sides and into the arms
Y0, Y1, Y2, Y3 = -0.32, -0.20, -0.02, 0.05
f = ss(Y0, Y1, y) * (1 - ss(Y2, Y3, y))
f *= 1 - ss(np.radians(55), np.radians(85), np.abs(th))
for nm in ('arm.L', 'arm.R'):
    f *= 1 - limb(nm, 0.0, 0.22, 0.06)
f *= nohead
print('points reshaped:', int((f > 0.01).sum()), 'of', len(U))

# smooth outer surface: farthest point per (height, angle) bin, blurred
band = (y > Y0 - 0.06) & (y < Y3 + 0.04) & (np.abs(th) < np.radians(110)) & (nohead > 0.5)
for nm in ('arm.L', 'arm.R'):
    band &= limb(nm, -0.02, 0.24, 0.01) < 0.5
yb = np.arange(Y0 - 0.06, Y3 + 0.0401, 0.01)
tb = np.radians(np.arange(-110, 111, 4))
iy = np.clip(np.digitize(y, yb) - 1, 0, len(yb) - 2)
it = np.clip(np.digitize(th, tb) - 1, 0, len(tb) - 2)
S = np.zeros((len(yb) - 1, len(tb) - 1))
np.maximum.at(S, (iy[band], it[band]), rho[band])
C = (S > 0).astype(float)
Sw = gaussian_filter(S, (4.0, 4.0), mode='nearest')
Cw = gaussian_filter(C, (4.0, 4.0), mode='nearest')
Rs = np.where(Cw > 1e-3, Sw / np.maximum(Cw, 1e-3), 0)
yc = (yb[:-1] + yb[1:]) / 2
tc = (tb[:-1] + tb[1:]) / 2
grid = RegularGridInterpolator((yc, tc), Rs, bounds_error=False, fill_value=None)
rs = grid(np.stack([np.clip(y, yc[0], yc[-1]), np.clip(th, tc[0], tc[-1])], 1))
outer = rho > 0.85 * rs                  # leave the hidden inner layers alone
newrho = rho + f * outer * (rs - rho)
U[:, 0] = np.sin(th) * newrho
U[:, 2] = ZC + np.cos(th) * newrho
d = np.abs(newrho - rho)
print('change: mean %.4f (where it moves), largest %.4f' % (d[f > 0.5].mean(), d.max()))

# vertex normals (area weighted) on the welded mesh, written where the shape changed
fn = np.cross(U[T[:, 1]] - U[T[:, 0]], U[T[:, 2]] - U[T[:, 0]])
VN = np.zeros_like(U)
for k in range(3):
    np.add.at(VN, T[:, k], fn)
VN /= np.linalg.norm(VN, axis=1, keepdims=True) + 1e-12

Pn = U[inv].astype(np.float32)
Nn = N.astype(np.float32).copy()
ch = np.linalg.norm(U - P[uid], axis=1)[inv] > 1e-6
Nn[ch] = VN[inv][ch]
buf[poff:poff + Pn.nbytes] = Pn.tobytes()
buf[noff:noff + Nn.nbytes] = Nn.tobytes()
a = g['accessors'][at['POSITION']]
a['min'] = Pn.min(0).tolist()
a['max'] = Pn.max(0).tolist()

jb = json.dumps(g, separators=(',', ':')).encode()
jb += b' ' * (-len(jb) % 4)
bb = bytes(buf) + b'\0' * (-len(buf) % 4)
o = struct.pack('<III', magic, version, 12 + 8 + len(jb) + 8 + len(bb))
o += struct.pack('<II', len(jb), 0x4E4F534A) + jb + struct.pack('<II', len(bb), 0x004E4942) + bb
open(dst, 'wb').write(o)
print('wrote', dst, len(o), 'bytes')
