"""Round off the chest and belly of the ChocChip plush.

Seen from the side, the scan's chest and belly drop straight down like a cylinder,
so in motion the front reads as a square slab sticking out of the belly. This script
reshapes the front of the torso, in cylinder coordinates around its vertical axis:
  * moves the outer skin onto a smooth surface (the distance from the axis averaged
    over a wide band of angle and height), which rounds corners and irons out the
    scan's fabric bumps, from the collar down to where the belly curves under,
  * adds a wide bulge around the belly that wraps round the sides, for a chubby plush tummy.
Only the outer skin moves: the scan has hidden inner layers under the belly, and
general mesh smoothing pushes the skin through them and crumples the surface.
The change fades out towards the head, arms, legs and back, so those stay as they
are. Duplicate vertices along UV seams are welded while reshaping so no cracks open.
Only POSITION and NORMAL bytes (and the POSITION min/max) are rewritten.

usage: python round_belly.py IN.glb OUT.glb
"""
import json
import struct
import sys

import numpy as np

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
nu = len(U)
T = inv[tri]

x, y, z = U[:, 0], U[:, 1], U[:, 2]
ZC = -0.03
th = np.arctan2(x, z - ZC)          # 0 = straight ahead, +-90 deg = the sides
rho = np.hypot(x, z - ZC)           # distance from the torso's vertical axis
# the tail curls round beside the body: keep it out of all of this
Jw, Ww = acc(at['JOINTS_0'])[0], acc(at['WEIGHTS_0'])[0].astype(float)
tailw = np.zeros(len(P))
for nm in names:
    if nm.startswith('tail'):
        tailw += (Ww * (Jw == names.index(nm))).sum(1)
notail = 1 - ss(0.05, 0.3, tailw[uid])


def limb(nm, s0, r0, soft):
    """0..1: how much a point belongs to the root/tube of a limb bone."""
    m = np.linalg.inv(ibm[names.index(nm)].T)
    p, d = m[:3, 3], m[:3, 1] / np.linalg.norm(m[:3, 1])
    v = U - p
    s = v @ d
    r = np.linalg.norm(v - np.outer(s, d), axis=1)
    w = ss(s0 - soft, s0, s) * (1 - ss(r0, r0 + soft, r))
    if nm.startswith('arm'):
        # below the shoulder the arm hangs free of the body: there only points out
        # beyond the torso's side are arm; the torso's own side wall is not
        w *= np.maximum(ss(0.44, 0.50, rho), ss(-0.30, -0.24, y))
    return w


# how much each point may move: 1 on the chest, belly and sides, fading out towards
# the back, the head, the arm roots and the legs
f = ss(-0.80, -0.64, y) * (1 - ss(-0.14, -0.04, y)) * ss(np.radians(-165), np.radians(-125), th) * (1 - ss(np.radians(125), np.radians(165), th))
for nm, s0, r0 in (('arm.L', -0.04, 0.24), ('arm.R', -0.04, 0.24), ('leg.L', 0.10, 0.22), ('leg.R', 0.10, 0.22)):
    f *= 1 - limb(nm, s0, r0, 0.07)
f *= notail
print('points reshaped:', int((f > 0.01).sum()), 'of', nu)

# --- large-scale rounding: in cylinder coordinates around the torso's vertical axis,
# the scan's fabric seams and wrinkles add bumps and the front has flat parts and
# corners. Average rho over a wide band of angle and height and move the points onto
# that smooth surface, so the torso becomes one smooth round piece like a stuffed toy.
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
torso = (y > -0.85) & (y < -0.08) & (notail > 0.5)   # stay below the big head
for nm, s0, r0 in (('arm.L', -0.06, 0.26), ('arm.R', -0.06, 0.26), ('leg.L', 0.05, 0.24), ('leg.R', 0.05, 0.24)):
    torso &= limb(nm, s0, r0, 0.01) < 0.5
yb = np.arange(-0.85, -0.079, 0.02)
tb_ = np.radians(np.arange(-180, 181, 5))
iy = np.clip(np.digitize(y, yb) - 1, 0, len(yb) - 2)
it = np.clip(np.digitize(th, tb_) - 1, 0, len(tb_) - 2)
S = np.zeros((len(yb) - 1, len(tb_) - 1))
C = np.zeros_like(S)
# the outer surface only: per bin keep the farthest points (inner folds would pull rho in)
np.maximum.at(S, (iy[torso], it[torso]), rho[torso])
C[S > 0] = 1
Sw = gaussian_filter(S, (3.0, 4.0), mode=('nearest', 'wrap'))
Cw = gaussian_filter(C, (3.0, 4.0), mode=('nearest', 'wrap'))
Rs = np.where(Cw > 1e-3, Sw / np.maximum(Cw, 1e-3), 0)
yc = (yb[:-1] + yb[1:]) / 2
tc = (tb_[:-1] + tb_[1:]) / 2
grid = RegularGridInterpolator((yc, tc), Rs, bounds_error=False, fill_value=None)
rs = grid(np.stack([np.clip(y, yc[0], yc[-1]), np.clip(th, tc[0], tc[-1])], 1))
# only move points that are on the outer skin (close to the bin's outer radius)
outer = rho > 0.85 * rs
newrho = rho + f * outer * (rs - rho)

# --- chubby: a wide bulge around the middle of the belly that wraps round to the
# sides (full at the front, still 3/4 at the sides, none on the back), so the
# tummy is plump all round instead of only at the front.
BULGE = 0.09
bell = np.exp(-0.5 * ((y + 0.40) / 0.20) ** 2)
around = np.clip(0.5 + 0.5 * np.cos(th), 0, 1) ** 0.4 * (1 - ss(np.radians(110), np.radians(150), np.abs(th)))
grow = BULGE * bell * around
body = torso * notail
# fade near the arm and leg roots so nothing pulls apart there
for nm, s0, r0 in (('arm.L', -0.06, 0.26), ('arm.R', -0.06, 0.26), ('leg.L', 0.0, 0.24), ('leg.R', 0.0, 0.24)):
    body *= 1 - limb(nm, s0, r0, 0.09)
newrho = newrho + grow * body * outer
U[:, 0] = np.sin(th) * newrho
U[:, 2] = ZC + np.cos(th) * newrho
print('rounding: mean |change| on the belly %.4f, largest %.4f' % (np.abs(newrho - rho)[f > 0.5].mean(), np.abs(newrho - rho).max()))


# vertex normals (area weighted) on the welded mesh, written where the shape changed
fn = np.cross(U[T[:, 1]] - U[T[:, 0]], U[T[:, 2]] - U[T[:, 0]])
VN = np.zeros_like(U)
for k in range(3):
    np.add.at(VN, T[:, k], fn)
VN /= np.linalg.norm(VN, axis=1, keepdims=True) + 1e-12

Pn = U[inv].astype(np.float32)
Nn = N.astype(np.float32).copy()
ch = np.linalg.norm(U - P[uid], axis=1)[inv] > 1e-6   # vertices whose shape changed
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
