"""Fill the creases where the arms join the body of the ChocChip plush.

In the scan, each arm is a separate lobe pressed against the body, so at the front
and back of every arm root there is a deep, sharp V crease, and the body wall that
runs into it is flat, which reads as sunken flesh. A sewn, firmly stuffed plush has
a round web there instead. This script, slice by slice through the height of the
arm roots:
  * fills the cross-section, then "closes" it with a disc (morphological closing),
    which bridges every crease narrower than the disc with a smooth round curve,
  * moves the outer-skin points that lie inside such a crease out onto that curve.
The moves are blended between neighbouring points so the web is smooth in height
too, and fade out above the shoulder, below the armpit and away from the arm roots,
so the rest of the body keeps its shape. Seam duplicates are welded so no cracks
open. Only POSITION and NORMAL bytes (and the POSITION min/max) are rewritten.

usage: python round_arms.py IN.glb OUT.glb
"""
import json
import struct
import sys

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

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

key = np.round(P / 1e-5).astype(np.int64)
_, uid, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
inv = inv.ravel()
U = P[uid].copy()
T = inv[tri]
y = U[:, 1]

# how close each point is to an arm root (1 at the root, 0 well away from it)
near = np.zeros(len(U))
for nm in ('arm.L', 'arm.R'):
    m = np.linalg.inv(ibm[names.index(nm)].T)
    p, d = m[:3, 3], m[:3, 1] / np.linalg.norm(m[:3, 1])
    v = U - p
    s = v @ d
    r = np.linalg.norm(v - np.outer(s, d), axis=1)
    near = np.maximum(near, ss(-0.22, -0.10, s) * (1 - ss(0.20, 0.32, s)) * (1 - ss(0.34, 0.44, r)))
YLO, YHI = -0.40, -0.02
near *= ss(YLO, YLO + 0.10, y) * (1 - ss(YHI - 0.06, YHI, y))

RES = 0.004           # grid cell size
CLOSE = 0.075         # radius of the disc that rounds the creases
X0, Z0 = -0.95, -0.55
NX, NZ = int(1.9 / RES), int(1.3 / RES)
disc = np.hypot(*np.mgrid[-19:20, -19:20]) <= CLOSE / RES

disp = np.zeros((len(U), 2))
cand = near > 0.01
for yc in np.arange(YLO, YHI + 1e-9, 0.006):
    sl = np.abs(y - yc) < 0.0045
    if sl.sum() < 50:
        continue
    q = U[sl][:, [0, 2]]
    ix = np.clip(((q[:, 0] - X0) / RES).astype(int), 0, NX - 1)
    iz = np.clip(((q[:, 1] - Z0) / RES).astype(int), 0, NZ - 1)
    # outline of the cross-section: cut every triangle that crosses this height
    # (the vertices alone are too sparse where triangles are large)
    ty = U[T][:, :, 1]
    hit = (ty.min(1) < yc) & (ty.max(1) > yc)
    V3 = U[T[hit]]
    segs = []
    for i0, i1 in ((0, 1), (1, 2), (2, 0)):
        a0, a1 = V3[:, i0], V3[:, i1]
        cross = (a0[:, 1] - yc) * (a1[:, 1] - yc) < 0
        u = np.where(cross, (yc - a0[:, 1]) / np.where(cross, a1[:, 1] - a0[:, 1], 1), np.nan)
        segs.append(a0[:, [0, 2]] + u[:, None] * (a1[:, [0, 2]] - a0[:, [0, 2]]))
    S3 = np.stack(segs, 1)                          # 3 candidate points, 2 of them valid
    ok = ~np.isnan(S3[:, :, 0])
    S3 = np.where(ok[:, :, None], S3, np.nan)
    A2 = np.array([s[o][0] for s, o in zip(S3, ok)]); B2 = np.array([s[o][-1] for s, o in zip(S3, ok)])
    tt = np.linspace(0, 1, 6)[None, :, None]
    pts = (A2[:, None] + tt * (B2 - A2)[:, None]).reshape(-1, 2)
    wall = np.zeros((NX, NZ), bool)
    wall[np.clip(((pts[:, 0] - X0) / RES).astype(int), 0, NX - 1), np.clip(((pts[:, 1] - Z0) / RES).astype(int), 0, NZ - 1)] = True
    solid = ndimage.binary_fill_holes(ndimage.binary_dilation(wall))
    closed = ndimage.binary_closing(np.pad(solid, 25), structure=disc)[25:-25, 25:-25] | solid
    # depth of each cell inside the original section, and inside the closed one
    d_orig = ndimage.distance_transform_edt(solid) * RES
    d_cl, (nx_, nz_) = ndimage.distance_transform_edt(closed, return_indices=True)
    d_cl = d_cl * RES
    idx = np.where(sl & cand)[0]
    jx, jz = ix[cand[sl]], iz[cand[sl]]
    on_skin = d_orig[jx, jz] < 0.035          # outer skin (and crease walls), not the hidden inner layers
    buried = d_cl[jx, jz] > 0.006             # inside a crease the disc bridged
    k = on_skin & buried
    tx = X0 + (nx_[jx[k], jz[k]] + 0.5) * RES
    tz = Z0 + (nz_[jx[k], jz[k]] + 0.5) * RES
    disp[idx[k]] = np.stack([tx, tz], 1) - U[idx[k]][:, [0, 2]]

disp *= near[:, None]
mag = np.linalg.norm(disp, axis=1)
disp *= (np.minimum(mag, 0.09) / np.maximum(mag, 1e-9))[:, None]
# blend the moves between neighbours so the web is smooth up and down the crease
moving = np.where(near > 0.01)[0]
tree = cKDTree(U[moving])
nb = tree.query_ball_point(U[moving], 0.02)
dm = disp[moving]
nz = np.linalg.norm(dm, axis=1) > 1e-6
sm = np.zeros_like(dm)
for i, n in enumerate(nb):
    n = np.array(n)
    k = n[nz[n]]
    if len(k):
        sm[i] = dm[k].mean(0) * ss(0.1, 0.4, len(k) / len(n))
disp[moving] = sm
U[:, 0] += disp[:, 0]
U[:, 2] += disp[:, 1]
mag = np.linalg.norm(disp, axis=1)
print('points moved:', int((mag > 0.002).sum()), 'mean %.4f, largest %.4f' % (mag[mag > 0.002].mean(), mag.max()))

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
