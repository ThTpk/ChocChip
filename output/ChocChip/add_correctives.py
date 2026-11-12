"""Add corrective shape keys (glTF morph targets) for the arm roots of the ChocChip plush.

When an arm is raised or swung forward, the skin between the arm and the body is
squeezed into a crease, so the body looks sunken under the arm or at the front of
the arm root. A stuffed plush instead bulges a little there. This script adds four
shape keys that push that skin outward, like stuffing pressed out of the seam:
  armL_up, armR_up    the underside of the arm root and the body side below it
  armL_fwd, armR_fwd  the front of the arm root and the chest beside it
Each key is a smooth bump along the surface normal, strongest at the arm root and
fading out along the arm, into the body and around to the other side. The web page
turns each key up as its arm rises or swings forward, so at rest nothing changes.
The keys are stored as sparse accessors (only the points that move), and their
names go in mesh.extras.targetNames. Everything else is copied as-is.

usage: python add_correctives.py IN.glb OUT.glb [strength]
"""
import json
import struct
import sys

import numpy as np
from scipy.spatial import cKDTree

src, dst = sys.argv[1], sys.argv[2]
AMP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
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
    return np.frombuffer(bytes(buf), CT[a['componentType']], a['count'] * NC[a['type']], off).reshape(a['count'], -1)


def ss(a, b, x):
    u = np.clip((x - a) / (b - a), 0, 1)
    return u * u * (3 - 2 * u)


mesh = g['meshes'][0]
prim = mesh['primitives'][0]
assert 'targets' not in prim, 'this file already has shape keys'
at = prim['attributes']
P = acc(at['POSITION']).astype(np.float64)
N = acc(at['NORMAL']).astype(np.float64)
names = [g['nodes'][j]['name'] for j in g['skins'][0]['joints']]
ibm = acc(g['skins'][0]['inverseBindMatrices']).reshape(-1, 4, 4)

# weld seam duplicates so a key never opens a crack along a UV seam
key = np.round(P / 1e-5).astype(np.int64)
_, uid, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
inv = inv.ravel()
U = P[uid]
NU = np.zeros_like(U)
np.add.at(NU, inv, N)
NU /= np.linalg.norm(NU, axis=1, keepdims=True) + 1e-12
tree = cKDTree(U)

# ---- pose keys, made in a posed shape. When an arm moves, the skin at its root is
# only part-carried by the arm and sticks out from the side of the body. The web page
# skins those points with plain linear blending (lbsMix below, the same as the page),
# which keeps most of it in; these keys do the rest. Each one is made like this: pose
# the arm with the page's own skinning, pull the skin that sticks out below the
# shoulder back in to the body wall, smooth, and turn the change back into the rest
# pose. The page blends the keys by how close the arm's direction is to each pose.
#   name: (raise, pitch, fwd) exactly as the page's arm() helper takes them
POSE_KEYS = {'mid': (60, 0, 0), 'high': (130, 0, 0), 'face': (115, 55, 0), 'front': (62, 75, 0),
             'beg': (-12, 0, 72), 'jab': (8, 0, 90), 'four': (48, 0, 72)}
Jall = acc(at['JOINTS_0']).astype(np.int64)
Wall = acc(at['WEIGHTS_0']).astype(np.float64)
tri = acc(prim['indices']).reshape(-1, 3).astype(np.int64)
T = inv[tri]
edges = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
edges = np.unique(np.sort(edges, 1), axis=0)
from scipy.sparse import coo_matrix
A = coo_matrix((np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
               shape=(len(U), len(U))).tocsr()
deg = np.asarray(A.sum(1)).ravel()


def qmul(a, b):
    w1, v1 = a[:, :1], a[:, 1:]
    w2, v2 = b[:, :1], b[:, 1:]
    return np.hstack([w1 * w2 - (v1 * v2).sum(1, keepdims=True), w1 * v2 + w2 * v1 + np.cross(v1, v2)])


def lbs_mix(w):
    """1 = plain linear blending, 0 = dual quaternions (same as the web page)"""
    return ss(0, 0.1, w) * (1 - ss(0.9, 1, w))


def rotm(ax, deg_):
    c, n = np.cos(np.radians(deg_)), np.sin(np.radians(deg_))
    if ax == 'x':
        return np.array([[1, 0, 0], [0, c, -n], [0, n, c]])
    if ax == 'y':
        return np.array([[c, 0, n], [0, 1, 0], [-n, 0, c]])
    return np.array([[c, -n, 0], [n, c, 0], [0, 0, 1]])


def arm_rot(sgn, raise_, pitch, fwd):
    """the page's arm(s, raise, pitch, fwd): about world Z, then X, then Y"""
    return rotm('y', -sgn * fwd) @ rotm('x', pitch) @ rotm('z', sgn * raise_)


def quat_of(Rm):
    w = np.sqrt(max(1e-12, 1 + np.trace(Rm))) / 2
    return np.array([w, (Rm[2, 1] - Rm[1, 2]) / (4 * w), (Rm[0, 2] - Rm[2, 0]) / (4 * w), (Rm[1, 0] - Rm[0, 1]) / (4 * w)])


def pose_key(nm, p, d, s, r, Rm, base):
    """shape key for this arm posed by rotation Rm; base = rest-space change already
    applied by the other keys in that pose (so the result lands where we want)"""
    bi = names.index(nm)
    w = (Wall * (Jall == bi)).sum(1)[uid]            # how much of each point the arm carries
    sgn = np.sign(p[0])
    qr = quat_of(Rm)
    t = p - Rm @ p
    qd = 0.5 * qmul(np.r_[0, t][None], qr[None])[0]
    real = (1 - w)[:, None] * np.array([1.0, 0, 0, 0]) + w[:, None] * qr
    dual = w[:, None] * qd
    n = np.linalg.norm(real, axis=1, keepdims=True)
    real, dual = real / n, dual / n
    rw, rv = real[:, :1], real[:, 1:]
    U0 = U + base
    Xd = U0 + 2 * np.cross(rv, np.cross(rv, U0) + rw * U0) + 2 * (rw * dual[:, 1:] - dual[:, :1] * rv + np.cross(rv, dual[:, 1:]))
    XL = (1 - w)[:, None] * U0 + w[:, None] * (U0 @ Rm.T + t)
    mix = lbs_mix(w)
    X = Xd + mix[:, None] * (XL - Xd)
    # each point's linear part, to turn posed changes back into the rest pose
    a_, b_, c_, d_ = real.T
    Rq = np.stack([np.stack([1 - 2 * (c_ * c_ + d_ * d_), 2 * (b_ * c_ - a_ * d_), 2 * (b_ * d_ + a_ * c_)], -1),
                   np.stack([2 * (b_ * c_ + a_ * d_), 1 - 2 * (b_ * b_ + d_ * d_), 2 * (c_ * d_ - a_ * b_)], -1),
                   np.stack([2 * (b_ * d_ - a_ * c_), 2 * (c_ * d_ + a_ * b_), 1 - 2 * (b_ * b_ + c_ * c_)], -1)], 1)
    Ml = (1 - w)[:, None, None] * np.eye(3) + w[:, None, None] * Rm
    Mx = (1 - mix)[:, None, None] * Rq + mix[:, None, None] * Ml
    # the arm root: the part-carried skin and a ring around it
    blend = ss(0.02, 0.15, w) * (1 - ss(0.85, 0.98, w))
    zone = ss(-0.20, -0.08, s) * (1 - ss(0.22, 0.32, s)) * (1 - ss(0.36, 0.46, r))
    m = np.clip(zone * (A @ blend / np.maximum(deg, 1) + blend), 0, 1)
    m = np.clip(m * 1.5, 0, 1)
    Y = X.copy()
    # pull the skin that sticks out below the shoulder back in to the body wall
    # (the body's own side at that height, from points the arm does not carry)
    body = (w < 0.03) & (s < 0.0) & (sgn * U[:, 0] > 0.05)
    yb = np.arange(-0.80, 0.10, 0.01)
    wall = np.full(len(yb), np.nan)
    ib = np.digitize(U[body, 1], yb) - 1
    for i in range(len(yb)):
        q = sgn * U[body, 0][ib == i]
        if len(q):
            wall[i] = q.max()
    ok = ~np.isnan(wall)
    wall = np.interp(yb, yb[ok], wall[ok])
    wall = np.convolve(np.pad(wall, 4, mode='edge'), np.ones(9) / 9, 'valid')
    wy = np.interp(Y[:, 1], yb, wall)
    below = ss(0.0, 0.08, p[1] - Y[:, 1])
    ex = sgn * Y[:, 0] - (wy + 0.02)
    inward = np.where(ex > 0, ex * 0.7, 0) * below * zone * (w < 0.98)
    Y[:, 0] -= sgn * inward
    m = np.maximum(m, np.clip(inward / 0.01, 0, 1) * zone)
    # Taubin smoothing (shrink, then grow back) irons out folds without deflating the root
    for _ in range(60):
        for lam in (0.5, -0.53):
            L = (A @ Y) / np.maximum(deg, 1)[:, None] - Y
            Y += lam * m[:, None] * L
    dX = Y - X
    D = np.zeros_like(dX)
    mv = np.linalg.norm(dX, axis=1) > 1e-6
    D[mv] = np.linalg.solve(Mx[mv], dX[mv][:, :, None])[:, :, 0]
    D[np.linalg.norm(D, axis=1) < 1e-5] = 0
    return D


targets = []
for nm, tag in (('arm.L', 'armL'), ('arm.R', 'armR')):
    m = np.linalg.inv(ibm[names.index(nm)].T)
    p, d = m[:3, 3], m[:3, 1] / np.linalg.norm(m[:3, 1])
    v = U - p
    s = v @ d                                   # along the arm, 0 at the shoulder joint
    r = np.linalg.norm(v - np.outer(s, d), axis=1)
    # (side, squeezed direction, centre and width along the arm, radius fade, strength, blend radius)
    # up:  only the armpit itself, and gently (a stronger, wider key made the sides of
    #      the body swell whenever the arms rose)
    # fwd: the front of the arm root and only the edge of the chest next to it
    #      (a wider key made the whole chest swell whenever an arm moved)
    for side, want, sc, sw, r0, r1, k, br in (
            ('up', np.array([0.0, -1.0, 0.0]), 0.10, 0.08, 0.22, 0.32, 0.5, 0.03),
            ('fwd', np.array([0.0, 0.0, 1.0]), 0.05, 0.12, 0.26, 0.38, 0.8, 0.035)):
        root = np.exp(-((s - sc) / sw) ** 2) * (1 - ss(r0, r1, r))
        # the side of the arm root that gets squeezed: below the arm when it rises,
        # in front of it when it swings forward
        e = want - (want @ d) * d
        e /= np.linalg.norm(e)
        h = v @ e
        f = root * ss(-0.03, 0.08, h)
        # stay on this arm's side of the chest and below the head
        f *= ss(0.10, 0.20, np.sign(p[0]) * U[:, 0]) * (1 - ss(-0.10, -0.02, U[:, 1]))
        dl = NU * (AMP * k * f)[:, None]
        # blend between neighbours so the bump is one smooth swell, not a copy of the scan's bumps
        nb = tree.query_ball_point(U, br)
        live = np.where(f > 1e-3)[0]
        sm = dl.copy()
        for i in live:
            sm[i] = dl[nb[i]].mean(0)
        targets.append((f'{tag}_{side}', sm[inv].astype(np.float32)))
        if side == 'up':
            upk = sm
        else:
            fwdk = sm
    sgn = np.sign(p[0])
    for kname, (ra, pi_, fw) in POSE_KEYS.items():
        Rm = arm_rot(sgn, ra, pi_, fw)
        dd = Rm @ d
        # the page's up / fwd drivers in this pose (the other pose keys are 0 here)
        up = float(ss(0, 1, np.clip((dd[1] - d[1] - 0.4) / 1.2, 0, 1)))
        fw_ = float(ss(0, 1, np.clip((dd[2] - d[2]) / 0.8, 0, 1)))
        targets.append((f'{tag}_{kname}', pose_key(nm, p, d, s, r, Rm, up * upk + fw_ * fwdk)[inv].astype(np.float32)))

# write each key as a sparse accessor: only the points that move
blob = bytearray()
acc_ids = []
for name, D in targets:
    idx = np.where(np.abs(D).max(1) > 1e-5)[0].astype(np.uint32)
    vals = D[idx]
    off_i = len(buf) + len(blob)
    blob += idx.tobytes()
    blob += b'\0' * (-len(blob) % 4)
    off_v = len(buf) + len(blob)
    blob += vals.astype(np.float32).tobytes()
    blob += b'\0' * (-len(blob) % 4)
    bvi = len(g['bufferViews']); g['bufferViews'].append({'buffer': 0, 'byteOffset': off_i, 'byteLength': idx.nbytes})
    bvv = len(g['bufferViews']); g['bufferViews'].append({'buffer': 0, 'byteOffset': off_v, 'byteLength': vals.nbytes})
    acc_ids.append(len(g['accessors']))
    g['accessors'].append({
        'componentType': 5126, 'count': len(P), 'type': 'VEC3',
        'min': np.minimum(D.min(0), 0).tolist(), 'max': np.maximum(D.max(0), 0).tolist(),
        'sparse': {'count': int(len(idx)),
                   'indices': {'bufferView': bvi, 'componentType': 5125},
                   'values': {'bufferView': bvv}}})
    print(f'{name}: {len(idx)} points, largest push {np.linalg.norm(D, axis=1).max():.3f}')

buf += blob
g['buffers'][0]['byteLength'] = len(buf)
prim['targets'] = [{'POSITION': a} for a in acc_ids]
mesh['weights'] = [0.0] * len(acc_ids)
mesh.setdefault('extras', {})['targetNames'] = [n for n, _ in targets]

jb = json.dumps(g, separators=(',', ':')).encode()
jb += b' ' * (-len(jb) % 4)
bb = bytes(buf) + b'\0' * (-len(buf) % 4)
o = struct.pack('<III', magic, version, 12 + 8 + len(jb) + 8 + len(bb))
o += struct.pack('<II', len(jb), 0x4E4F534A) + jb + struct.pack('<II', len(bb), 0x004E4942) + bb
open(dst, 'wb').write(o)
print('wrote', dst, len(o), 'bytes')
