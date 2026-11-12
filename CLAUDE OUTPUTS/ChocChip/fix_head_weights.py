"""Clean up the auto-generated skin weights of the ChocChip rig so the plush stays round.

The Blender auto-weights let the ear bones own ~40% of the skull and let the arm
bones reach the cheeks, so moving an ear or arm dents the face. This script:
  * keeps ear weights only on the ear bumps,
  * keeps arm weights only inside the arm's radius, fading in along the arm's
    root so raising an arm bends the arm itself instead of pulling the body side in,
  * keeps leg and tail.1 weights only inside the leg/tail tube,
  * removes spine weight above the neck,
and gives the removed weight to the head (or the spine below the neck), or to the
hips for the legs and tail, so the body moves as one round piece.
Only WEIGHTS_0 / JOINTS_0 bytes are rewritten; everything else is copied as-is.

usage: python fix_head_weights.py IN.glb OUT.glb
"""
import json
import struct
import sys

import numpy as np

src, dst = sys.argv[1], sys.argv[2]
# arm keep zone: r fades out between ARM[0]..ARM[1], s fades in between ARM[2]..ARM[3]
# (a wide fade, so the skin at the arm root blends from body to arm instead of folding)
ARM = [float(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else [0.20, 0.32, -0.04, 0.20]
# arm-root ramp: the arm weight rises smoothly from 0 to 1 between these distances along the arm
RAMP = [float(x) for x in sys.argv[4].split(',')] if len(sys.argv) > 4 else [-0.06, 0.34]
data = open(src, 'rb').read()
magic, version, _ = struct.unpack_from('<III', data, 0)
jlen, jtype = struct.unpack_from('<II', data, 12)
gltf = json.loads(data[20:20 + jlen])
boff = 20 + jlen
blen, btype = struct.unpack_from('<II', data, boff)
buf = bytearray(data[boff + 8: boff + 8 + blen])

CT = {5126: np.float32, 5123: np.uint16, 5121: np.uint8}
NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


def view(i):
    a = gltf['accessors'][i]
    bv = gltf['bufferViews'][a['bufferView']]
    assert bv.get('byteStride') in (None, np.dtype(CT[a['componentType']]).itemsize * NC[a['type']])
    off = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
    n = NC[a['type']]
    return np.frombuffer(buf, dtype=CT[a['componentType']], count=a['count'] * n, offset=off).reshape(a['count'], n), off


def ss(a, b, x):
    u = np.clip((x - a) / (b - a), 0, 1)
    return u * u * (3 - 2 * u)


skin = gltf['skins'][0]
names = [gltf['nodes'][k]['name'] for k in skin['joints']]
J = {n: i for i, n in enumerate(names)}
ibm, _ = view(skin['inverseBindMatrices'])
bind = [np.linalg.inv(m.reshape(4, 4).T) for m in ibm]  # glTF matrices are column-major


def frame(n):
    m = bind[J[n]]
    d = m[:3, 1] / np.linalg.norm(m[:3, 1])
    return m[:3, 3], d


for mesh in gltf['meshes']:
    for prim in mesh['primitives']:
        at = prim['attributes']
        if 'WEIGHTS_0' not in at:
            continue
        P, _ = view(at['POSITION'])
        Jv, joff = view(at['JOINTS_0'])
        Wv, woff = view(at['WEIGHTS_0'])
        nv = len(P)
        M = np.zeros((nv, len(names)))
        for c in range(4):
            np.add.at(M, (np.arange(nv), Jv[:, c].astype(int)), Wv[:, c].astype(float))
        before = M.copy()

        removed = np.zeros(nv)      # weight handed to head (above neck) / spine (below)
        to_hips = np.zeros(nv)      # weight handed to the hips
        # (bone, keep(s, r), receiver): s = distance along the bone, r = distance from its axis
        rules = [(e, lambda s, r: ss(0.06, 0.16, s) * (1 - ss(0.14, 0.22, r)), 'auto') for e in ('ear.L', 'ear.R')] +                 [(a, lambda s, r: (1 - ss(ARM[0], ARM[1], r)) * ss(ARM[2], ARM[3], s), 'auto') for a in ('arm.L', 'arm.R')] +                 [(g, lambda s, r: (1 - ss(0.20, 0.26, r)) * ss(-0.06, 0.04, s), 'hips') for g in ('leg.L', 'leg.R')] +                 [('tail.1', lambda s, r: (1 - ss(0.15, 0.22, r)) * ss(-0.02, 0.08, s), 'hips')]
        for name, keep, recv in rules:
            p, d = frame(name)
            v = P - p
            s = v @ d
            r = np.linalg.norm(v - np.outer(s, d), axis=1)
            k = J[name]
            nw = M[:, k] * keep(s, r)
            if recv == 'hips':
                to_hips += M[:, k] - nw
            else:
                removed += M[:, k] - nw
            M[:, k] = nw
        k = J['spine']
        nw = M[:, k] * (1 - ss(0.04, 0.14, P[:, 1]))
        removed += M[:, k] - nw
        M[:, k] = nw
        hf = ss(-0.05, 0.08, P[:, 1])
        M[:, J['head']] += removed * hf
        M[:, J['spine']] += removed * (1 - hf)
        M[:, J['hips']] += to_hips
        removed += to_hips

        # torso: share the hips/spine weight by height along one long, even ramp, so a
        # bending pose curves the chest and belly smoothly instead of folding them at
        # one line (the auto-weights switch from hips to spine within ~0.1)
        hs = M[:, J['hips']] + M[:, J['spine']]
        torso = hs > 0.5
        up = ss(-0.62, -0.05, P[:, 1])
        M[torso, J['hips']] = hs[torso] * (1 - up[torso])
        M[torso, J['spine']] = hs[torso] * up[torso]

        # arm root: the auto-weights jump from ~0.5 to ~1 within a few cm partway along
        # the arm, so a high raise bends the whole arm at that one line and the skin folds.
        # Inside the arm's tube (below the head) the arm weight is replaced by one long,
        # smooth ramp along the arm, so the bend spreads over the whole root; the other
        # bones give up (or take back) weight in proportion.
        for name in ('arm.L', 'arm.R'):
            p, d = frame(name)
            v = P - p
            s = v @ d
            r = np.linalg.norm(v - np.outer(s, d), axis=1)
            k = J[name]
            # the tube around the arm, minus the head (points the head bone owns)
            zone = (1 - ss(ARM[0], ARM[1], r)) * (1 - ss(0.3, 0.6, M[:, J['head']]))
            old = M[:, k].copy()
            # before the ramp's end follow the ramp; past it the arm keeps its own weight
            new = np.where(s < RAMP[1], ss(RAMP[0], RAMP[1], s) * zone, old)
            rest_old = 1 - old
            scale = np.where(rest_old > 1e-6, (1 - new) / np.maximum(rest_old, 1e-6), 0)
            for j in range(len(names)):
                if j != k:
                    M[:, j] *= scale
            # a point that had only arm weight and now loses some gives it to the spine
            lost = (rest_old <= 1e-6) & (new < old)
            M[lost, J['spine']] += old[lost] - new[lost]
            M[:, k] = new
            removed += np.abs(new - old)

        # back to 4 influences per vertex, normalised
        top = np.argsort(-M, axis=1)[:, :4]
        tw = np.take_along_axis(M, top, axis=1)
        tw[tw < 1e-4] = 0
        tw /= tw.sum(axis=1, keepdims=True)
        top[tw == 0] = 0
        buf[joff:joff + Jv.nbytes] = top.astype(Jv.dtype).tobytes()
        buf[woff:woff + Wv.nbytes] = tw.astype(Wv.dtype).tobytes()

        head_zone = P[:, 1] > 0.05
        hj = [J['head'], J['ear.L'], J['ear.R']]
        for label, Mx in (('before', before), ('after', M)):
            print(f"{label}: head weight on head-zone verts avg {Mx[head_zone, J['head']].mean():.3f}; "
                  f"arm influence on head verts {((Mx[head_zone, J['arm.L']] + Mx[head_zone, J['arm.R']]) > 0.05).sum()}; "
                  f"ear-owned head verts (>0.05) {((Mx[head_zone, J['ear.L']] + Mx[head_zone, J['ear.R']]) > 0.05).sum()}")
        print('changed vertices:', int((removed > 1e-4).sum()), 'of', nv)

out = bytearray()
jb = json.dumps(gltf, separators=(',', ':')).encode()
jb += b' ' * (-len(jb) % 4)
bb = bytes(buf) + b'\0' * (-len(buf) % 4)
out += struct.pack('<III', magic, version, 12 + 8 + len(jb) + 8 + len(bb))
out += struct.pack('<II', len(jb), 0x4E4F534A) + jb
out += struct.pack('<II', len(bb), 0x004E4942) + bb
open(dst, 'wb').write(out)
print('wrote', dst, len(out), 'bytes')
