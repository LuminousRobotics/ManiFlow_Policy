"""Round-trip label-consistency test for the _augment_so2 co-rotation math.
Replicates the EXACT transform used in the dataset and asserts:
 (A) a camera-frame goal position projects to the SAME pixel the image rotation moves it to;
 (B) co-rotating a rotvec by R_z(a) equals the correct camera-frame conjugation of that rotation.
"""
import numpy as np
import torch
import torchvision.transforms.functional as TF

S, f = 128, 200.0
c = (S - 1) / 2.0
def project(P): return np.array([f * P[0] / P[2] + c, f * P[1] / P[2] + c])
def blob(u, v):
    yy, xx = np.mgrid[0:S, 0:S]
    return np.exp(-(((xx - u) ** 2 + (yy - v) ** 2) / (2 * 2.0 ** 2))).astype(np.float32)
def centroid(im):
    yy, xx = np.mgrid[0:S, 0:S]; w = im / (im.sum() + 1e-9)
    return np.array([(xx * w).sum(), (yy * w).sum()])
def rodrigues(rv):
    th = np.linalg.norm(rv)
    if th < 1e-9: return np.eye(3)
    k = rv / th; K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3) + np.sin(th)*K + (1-np.cos(th))*(K@K)
def rotvec_of(R):
    ang = np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))
    if ang < 1e-9: return np.zeros(3)
    return ang/(2*np.sin(ang)) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])

# ---- EXACT transform from _augment_so2 (SO2_SIGN = -1) ----
def co_rotate(arr6, phi_deg):
    a = np.radians(-phi_deg)
    ca, sa = np.cos(a), np.sin(a)
    Rz_T = np.array([[ca, sa, 0.0], [-sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)  # R_z(a).T
    out = arr6.copy()
    out[..., :3] = arr6[..., :3] @ Rz_T
    out[..., 3:] = arr6[..., 3:] @ Rz_T
    return out, a

phi_deg = 12.0
# (A) position round-trip
gpos = np.array([0.15, 0.08, 1.0], dtype=np.float32)
gvec = np.concatenate([gpos, np.zeros(3, np.float32)])
p0 = project(gpos)
p_img = centroid(TF.rotate(torch.from_numpy(blob(*p0))[None,None], angle=phi_deg,
                           interpolation=TF.InterpolationMode.BILINEAR)[0,0].numpy())
gvec_rot, a = co_rotate(gvec, phi_deg)
p_lab = project(gvec_rot[:3])
errA = np.linalg.norm(p_img - p_lab)
print(f"(A) position: image-rotated pixel {p_img.round(2)} vs label-projected {p_lab.round(2)}  err={errA:.3f}px")

# (B) rotvec conjugation identity: co_rotate's R_z(a)@rv  ==  rotvec(R_z(a) @ R(rv) @ R_z(a).T)
rv = np.array([0.05, -0.12, 0.30], dtype=np.float32)
vec = np.concatenate([np.zeros(3, np.float32), rv])
vec_rot, a = co_rotate(vec, phi_deg)
rv_co = vec_rot[3:]
ca, sa = np.cos(a), np.sin(a)
Rz = np.array([[ca,-sa,0],[sa,ca,0],[0,0,1.0]])
rv_conj = rotvec_of(Rz @ rodrigues(rv) @ Rz.T)
errB = np.linalg.norm(rv_co - rv_conj)
print(f"(B) rotvec: co_rotate {rv_co.round(4)} vs true conjugation {rv_conj.round(4)}  err={errB:.5f}")

assert errA < 1.0, "position co-rotation NOT label-consistent with the image rotation"
assert errB < 1e-4, "rotvec co-rotation != camera-frame conjugation"
print("\nPASS: SO(2) co-rotation is label-consistent for BOTH position and rotation.")
