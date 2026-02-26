import math
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pygame
from PIL import Image
from numba import njit


# ============================================================
#  Math primitives
# ============================================================

@dataclass(frozen=True)
class Vec2:
    """
    2D vector for texture coordinates (u, v) or 2D arithmetic.

    Note:
      - We keep this immutable (frozen) for safer math style.
      - Operations return new objects (no in-place changes).
    """
    x: float
    y: float

    def __add__(self, o): return Vec2(self.x + o.x, self.y + o.y)
    def __sub__(self, o): return Vec2(self.x - o.x, self.y - o.y)
    def __mul__(self, k: float): return Vec2(self.x * k, self.y * k)


@dataclass(frozen=True)
class Vec3:
    """
    3D vector for positions and normals.

    Used in:
      - OBJ vertices (positions)
      - OBJ normals (vn)
      - intermediate view-space coordinates for clipping
      - light direction (normalized)
    """
    x: float
    y: float
    z: float

    def __add__(self, o): return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)
    def __sub__(self, o): return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)
    def __mul__(self, k: float): return Vec3(self.x * k, self.y * k, self.z * k)

    def dot(self, o) -> float:
        """Dot product (scalar product)."""
        return self.x * o.x + self.y * o.y + self.z * o.z

    def cross(self, o):
        """Cross product (vector product)."""
        return Vec3(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x
        )

    def norm(self) -> float:
        """Euclidean length."""
        return math.sqrt(self.dot(self))

    def normalize(self):
        """Return normalized vector (length=1)."""
        n = self.norm()
        if n <= 1e-12:
            return Vec3(0.0, 0.0, 0.0)
        return self * (1.0 / n)


@dataclass(frozen=True)
class Vec4:
    """
    4D homogeneous vector.
    Used for matrix multiplication in 3D transforms and projections.
    """
    x: float
    y: float
    z: float
    w: float


class Mat4:
    """
    4x4 matrix (row-major).

    We use Mat4 for:
      - Model matrix (scale + rotation)
      - View matrix (camera translation)
      - Projection matrix (perspective)
      - Axonometric transform (isometric rotation)

    Multiplication:
      - Matrix * Matrix => Mat4
      - Matrix * Vec4   => Vec4
    """
    def __init__(self, m: Optional[List[List[float]]] = None):
        self.m = m if m is not None else [[0.0]*4 for _ in range(4)]

    @staticmethod
    def identity():
        """Create identity matrix."""
        m = Mat4()
        for i in range(4):
            m.m[i][i] = 1.0
        return m

    def __matmul__(self, o: "Mat4") -> "Mat4":
        """Matrix multiplication (Mat4 @ Mat4)."""
        r = Mat4()
        for i in range(4):
            for j in range(4):
                s = 0.0
                for k in range(4):
                    s += self.m[i][k] * o.m[k][j]
                r.m[i][j] = s
        return r

    def mul_vec4(self, v: Vec4) -> Vec4:
        """Multiply matrix by a Vec4 (Mat4 * Vec4)."""
        x = self.m[0][0]*v.x + self.m[0][1]*v.y + self.m[0][2]*v.z + self.m[0][3]*v.w
        y = self.m[1][0]*v.x + self.m[1][1]*v.y + self.m[1][2]*v.z + self.m[1][3]*v.w
        z = self.m[2][0]*v.x + self.m[2][1]*v.y + self.m[2][2]*v.z + self.m[2][3]*v.w
        w = self.m[3][0]*v.x + self.m[3][1]*v.y + self.m[3][2]*v.z + self.m[3][3]*v.w
        return Vec4(x, y, z, w)


def vec3_to_vec4(v: Vec3, w: float = 1.0) -> Vec4:
    """Convert Vec3 to homogeneous Vec4."""
    return Vec4(v.x, v.y, v.z, w)


# ============================================================
#  3D transforms
# ============================================================

def translate(tx, ty, tz) -> Mat4:
    """
    Translation matrix.

    Applies: (x, y, z) -> (x + tx, y + ty, z + tz)
    """
    m = Mat4.identity()
    m.m[0][3] = tx
    m.m[1][3] = ty
    m.m[2][3] = tz
    return m

def scale(sx, sy, sz) -> Mat4:
    """
    Scaling matrix.

    Applies: (x, y, z) -> (sx*x, sy*y, sz*z)
    """
    m = Mat4.identity()
    m.m[0][0] = sx
    m.m[1][1] = sy
    m.m[2][2] = sz
    return m

def rotate_x(a) -> Mat4:
    """Rotation around X axis by angle a (radians)."""
    c, s = math.cos(a), math.sin(a)
    m = Mat4.identity()
    m.m[1][1] = c
    m.m[1][2] = -s
    m.m[2][1] = s
    m.m[2][2] = c
    return m

def rotate_y(a) -> Mat4:
    """Rotation around Y axis by angle a (radians)."""
    c, s = math.cos(a), math.sin(a)
    m = Mat4.identity()
    m.m[0][0] = c
    m.m[0][2] = s
    m.m[2][0] = -s
    m.m[2][2] = c
    return m

def rotate_z(a) -> Mat4:
    """Rotation around Z axis by angle a (radians)."""
    c, s = math.cos(a), math.sin(a)
    m = Mat4.identity()
    m.m[0][0] = c
    m.m[0][1] = -s
    m.m[1][0] = s
    m.m[1][1] = c
    return m


# ============================================================
#  Projections
# ============================================================

def perspective(fov_y, aspect, z_near, z_far) -> Mat4:
    """
    Perspective projection matrix.

    Parameters:
      fov_y  - vertical field of view in radians
      aspect - width / height
      z_near - near plane distance (positive)
      z_far  - far plane distance (positive)

    Notes:
      - In our pipeline, camera looks towards -Z in view space.
      - This projection produces clip-space with w = -z_view.
    """
    f = 1.0 / math.tan(fov_y / 2.0)
    m = Mat4()
    m.m[0][0] = f / aspect
    m.m[1][1] = f
    m.m[2][2] = (z_far + z_near) / (z_near - z_far)
    m.m[2][3] = (2.0 * z_far * z_near) / (z_near - z_far)
    m.m[3][2] = -1.0
    return m

def axonometric_isometric() -> Mat4:
    """
    Isometric (axonometric) viewing transform.

    Classic isometric orientation:
      rotate_x(35.264°) then rotate_y(45°)
    """
    return rotate_x(math.radians(35.264)) @ rotate_y(math.radians(45.0))


# ============================================================
#  OBJ loader
# ============================================================

@dataclass
class Face:
    """
    Single triangle face, indices into:
      - v:  vertex positions
      - vt: texture coords
      - vn: vertex normals

    Indices are 0-based (we subtract 1 when parsing OBJ).
    """
    v: Tuple[int, int, int]
    vt: Tuple[int, int, int]
    vn: Tuple[int, int, int]

class OBJModel:
    """
    Minimal OBJ parser for triangular meshes.

    Supported:
      v  x y z
      vt u v
      vn x y z
      f  v/vt/vn v/vt/vn v/vt/vn  (triangles only)

    If your OBJ includes quads, you must triangulate or extend parser.
    """
    def __init__(self, path: str):
        self.verts: List[Vec3] = []
        self.uvs: List[Vec2] = []
        self.normals: List[Vec3] = []
        self.faces: List[Face] = []
        self._load(path)

    def _load(self, path: str):
        """Read OBJ file and populate verts/uvs/normals/faces."""
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if parts[0] == "v" and len(parts) >= 4:
                    self.verts.append(Vec3(float(parts[1]), float(parts[2]), float(parts[3])))
                elif parts[0] == "vt" and len(parts) >= 3:
                    self.uvs.append(Vec2(float(parts[1]), float(parts[2])))
                elif parts[0] == "vn" and len(parts) >= 4:
                    self.normals.append(Vec3(float(parts[1]), float(parts[2]), float(parts[3])))
                elif parts[0] == "f" and len(parts) == 4:
                    v_idx, vt_idx, vn_idx = [], [], []
                    for i in range(1, 4):
                        comps = parts[i].split("/")
                        vi = int(comps[0]) - 1
                        vti = int(comps[1]) - 1 if len(comps) > 1 and comps[1] else -1
                        vni = int(comps[2]) - 1 if len(comps) > 2 and comps[2] else -1
                        v_idx.append(vi); vt_idx.append(vti); vn_idx.append(vni)
                    self.faces.append(Face(tuple(v_idx), tuple(vt_idx), tuple(vn_idx)))


def safe_uv(model: OBJModel, idx: int) -> Vec2:
    """Return UV by index, or (0,0) if missing."""
    if idx < 0 or idx >= len(model.uvs):
        return Vec2(0.0, 0.0)
    return model.uvs[idx]

def safe_n(model: OBJModel, idx: int) -> Vec3:
    """Return normal by index, or default (0,0,1) if missing."""
    if idx < 0 or idx >= len(model.normals):
        return Vec3(0.0, 0.0, 1.0)
    return model.normals[idx]


# ============================================================
#  Bresenham line
# ============================================================

def draw_line(x0, y0, x1, y1, set_pixel, color):
    """
    Bresenham integer line drawing.

    Parameters:
      x0, y0, x1, y1  - endpoints
      set_pixel(x,y,color) - callback for plotting
      color - RGB tuple

    Used in wireframe mode to draw triangle edges efficiently.
    """
    steep = abs(x0 - x1) < abs(y0 - y1)
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    dx = x1 - x0
    dy = abs(y1 - y0)
    error2 = 0
    y = y0
    ystep = 1 if y1 > y0 else -1

    for x in range(x0, x1 + 1):
        if steep:
            set_pixel(y, x, color)
        else:
            set_pixel(x, y, color)
        error2 += 2 * dy
        if error2 > dx:
            y += ystep
            error2 -= 2 * dx


# ============================================================
#  Utilities
# ============================================================

def to_screen(ndc_x, ndc_y, W, H):
    """
    Convert NDC coordinates [-1..1] to screen coordinates [0..W-1], [0..H-1].

    NDC:
      x=-1 left, x=+1 right
      y=-1 bottom, y=+1 top

    Screen:
      x=0 left, x=W-1 right
      y=0 top, y=H-1 bottom
    """
    sx = (ndc_x + 1.0) * 0.5 * (W - 1)
    sy = (1.0 - (ndc_y + 1.0) * 0.5) * (H - 1)
    return sx, sy

def transform_normal(mat_vm: Mat4, n: Vec3) -> Vec3:
    """
    Transform a normal vector by the upper-left 3x3 of vm matrix.

    Note:
      Correct for uniform scale. For non-uniform scale you should use
      inverse-transpose of the 3x3 part. For this assignment, uniform scale is OK.
    """
    x = mat_vm.m[0][0]*n.x + mat_vm.m[0][1]*n.y + mat_vm.m[0][2]*n.z
    y = mat_vm.m[1][0]*n.x + mat_vm.m[1][1]*n.y + mat_vm.m[1][2]*n.z
    z = mat_vm.m[2][0]*n.x + mat_vm.m[2][1]*n.y + mat_vm.m[2][2]*n.z
    return Vec3(x, y, z).normalize()


# ============================================================
#  Near-plane clipping
# ============================================================

def clip_poly_near(pts: List[Vec3], uvs: List[Vec2], nrms: List[Vec3], near_z: float):
    """
    Clip polygon against the near plane in VIEW space.

    View-space convention:
      - camera looks towards -Z
      - points in front of camera have z < 0
      - we keep points satisfying z <= near_z, where near_z is negative (e.g., -0.12)

    This prevents "holes" / "tearing" when triangles cross the near plane.

    Implementation:
      Sutherland–Hodgman clipping for a single plane.
      Attributes (UV, normal) are linearly interpolated at intersection.
    """
    out_pts: List[Vec3] = []
    out_uvs: List[Vec2] = []
    out_nrms: List[Vec3] = []

    def inside(p: Vec3) -> bool:
        return p.z <= near_z

    def lerp3(a: Vec3, b: Vec3, t: float) -> Vec3:
        return Vec3(a.x + (b.x-a.x)*t, a.y + (b.y-a.y)*t, a.z + (b.z-a.z)*t)

    def lerp2(a: Vec2, b: Vec2, t: float) -> Vec2:
        return Vec2(a.x + (b.x-a.x)*t, a.y + (b.y-a.y)*t)

    n = len(pts)
    for i in range(n):
        pA, pB = pts[i], pts[(i+1) % n]
        uvA, uvB = uvs[i], uvs[(i+1) % n]
        nA, nB = nrms[i], nrms[(i+1) % n]

        A_in, B_in = inside(pA), inside(pB)

        if A_in and B_in:
            # keep end vertex B
            out_pts.append(pB); out_uvs.append(uvB); out_nrms.append(nB)
        elif A_in and not B_in:
            # leaving visible region => add intersection only
            t = (near_z - pA.z) / (pB.z - pA.z)
            out_pts.append(lerp3(pA, pB, t))
            out_uvs.append(lerp2(uvA, uvB, t))
            out_nrms.append(lerp3(nA, nB, t).normalize())
        elif (not A_in) and B_in:
            # entering visible region => intersection + B
            t = (near_z - pA.z) / (pB.z - pA.z)
            out_pts.append(lerp3(pA, pB, t))
            out_uvs.append(lerp2(uvA, uvB, t))
            out_nrms.append(lerp3(nA, nB, t).normalize())
            out_pts.append(pB); out_uvs.append(uvB); out_nrms.append(nB)

    return out_pts, out_uvs, out_nrms

def triangulate_fan(poly_pts: List[Vec3], poly_uvs: List[Vec2], poly_nrms: List[Vec3]):
    """
    Convert a convex polygon (3..N vertices) into triangles using a fan:
      (0,1,2), (0,2,3), ..., (0,N-2,N-1)

    We use this after near-plane clipping, because the clipped polygon may have 4 vertices.
    """
    tris = []
    if len(poly_pts) < 3:
        return tris
    p0, uv0, n0 = poly_pts[0], poly_uvs[0], poly_nrms[0]
    for i in range(1, len(poly_pts)-1):
        tris.append((
            (p0, poly_pts[i], poly_pts[i+1]),
            (uv0, poly_uvs[i], poly_uvs[i+1]),
            (n0, poly_nrms[i], poly_nrms[i+1]),
        ))
    return tris


# ============================================================
#  Numba rasterizers
# ============================================================

@njit(cache=True)
def _barycentric(ax, ay, bx, by, cx, cy, px, py):
    """
    Compute barycentric coordinates for point (px,py) inside triangle (A,B,C)
    in 2D screen space.

    Returns (alpha, beta, gamma). If triangle is degenerate => (-1,-1,-1).
    """
    v0x, v0y = bx - ax, by - ay
    v1x, v1y = cx - ax, cy - ay
    v2x, v2y = px - ax, py - ay
    den = v0x * v1y - v1x * v0y
    if abs(den) < 1e-12:
        return -1.0, -1.0, -1.0
    inv = 1.0 / den
    beta = (v2x * v1y - v1x * v2y) * inv
    gamma = (v0x * v2y - v2x * v0y) * inv
    alpha = 1.0 - beta - gamma
    return alpha, beta, gamma


@njit(cache=True)
def draw_triangle_solid_flat(img, zbuf,
                             x0, y0, iw0,
                             x1, y1, iw1,
                             x2, y2, iw2,
                             r, g, b):
    """
    Rasterize a filled triangle with a constant color (no texture).

    Z-buffer:
      - we store inv_w (1/w) in zbuf for each pixel
      - pixel is drawn if invw > zbuf[x,y] (closer to camera)

    img:
      - pygame.surfarray.pixels3d -> shape (W,H,3), dtype=uint8
      - IMPORTANT: index order is [x,y,color] for pygame surfarray
    """
    W, H, _ = img.shape

    minx = max(0, int(math.floor(min(x0, x1, x2))))
    maxx = min(W - 1, int(math.ceil(max(x0, x1, x2))))
    miny = max(0, int(math.floor(min(y0, y1, y2))))
    maxy = min(H - 1, int(math.ceil(max(y0, y1, y2))))

    for y in range(miny, maxy + 1):
        py = y + 0.5
        for x in range(minx, maxx + 1):
            px = x + 0.5
            a, b0, c = _barycentric(x0, y0, x1, y1, x2, y2, px, py)
            if a < 0.0 or b0 < 0.0 or c < 0.0:
                continue

            invw = a*iw0 + b0*iw1 + c*iw2
            if invw <= zbuf[x, y]:
                continue
            zbuf[x, y] = invw

            img[x, y, 0] = r
            img[x, y, 1] = g
            img[x, y, 2] = b


@njit(cache=True)
def draw_triangle_tex_flat(img, zbuf, tex,
                           x0, y0, iw0, u0, v0,
                           x1, y1, iw1, u1, v1,
                           x2, y2, iw2, u2, v2,
                           intensity):
    """
    Rasterize a textured triangle with FLAT lighting (constant intensity per triangle).

    Texture mapping:
      - Uses perspective-correct interpolation for (u,v) via inv_w:
          invw = a*iw0 + b*iw1 + c*iw2
          u = (a*u0*iw0 + b*u1*iw1 + c*u2*iw2) / invw
          v = (a*v0*iw0 + b*v1*iw1 + c*v2*iw2) / invw

    Lighting:
      - intensity is constant for the triangle (computed in Python), then applied to texel RGB.
    """
    W, H, _ = img.shape
    th, tw, _ = tex.shape

    minx = max(0, int(math.floor(min(x0, x1, x2))))
    maxx = min(W - 1, int(math.ceil(max(x0, x1, x2))))
    miny = max(0, int(math.floor(min(y0, y1, y2))))
    maxy = min(H - 1, int(math.ceil(max(y0, y1, y2))))

    if intensity < 0.0:
        intensity = 0.0
    if intensity > 1.0:
        intensity = 1.0

    for y in range(miny, maxy + 1):
        py = y + 0.5
        for x in range(minx, maxx + 1):
            px = x + 0.5
            a, b0, c = _barycentric(x0, y0, x1, y1, x2, y2, px, py)
            if a < 0.0 or b0 < 0.0 or c < 0.0:
                continue

            invw = a*iw0 + b0*iw1 + c*iw2
            if invw <= zbuf[x, y]:
                continue
            zbuf[x, y] = invw

            uu = (a*u0*iw0 + b0*u1*iw1 + c*u2*iw2) / invw
            vv = (a*v0*iw0 + b0*v1*iw1 + c*v2*iw2) / invw

            # clamp UV to [0..1] to avoid out-of-bounds
            if uu < 0.0: uu = 0.0
            if uu > 1.0: uu = 1.0
            if vv < 0.0: vv = 0.0
            if vv > 1.0: vv = 1.0

            tx = int(uu * (tw - 1))
            ty = int((1.0 - vv) * (th - 1))

            img[x, y, 0] = int(tex[ty, tx, 0] * intensity)
            img[x, y, 1] = int(tex[ty, tx, 1] * intensity)
            img[x, y, 2] = int(tex[ty, tx, 2] * intensity)


@njit(cache=True)
def draw_triangle_tex_phong(img, zbuf, tex,
                            x0, y0, iw0, u0, v0, nx0, ny0, nz0,
                            x1, y1, iw1, u1, v1, nx1, ny1, nz1,
                            x2, y2, iw2, u2, v2, nx2, ny2, nz2,
                            lx, ly, lz):
    """
    Rasterize a textured triangle with per-pixel normal interpolation
    (Phong-style shading for diffuse term).

    Differences from flat:
      - normal is interpolated per pixel (perspective-correct), normalized
      - intensity computed per pixel as max(0, dot(N, L))

    Result:
      - smooth shading across triangles (reduces visible triangle facets).
    """
    W, H, _ = img.shape
    th, tw, _ = tex.shape

    minx = max(0, int(math.floor(min(x0, x1, x2))))
    maxx = min(W - 1, int(math.ceil(max(x0, x1, x2))))
    miny = max(0, int(math.floor(min(y0, y1, y2))))
    maxy = min(H - 1, int(math.ceil(max(y0, y1, y2))))

    for y in range(miny, maxy + 1):
        py = y + 0.5
        for x in range(minx, maxx + 1):
            px = x + 0.5
            a, b0, c = _barycentric(x0, y0, x1, y1, x2, y2, px, py)
            if a < 0.0 or b0 < 0.0 or c < 0.0:
                continue

            invw = a*iw0 + b0*iw1 + c*iw2
            if invw <= zbuf[x, y]:
                continue
            zbuf[x, y] = invw

            # perspective-correct UV
            uu = (a*u0*iw0 + b0*u1*iw1 + c*u2*iw2) / invw
            vv = (a*v0*iw0 + b0*v1*iw1 + c*v2*iw2) / invw

            # perspective-correct normal
            nnx = (a*nx0*iw0 + b0*nx1*iw1 + c*nx2*iw2) / invw
            nny = (a*ny0*iw0 + b0*ny1*iw1 + c*ny2*iw2) / invw
            nnz = (a*nz0*iw0 + b0*nz1*iw1 + c*nz2*iw2) / invw

            # normalize normal
            nlen = math.sqrt(nnx*nnx + nny*nny + nnz*nnz) + 1e-12
            nnx /= nlen; nny /= nlen; nnz /= nlen

            intensity = nnx*lx + nny*ly + nnz*lz
            if intensity < 0.0: intensity = 0.0
            if intensity > 1.0: intensity = 1.0

            # clamp UV
            if uu < 0.0: uu = 0.0
            if uu > 1.0: uu = 1.0
            if vv < 0.0: vv = 0.0
            if vv > 1.0: vv = 1.0

            tx = int(uu * (tw - 1))
            ty = int((1.0 - vv) * (th - 1))

            img[x, y, 0] = int(tex[ty, tx, 0] * intensity)
            img[x, y, 1] = int(tex[ty, tx, 1] * intensity)
            img[x, y, 2] = int(tex[ty, tx, 2] * intensity)


# ============================================================
#  Render modes (runtime toggles)
# ============================================================

RENDER_POINTS = 1
RENDER_WIREFRAME = 2
RENDER_SOLID_FLAT = 3
RENDER_TEX_FLAT = 4
RENDER_TEX_PHONG = 5


# ============================================================
#  Main loop
# ============================================================

def main():
    """
    Main interactive loop:
      - handle input
      - rebuild matrices
      - render selected mode each frame
    """

    obj_path = "assets/african_head.obj"
    tex_path = "assets/african_head_diffuse.tga"

    # --- Load model + texture
    model = OBJModel(obj_path)
    texture = Image.open(tex_path).convert("RGB")
    tex_np = np.array(texture, dtype=np.uint8)  # HxWx3

    # --- Window size
    W, H = 900, 900
    RW, RH = 900, 900  # render resolution

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Software Renderer: modes (1..5), P/O projections")
    render_surface = pygame.Surface((RW, RH))

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)

    # --- Pre-warm Numba (first call triggers compilation)
    dummy_img = np.zeros((4, 4, 3), dtype=np.uint8)
    dummy_z = np.full((4, 4), -1e9, dtype=np.float32)
    dummy_tex = np.zeros((2, 2, 3), dtype=np.uint8)
    draw_triangle_solid_flat(dummy_img, dummy_z, 0,0,1, 1,0,1, 0,1,1, 255,255,255)
    draw_triangle_tex_flat(dummy_img, dummy_z, dummy_tex, 0,0,1,0,0, 1,0,1,1,0, 0,1,1,0,1, 1.0)
    draw_triangle_tex_phong(dummy_img, dummy_z, dummy_tex,
                            0,0,1,0,0,0,0,1,
                            1,0,1,1,0,0,0,1,
                            0,1,1,0,1,0,0,1,
                            0,0,1)

    # --- Runtime state (camera/object/light)
    yaw = 0.0
    pitch = 0.0
    pos = Vec3(0.0, 0.0, 2.7)  # camera position (implemented as inverse translate)
    obj_scale = 1.25

    projection_perspective = True
    near_z = -0.12  # view-space near clip plane (negative)

    render_mode = RENDER_TEX_PHONG
    cull_backfaces = True

    light_yaw = 0.6
    light_pitch = 0.35

    dragging = False
    last_mouse = (0, 0)

    def light_dir() -> Vec3:
        """
        Build a unit directional light vector from yaw/pitch angles.
        This yields a consistent controllable light direction.
        """
        cy, sy = math.cos(light_yaw), math.sin(light_yaw)
        cp, sp = math.cos(light_pitch), math.sin(light_pitch)
        return Vec3(sy * cp, sp, cy * cp).normalize()

    def set_pixel_np(x, y, color):
        """
        Pixel setter for wireframe mode only.
        We use pygame surface plotting here since edges are relatively few pixels.
        """
        if 0 <= x < RW and 0 <= y < RH:
            render_surface.set_at((x, y), color)

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        # ====================================================
        #  Input handling
        # ====================================================
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                # Projection toggles
                elif event.key == pygame.K_p:
                    projection_perspective = True
                elif event.key == pygame.K_o:
                    projection_perspective = False

                # Render mode toggles
                elif event.key == pygame.K_1:
                    render_mode = RENDER_POINTS
                elif event.key == pygame.K_2:
                    render_mode = RENDER_WIREFRAME
                elif event.key == pygame.K_3:
                    render_mode = RENDER_SOLID_FLAT
                elif event.key == pygame.K_4:
                    render_mode = RENDER_TEX_FLAT
                elif event.key == pygame.K_5:
                    render_mode = RENDER_TEX_PHONG

                # Backface culling
                elif event.key == pygame.K_c:
                    cull_backfaces = not cull_backfaces

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging = True
                    last_mouse = event.pos

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False

            elif event.type == pygame.MOUSEMOTION and dragging:
                # Mouse drag rotates the model (yaw/pitch)
                mx, my = event.pos
                lx, ly = last_mouse
                dx, dy = mx - lx, my - ly
                last_mouse = (mx, my)
                yaw += dx * 0.01
                pitch += dy * 0.01
                pitch = max(-1.4, min(1.4, pitch))

        # Continuous key controls
        keys = pygame.key.get_pressed()

        # Camera movement
        speed = 1.8 * dt
        if keys[pygame.K_w]:
            pos = Vec3(pos.x, pos.y, pos.z - speed)
        if keys[pygame.K_s]:
            pos = Vec3(pos.x, pos.y, pos.z + speed)
        if keys[pygame.K_a]:
            pos = Vec3(pos.x - speed, pos.y, pos.z)
        if keys[pygame.K_d]:
            pos = Vec3(pos.x + speed, pos.y, pos.z)
        if keys[pygame.K_q]:
            pos = Vec3(pos.x, pos.y - speed, pos.z)
        if keys[pygame.K_e]:
            pos = Vec3(pos.x, pos.y + speed, pos.z)

        # Model scale
        if keys[pygame.K_z]:
            obj_scale = max(0.15, obj_scale - 0.9 * dt)
        if keys[pygame.K_x]:
            obj_scale = min(8.0, obj_scale + 0.9 * dt)

        # Light control
        if keys[pygame.K_j]:
            light_yaw -= 1.5 * dt
        if keys[pygame.K_l]:
            light_yaw += 1.5 * dt
        if keys[pygame.K_i]:
            light_pitch += 1.5 * dt
        if keys[pygame.K_k]:
            light_pitch -= 1.5 * dt
        light_pitch = max(-1.2, min(1.2, light_pitch))

        L = light_dir()

        # ====================================================
        #  Framebuffer setup
        # ====================================================
        render_surface.fill((10, 10, 18))

        img = None
        zbuf = None

        # For filled modes, we need direct access to pixel buffer and z-buffer
        if render_mode in (RENDER_SOLID_FLAT, RENDER_TEX_FLAT, RENDER_TEX_PHONG):
            img = pygame.surfarray.pixels3d(render_surface)  # shape: (RW,RH,3)
            img[:, :, :] = (10, 10, 18)
            zbuf = np.full((RW, RH), -1e9, dtype=np.float32)

        # ====================================================
        #  Build transform matrices
        # ====================================================
        # Model: rotate + scale (object space -> world-ish)
        model_m = rotate_y(yaw) @ rotate_x(pitch) @ scale(obj_scale, obj_scale, obj_scale)
        # View: inverse camera translation (world -> view)
        view_m = translate(-pos.x, -pos.y, -pos.z)
        # VM: object -> view
        vm = view_m @ model_m

        # Projection selection
        if projection_perspective:
            proj_m = perspective(math.radians(60.0), RW / RH, 0.1, 100.0)
            use_persp = True
        else:
            # Axonometric: rotate view space by isometric transform, then use orthographic mapping.
            vm = axonometric_isometric() @ vm
            proj_m = Mat4.identity()
            use_persp = False

        # ====================================================
        #  Render all faces
        # ====================================================
        for face in model.faces:
            i0, i1, i2 = face.v
            vt0, vt1, vt2 = face.vt
            vn0, vn1, vn2 = face.vn

            # object-space positions
            v0 = model.verts[i0]
            v1 = model.verts[i1]
            v2 = model.verts[i2]

            # view-space positions (for clipping)
            pv0_4 = vm.mul_vec4(vec3_to_vec4(v0))
            pv1_4 = vm.mul_vec4(vec3_to_vec4(v1))
            pv2_4 = vm.mul_vec4(vec3_to_vec4(v2))
            pv0 = Vec3(pv0_4.x, pv0_4.y, pv0_4.z)
            pv1 = Vec3(pv1_4.x, pv1_4.y, pv1_4.z)
            pv2 = Vec3(pv2_4.x, pv2_4.y, pv2_4.z)

            # vertex normals transformed to view space
            n0 = transform_normal(vm, safe_n(model, vn0))
            n1 = transform_normal(vm, safe_n(model, vn1))
            n2 = transform_normal(vm, safe_n(model, vn2))

            # texture coordinates
            uv0 = safe_uv(model, vt0)
            uv1 = safe_uv(model, vt1)
            uv2 = safe_uv(model, vt2)

            # Near-plane clipping to avoid tearing / holes
            poly_pts, poly_uvs, poly_nrms = clip_poly_near([pv0, pv1, pv2],
                                                           [uv0, uv1, uv2],
                                                           [n0, n1, n2],
                                                           near_z=near_z)
            if len(poly_pts) < 3:
                continue

            # Clipping can yield 3..4 vertices => triangulate
            tris = triangulate_fan(poly_pts, poly_uvs, poly_nrms)

            for (tp0, tp1, tp2), (tuv0, tuv1, tuv2), (tn0, tn1, tn2) in tris:
                # Project view-space points to clip space
                if use_persp:
                    c0 = proj_m.mul_vec4(Vec4(tp0.x, tp0.y, tp0.z, 1.0))
                    c1 = proj_m.mul_vec4(Vec4(tp1.x, tp1.y, tp1.z, 1.0))
                    c2 = proj_m.mul_vec4(Vec4(tp2.x, tp2.y, tp2.z, 1.0))

                    # Perspective divide requires w > 0
                    if c0.w <= 1e-6 or c1.w <= 1e-6 or c2.w <= 1e-6:
                        continue

                    ndc0x, ndc0y = c0.x / c0.w, c0.y / c0.w
                    ndc1x, ndc1y = c1.x / c1.w, c1.y / c1.w
                    ndc2x, ndc2y = c2.x / c2.w, c2.y / c2.w

                    # inv_w for z-buffer + perspective-correct interpolation
                    iw0 = 1.0 / c0.w
                    iw1 = 1.0 / c1.w
                    iw2 = 1.0 / c2.w
                else:
                    # Axonometric: treat view coords as NDC-ish with scaling
                    s = 0.7
                    ndc0x, ndc0y = tp0.x * s, tp0.y * s
                    ndc1x, ndc1y = tp1.x * s, tp1.y * s
                    ndc2x, ndc2y = tp2.x * s, tp2.y * s
                    iw0 = 1.0
                    iw1 = 1.0
                    iw2 = 1.0

                # NDC -> screen pixels
                sx0, sy0 = to_screen(ndc0x, ndc0y, RW, RH)
                sx1, sy1 = to_screen(ndc1x, ndc1y, RW, RH)
                sx2, sy2 = to_screen(ndc2x, ndc2y, RW, RH)

                # trivial reject if triangle bbox is fully off-screen
                if (max(sx0, sx1, sx2) < 0) or (min(sx0, sx1, sx2) > RW-1) or \
                   (max(sy0, sy1, sy2) < 0) or (min(sy0, sy1, sy2) > RH-1):
                    continue

                # Backface culling in screen space by signed area
                area2 = (sx1 - sx0) * (sy2 - sy0) - (sy1 - sy0) * (sx2 - sx0)
                if cull_backfaces and area2 >= 0:
                    continue

                # ====================================================
                #  Mode dispatch
                # ====================================================
                if render_mode == RENDER_POINTS:
                    # vertices only
                    render_surface.set_at((int(sx0), int(sy0)), (255, 255, 255))
                    render_surface.set_at((int(sx1), int(sy1)), (255, 255, 255))
                    render_surface.set_at((int(sx2), int(sy2)), (255, 255, 255))

                elif render_mode == RENDER_WIREFRAME:
                    # edges only (Bresenham)
                    col = (200, 200, 220)
                    draw_line(int(sx0), int(sy0), int(sx1), int(sy1), set_pixel_np, col)
                    draw_line(int(sx1), int(sy1), int(sx2), int(sy2), set_pixel_np, col)
                    draw_line(int(sx2), int(sy2), int(sx0), int(sy0), set_pixel_np, col)

                else:
                    # filled modes require framebuffer + z-buffer
                    if img is None or zbuf is None:
                        continue

                    if render_mode == RENDER_SOLID_FLAT:
                        # Flat normal for triangle (average vertex normals)
                        nn = Vec3(tn0.x + tn1.x + tn2.x,
                                  tn0.y + tn1.y + tn2.y,
                                  tn0.z + tn1.z + tn2.z).normalize()
                        intensity = max(0.0, min(1.0, nn.dot(L)))
                        shade = int(255 * intensity)
                        draw_triangle_solid_flat(img, zbuf,
                                                 sx0, sy0, iw0,
                                                 sx1, sy1, iw1,
                                                 sx2, sy2, iw2,
                                                 shade, shade, shade)

                    elif render_mode == RENDER_TEX_FLAT:
                        nn = Vec3(tn0.x + tn1.x + tn2.x,
                                  tn0.y + tn1.y + tn2.y,
                                  tn0.z + tn1.z + tn2.z).normalize()
                        intensity = nn.dot(L)  # flat per-triangle
                        draw_triangle_tex_flat(img, zbuf, tex_np,
                                               sx0, sy0, iw0, tuv0.x, tuv0.y,
                                               sx1, sy1, iw1, tuv1.x, tuv1.y,
                                               sx2, sy2, iw2, tuv2.x, tuv2.y,
                                               intensity)

                    elif render_mode == RENDER_TEX_PHONG:
                        draw_triangle_tex_phong(img, zbuf, tex_np,
                                                sx0, sy0, iw0, tuv0.x, tuv0.y, tn0.x, tn0.y, tn0.z,
                                                sx1, sy1, iw1, tuv1.x, tuv1.y, tn1.x, tn1.y, tn1.z,
                                                sx2, sy2, iw2, tuv2.x, tuv2.y, tn2.x, tn2.y, tn2.z,
                                                L.x, L.y, L.z)

        # Delete img view to unlock surface for blitting
        if img is not None:
            del img

        # ====================================================
        #  Present frame
        # ====================================================
        if (RW, RH) != (W, H):
            pygame.transform.scale(render_surface, (W, H), screen)
        else:
            screen.blit(render_surface, (0, 0))

        proj_name = "PERSPECTIVE (P)" if projection_perspective else "AXONOMETRIC (O)"
        mode_name = {
            RENDER_POINTS: "POINTS (1)",
            RENDER_WIREFRAME: "WIREFRAME (2)",
            RENDER_SOLID_FLAT: "SOLID FLAT (3)",
            RENDER_TEX_FLAT: "TEX FLAT (4)",
            RENDER_TEX_PHONG: "TEX PHONG (5)",
        }[render_mode]

        hud = [
            f"{proj_name} | {mode_name} | Cull(C): {cull_backfaces} | Faces: {len(model.faces)} | FPS: {clock.get_fps():.1f}",
            "P/O proj | 1..5 modes | LMB drag rotate | WASD/QE move | Z/X scale | IJKL light | ESC exit",
        ]
        y = 10
        for line in hud:
            screen.blit(font.render(line, True, (235, 235, 235)), (10, y))
            y += 18

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()