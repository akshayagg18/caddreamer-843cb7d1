"""Minimal trimesh-backed shim for the small PyMesh API surface CADDreamer uses.

The real PyMesh is a heavy CGAL-backed C++ build. CADDreamer's actual B-rep
STEP generation goes through OpenCascade (neus/fit_and_intersection.py), NOT
PyMesh; PyMesh is only used by a couple of auxiliary mesh-export helpers in
neus/fit_surfaces/io_utils.py (form_mesh / merge_meshes / *_self_intersection /
separate_mesh / remove_duplicated_vertices). This shim provides those with
trimesh/numpy so the module imports and the OCC path runs. Self-intersection
"resolution" here is a no-op cleanup (trimesh has no exact CGAL arrangement),
which is fine because these helpers are off the STEP critical path.
"""
import numpy as np
import trimesh as _tri


class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        self._attrs = {}

    # PyMesh attribute API used by io_utils ("face_sources")
    def add_attribute(self, name):
        if name not in self._attrs:
            self._attrs[name] = np.zeros(len(self.faces), dtype=np.float64)

    def set_attribute(self, name, value):
        self._attrs[name] = np.asarray(value)

    def get_attribute(self, name):
        if name not in self._attrs:
            # default: each face maps to itself
            self._attrs[name] = np.arange(len(self.faces), dtype=np.float64)
        return self._attrs[name]

    def has_attribute(self, name):
        return name in self._attrs

    @property
    def num_vertices(self):
        return len(self.vertices)

    @property
    def num_faces(self):
        return len(self.faces)

    def to_trimesh(self):
        return _tri.Trimesh(vertices=self.vertices, faces=self.faces,
                            process=False)


def form_mesh(vertices, faces, *args, **kwargs):
    return Mesh(vertices, faces)


def merge_meshes(meshes):
    vs, fs, srcs = [], [], []
    off = 0
    for i, m in enumerate(meshes):
        vs.append(m.vertices)
        fs.append(m.faces + off)
        srcs.append(np.full(len(m.faces), i, dtype=np.float64))
        off += len(m.vertices)
    merged = Mesh(np.vstack(vs) if vs else np.zeros((0, 3)),
                  np.vstack(fs) if fs else np.zeros((0, 3), dtype=np.int64))
    merged.set_attribute("face_sources", np.concatenate(srcs) if srcs
                         else np.zeros(0))
    return merged


def detect_self_intersection(mesh):
    # exact detection needs CGAL; report none (off the STEP critical path)
    return np.zeros((0, 2), dtype=np.int64)


def resolve_self_intersection(mesh):
    # no-op resolution: pass the mesh through, preserving face_sources identity
    out = Mesh(mesh.vertices.copy(), mesh.faces.copy())
    out.set_attribute("face_sources",
                      np.arange(len(mesh.faces), dtype=np.float64))
    return out


def separate_mesh(mesh, *args, **kwargs):
    tm = mesh.to_trimesh()
    parts = tm.split(only_watertight=False)
    if len(parts) == 0:
        return [mesh]
    return [Mesh(p.vertices, p.faces) for p in parts]


def remove_duplicated_vertices(mesh, tol=1e-6, importance=None):
    tm = mesh.to_trimesh()
    tm.merge_vertices()
    out = Mesh(tm.vertices, tm.faces)
    return out, {"num_vertex_merged": int(mesh.num_vertices - out.num_vertices)}


def remove_duplicated_vertices_raw(vertices, faces, tol=1e-6):
    m = Mesh(vertices, faces)
    return remove_duplicated_vertices(m, tol=tol)
