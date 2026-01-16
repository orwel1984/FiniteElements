import numpy as np
import warnings

import os
import numpy as np
import matplotlib.pyplot as plt

def readquad(ext=None, path="."):
    """
    Read quadrilateral FEM mesh from files in a given directory.

    Parameters
    ----------
    ext : str or None
        File extension (e.g. 'alt', 'sect')
    path : str
        Directory containing mesh files

    Returns
    -------
    nodes, xnod, ynod, bnod, nele, nno
    """

    ex = ""
    if ext is not None:
        ex = f".{ext}"

    def fname(base):
        return os.path.join(path, f"{base}{ex}")

    xnod = np.loadtxt(fname("xcoor"))
    ynod = np.loadtxt(fname("ycoor"))
    bnod = np.loadtxt(fname("bnode"), dtype=int)

    nno = xnod.size

    if ynod.size != nno or bnod.size != nno:
        raise ValueError("Global coordinates erroneous ...")

    if not np.all((bnod == 0) | (bnod == 1)):
        raise ValueError("Wrong boundary indicator ...")

    nodes_flat = np.loadtxt(fname("nodes"), dtype=int)

    if nodes_flat.size % 4 != 0:
        raise ValueError("Wrong number of numbers in nodes ...")

    nele = nodes_flat.size // 4
    nodes = nodes_flat.reshape((nele, 4))

    # MATLAB → Python indexing
    nodes -= 1

    nodes = meshtest(nodes, xnod, ynod)

    return nodes, xnod, ynod, bnod, nele, nno


def meshtest(nodes, xnod, ynod):
    """
    Apply tests to ensure quadrilateral mesh is admissible.
    Ensures all quads are convex and counterclockwise.
    
    Parameters
    ----------
    nodes : (nele, 4) ndarray of int (0-based)
        Element connectivity
    xnod, ynod : (nno,) ndarray
        Node coordinates

    Returns
    -------
    nodesn : (nele, 4) ndarray
        Possibly re-ordered nodes with positive orientation
    """

    nele = nodes.shape[0]
    
    # --- 1. Degeneracy checks ---
    for i in range(nele):
        if len(set(nodes[i, :])) < 4:
            raise ValueError("Degenerated quadrilateral ...")

    # --- 2. Convexity checks using barycentric coordinates ---
    for i in range(nele):
        quad = nodes[i, :]
        # List of triangles to test
        tris = [
            (quad[[0, 1, 2]], quad[3]),
            (quad[[0, 1, 3]], quad[2]),
            (quad[[0, 2, 3]], quad[1]),
            (quad[[1, 2, 3]], quad[0])
        ]
        for tri, j in tris:
            b = np.vstack([xnod[tri], ynod[tri], np.ones(3)])
            vec = np.array([xnod[j], ynod[j], 1.0])
            try:
                x = np.linalg.solve(b, vec)
            except np.linalg.LinAlgError:
                raise ValueError("Singular triangle in convexity test ...")
            if np.all(x > -1e-5):
                raise ValueError("Nonconvex quadrilateral ...")

    # --- 3. Severe nonconvexity checks (diagonal method) ---
    for i in range(nele):
        for pairs in [(0, 1, 2, 3), (1, 2, 3, 0)]:
            i1, i2, i3, i4 = [nodes[i, idx] for idx in pairs]
            b = np.array([
                [xnod[i1]-xnod[i3], xnod[i2]-xnod[i4]],
                [ynod[i1]-ynod[i3], ynod[i2]-ynod[i4]]
            ])
            if np.linalg.cond(b) < 1/1e-14:  # MATLAB rcond > 1e-14
                x = np.linalg.solve(b, [xnod[i4]-xnod[i3], ynod[i4]-ynod[i3]])
                if np.all((x >= 0) & (x <= 1)):
                    raise ValueError("Severely nonconvex quadrilateral ...")

    # --- 4. Orientation correction ---
    for i in range(nele):
        tri = nodes[i, [0, 1, 2]]
        mat = np.vstack([xnod[tri], ynod[tri], np.ones(3)])
        if np.linalg.det(mat) < 0:
            warnings.warn("Introducing positive orientation ...")
            # Swap nodes to make counterclockwise
            nodes[i, :] = nodes[i, [0, 3, 2, 1]]

    # --- 5. Double edge check (planarity) ---
    nno = nodes.max() + 1  # 0-based
    M = 10*nele + 1
    htable = np.zeros(M, dtype=int)
    nodesn = nodes.copy()
    nodesn_ext = np.hstack([nodesn, nodesn[:, [0]]])  # append first node to close quad

    for k in range(nele):
        for i in range(4):
            key = nodesn_ext[k, i] + (nno-1)*nodesn_ext[k, i+1]
            pos = key % M
            while True:
                if htable[pos] == 0:
                    htable[pos] = key
                    break
                elif htable[pos] == key:
                    raise ValueError("Double edge ...")
                else:
                    pos = (pos + 1) % M

    return nodesn


def plotmesh(xnod, ynod, nodes):
    """
    Plot an unstructured 2D quadrilateral mesh.

    Parameters
    ----------
    xnod, ynod : (nno,) ndarray
        Node coordinates
    nodes : (nele, 4) ndarray of int
        Element connectivity (0-based)
    """

    # Flatten connectivity
    nn = nodes.ravel()

    # Coordinates corresponding to nodes
    xx = xnod[nn]
    yy = ynod[nn]

    # Reshape back to element-wise arrays
    xplot = xx.reshape(nodes.shape)
    yplot = yy.reshape(nodes.shape)

    plt.clf()
    plt.gca().set_aspect('equal', adjustable='box')

    # Matplotlib fill: each row is one polygon
    plt.fill(xplot.T, yplot.T, facecolor='white', edgecolor='black')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Quadrilateral FEM Mesh")
    plt.show()
