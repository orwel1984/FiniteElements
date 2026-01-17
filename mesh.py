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


def refine_quad_mesh(nodes, bnode, xnod, ynod):
    """
    Refine a quadrilateral mesh once.

    nodes : (nele,4) element connectivity (0-based indexing)
    bnode : (nno,) boundary flags, 1 for boundary, 0 otherwise
    xnod, ynod : node coordinates

    Returns:
    nodesn, bnodn, xnodn, ynodn, nele, nno
    """
    nele = nodes.shape[0]
    nno = len(xnod)
    nbn = np.sum(bnode)

    # estimate number of new nodes
    nnon = nno + 3*nele + nbn//2

    nodesn = np.zeros((4*nele, 4), dtype=int)
    bnodn = np.zeros(nnon, dtype=int)
    xnodn = np.zeros(nnon)
    ynodn = np.zeros(nnon)

    # copy old nodes
    bnodn[:nno] = bnode
    xnodn[:nno] = xnod
    ynodn[:nno] = ynod

    # center weights for element center
    sum1 = np.ones(4)/4.0

    # map to track mid-edge nodes
    check = -np.ones((nno, nno), dtype=int)

    idxn = 0
    nno_new = nno

    for iel in range(nele):
        iv = nodes[iel]

        # element center
        xnodn[nno_new] = np.dot(sum1, xnod[iv])
        ynodn[nno_new] = np.dot(sum1, ynod[iv])
        nodmid = nno_new
        nno_new += 1

        # loop over edges (1-2,2-3,3-4,4-1)
        edges = [(0,1),(1,2),(2,3),(3,0)]
        midnodes = []
        for a,b in edges:
            if check[iv[a], iv[b]] == -1:
                xnodn[nno_new] = 0.5*(xnod[iv[a]] + xnod[iv[b]])
                ynodn[nno_new] = 0.5*(ynod[iv[a]] + ynod[iv[b]])
                check[iv[a], iv[b]] = nno_new
                check[iv[b], iv[a]] = nno_new
                # set boundary if both nodes are boundary
                if bnodn[iv[a]] == 1 and bnodn[iv[b]] == 1:
                    bnodn[nno_new] = 1
                nno_new += 1
            midnodes.append(check[iv[a], iv[b]])

        # define the 4 new elements
        nodesn[idxn]   = [iv[0], midnodes[0], nodmid, midnodes[3]]
        nodesn[idxn+1] = [midnodes[0], iv[1], midnodes[1], nodmid]
        nodesn[idxn+2] = [nodmid, midnodes[1], iv[2], midnodes[2]]
        nodesn[idxn+3] = [midnodes[3], nodmid, midnodes[2], iv[3]]

        idxn += 4

    nele_new = 4*nele
    nno_final = nno_new

    return nodesn, bnodn[:nno_final], xnodn[:nno_final], ynodn[:nno_final], nele_new, nno_final
