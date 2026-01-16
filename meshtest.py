import numpy as np
import warnings

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
