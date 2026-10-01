#!/usr/bin/env python3
"""Reproduce the exact calculations of the BB-metacheck manuscript.

Companion code for

    M. Rowshan, "Metachecks in Bivariate Bicycle Codes: Syndrome Distance, Measurement Faults, and Repair Limits," 2026, arXiv:2609.39121.

Everything computed here is exact (GF(2) linear algebra and exhaustive
searches); nothing depends on an iterative decoder or on random sampling.
The script reproduces

  * tab:standard_codes and the syndrome-side columns of tab:diagnostic_codes
    (k, r_M, d_S, K_M, |A^x|, u_1, w_amb, mu_bar),
  * the leader histograms (tab:leader_hist) and exact repair curves (fig:exact_repair),
  * the unit-group bound (prop:unitgroup) and partial-repetition cost (eq:partial_cost),
  * the sparse metacheck row-weight profiles (eq:sparse_weights),
  * the dimension statement dim L_ann = dim L_col = k/2 (prop:logical_decomp) and
    the one-sided logical witnesses.

References use the manuscript's LaTeX labels, which survive renumbering.
Every value is compared with the manuscript; the script exits with status 1
if anything disagrees.

Not included: the logical component distances d_ann, d_col (certified with the
companion package of Rowshan and Devitt, see README) and the sustained-memory
BP+OSD Monte Carlo experiment.

Coordinates follow the manuscript: x^i y^j is stored at index i + l*j.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, deque
from dataclasses import dataclass
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# GF(2) linear algebra
# ---------------------------------------------------------------------------


def rref(a: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Reduced row-echelon form over GF(2), together with pivot columns."""
    a = np.asarray(a, dtype=np.uint8).copy() & 1
    nrow, ncol = a.shape
    row = 0
    pivots: list[int] = []
    for col in range(ncol):
        if row == nrow:
            break
        candidates = np.flatnonzero(a[row:, col])
        if candidates.size == 0:
            continue
        pivot = row + int(candidates[0])
        a[[row, pivot]] = a[[pivot, row]]
        other = np.flatnonzero(a[:, col])
        a[other[other != row]] ^= a[row]
        pivots.append(col)
        row += 1
    return a[:row], pivots


def gf2_rank(a: np.ndarray) -> int:
    return 0 if a.size == 0 else len(rref(a)[1])


def nullspace(a: np.ndarray) -> np.ndarray:
    """Rows form a basis of ker(a) over GF(2)."""
    rr, pivots = rref(a)
    n = a.shape[1]
    pivot_set = set(pivots)
    basis = []
    for free in range(n):
        if free in pivot_set:
            continue
        v = np.zeros(n, dtype=np.uint8)
        v[free] = 1
        for i, p in enumerate(pivots):
            if rr[i, free]:
                v[p] = 1
        basis.append(v)
    return np.asarray(basis, dtype=np.uint8).reshape(len(basis), n)


def solve_gf2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """One solution of a x = b over GF(2) (raises if inconsistent)."""
    a = np.asarray(a, dtype=np.uint8) & 1
    aug = np.hstack((a, np.asarray(b, dtype=np.uint8).reshape(-1, 1) & 1))
    rr, pivots = rref(aug)
    ncol = a.shape[1]
    if ncol in pivots:  # a pivot in the right-hand column means 0 = 1
        raise ValueError("GF(2) system is inconsistent")
    x = np.zeros(ncol, dtype=np.uint8)
    for i, col in enumerate(pivots):
        x[col] = rr[i, ncol]
    return x


def mat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a.astype(np.int64) @ b.astype(np.int64)) & 1).astype(np.uint8)


def bits_to_int(v: np.ndarray) -> int:
    """Pack a binary vector into a Python int (bit j = coordinate j)."""
    out = 0
    for j in np.flatnonzero(v):
        out |= 1 << int(j)
    return out


class BinarySpan:
    """Incremental GF(2) span of packed vectors (used for greedy bases)."""

    def __init__(self, vectors: Iterable[int] = ()):
        self.pivots: dict[int, int] = {}
        for v in vectors:
            self.add(int(v))

    def reduce(self, v: int) -> int:
        while v:
            p = v.bit_length() - 1
            if p not in self.pivots:
                break
            v ^= self.pivots[p]
        return v

    def add(self, v: int) -> bool:
        """Insert v; return True if it enlarged the span."""
        v = self.reduce(v)
        if v == 0:
            return False
        self.pivots[v.bit_length() - 1] = v
        return True


# ---------------------------------------------------------------------------
# BB ring R = F2[x,y]/(x^l-1, y^m-1) and the check matrices
# ---------------------------------------------------------------------------


def index(l: int, m: int, i: int, j: int) -> int:
    """Index of x^i y^j in the manuscript's monomial ordering."""
    return (i % l) + l * (j % m)


def coord(l: int, idx: int) -> tuple[int, int]:
    return idx % l, idx // l


def multiplication_matrix(l: int, m: int, terms: Iterable[tuple[int, int]]) -> np.ndarray:
    """L_c: column h holds the coefficient vector of c*h."""
    n = l * m
    out = np.zeros((n, n), dtype=np.uint8)
    for i in range(l):
        for j in range(m):
            col = index(l, m, i, j)
            for di, dj in terms:
                out[index(l, m, i + di, j + dj), col] ^= 1
    return out


def translate(v: np.ndarray, l: int, m: int, di: int, dj: int) -> np.ndarray:
    """Multiply a ring element (coefficient vector) by the monomial x^di y^dj."""
    out = np.zeros_like(v)
    for h in range(l * m):
        i, j = coord(l, h)
        out[index(l, m, i + di, j + dj)] = v[h]
    return out


def ring_product(u: np.ndarray, v: np.ndarray, l: int, m: int) -> np.ndarray:
    out = np.zeros(l * m, dtype=np.uint8)
    for p in np.flatnonzero(u):
        i, j = coord(l, int(p))
        for q in np.flatnonzero(v):
            r, s = coord(l, int(q))
            out[index(l, m, i + r, j + s)] ^= 1
    return out


def format_monomial(g: tuple[int, int]) -> str:
    i, j = g
    pieces = []
    if i:
        pieces.append("x" if i == 1 else f"x^{i}")
    if j:
        pieces.append("y" if j == 1 else f"y^{j}")
    return "".join(pieces) or "1"


@dataclass(frozen=True)
class BBCode:
    name: str
    l: int
    m: int
    a: tuple[tuple[int, int], ...]   # monomials (i, j) of a, i.e. x^i y^j
    b: tuple[tuple[int, int], ...]
    reported_d: int | None = None    # quantum distance quoted in the manuscript

    @property
    def N(self) -> int:
        return self.l * self.m

    @property
    def n(self) -> int:
        return 2 * self.N


GROSS_A = ((3, 0), (0, 1), (0, 2))   # x^3 + y + y^2
GROSS_B = ((0, 3), (1, 0), (2, 0))   # y^3 + x + x^2

STANDARD_CODES = (
    BBCode("[[72,12,6]]", 6, 6, GROSS_A, GROSS_B, 6),
    BBCode("[[90,8,10]]", 15, 3, ((9, 0), (0, 1), (0, 2)), ((0, 0), (2, 0), (7, 0)), 10),
    BBCode("[[108,8,10]]", 9, 6, GROSS_A, GROSS_B, 10),
    BBCode("Gross [[144,12,12]]", 12, 6, GROSS_A, GROSS_B, 12),
    BBCode("[[288,12,18]]", 12, 12, ((3, 0), (0, 2), (0, 7)), GROSS_B, 18),
)

DIAGNOSTIC_CODES = (
    BBCode("BB(6,3)", 6, 3, GROSS_A, GROSS_B, 4),
    BBCode("BB(9,3)", 9, 3, GROSS_A, GROSS_B),
    BBCode("BB(3,3)", 3, 3, ((0, 0), (1, 0), (2, 0)), ((0, 0), (0, 1), (0, 2))),
    BBCode("BB(6,6) separable", 6, 6, ((0, 0), (1, 0), (2, 0)), ((0, 0), (0, 1), (0, 2))),
    BBCode("BB(4,4) symmetric", 4, 4, ((0, 0), (0, 1)), ((0, 0), (0, 1)), 2),
    BBCode("BB(6,6) symmetric", 6, 6, ((0, 0), (1, 0), (0, 1), (1, 2)),
           ((0, 0), (1, 0), (0, 1), (1, 2)), 2),
    BBCode("BB(8,8) symmetric", 8, 8, ((0, 0), (0, 1)), ((0, 0), (0, 1)), 2),
)


def build_code(code: BBCode) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """L_a, L_b, H_X = [L_a | L_b] and H = H_Z = [L_b^T | L_a^T] (eq:bbchecks)."""
    la = multiplication_matrix(code.l, code.m, code.a)
    lb = multiplication_matrix(code.l, code.m, code.b)
    hx = np.hstack((la, lb))
    hz = np.hstack((lb.T, la.T))
    if mat_mul(hx, hz.T).any():
        raise AssertionError(f"{code.name}: H_X H_Z^T != 0")
    return la, lb, hx, hz


# ---------------------------------------------------------------------------
# Syndrome code S = im H, metachecks C_M = ker H^T, and K_M  (sec:ds2)
# ---------------------------------------------------------------------------


def column_masks(m: np.ndarray) -> list[int]:
    return [bits_to_int(m[:, j]) for j in range(m.shape[1])]


def syndrome_distance(m: np.ndarray) -> int | None:
    """Minimum distance of ker(m), tested through weight four (None if larger).

    Weight w words of ker(m) are w columns of m summing to zero: a zero column
    (w=1), two equal columns (w=2), a column equal to the sum of two others (w=3),
    or two disjoint pairs with equal sums (w=4).
    """
    cols = column_masks(m)
    if 0 in cols:
        return 1
    if len(set(cols)) < len(cols):
        return 2
    colset = set(cols)
    if any((ci ^ cj) in colset for ci, cj in combinations(cols, 2)):
        return 3
    seen: set[int] = set()
    for ci, cj in combinations(cols, 2):
        if (ci ^ cj) in seen:   # columns are distinct, so the two pairs are disjoint
            return 4
        seen.add(ci ^ cj)
    return None


def translation_subgroup(mcheck: np.ndarray, code: BBCode) -> list[tuple[int, int]]:
    """K_M = translations fixing every metacheck (eq:KM).  Identity comes first."""
    return [
        (di, dj)
        for dj in range(code.m)
        for di in range(code.l)
        if all(np.array_equal(translate(row, code.l, code.m, di, dj), row) for row in mcheck)
    ]


def subgroup_orbits(code: BBCode, subgroup: list[tuple[int, int]]) -> list[list[int]]:
    unseen = set(range(code.N))
    orbits = []
    while unseen:
        i, j = coord(code.l, min(unseen))
        orbit = sorted({index(code.l, code.m, i + di, j + dj) for di, dj in subgroup})
        orbits.append(orbit)
        unseen.difference_update(orbit)
    return orbits


def subgroup_generators(subgroup: list[tuple[int, int]], code: BBCode) -> list[tuple[int, int]]:
    """A small generating set of K_M, for display only."""
    target = set(subgroup)
    chosen: list[tuple[int, int]] = []
    generated = {(0, 0)}
    for g in sorted(subgroup, key=lambda t: (t[1], t[0])):
        if g in generated:
            continue
        chosen.append(g)
        frontier = list(generated)
        while frontier:   # close under addition of the chosen generators
            h = frontier.pop()
            for c in chosen:
                s = ((h[0] + c[0]) % code.l, (h[1] + c[1]) % code.m)
                if s not in generated:
                    generated.add(s)
                    frontier.append(s)
        if generated == target:
            break
    return chosen


def coset_leaders(mcheck: np.ndarray) -> Counter:
    """Histogram of minimum-weight leaders over all 2^{r_M} metasyndromes.

    Breadth-first search adds one syndrome coordinate at a time, so the first
    visit to each metasyndrome is through a minimum-weight leader.
    """
    cols = column_masks(mcheck)
    target = 1 << mcheck.shape[0]
    weight: dict[int, int] = {0: 0}
    frontier = deque([0])
    while frontier and len(weight) < target:
        s = frontier.popleft()
        for col in cols:
            if (s ^ col) not in weight:
                weight[s ^ col] = weight[s] + 1
                frontier.append(s ^ col)
    if len(weight) != target:
        raise AssertionError("metacheck rows are not independent")
    return Counter(weight.values())


def sparse_metacheck_weights(raw_basis: np.ndarray) -> list[int]:
    """Row weights of the deterministic sparse basis fed to BP (eq:sparse_weights).

    Enumerate C_M, sort by (Hamming weight, packed integer value), and keep a
    vector whenever it increases the rank.  Matroid greedy => minimum total weight.
    """
    masks = [bits_to_int(row) for row in raw_basis]
    candidates = []
    for selector in range(1, 1 << len(masks)):
        v = 0
        for i, row in enumerate(masks):
            if (selector >> i) & 1:
                v ^= row
        candidates.append(v)
    candidates.sort(key=lambda v: (v.bit_count(), v))
    span, weights = BinarySpan(), []
    for v in candidates:
        if span.add(v):
            weights.append(v.bit_count())
            if len(weights) == len(masks):
                break
    return weights


def verify_partial_repetition(code: BBCode, h: np.ndarray, subgroup) -> tuple[int, int | None]:
    """Partial repetition (eq:partial_cost): remeasure all but one check per orbit.

    Builds the augmented code S_T = {(s, s|_T)} and returns |T| together with
    its distance, which must be >= 3 for every single fault to be resolvable.
    """
    repeated = sorted(hh for orbit in subgroup_orbits(code, subgroup) for hh in orbit[1:])
    _, pivots = rref(h)
    gens = h[:, pivots].T                                   # rows span S = im H
    augmented = np.hstack((gens, gens[:, repeated]))        # rows span S_T
    return len(repeated), syndrome_distance(nullspace(augmented))


# ---------------------------------------------------------------------------
# Quotient algebra A = R/S and its unit group  (prop:unitgroup)
# ---------------------------------------------------------------------------


def quotient_unit_count(mcheck: np.ndarray, code: BBCode) -> int:
    """|A^x|, counted by testing every nonzero class for invertibility.

    M identifies R/S with F2^{r_M}.  Representatives r_j with M r_j = e_j give a
    basis; an element is a unit iff its multiplication map on A has full rank.
    """
    r = mcheck.shape[0]
    reps = np.asarray([solve_gf2(mcheck, np.eye(r, dtype=np.uint8)[j]) for j in range(r)]).T
    # structure[:, i, j] = coordinates in A of r_i * r_j
    structure = np.zeros((r, r, r), dtype=np.uint8)
    for i in range(r):
        for j in range(r):
            structure[:, i, j] = mat_mul(mcheck, ring_product(reps[:, i], reps[:, j], code.l, code.m))
    units = 0
    for mask in range(1, 1 << r):
        mult = np.zeros((r, r), dtype=np.uint8)
        for i in range(r):
            if (mask >> i) & 1:
                mult ^= structure[:, i, :]
        units += gf2_rank(mult) == r
    return units


# ---------------------------------------------------------------------------
# Logical quantities  (sec:logical)
# ---------------------------------------------------------------------------


def all_span_masks(basis: Iterable[np.ndarray]) -> list[int]:
    out = [0]
    for v in (bits_to_int(row) for row in basis):
        out += [x ^ v for x in out]
    return out


def one_sided_minimum(kernel: np.ndarray, slice_: np.ndarray, *, exact: bool) -> tuple[int, int | None]:
    """dim(K/B) and min weight of K \\ B for a one-block kernel K and slice B.

    The exact minimum enumerates every nontrivial coset of B (2^dim B elements
    each), which is the expensive step for [[288,12,18]].
    """
    b_basis, _ = rref(slice_) if slice_.size else (slice_, [])
    span = BinarySpan(bits_to_int(v) for v in b_basis)
    complement = [v for v in kernel if span.add(bits_to_int(v))]
    q = len(complement)
    if q == 0 or not exact:
        return q, None
    b_masks = all_span_masks(b_basis)
    best = kernel.shape[1] + 1
    for c in all_span_masks(complement)[1:]:
        best = min(best, min((c ^ b).bit_count() for b in b_masks))
    return q, best


def logical_quantities(code: BBCode, la: np.ndarray, lb: np.ndarray, *, exact: bool) -> dict[str, object]:
    """dim L_ann, dim L_col (both k/2, prop:logical_decomp) and one-sided witnesses.

    In the selected X sector, L_{a*} = L_a^T and L_{b*} = L_b^T.
      left  block: Ann(b*) / a* Ann(b*)   = L_ann (every class has a left rep)
      right block: Ann(a*) / b* Ann(a*)   (right one-sided classes)
      colon      : (b*:a*) / (b*)          = L_col
    The left one-sided minimum is an upper bound on d_ann.
    """
    la_s, lb_s = la.T, lb.T
    k_left, k_right = nullspace(lb_s), nullspace(la_s)
    dim_ann, w_left = one_sided_minimum(k_left, mat_mul(la_s, k_left.T).T, exact=exact)
    _, w_right = one_sided_minimum(k_right, mat_mul(lb_s, k_right.T).T, exact=exact)

    # (b*:a*) = {r : a* r in im L_{b*}} = ker(P L_{a*}), with P a parity check of im L_{b*}.
    parity = nullspace(lb_s.T)
    dim_colon = code.N - gf2_rank(mat_mul(parity, la_s))
    dim_col = dim_colon - gf2_rank(lb_s)
    return dict(dim_L_ann=dim_ann, dim_L_col=dim_col, onesided_left=w_left, onesided_right=w_right)


# ---------------------------------------------------------------------------
# Ambiguity cost mu(g)  (sec:soundness)
# ---------------------------------------------------------------------------


def min_preimage_weight(h: np.ndarray, target: int, max_weight: int) -> int | None:
    """Exact min wt(e) with H e = target, or None if it exceeds max_weight.

    Iterative-deepening search that always branches on the lowest-index check
    where the current syndrome differs from the target.  If e is any solution
    and the chosen set C is a subset of e, that check is touched by some qubit
    of e \\ C, so every solution is reachable and the first depth that succeeds
    is the minimum.  The bound |e \\ C| >= ceil(mismatch / max column weight)
    prunes the tree.  Works for any column parity (the symmetric rows have
    even-weight columns, so odd preimage weights must be allowed).
    """
    check_qubits = [np.flatnonzero(h[r]).tolist() for r in range(h.shape[0])]
    cols = column_masks(h)
    cmax = int(h.sum(axis=0).max())

    def dfs(mismatch: int, chosen: set[int], limit: int) -> bool:
        if mismatch == 0:
            return True
        if len(chosen) + -(-mismatch.bit_count() // cmax) > limit:
            return False
        row = (mismatch & -mismatch).bit_length() - 1
        for q in check_qubits[row]:
            if q not in chosen:
                chosen.add(q)
                if dfs(mismatch ^ cols[q], chosen, limit):
                    return True
                chosen.discard(q)
        return False

    for w in range(1, max_weight + 1):
        if dfs(target, set(), w):
            return w
    return None


def min_preimage_weight_mitm(h: np.ndarray, target: int) -> int | None:
    """Independent cross-check through weight six by meet-in-the-middle.

    Stores syndromes of all column subsets of size <= 3 and looks for two
    disjoint subsets whose syndromes add up to the target.  Memory grows like
    n^3, so it is only run for n <= 144.
    """
    cols = column_masks(h)
    by_size: dict[int, dict[int, list[int]]] = {}
    for size in (1, 2, 3):
        table: dict[int, list[int]] = {}
        for subset in combinations(range(len(cols)), size):
            syn, mask = 0, 0
            for q in subset:
                syn ^= cols[q]
                mask |= 1 << q
            table.setdefault(syn, []).append(mask)
        by_size[size] = table
    if target in by_size[1]:
        return 1
    for w in range(2, 7):
        for s1 in range(1, 4):
            s2 = w - s1
            if not 0 <= s2 <= 3 or s2 > s1:
                continue
            if s2 == 0:
                if target in by_size[s1]:
                    return w
                continue
            for syn, masks_a in by_size[s1].items():
                masks_b = by_size[s2].get(syn ^ target, ())
                if any((ma & mb) == 0 for ma in masks_a for mb in masks_b):
                    return w
    return None


def ambiguity_costs(h, code, subgroup, max_weight, cross_check):
    """mu(g) for each nontrivial g in K_M (eq:mu_g); w_amb = min, mu_bar = mean."""
    mu: dict[tuple[int, int], int | None] = {}
    for g in subgroup[1:]:
        # By translation equivariance it suffices to target e_1 + e_g.
        target = (1 << index(code.l, code.m, 0, 0)) | (1 << index(code.l, code.m, *g))
        mu[g] = min_preimage_weight(h, target, max_weight)
        if cross_check and code.n <= 144:
            lo = min_preimage_weight_mitm(h, target)
            if lo != (mu[g] if mu[g] is not None and mu[g] <= 6 else None):
                raise AssertionError(f"{code.name}: mu{g} search and meet-in-the-middle disagree")
    return mu


# ---------------------------------------------------------------------------
# Exact repair curves  (eq:exact_repair, fig:exact_repair)
# ---------------------------------------------------------------------------


P_GRID = np.asarray([0.001, 0.0013, 0.0016, 0.002, 0.0025, 0.003, 0.0037,
                     0.0045, 0.0055, 0.0067, 0.0082, 0.010, 0.012, 0.015])


def exact_repair_failure(hist: Counter, n: int, p: float) -> float:
    """P_repair^MW(p) = 1 - sum_w c_w p^w (1-p)^(N-w)."""
    return 1.0 - sum(c * p**w * (1.0 - p) ** (n - w) for w, c in hist.items())


def write_repair_curves(outdir: Path, selected, make_plot: bool) -> None:
    with (outdir / "exact_repair_curves.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["code", "p_M", "P_repair_MW", "orbit_bound"])
        for label, (code, hist, u1) in selected.items():
            for p in P_GRID:
                bound = 1 - (1 - p) ** u1 if u1 else ""
                writer.writerow([label, p, exact_repair_failure(hist, code.N, p), bound])
    if not make_plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: wrote the CSV, skipped the plot")
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    p_dense = np.geomspace(P_GRID.min(), P_GRID.max(), 200)
    for label, (code, hist, u1) in selected.items():
        line, = ax.loglog(P_GRID, [exact_repair_failure(hist, code.N, p) for p in P_GRID],
                          marker="o", label=label)
        if u1:   # finite-rate orbit bound of cor:ambiguity
            ax.loglog(p_dense, 1 - (1 - p_dense) ** u1, linestyle="--",
                      color=line.get_color(), alpha=0.7, label=f"{label}: orbit bound")
    ax.set_xlabel("measurement-flip probability $p_M$")
    ax.set_ylabel("exact reconstruction failure $P^{\\rm MW}_{\\rm repair}$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "exact_repair_curves.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Values printed in the manuscript (used as regression targets)
# ---------------------------------------------------------------------------

# hist = (c_0, c_1, ...) from tab:leader_hist; sparse = eq:sparse_weights;
# w_amb, mu_bar and the one-sided witnesses from tab:standard_codes and sec:logical.
EXPECTED = {
    "[[72,12,6]]": dict(k=12, d_S=3, K_M=1, units=36, u1=0, hist=(1, 36, 27),
                        sparse=[16, 16, 16, 16, 18, 18], onesided_left=6),
    "[[90,8,10]]": dict(k=8, d_S=2, K_M=5, units=9, u1=36, hist=(1, 9, 6),
                        sparse=[20] * 4, w_amb=4, mu_bar=4.0, onesided_left=10),
    "[[108,8,10]]": dict(k=8, d_S=2, K_M=6, units=9, u1=45, hist=(1, 9, 6),
                         sparse=[24] * 4, w_amb=4, mu_bar=5.2, onesided_left=12),
    "Gross [[144,12,12]]": dict(k=12, d_S=2, K_M=2, units=36, u1=36, hist=(1, 36, 27),
                                sparse=[32, 32, 32, 32, 36, 36], w_amb=6, mu_bar=6.0,
                                onesided_left=12),
    "[[288,12,18]]": dict(k=12, d_S=2, K_M=4, units=36, u1=108, hist=(1, 36, 27),
                          # manuscript prints w_amb >= 8; the exact search gives 14
                          sparse=[64, 64, 64, 64, 72, 72], w_amb_at_least=8,
                          onesided_left=18),
    # tab:diagnostic_codes and the text around prop:unitgroup
    "BB(6,3)": dict(k=8, d_S=2, K_M=2, units=9, u1=9, hist=(1, 9, 6), w_amb=2, mu_bar=2.0),
    "BB(9,3)": dict(k=8, d_S=2, K_M=3, units=9, u1=18, hist=(1, 9, 6)),
    "BB(3,3)": dict(k=8, d_S=3, K_M=1, units=9, u1=0, hist=(1, 9, 6)),
    "BB(6,6) separable": dict(k=8, d_S=2, K_M=4, units=9, u1=27),
    "BB(4,4) symmetric": dict(k=8, d_S=2, K_M=4, units=8, u1=12),
    "BB(6,6) symmetric": dict(k=20, d_S=4, K_M=1, units=288, u1=0,
                              sparse=[6] * 6 + [12] * 4),
    "BB(8,8) symmetric": dict(k=16, d_S=2, K_M=8, units=128, u1=56),
}

EXPECTED_REPAIR_AT_1E2 = {"[[72,12,6]]": 0.0484, "Gross [[144,12,12]]": 0.3373, "BB(6,3)": 0.0891}


def compare_with_manuscript(row: dict, raw: dict) -> list[str]:
    """Return a list of human-readable mismatches (empty if all agree)."""
    exp = EXPECTED.get(row["code"], {})
    bad = []

    def check(label, got, want):
        if got != want:
            bad.append(f"{row['code']}: {label} = {got}, manuscript {want}")

    for key, field in (("k", "k"), ("d_S", "d_S"), ("K_M", "K_M_size"),
                       ("units", "unit_group_size"), ("u1", "u1")):
        if key in exp:
            check(key, row[field], exp[key])
    if "hist" in exp:
        check("leader histogram", raw["hist"], exp["hist"])
    if "sparse" in exp:
        check("sparse metacheck weights", raw["sparse"], exp["sparse"])
    if "w_amb" in exp and raw["w_amb"] is not None:
        check("w_amb", raw["w_amb"], exp["w_amb"])
        check("mu_bar", round(raw["mu_bar"], 6), exp["mu_bar"])
    if "w_amb_at_least" in exp and raw["w_amb"] is not None:
        if raw["w_amb"] < exp["w_amb_at_least"]:
            bad.append(f"{row['code']}: w_amb = {raw['w_amb']} < {exp['w_amb_at_least']}")
    if "onesided_left" in exp and row["onesided_left"] != "skipped":
        check("one-sided left witness", row["onesided_left"], exp["onesided_left"])

    # Identities that must hold for every code
    k = row["k"]
    check("r_M (= k/2)", row["r_M"], k // 2)
    check("dim L_ann (= k/2)", row["dim_L_ann"], k // 2)
    check("dim L_col (= k/2)", row["dim_L_col"], k // 2)
    check("c_1 (= N/|K_M|)", raw["hist"][1], row["N"] // row["K_M_size"])
    check("partial repetition cost (= u_1)", row["partial_repetition_min"], row["u1"])
    if (row["d_S"] == 2) != (row["K_M_size"] > 1):
        bad.append(f"{row['code']}: d_S = 2 iff |K_M| > 1 fails")
    if row["partial_augmented_dS"] is not None and row["partial_augmented_dS"] < 3:
        bad.append(f"{row['code']}: augmented syndrome code still has distance < 3")
    if row["N"] // row["K_M_size"] > row["unit_group_size"]:
        bad.append(f"{row['code']}: |G/K_M| > |A^x| violates prop:unitgroup")
    return bad


# ---------------------------------------------------------------------------
# Audit of one code
# ---------------------------------------------------------------------------


def audit_code(code: BBCode, *, exact_onesided: bool, mu_max: int | None, cross_check: bool):
    la, lb, hx, h = build_code(code)
    k = code.n - gf2_rank(hx) - gf2_rank(h)

    mcheck = nullspace(h.T)                    # rows = basis of C_M = ker H^T
    d_s = syndrome_distance(mcheck)
    subgroup = translation_subgroup(mcheck, code)
    kappa = len(subgroup)
    u1 = code.N - code.N // kappa              # eq:u1; zero when K_M is trivial
    hist_counter = coset_leaders(mcheck)
    hist = tuple(hist_counter.get(w, 0) for w in range(max(hist_counter) + 1))
    units = quotient_unit_count(mcheck, code)
    sparse = sparse_metacheck_weights(mcheck)
    logical = logical_quantities(code, la, lb, exact=exact_onesided)
    partial_cost, partial_ds = verify_partial_repetition(code, h, subgroup)

    mu, w_amb, mu_bar = {}, None, None
    if mu_max is not None and kappa > 1:
        mu = ambiguity_costs(h, code, subgroup, mu_max, cross_check)
        if all(v is not None for v in mu.values()):
            w_amb = min(mu.values())
            mu_bar = sum(mu.values()) / len(mu)

    def fmt(v):
        return f">{mu_max}" if v is None else v

    row = {
        "code": code.name, "l": code.l, "m": code.m, "n": code.n, "N": code.N,
        "k": k, "r_M": len(mcheck),
        "dim_L_ann": logical["dim_L_ann"], "dim_L_col": logical["dim_L_col"],
        "onesided_left": fmt_opt(logical["onesided_left"]),
        "onesided_right": fmt_opt(logical["onesided_right"]),
        "reported_d": code.reported_d if code.reported_d is not None else "",
        "d_S": d_s, "K_M_size": kappa,
        "K_M_generators": ";".join(format_monomial(g) for g in subgroup_generators(subgroup, code)),
        "unit_group_size": units,
        "unit_map_surjective": code.N // kappa == units,
        "u1": u1,
        "partial_repetition_min": partial_cost,
        "partial_augmented_dS": partial_ds,
        "leader_hist": "(" + ",".join(map(str, hist)) + ")",
        "sparse_metacheck_weights": " ".join(map(str, sparse)),
        "mu_g": ";".join(f"{format_monomial(g)}:{fmt(v)}" for g, v in mu.items()),
        "w_amb": w_amb if w_amb is not None else (f">{mu_max}" if mu else ""),
        "mu_bar": round(mu_bar, 6) if mu_bar is not None else "",
    }
    raw = dict(hist=hist, hist_counter=hist_counter, sparse=sparse, w_amb=w_amb, mu_bar=mu_bar)
    return row, raw


def fmt_opt(v):
    return "skipped" if v is None else v


# ---------------------------------------------------------------------------
# Command-line driver
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exact calculations of the BB-metacheck manuscript (see README.md).")
    parser.add_argument("--outdir", type=Path, default=Path("results"),
                        help="directory for the CSV and PNG outputs (default: results)")
    parser.add_argument("--include-diagnostics", action="store_true",
                        help="also audit the seven diagnostic rows of tab:diagnostic_codes")
    parser.add_argument("--quick", action="store_true",
                        help="skip the two slow [[288,12,18]] steps (one-sided enumeration, mu(g))")
    parser.add_argument("--mu-max-weight", type=int, default=16,
                        help="largest preimage weight tried in the mu(g) search (default 16)")
    parser.add_argument("--cross-check", action="store_true",
                        help="re-derive mu(g) by meet-in-the-middle for codes with n <= 144")
    parser.add_argument("--no-plot", action="store_true", help="do not write the PNG")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    codes = list(STANDARD_CODES) + (list(DIAGNOSTIC_CODES) if args.include_diagnostics else [])
    rows, raws, problems = [], {}, []

    header = f"{'code':22s} {'k':>3} {'d_S':>3} {'|K_M|':>5} {'|A^x|':>5} {'u1':>4} {'w_amb':>6} {'1-sided':>7}"
    print(header)
    print("-" * len(header))
    for code in codes:
        heavy = code.name == "[[288,12,18]]" and args.quick
        row, raw = audit_code(code, exact_onesided=not heavy,
                              mu_max=None if heavy else args.mu_max_weight,
                              cross_check=args.cross_check)
        problems += compare_with_manuscript(row, raw)
        rows.append(row)
        raws[code.name] = (code, raw)
        print(f"{code.name:22s} {row['k']:>3} {row['d_S']:>3} {row['K_M_size']:>5} "
              f"{row['unit_group_size']:>5} {row['u1']:>4} {str(row['w_amb']):>6} "
              f"{str(row['onesided_left']):>7}")

    with (args.outdir / "structural_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # fig:exact_repair uses [[72,12,6]], Gross and BB(6,3); add BB(6,3) if not audited.
    if "BB(6,3)" not in raws:
        _, raw = audit_code(DIAGNOSTIC_CODES[0], exact_onesided=False, mu_max=None, cross_check=False)
        raws["BB(6,3)"] = (DIAGNOSTIC_CODES[0], raw)
    u1_of = {r["code"]: r["u1"] for r in rows}
    u1_of.setdefault("BB(6,3)", 9)
    selected = {
        label: (raws[name][0], raws[name][1]["hist_counter"], u1_of[name])
        for label, name in (("[[72,12,6]]", "[[72,12,6]]"),
                            ("Gross [[144,12,12]]", "Gross [[144,12,12]]"),
                            ("BB(6,3) [[36,8,4]]", "BB(6,3)"))
    }
    write_repair_curves(args.outdir, selected, make_plot=not args.no_plot)

    print("\nExact minimum-weight repair failure at p_M = 0.01 (orbit bound in brackets)")
    for name, want in EXPECTED_REPAIR_AT_1E2.items():
        code, raw = raws[name]
        got = exact_repair_failure(raw["hist_counter"], code.N, 0.01)
        bound = f"{1 - 0.99 ** u1_of[name]:.4f}" if u1_of[name] else "none, d_S = 3"
        print(f"  {name:22s} {got:.4f}   [{bound}]")
        if round(got, 4) != want:
            problems.append(f"{name}: P_repair(0.01) = {got:.4f}, manuscript {want}")

    # With d_S = 3, failures start at weight two: the coefficient of p^2 is the
    # number of weight-two patterns that are not selected leaders (eq:603).
    coeff = comb(36, 2) - raws["[[72,12,6]]"][1]["hist_counter"][2]
    print(f"  [[72,12,6]] low-noise coefficient: {coeff} p_M^2")
    if coeff != 603:
        problems.append(f"[[72,12,6]]: p^2 coefficient {coeff}, manuscript 603")

    print(f"\nWrote {args.outdir}/structural_results.csv and exact_repair_curves.csv"
          + ("" if args.no_plot else " (+ .png)"))
    if problems:
        print(f"\n{len(problems)} disagreement(s) with the manuscript:")
        for p in problems:
            print("  " + p)
        return 1
    print("\nAll values agree with the manuscript.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
