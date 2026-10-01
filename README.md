# BB-Codes-Metachecks-Syndrome-Repair

Companion code for

> M. Rowshan, *Bivariate Bicycle Codes and Metachecks: Syndrome Redundancy,
> Repair Limits, and Logical Obstructions*, submitted to IEEE Transactions on
> Information Theory.

The script `bb_manuscript_reproduce.py` recomputes every exact quantity in the
paper that concerns syndrome redundancy and checks it against the printed
value. It uses only GF(2) linear algebra and exhaustive searches, so the
results do not depend on a decoder, a random seed, or floating-point
tolerances. A run ends with either `All values agree with the manuscript.`
(exit status 0) or a list of disagreements (exit status 1).

## Quick start

```bash
git clone https://github.com/mohammad-rowshan/BB-Codes-Metachecks-Syndrome-Repair.git
cd BB-Codes-Metachecks-Syndrome-Repair
pip install -r requirements.txt          # numpy; matplotlib is optional
python bb_manuscript_reproduce.py --include-diagnostics
```

This takes about 40 seconds on a laptop, almost all of it spent on the
[[288,12,18]] code. `--quick` skips those two steps and finishes in about a
second. Python 3.10 or newer is required (the code uses `int.bit_count`).

## What is reproduced

| Quantity | Manuscript location | Notes |
| --- | --- | --- |
| k, r_M = k/2 | eq. (rm) | rank calculation |
| dim L_ann = dim L_col = k/2 | Prop. *logical_decomp* | the colon quotient (b\*:a\*)/(b\*) is computed directly |
| one-sided logical witnesses | Sec. *logical* | exhaustive over the cosets of the stabilizer slice; the left value is an upper bound on d_ann |
| syndrome distance d_S | Thm. *ds2* | exact through weight four |
| K_M, its generators, u_1 | eq. (KM), Cor. *ambiguity* | each translation tested on a basis of the metacheck space |
| \|A^x\| and surjectivity of G -> A^x | Prop. *unitgroup* | every nonzero class of A = R/S tested for invertibility |
| partial-repetition cost | eq. (partial_cost) | the augmented syndrome code is built and its distance checked to be at least 3 |
| leader histograms (c_0, c_1, c_2, ...) | Table *leader_hist* | breadth-first search over all 2^{r_M} metasyndromes |
| sparse metacheck row weights | eq. (sparse_weights) | the deterministic basis given to BP+OSD |
| mu(g), w_amb, mu_bar | eq. (mu_g), Table *standard_codes* | exact search, see below |
| exact repair curves, the 603 p^2 coefficient | Fig. *exact_repair*, eq. (603) | evaluated from the histograms, no sampling |

Locations are given by the LaTeX labels of the manuscript, so they remain
valid if the numbering changes. The five standard codes are always audited.
`--include-diagnostics` adds the seven rows of Table *diagnostic_codes*,
including the three symmetric codes for which G -> A^x is not surjective.

### The ambiguity search

mu(g) is the lightest data error whose syndrome equals the weight-two
ambiguity e_1 + e_g. The script finds it by iterative deepening: at each step
it branches on the qubits of the lowest-index check whose syndrome bit is
still wrong. Any solution must contain one of those qubits, so the first depth
at which the search succeeds is the exact minimum. Odd weights are allowed,
which matters for the symmetric codes, whose columns have even weight.

With `--cross-check`, every mu(g) with n <= 144 is recomputed independently by
the meet-in-the-middle search described in the manuscript (all column subsets
of size at most three), and the two methods must agree through weight six.
The iterative search also reaches [[288,12,18]], where it returns
mu(g) = 14 for all three nontrivial translations, so w_amb = mu_bar = 14.
The manuscript states the weaker bound w_amb >= 8, which the script checks.

## Command-line options

| Option | Effect |
| --- | --- |
| `--include-diagnostics` | also audit the seven diagnostic rows |
| `--quick` | skip the [[288,12,18]] one-sided enumeration and mu(g) search |
| `--cross-check` | recompute mu(g) by meet-in-the-middle for n <= 144 |
| `--mu-max-weight W` | largest preimage weight tried in the mu(g) search (default 16) |
| `--outdir DIR` | output directory (default `results/`) |
| `--no-plot` | do not write the PNG |

## Output files

`reference_results/` holds the output of
`python bb_manuscript_reproduce.py --include-diagnostics --cross-check`, so a
fresh run can be compared with `diff`.

`structural_results.csv` has one row per code:

| Column | Meaning |
| --- | --- |
| `k`, `r_M` | encoded dimension and number of independent metachecks |
| `dim_L_ann`, `dim_L_col` | dimensions of Ann(b\*)/a\*Ann(b\*) and (b\*:a\*)/(b\*) |
| `onesided_left`, `onesided_right` | lightest nontrivial logical supported on one block |
| `reported_d` | quantum distance quoted in the manuscript |
| `d_S` | syndrome distance |
| `K_M_size`, `K_M_generators` | translation subgroup and a generating set |
| `unit_group_size`, `unit_map_surjective` | \|A^x\| and whether \|G/K_M\| = \|A^x\| |
| `u1` | unavoidable single-fault mis-reconstructions of metasyndrome-only repair |
| `partial_repetition_min`, `partial_augmented_dS` | checks remeasured, and the distance this achieves |
| `leader_hist` | (c_0, c_1, c_2, ...) |
| `sparse_metacheck_weights` | row weights of the BP metacheck basis |
| `mu_g`, `w_amb`, `mu_bar` | ambiguity cost per translation, its minimum and mean |

`exact_repair_curves.csv` lists P_repair^MW(p_M) on the 14-point grid of
the manuscript, with the orbit bound 1 - (1 - p_M)^{u_1} alongside;
`exact_repair_curves.png` plots both.

## Conventions

The monomial order is `1, x, ..., x^(l-1), y, xy, ..., x^(l-1) y, ...`, so
x^i y^j has coordinate `i + l*j`. The checks are H_X = [L_a | L_b] and
H = H_Z = [L_b^T | L_a^T], and the selected sector decodes X errors from the
Z syndrome, as in Sec. II of the manuscript.

## Logical component distances

The values d_ann and d_col in Table *standard_codes* are certified with the
companion package
[BB-Codes-Distance-Logical-Decomposition](https://github.com/mohammad-rowshan/BB-Codes-Distance-Logical-Decomposition)
(Rowshan and Devitt). That package works with Z-type logicals and the
projection v + (a). Under the weight-preserving map (u, v) -> (v\*, u\*), the
components of this manuscript correspond to that package's components with
**a and b interchanged**. To reproduce the manuscript's labels, pass the
polynomials in the order `b a`:

```bash
git clone https://github.com/mohammad-rowshan/BB-Codes-Distance-Logical-Decomposition.git
cd BB-Codes-Distance-Logical-Decomposition && gcc -O3 -o cluster_lb cluster_lb.c

# [[108,8,10]]: no colon-class logical below weight 10, one at weight 10, so d_col = 10
./cluster_lb -c 9 6 9  "y^3+x+x^2" "x^3+y+y^2"
./cluster_lb -c 9 6 10 "y^3+x+x^2" "x^3+y+y^2"

# [[288,12,18]]: d >= 18, and a weight-18 colon-class logical exists
./cluster_lb    12 12 17 "y^3+x+x^2" "x^3+y^2+y^7"
./cluster_lb -c 12 12 18 "y^3+x+x^2" "x^3+y^2+y^7"
```

For [[108,8,10]], d_ann = 12 follows from enumerating every logical of weight
at most 11 (complete because 11 < 2 w_K = 12) and checking that each one is a
colon class; the weight-12 left-block witness reported here as `onesided_left`
then attains the bound:

```python
from bbcode import BBCode, cluster_search
code = BBCode(9, 6, "y^3+x+x^2", "x^3+y+y^2")      # order b, a
status, hits, _ = cluster_search(code, 11, mode="logical", enumerate_all=True)
print(sorted({(bin(z).count("1"), code.component(z)) for z in hits}))
# [(10, 'C')]: only weight 10, and only colon classes
```

## Not included

The sustained phenomenological memory experiment (Fig. *sustained*,
Table *sustained_codes*) depends on the `ldpc` package (version 2.4.1) and on
adaptive Monte Carlo sampling. It is not part of this repository.

## Citation

```bibtex
@article{rowshan_bb_metachecks,
  author  = {Rowshan, Mohammad},
  title   = {BMetachecks in Bivariate Bicycle Codes: Syndrome Distance, Measurement Faults, and Repair Limits},
  year    = {2026},
  note = {arXiv:2609.39121}
}
```

The coprime-period case was presented at ISIT 2026
([arXiv:2601.01137](https://arxiv.org/abs/2601.01137)).

## License

MIT, see `LICENSE`.
