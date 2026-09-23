# Elsevier submission package

A compact rewrite of the IEEE Access draft in `../paper/`, prepared to the
Elsevier guide for authors and held to 18 pages including all figures and
tables. Both versions are kept: this directory does not replace `../paper/`.

## Files

| File | Purpose |
| --- | --- |
| `manuscript.tex` | The manuscript. Pulls in `sections/*.tex`, `tables/*.tex` and `figures/Figure_N.pdf`. **18 pages.** |
| `highlights.tex` | Article highlights, submitted separately. 5 bullets, each ≤ 85 characters. |
| `supplementary-material.tex` | Tables S1–S8 and Figures S1–S6, cited from the manuscript. 9 pages. |
| `author-biographies.tex` | Vitae, submitted separately as the guide asks, with photographs to follow as `Photo_1`, `Photo_2`. |
| `refs.bib`, `refs-methods.bib` | Bibliography. Local copies: edits here do not affect the IEEE version. |

Figures are separate files named `Figure_1` … `Figure_4` (manuscript) and
`Figure_S1` … `Figure_S6` (supplementary), all vector PDF.

## Build

```
pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript
pdflatex highlights
pdflatex supplementary-material && pdflatex supplementary-material
pdflatex author-biographies
```

The class is `elsarticle` with `final,5p,times,twocolumn`: the journal's own
two-column final format, which the guide permits for LaTeX submissions. The
single-column `preprint` option produces the same content in 28 pages, over
the 18-page target.

## What the authors still need to fill in

Target journal: **Future Generation Computer Systems** (Elsevier), set in
`\journal{}`.

Settled: both authors and their emails, no competing interests, no funding,
the generative-AI declaration, and the data availability statement.

Outstanding, all written in square brackets so `grep -rn "\[" *.tex
sections/*.tex` finds them:

1. Whether the first author needs a "present address" footnote, if he has
   left the institute since the work was carried out. Elsevier keeps the
   address where the work was done as the main affiliation and adds the
   current one as a numbered footnote.
2. CRediT roles for Suseta Datta — Supervision, Validation, Writing – review
   and editing are proposed in `sections/08-back-matter.tex`; confirm or edit.
3. Author biographies (≤ 100 words each) in `author-biographies.tex`, and
   passport-type photographs as `Photo_1` and `Photo_2`.
4. A persistent identifier for the software reference `mondal2026arbiter`,
   once a release is deposited somewhere that issues a DOI. Until then the
   entry carries a bracketed note instead of a DOI.

The affiliation is complete: both authors are in the Department of Computer
Application, Narula Institute of Technology, 81 Nilgunj Road, Agarpara,
Kolkata 700109, West Bengal, India.

Author order is Suseta Datta, then Indrajeet Mondal. Correspondence still
comes to Indrajeet Mondal; a comment in `manuscript.tex` marks where to move
`\corref` if it should go to the first author instead. Because the order
changed, the passages describing the blind human labelling now say "the
second author", which is who produced those labels.

## Notes

- The abstract is exactly 250 words, the guide's limit.
- The data availability statement points to the corresponding author for the
  generated outputs, labels and analysis code, which is the guide's Option C
  fallback when data is not deposited.
- BibTeX reports 16 `empty pages` warnings for conference papers and preprints
  that have no page range; they do not affect the output.
- Material that did not fit the 18 pages was moved to the supplementary file
  rather than deleted, and each item is cited from the manuscript text.
