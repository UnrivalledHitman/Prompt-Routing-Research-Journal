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

Every placeholder is written in square brackets, so `grep -rn "\[" *.tex
sections/*.tex` finds them all.

1. Co-author name, and the full affiliation with postal address and country.
2. Corresponding author's email address.
3. CRediT roles for the co-author.
4. Declaration of competing interest, including the pending patent application.
5. Declaration on generative AI in the writing process.
6. Acknowledgements and funding.
7. Author biographies (≤ 100 words each) and passport-type photographs.
8. A persistent identifier for the software reference `mondal2026arbiter`,
   once a release is deposited somewhere that issues a DOI.

## Notes

- The abstract is exactly 250 words, the guide's limit.
- The data availability statement explains why the generated outputs and
  analysis code are not deposited publicly (pending patent application) and
  points to the corresponding author, which is the guide's Option C fallback.
- BibTeX reports 16 `empty pages` warnings for conference papers and preprints
  that have no page range; they do not affect the output.
- Material that did not fit the 18 pages was moved to the supplementary file
  rather than deleted, and each item is cited from the manuscript text.
