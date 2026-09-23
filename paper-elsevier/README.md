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

## Build the submission package

```
python build_submission.py
```

This writes `submission/` and `FGCS_submission.zip`: the files to upload,
with a `README.txt` saying which Editorial Manager item each one is. The
script stops with an error instead of producing a package if the manuscript
exceeds 18 pages, the abstract exceeds 250 words, or a highlight exceeds 85
characters. It needs pdflatex, bibtex and pandoc (set `PANDOC` if pandoc is
not on PATH), and passes `--disable-installer` to MiKTeX so a build never
downloads packages.

Editorial Manager compiles LaTeX with every uploaded file in one directory,
so the packaged `manuscript.tex` is a single file with the sections, tables
and bibliography inlined; the script checks it renders identically to the
multi-file source. Edit the files here, not the packaged copy.

The class is `elsarticle` with `final,5p,times,twocolumn`, which is what FGCS
requires: a Full Length Article "should be strictly double column, single
spaced and limited to 18 pages; including all figures, tables, references. A
manuscript longer than 18 pages will not be considered for review and
returned to author to be revised to the correct format."

The manuscript is at exactly 18 pages, so there is no slack. Anything added
has to be paid for by cutting something else, or moved to the supplementary
file, which is separate and does not count.

FGCS does not use graphical abstracts; the term does not appear in its guide.

## What the authors still need to fill in

Target journal: **Future Generation Computer Systems** (Elsevier), set in
`\journal{}`.

Settled: both authors and their emails, no competing interests, no funding,
the generative-AI declaration, and the data availability statement.

Settled since: the biographies (96 and 85 words) and the photographs. The
photographs live in `photos/`, which git ignores so personal images stay out
of the repository's history; `build_submission.py` copies them into the
package and warns if they are missing. Highlights and biographies are
packaged as `.docx`.

Outstanding:

1. **Reference [61], the software citation**, prints a bracketed note in
   place of a persistent identifier. Replace it with a DOI once a release is
   deposited, or reword it; do not submit with the note in place.
2. CRediT roles for Suseta Datta — Supervision, Validation, Writing – review
   and editing are proposed in `sections/08-back-matter.tex`; confirm or edit.
3. **ORCID for the corresponding author.** FGCS verifies author identity
   through it: "the visibility of the corresponding author's ORCID is a step
   in this process… the manuscript will be rejected until the record owner
   updates their setting." Register at orcid.org if needed, and set the
   record's visibility to Everyone, not Trusted parties or Only me.
4. Whether the first author needs a "present address" footnote, if he has
   left the institute since the work was carried out.

The affiliation is complete: both authors are in the Department of Computer
Application, Narula Institute of Technology, 81 Nilgunj Road, Agarpara,
Kolkata 700109, West Bengal, India.

Author order is Indrajeet Mondal (first and corresponding author), then
Suseta Datta. The passages describing the blind human labelling say "the
first author", who produced those labels; if the order ever changes, those
five passages in `sections/04-experimental-setup.tex` and
`sections/05-results.tex` must change with it.

## Notes

- The abstract is 249 words against the guide's 250.
- The data availability statement points to the corresponding author for the
  generated outputs, labels and analysis code, which is the guide's Option C
  fallback when data is not deposited.
- BibTeX reports 16 `empty pages` warnings for conference papers and preprints
  that have no page range; they do not affect the output.
- Material that did not fit the 18 pages was moved to the supplementary file
  rather than deleted, and each item is cited from the manuscript text.
