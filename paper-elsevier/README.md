# Elsevier submission package — Expert Systems with Applications

A compact rewrite of the IEEE Access draft in `../paper/`, prepared for
**Expert Systems with Applications** (ESWA, Elsevier). Both versions are
kept: this directory does not replace `../paper/`.

ESWA reviews **double-anonymised**: reviewers never see who wrote the paper.
The manuscript, the supplementary material and the highlights therefore
carry no author names, affiliations, e-mail addresses, ORCIDs or
acknowledgements. Those live on the separate title page, and the build
refuses to package if any of them leaks into a file reviewers see.

## Files

| File | Purpose |
| --- | --- |
| `manuscript.tex` | The anonymised manuscript. Pulls in `sections/01`–`08`, `tables/*.tex` and `figures/Figure_N.pdf`. 18 pages. |
| `title-page.tex` | The title page, the only file naming the authors: authors, affiliation, the corresponding author's full address and e-mail, ORCIDs, and `sections/author-statements.tex` (acknowledgements, competing interests, CRediT). |
| `highlights.tex` | Article highlights. 5 bullets, each ≤ 85 characters. |
| `supplementary-material.tex` | Tables S1–S8 and Figures S1–S6, anonymised, cited from the manuscript. 9 pages. |
| `cover-letter.tex` | Cover letter to the editor from Suseta Datta, one page, dated on each build. Every figure it quotes is taken from the manuscript. |
| `author-biographies.tex` | Vitae written for FGCS. ESWA does not ask for vitae or photographs, so they are not packaged. |
| `refs.bib`, `refs-methods.bib` | Bibliography. Local copies: edits here do not affect the IEEE version. |

Figures are separate files named `Figure_1` … `Figure_4` (manuscript) and
`Figure_S1` … `Figure_S6` (supplementary), all vector PDF with fonts embedded.

## Authors

1. **Suseta Datta** — first author. Assistant Professor, Department of
   Computer Application, Narula Institute of Technology. suseta.datta@nit.ac.in,
   ORCID 0000-0002-1661-1821.
2. **Indrajeet Mondal** — corresponding author. Same department.
   indrajeetmondal120703@gmail.com, ORCID 0009-0009-7966-2683.

Affiliation: Department of Computer Application, Narula Institute of
Technology, 81 Nilgunj Road, Agarpara, Kolkata 700109, West Bengal, India.

The blind human scoring of open-ended answers was done by Indrajeet Mondal.
The manuscript refers to him as "one of the authors", which keeps review
anonymous and stays true whatever the author order.

## Build the submission package

```
python build_submission.py
```

This writes `submission/` and `ESWA_submission.zip`: the files to upload, with
a `README.txt` saying which Editorial Manager item each one is. The script
stops instead of packaging if the abstract exceeds 250 words, there are more
than 7 keywords, a highlight exceeds 85 characters, any document has a LaTeX
error, undefined reference or overfull line, or an author name, affiliation,
e-mail or ORCID appears in the manuscript, supplementary material or
highlights — comments included, since the manuscript is uploaded as source.
It needs pdflatex, bibtex and pandoc (set `PANDOC` if pandoc is not on PATH),
and passes `--disable-installer` to MiKTeX so a build never downloads packages.

Editorial Manager compiles LaTeX with every uploaded file in one directory,
so the packaged `manuscript.tex` is a single file with the sections, tables
and bibliography inlined; the script checks it renders identically to the
multi-file source. Edit the files here, not the packaged copy.

## ESWA's rules, as checked against its guide for authors

- Double-anonymised review; title page and anonymised manuscript as separate
  files. Acknowledgements, the competing-interest declaration and the
  corresponding author's full address go on the title page.
- Abstract ≤ 250 words (this one: 249); 1–7 keywords (7); highlights
  encouraged, 3–5 bullets of ≤ 85 characters.
- No page limit. The two-column `elsarticle` layout is permitted for LaTeX;
  Word submissions must be single-column.
- The declarations tool in Editorial Manager "should always be completed".
- No vitae, photographs or ORCID requirement.

## Before upload

1. Suseta Datta's approval of the final manuscript. Elsevier requires every
   author to approve the version submitted, and the cover letter says both
   have.
2. The authors' own read of the manuscript, highlights and title page.
3. In Editorial Manager, enter the authors in the title page's order.

## Notes

- The data availability statement offers the generated outputs and labels
  from the corresponding author, which is Elsevier's fallback when data is
  not deposited. The code is neither cited nor offered.
- The generative-AI declaration mentions code because it discloses how AI
  was used; it names no one, so it stays in the anonymised manuscript.
- BibTeX reports 16 `empty pages` warnings for conference papers and preprints
  that have no page range; they do not affect the output.
