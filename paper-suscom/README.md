# Submission package — Sustainable Computing: Informatics and Systems

The manuscript prepared for **Sustainable Computing: Informatics and Systems**
(SUSCOM, Elsevier), submitted through Editorial Manager at
<https://www.editorialmanager.com/suscom/default.aspx>. It is built on the ESWA
package in `../paper-elsevier/`, which is left unchanged.

## What differs from the ESWA package

| | ESWA (`paper-elsevier/`) | SUSCOM (this directory) |
| --- | --- | --- |
| Review | Double-anonymised | **Single anonymised**: authors are named |
| Title page | Separate file | Part of the manuscript (`\author`, `\affiliation`, corresponding author) |
| Author statements | Separate title page | CRediT, competing interests, generative-AI declaration and acknowledgements in the manuscript, before the references |
| Build checks | Anonymity scan | No anonymity scan; checks for vertical rules in tables and for missing figures |
| Figures | `Figure_1`–`4` | `Figure_1`–`5`; Figure 1 is the new architecture diagram, the old figures shift up by one |
| Supplementary tables | S1 is the prior-work comparison | That comparison moved into the manuscript as Table 1, so the other tables shift: gates S1–S2, overhead S3, carbon S4, family accuracy S5, public pool S6, hosted cells S7 |
| Data | Statement offers data on request | Option C applies: deposit in a repository and link, or explain why not (see below) |

## Additions

- **Architecture diagram** — Figure 1, Section 3.1. Source in
  `figures-src/architecture.tex` (TikZ); `build_submission.py` rebuilds it into
  `figures/Figure_1.pdf`.
- **Comparison with related work** — Table 1 in Section 2.4, with prose reading it.
- **Datasets** — new Section 4: corpus table, splits, records collected, label
  statistics, the public and hosted pools, licences. The duplicated corpus text
  in Section 5 now points to it.

## Rules taken from the guide for authors

Abstract ≤ 250 words (249); 1–7 keywords (7); highlights 3–5 bullets of ≤ 85
characters in a separate editable file with "highlights" in its name;
numbered sections and subsections; tables as editable text without vertical
rules; each figure a separate vector file with fonts embedded; references
numbered in order of appearance; the generative-AI declaration in its own
section before the references; acknowledgements directly before the
references; `.tex` source rather than PDF; two-column layout is allowed for
LaTeX.

## Build

```
python build_submission.py
```

Writes `submission/` and `SUSCOM_submission.zip`, with a `README.txt` saying
which Editorial Manager item each file is. It stops if a rule above is broken
or if LaTeX reports an error, an undefined reference or an overfull line. It
needs pdflatex, bibtex and pandoc.

## Before upload

1. **Data statement.** The guide requires depositing the research data in a
   repository and linking it, or stating why that is not possible. The
   Data availability section still offers the data on request, which is the
   ESWA wording and does not meet Option C. Decide and update the section and
   the Editorial Manager data statement together.
2. **Prior submission.** The cover letter says the manuscript is not under
   consideration elsewhere. That is true only if the earlier submission to
   another journal has been withdrawn or decided.
3. Suseta Datta's approval of the final text, which the cover letter states.
4. The generative-AI declaration now mentions the architecture diagram.
   Confirm it describes your use accurately.
