"""Build the Sustainable Computing: Informatics and Systems submission package.

    python build_submission.py

Writes submission/ and SUSCOM_submission.zip next to this file, holding every
file the journal's guide for authors asks for, and stops with an error if a
rule is broken: the 250-word abstract, 1-7 keywords, 3-5 highlights of at
most 85 characters, a vertical rule in a table, a missing figure file, a
LaTeX error, an undefined reference or an overfull line.

The journal follows a single anonymised review: reviewers do not see who the
authors are, but the authors are named, so nothing is stripped from the
manuscript and there is no separate title page.

Editorial Manager compiles LaTeX with every uploaded file in one directory,
so the manuscript is flattened into a single .tex with the sections, tables
and bibliography inlined and the figures referenced by bare file name.
Figure 1 is drawn from figures-src/architecture.tex and rebuilt here.

Needs pdflatex and bibtex, and pandoc for the Word file. Set PANDOC to
pandoc's path if it is not on PATH.
"""

import glob
import os
import pathlib
import re
import shutil
import subprocess
import sys
import zipfile
from typing import NoReturn

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "submission"
ZIP = HERE / "SUSCOM_submission.zip"
ABSTRACT_LIMIT = 250
KEYWORD_LIMIT = 7
HIGHLIGHT_CHARS = 85
FIGURES = [f"Figure_{n}.pdf" for n in range(1, 6)]
BUILD_JUNK = (".aux", ".log", ".out", ".spl", ".blg", ".bbl")


def fail(msg: str) -> NoReturn:
    """Stop the build with an error message."""
    sys.exit(f"build_submission: {msg}")


def run(cmd: list[str], cwd: pathlib.Path) -> None:
    """Run a command quietly; its log file, not its exit code, is checked."""
    subprocess.run(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def is_miktex() -> bool:
    """Return whether the pdflatex on PATH belongs to MiKTeX."""
    out = subprocess.run(["pdflatex", "--version"], capture_output=True, text=True)
    return "MiKTeX" in out.stdout


# MiKTeX installs missing packages on the fly; never let a build download.
NO_INSTALL = ["--disable-installer"] if is_miktex() else []


def latex(name: str, cwd: pathlib.Path, bib: bool = False) -> str:
    """Compile name.tex in cwd and return its log text."""
    tex = ["pdflatex", *NO_INSTALL, "--interaction=nonstopmode", f"{name}.tex"]
    run(tex, cwd)
    if bib:
        run(["bibtex", name], cwd)
    run(tex, cwd)
    run(tex, cwd)
    return (cwd / f"{name}.log").read_text(encoding="utf-8", errors="replace")


def check_log(log: str, name: str) -> int:
    """Fail on errors, undefined references or overfull boxes; return the page count."""
    errors = re.findall(r"(?m)^! .*", log)
    undefined = re.findall(r"(?:Citation|Reference) `([^']+)'[^\n]*undefined", log)
    overfull = log.count("Overfull \\hbox")
    pages = re.search(rf"Output written on {name}\.pdf \((\d+) pages?", log)
    if errors:
        fail(f"{name}: LaTeX errors: {errors[:3]}")
    if undefined:
        fail(f"{name}: undefined references: {sorted(set(undefined))}")
    if not pages:
        fail(f"{name}: no PDF was written")
    if overfull:
        fail(f"{name}: {overfull} overfull box(es); something runs into the margin")
    return int(pages.group(1))


def build_architecture() -> None:
    """Draw Figure 1 from its TikZ source and place it with the other figures."""
    src = HERE / "figures-src"
    log = latex("architecture", src)
    errors = re.findall(r"(?m)^! .*", log)
    if errors:
        fail(f"architecture: LaTeX errors: {errors[:3]}")
    if "Output written on architecture.pdf" not in log:
        fail("architecture: no PDF was written")
    shutil.copy2(src / "architecture.pdf", HERE / "figures" / "Figure_1.pdf")
    for junk in src.iterdir():
        if junk.suffix in BUILD_JUNK:
            junk.unlink()


def check_figures() -> None:
    """Fail if a figure the manuscript includes is missing, or one is unused."""
    for fig in FIGURES:
        if not (HERE / "figures" / fig).exists():
            fail(f"missing figure file figures/{fig}")
    text = "".join(p.read_text(encoding="utf-8") for p in (HERE / "sections").glob("*.tex"))
    used = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{(Figure_\d)\}", text))
    expected = {f.removesuffix(".pdf") for f in FIGURES}
    if used != expected:
        fail(f"manuscript includes {sorted(used)}, expected {sorted(expected)}")


def check_tables(text: str, name: str) -> None:
    """Fail on a vertical rule in a tabular column specification.

    The guide asks authors to avoid vertical rules and shading in tables.
    """
    for spec in re.findall(
        r"\\begin\{tabular[x*]?\}(?:\[[^\]]*\])?(?:\{[^}]*\})?\{([^}]*)\}", text
    ):
        if "|" in spec:
            fail(f"{name}: vertical rule in table column specification '{spec}'")


def inline_inputs(text: str, base: pathlib.Path) -> str:
    r"""Replace every \input{...} with the file's contents, recursively."""

    def repl(m: re.Match[str]) -> str:
        path = base / m.group(1)
        if path.suffix != ".tex":
            path = path.with_suffix(".tex")
        return inline_inputs(path.read_text(encoding="utf-8"), base)

    return re.sub(r"\\input\{([^}]+)\}", repl, text)


def flatten_manuscript() -> str:
    """Return the manuscript as one file, with inputs and bibliography inlined."""
    text = inline_inputs((HERE / "manuscript.tex").read_text(encoding="utf-8"), HERE)
    bbl = (HERE / "manuscript.bbl").read_text(encoding="utf-8")
    text = text.replace("\\bibliographystyle{elsarticle-num}\n", "")
    text = text.replace("\\bibliography{refs,refs-methods}", bbl.strip())
    text = re.sub(r"\\graphicspath\{\{[^}]*\}\}\n", "", text)
    text = re.sub(r"(?m)^[ \t]*%.*\n", "", text)
    header = (
        "% The escalation tax -- manuscript for Sustainable Computing: Informatics\n"
        "% and Systems. Single-file source generated by build_submission.py:\n"
        "% sections, tables and bibliography are inlined, and figures are the\n"
        "% separately uploaded Figure_1.pdf to Figure_5.pdf.\n"
    )
    return header + text


def check_abstract(text: str) -> int:
    """Fail if the abstract exceeds the word limit; return its word count."""
    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.S)
    if match is None:
        fail("manuscript has no abstract environment")
    body = match.group(1)
    body = body.replace("{,}", ",")  # 3{,}612 is one word
    body = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", body)
    body = body.translate(str.maketrans("{}~$\\", "     "))
    words = [w for w in body.split() if any(c.isalnum() for c in w)]
    if len(words) > ABSTRACT_LIMIT:
        fail(f"abstract is {len(words)} words, limit {ABSTRACT_LIMIT}")
    return len(words)


def check_keywords(text: str) -> int:
    """Fail unless there are 1 to 7 keywords; return how many."""
    match = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", text, re.S)
    if match is None:
        fail("manuscript has no keyword environment")
    block = match.group(1)
    count = len([k for k in block.split("\\sep") if k.strip()])
    if not 1 <= count <= KEYWORD_LIMIT:
        fail(f"{count} keywords, need 1 to {KEYWORD_LIMIT}")
    return count


def check_highlights() -> int:
    """Fail unless there are 3-5 highlights within the length limit; return the longest."""
    text = (HERE / "highlights.tex").read_text(encoding="utf-8")
    items = [
        " ".join(i.split())
        for i in re.findall(r"\\item(.*?)(?=\\item|\\end\{itemize\})", text, re.S)
    ]
    if not 3 <= len(items) <= 5:
        fail(f"{len(items)} highlights, need 3 to 5")
    long = [i for i in items if len(i) > HIGHLIGHT_CHARS]
    if long:
        fail(f"highlights over {HIGHLIGHT_CHARS} characters: {long}")
    return max(len(i) for i in items)


def find_pandoc() -> str:
    """Return the path to pandoc, from PANDOC, PATH or a per-user install."""
    candidates = [os.environ.get("PANDOC"), shutil.which("pandoc")]
    local = os.environ.get("LOCALAPPDATA", "")
    candidates += sorted(glob.glob(os.path.join(local, "Programs", "pandoc-*", "pandoc.exe")))
    for c in candidates:
        if c and pathlib.Path(c).exists():
            return c
    fail("pandoc not found; set PANDOC to its path")


def to_docx(pandoc: str, src: pathlib.Path, dest: pathlib.Path) -> None:
    """Convert a LaTeX file to Word with British English spell-checking."""
    subprocess.run(
        [
            pandoc,
            str(src),
            "-f",
            "latex",
            "-t",
            "docx",
            "-M",
            "lang=en-GB",
            "-o",
            str(dest),
        ],
        cwd=HERE,
        check=True,
    )


README = """SUSCOM submission package -- The escalation tax
===============================================

Sustainable Computing: Informatics and Systems follows a single anonymised
review: reviewers do not see who the authors are, but the authors are named,
so the manuscript carries the author block, the CRediT statement, the
competing-interest and generative-AI declarations and the acknowledgements.
Submit at https://www.editorialmanager.com/suscom/default.aspx.

Upload to Editorial Manager as follows.

  Cover_letter.pdf          Cover letter, from Suseta Datta. It states that
                            both authors have approved the manuscript; make
                            sure that is true before uploading.
  manuscript.tex            Manuscript (LaTeX source), a single file. Upload
                            together with the five figures, which it
                            includes by file name. A PDF is not accepted as
                            a source file.
  Figure_1.pdf ... _5.pdf   Figures, one file each, vector PDF with fonts
                            embedded. Figure_1 is the architecture diagram.
  highlights.docx           Highlights (5 bullets, each <= 85 characters),
                            a separate editable file with "highlights" in
                            its name.
  Supplementary_material.pdf  Supplementary material (Tables S1-S7,
                            Figures S1-S6), cited from the manuscript.
  manuscript.pdf            For checking only: the PDF this source builds.
                            Editorial Manager builds its own from the .tex.

In the submission system, not as files:
  - the authors in this order: Suseta Datta, then Indrajeet Mondal, who is
    the corresponding author; the order must match the manuscript;
  - the data statement. For this journal Option C of Elsevier's research data
    guidelines applies: deposit the research data in a repository and link it
    here, or state why it cannot be shared. The manuscript's Data availability
    section must say the same.
  - the competing-interest declaration and the generative-AI disclosure,
    matching the manuscript.
"""


def main() -> None:
    """Build, check and package every file the submission needs."""
    build_architecture()
    check_figures()
    pages = check_log(latex("manuscript", HERE, bib=True), "manuscript")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()

    flat = flatten_manuscript()
    words = check_abstract(flat)
    keywords = check_keywords(flat)
    check_tables(flat, "manuscript")
    (OUT / "manuscript.tex").write_text(flat, encoding="utf-8")
    for fig in FIGURES:
        shutil.copy2(HERE / "figures" / fig, OUT / fig)

    # The flattened source must build on its own, as Editorial Manager will.
    flat_pages = check_log(latex("manuscript", OUT), "manuscript")
    if flat_pages != pages:
        fail(f"flattened source builds to {flat_pages} pages, not {pages}")
    shutil.copy2(OUT / "manuscript.pdf", HERE / "manuscript.pdf")
    for junk in OUT.iterdir():
        if junk.suffix in BUILD_JUNK:
            junk.unlink()

    supplementary = (HERE / "supplementary-material.tex").read_text(encoding="utf-8")
    check_tables(supplementary, "supplementary-material.tex")
    supp_pages = check_log(latex("supplementary-material", HERE), "supplementary-material")
    shutil.copy2(HERE / "supplementary-material.pdf", OUT / "Supplementary_material.pdf")

    letter_pages = check_log(latex("cover-letter", HERE), "cover-letter")
    shutil.copy2(HERE / "cover-letter.pdf", OUT / "Cover_letter.pdf")

    longest = check_highlights()
    pandoc = find_pandoc()
    to_docx(pandoc, HERE / "highlights.tex", OUT / "highlights.docx")

    (OUT / "README.txt").write_text(README, encoding="utf-8")

    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.iterdir()):
            z.write(f, f.name)

    print(f"manuscript: {pages} pages; flattened source builds identically")
    print(
        f"abstract: {words} words (limit {ABSTRACT_LIMIT}); keywords: {keywords} "
        f"(limit {KEYWORD_LIMIT}); highlights: longest {longest} characters "
        f"(limit {HIGHLIGHT_CHARS})"
    )
    print(f"supplementary: {supp_pages} pages; cover letter: {letter_pages} page(s)")
    print(f"wrote {OUT.name}/ ({len(list(OUT.iterdir()))} files) and {ZIP.name}")


if __name__ == "__main__":
    main()
