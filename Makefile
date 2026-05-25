# TwoPhase PageRank — reproducible pipeline.
#
# Targets fall into four groups: data -> evaluation -> figures -> documents.
# Each stage's output is a file, so Make will only rerun the parts whose
# inputs actually changed.
#
#   make all         everything below
#   make pipeline    data/ artefacts (popularity, smoothing, PageRank vector)
#   make evaluate    results/ (metrics + evaluation plots)
#   make figures     figures/paper/ (paper PDFs)
#   make docs        documents/ (paper, walkthrough, slides, combined note)
#   make clean       remove derived data/results/figures; keep documents
#   make distclean   clean + remove built PDFs
#
# The raw input (GPS CSVs, matched_routes.json, slc_network.xml) is treated
# as a primary source and never regenerated here.

PYTHON ?= python3
LATEX  ?= pdflatex -interaction=nonstopmode -halt-on-error
PROJ    = $(CURDIR)

# ---------- Pipeline artefacts ----------
POP_RAW      = data/popularity_results.npz
POP_SMOOTH   = data/popularity_results_smoothed_026.npz
PR_VECTOR    = data/two_phase_pagerank_vector.json

MATCHED      = data/matched_routes.json
GRAPH        = data/city_graph_full.json
NETWORK_XML  = data/slc_network.xml

# ---------- Results ----------
EVAL_JSON    = results/evaluation_results.json

# ---------- Figures ----------
FIG_STAMP    = figures/paper/.stamp

# ---------- Documents ----------
PAPER_PDF        = documents/paper/main.pdf
WALKTHROUGH_PDF  = documents/walkthrough/main.pdf
SLIDES_PDF       = documents/presentation/slides.pdf
NOTE_PDF         = documents/notes/two_phase_pagerank_combined/two_phase_pagerank_combined.pdf

# ===========================================================================
.PHONY: all pipeline evaluate figures docs paper walkthrough slides note \
        clean distclean

all: pipeline evaluate figures docs

pipeline: $(POP_RAW) $(POP_SMOOTH) $(PR_VECTOR)

evaluate: $(EVAL_JSON)

figures: $(FIG_STAMP)

docs: paper walkthrough slides note

paper: $(PAPER_PDF)
walkthrough: $(WALKTHROUGH_PDF)
slides: $(SLIDES_PDF)
note: $(NOTE_PDF)

# ---------- Pipeline rules ----------
$(POP_RAW): pipeline/analyze_popularity.py $(MATCHED) config.py
	$(PYTHON) pipeline/analyze_popularity.py

$(POP_SMOOTH): pipeline/generate_smoothed.py $(POP_RAW) $(GRAPH) config.py
	$(PYTHON) pipeline/generate_smoothed.py --gamma 0.26

$(PR_VECTOR): core/pagerank.py $(POP_SMOOTH) $(GRAPH) config.py
	$(PYTHON) core/pagerank.py

# ---------- Evaluation ----------
$(EVAL_JSON): analysis/evaluate.py $(PR_VECTOR) $(POP_SMOOTH) $(GRAPH)
	$(PYTHON) analysis/evaluate.py

# ---------- Figures ----------
$(FIG_STAMP): figures/paper/generate_figures.py $(POP_RAW) $(EVAL_JSON) \
              documents/reports/data/smoothing_grid_search_results.json \
              documents/reports/data/verification_results.json
	$(PYTHON) figures/paper/generate_figures.py
	@touch $@

# ---------- Documents ----------
$(PAPER_PDF): documents/paper/main.tex documents/paper/references.bib \
              $(FIG_STAMP) | _paper_figures_dep
	cd documents/paper && $(LATEX) main.tex && bibtex main || true
	cd documents/paper && $(LATEX) main.tex
	cd documents/paper && $(LATEX) main.tex

$(WALKTHROUGH_PDF): documents/walkthrough/main.tex
	cd documents/walkthrough && $(LATEX) main.tex
	cd documents/walkthrough && $(LATEX) main.tex

$(SLIDES_PDF): documents/presentation/slides.tex $(FIG_STAMP)
	cd documents/presentation && $(LATEX) slides.tex
	cd documents/presentation && $(LATEX) slides.tex

$(NOTE_PDF): documents/notes/two_phase_pagerank_combined/two_phase_pagerank_combined.tex
	cd documents/notes/two_phase_pagerank_combined && \
	    $(LATEX) two_phase_pagerank_combined.tex && \
	    $(LATEX) two_phase_pagerank_combined.tex

# The paper also pulls in paper-only figures from its local figures/ dir.
# This empty rule exists so Make re-evaluates the paper if any of those
# assets change (they are not produced by our figure generator).
_paper_figures_dep: $(wildcard documents/paper/figures/*.png)
	@true

# ---------- Housekeeping ----------
clean:
	rm -f $(POP_RAW) $(POP_SMOOTH) $(PR_VECTOR) $(EVAL_JSON) \
	      $(FIG_STAMP) figures/paper/*.pdf figures/evaluation/*.pdf \
	      figures/smoothing/*.pdf
	rm -f results/evaluation_results.txt

distclean: clean
	rm -f $(PAPER_PDF) $(WALKTHROUGH_PDF) $(SLIDES_PDF) $(NOTE_PDF)
	find documents -type f \( -name '*.aux' -o -name '*.log' -o -name '*.out' \
	    -o -name '*.bbl' -o -name '*.blg' -o -name '*.toc' \) -delete
