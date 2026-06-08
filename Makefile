# Makefile for llm-eval-platform — convenience wrappers over the `eval-platform` CLI.
# Override the interpreter on Windows if needed:  make smoke PYTHON=py
PYTHON ?= python
ITEMS ?= data/sample_eval.jsonl
CANDIDATES ?= data/sample_candidates.jsonl

.PHONY: help install verify-stats smoke run-pointwise run-pairwise test report dashboard clean

help:                ## list targets
	@echo "Targets: install verify-stats smoke run-pointwise run-pairwise test report dashboard clean"

install:             ## editable install with all optional extras (litellm, streamlit, pytest, ...)
	$(PYTHON) -m pip install -e .[all]

verify-stats:        ## check the hand-coded stats against scipy/sklearn/fixtures
	$(PYTHON) handcode/verify_stats.py

smoke:               ## wire-check provider+store+cache+retry+ratelimit WITHOUT the judge
	$(PYTHON) -m evalplatform.cli init-db
	$(PYTHON) -m evalplatform.cli smoke --n 5

run-pointwise:       ## run a pointwise eval over the sample data (needs the hand-coded judge)
	$(PYTHON) -m evalplatform.cli run --mode pointwise --items $(ITEMS) --candidates $(CANDIDATES)

run-pairwise:        ## run a pairwise eval over the sample data (needs the hand-coded judge + swap)
	$(PYTHON) -m evalplatform.cli run --mode pairwise --items $(ITEMS) --candidates $(CANDIDATES)

test:                ## run the test suite (requires .[dev])
	$(PYTHON) -m pytest

report:              ## build the static HTML report for the latest (or RUN_ID=...) run
	$(PYTHON) -m evalplatform.cli report $(if $(RUN_ID),--run-id $(RUN_ID),)

dashboard:           ## launch the Streamlit dashboard (requires .[dashboard])
	$(PYTHON) -m evalplatform.cli dashboard

clean:               ## remove the local DB, reports, caches, and build artifacts
	$(PYTHON) -c "import shutil,glob,os; [shutil.rmtree(p,ignore_errors=True) for p in ['reports','.pytest_cache','build','dist']]; [os.remove(p) for p in glob.glob('data/*.db')+glob.glob('data/*.db-*')]; [shutil.rmtree(p,ignore_errors=True) for p in glob.glob('**/__pycache__',recursive=True)+glob.glob('*.egg-info')+glob.glob('src/*.egg-info')]"
