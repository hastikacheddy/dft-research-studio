# Makefile — DFT Research Studio
# Usage: make <target>

.PHONY: install pin-deps test test-fast coverage lint format \
        serve-api serve-app serve-standalone \
        docker-build docker-up docker-down \
        debug run resume setup-wandb

# ── Install ──────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm
	python -m nltk.downloader punkt punkt_tab

# Pin exact dependency versions for full reproducibility
pin-deps:
	pip freeze | grep -v "^-e" > requirements.lock
	@echo "Pinned versions saved to requirements.lock"

# ── Tests ─────────────────────────────────────────────────────────────
test:
	pytest tests/ -m "not integration and not slow"

test-fast:
	pytest tests/ -x --tb=short -m "not integration and not slow"

coverage:
	pytest tests/ -m "not integration and not slow" \
	    --cov=dft_research_studio \
	    --cov-report=term-missing \
	    --cov-report=html:htmlcov \
	    --cov-fail-under=70

# ── Lint & format ─────────────────────────────────────────────────────
lint:
	ruff check dft_research_studio/ tests/ serving/

format:
	ruff format dft_research_studio/ tests/ serving/
	ruff check --fix dft_research_studio/ tests/ serving/

# ── Serving ───────────────────────────────────────────────────────────
serve-api:
	python serving/litserve_api.py

serve-app:
	python app.py

serve-standalone:
	python app.py --standalone

# ── Docker ────────────────────────────────────────────────────────────
docker-build:
	docker build -f Dockerfile.serve -t dft-serve .
	docker build -f Dockerfile.app   -t dft-app   .

docker-up:
	docker compose up --build

docker-down:
	docker compose down

# ── Experiment pipeline ───────────────────────────────────────────────
debug:
	python main.py --debug

run:
	python main.py

resume:
	python main.py --resume

# Run with W&B tracking (set WANDB_API_KEY first)
run-tracked:
	python main.py --wandb dft-research-studio

# ── W&B setup ─────────────────────────────────────────────────────────
setup-wandb:
	pip install wandb
	wandb login
