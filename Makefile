# Phase 0 — one-command setup and reproducible entry points.
# Usage: make setup | make train | make backtest | make test | make lock | make repro
SEED ?= 42
EPISODES ?= 500
CONFIG ?= tuning/best_config.json

SEEDS ?= 0 1 2 3 4

.PHONY: help setup lock train backtest tune test repro evaluate clean

help:
	@echo "Targets:"
	@echo "  setup     Create the conda env from environment.yml"
	@echo "  lock      Freeze the current env into requirements.lock.txt"
	@echo "  train     Train (SEED=$(SEED) EPISODES=$(EPISODES) CONFIG=$(CONFIG))"
	@echo "  backtest  Backtest the saved checkpoint (SEED=$(SEED))"
	@echo "  tune      Run Ray Tune HPO"
	@echo "  test      Run the pytest suite"
	@echo "  repro     Train twice with the same seed and diff the logs"
	@echo "  evaluate  Phase 1 multi-seed eval (SEEDS=$(SEEDS) EPISODES=$(EPISODES) CONFIG=$(CONFIG))"

setup:
	conda env create -f environment.yml || conda env update -f environment.yml

# Capture the exact resolved versions from the active environment.
lock:
	pip freeze > requirements.lock.txt
	@echo "Wrote requirements.lock.txt"

train:
	python main.py --mode train --seed $(SEED) --episodes $(EPISODES) --config $(CONFIG)

backtest:
	python main.py --mode backtest --seed $(SEED)

tune:
	python main.py --mode tune

test:
	python -m pytest test/ -q

# Phase 0 acceptance check: two same-seed runs must produce identical logs.
repro:
	python main.py --mode train --seed $(SEED) --episodes 5
	cp checkpoints/training_log.csv /tmp/run_a.csv
	python main.py --mode train --seed $(SEED) --episodes 5
	cp checkpoints/training_log.csv /tmp/run_b.csv
	@diff -q /tmp/run_a.csv /tmp/run_b.csv && echo "REPRODUCIBLE: logs identical" \
		|| echo "NOT reproducible: logs differ"

# Phase 1: honest multi-seed evaluation with CIs, significance test, diagnostics.
evaluate:
	python experiments/multi_seed.py --seeds $(SEEDS) --episodes $(EPISODES) --config $(CONFIG)

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache
