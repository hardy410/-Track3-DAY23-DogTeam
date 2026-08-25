.PHONY: install test lint typecheck run-scenarios grade-local ui clean

install:
	pip install -e '.[dev,openai,sqlite,ui]'

test:
	pytest

lint:
	ruff check src tests

typecheck:
	mypy src

run-scenarios:
	python -m langgraph_agent_lab.cli run-scenarios --config configs/lab.yaml --output outputs/metrics.json

grade-local:
	python -m langgraph_agent_lab.cli validate-metrics --metrics outputs/metrics.json

ui:
	streamlit run src/langgraph_agent_lab/ui.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov dist build *.egg-info outputs/*.json
