#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = map_completeness_estimation

-include .python-version.mk
#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

.PHONY: pyver
pyver:
	@echo "Finding highest Python version..."
	@highest_python=$$(find /usr/bin /usr/local/bin -name "python3*" -type f -executable 2>/dev/null | grep -Ev "config|m$$" | while read py; do \
		if [ -x "$$py" ]; then \
			$$py -c "import sys; print('{}.{}.{} {}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2], '$$py'))"; \
		fi; \
	done | sort -V | tail -1 | awk '{print $$NF}'); \
	echo "Highest Python version found: $$highest_python"; \
	echo "PYTHON_INTERPRETER := $$highest_python" > .python-version.mk; \
	echo "Set PYTHON_INTERPRETER to $$highest_python"


## Set up Python interpreter environment
.PHONY: create_environment
create_environment: pyver
	@bash -c "$(PYTHON_INTERPRETER) -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install -e ."
	@echo ">>> Environment created and configured. Remember to activate it with:\nsource .venv/bin/activate"
	


## Make dataset from raw images 
.PHONY: preprocess_data
preprocess_data:
	@bash -c "source .venv/bin/activate && python3 src/script/csv_from_raw_data.py"
	

## Launch hyperparameter optimization study of each model
.PHONY: optimize_hyperparameters
optimize_hyperparameters:
	@bash -c "source .venv/bin/activate && python3 src/script/hyperopt.py"

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
