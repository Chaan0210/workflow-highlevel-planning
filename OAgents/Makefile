.PHONY: quality style test docs utils

check_dirs := src example

# Check code quality of the source code
quality:
	ruff check $(check_dirs)
	ruff format --check $(check_dirs)
	@if [ -f utils/check_tests_in_ci.py ]; then python utils/check_tests_in_ci.py; else echo "Skipping utils/check_tests_in_ci.py (not found)"; fi

# Format source code automatically
style:
	ruff check $(check_dirs) --fix
	ruff format $(check_dirs)
	
# Run smolagents tests
test:
	pytest ./tests/
