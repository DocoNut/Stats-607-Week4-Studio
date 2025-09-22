#!/bin/bash

# Bash script to run tests in test_bootstrap.py

# Exit immediately if a command exits with a non-zero status
set -e

# Check if pytest is installed
if ! command -v pytest &> /dev/null
then
    echo "pytest is not installed. Please install it using: pip install pytest"
    exit 1
fi

# Run test_bootstrap.py using pytest
pytest test_bootstrap.py

# Exit with pytest's exit code
exit $?
