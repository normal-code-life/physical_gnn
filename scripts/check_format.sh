#!/usr/bin/env bash

# Run the repository's configured static style check.
echo "-- flake8 --"
flake8 --max-line-length=120
