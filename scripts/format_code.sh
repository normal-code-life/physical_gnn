#!/usr/bin/env bash

# Apply the repository's Python formatting and import-order conventions.
echo "-- black --"
black --line-length=120 .

echo "-- isort --"
isort .
