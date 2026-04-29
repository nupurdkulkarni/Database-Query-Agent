#!/bin/bash
set -e

echo "🔍 Running Python Linter (Ruff)..."
python3 -m ruff check .

echo "✨ Running Python Formatter Check (Ruff)..."
python3 -m ruff format --check .

echo "⚛️ Running Frontend Linter (ESLint)..."
cd frontend && npm run lint

echo "✅ All linting checks passed!"
