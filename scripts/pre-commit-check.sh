#!/bin/bash
# Pre-commit check script
# Run this before committing to ensure code quality

set -e  # Exit on error

echo "🔍 Running pre-commit checks..."

# 1. Formatting check
echo ""
echo "1️⃣ Checking code formatting..."
uv run ruff format --check .
echo "✅ Formatting check passed"

# 2. Linting check
echo ""
echo "2️⃣ Running linter..."
uv run ruff check .
echo "✅ Linting check passed"

# 3. Type checking
echo ""
echo "3️⃣ Running type checker..."
npx pyright
echo "✅ Type check passed"

# 4. Build documentation
echo ""
echo "4️⃣ Building documentation..."
uv run mkdocs build --strict
echo "✅ Documentation built successfully"

echo ""
echo "✨ All pre-commit checks passed!"
echo "You can now commit and push your changes."
