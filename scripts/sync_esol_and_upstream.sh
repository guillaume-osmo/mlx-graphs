#!/usr/bin/env bash
#
# Sync guillaume-osmo/mlx-graphs with:
#   1) thegodone/mlx-graphs  -> get and push the "esol" branch
#   2) mlx-graphs/mlx-graphs -> upstream, sync main
#
# Run from repo root:  ./scripts/sync_esol_and_upstream.sh
# You must have push access to https://github.com/guillaume-osmo/mlx-graphs
#
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Remotes: push target = guillaume-osmo; sources = thegodone (esol), upstream (main)
GUILLAUME_URL="https://github.com/guillaume-osmo/mlx-graphs.git"
THEGODONE_URL="https://github.com/thegodone/mlx-graphs.git"
UPSTREAM_URL="https://github.com/mlx-graphs/mlx-graphs.git"

echo "Adding remotes if missing..."
git remote add guillaume "$GUILLAUME_URL" 2>/dev/null || true
git remote add thegodone "$THEGODONE_URL" 2>/dev/null || true
git remote add upstream "$UPSTREAM_URL" 2>/dev/null || true

echo "Fetching thegodone (for esol branch)..."
git fetch thegodone

echo "Fetching upstream (mlx-graphs/mlx-graphs)..."
git fetch upstream

# --- 1) Esol branch: create/update from thegodone/esol and push to guillaume-osmo ---
if git show-ref --verify --quiet refs/remotes/thegodone/esol; then
  echo "Creating/updating local 'esol' from thegodone/esol..."
  git branch -f esol thegodone/esol 2>/dev/null || git checkout -b esol thegodone/esol
  echo "Pushing esol to guillaume-osmo/mlx-graphs..."
  git push guillaume esol
else
  echo "WARNING: branch 'esol' not found on thegodone/mlx-graphs. List branches: git branch -r"
  git branch -r | grep thegodone || true
fi

# --- 2) Main: merge upstream/main and push to guillaume-osmo ---
echo "Updating main from upstream (mlx-graphs/mlx-graphs)..."
git checkout main
git merge upstream/main --no-edit
echo "Pushing main to guillaume-osmo/mlx-graphs..."
git push guillaume main

echo "Done. guillaume-osmo/mlx-graphs now has: main (synced with mlx-graphs/mlx-graphs), esol (from thegodone/mlx-graphs)."
