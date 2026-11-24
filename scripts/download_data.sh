#!/usr/bin/env bash
# Download the four raw Steam files into data/ (about 84 MB in total).
set -euo pipefail
cd "$(dirname "$0")/../data" 2>/dev/null || { mkdir -p "$(dirname "$0")/../data"; cd "$(dirname "$0")/../data"; }
for url in \
  https://mcauleylab.ucsd.edu/public_datasets/data/steam/australian_users_items.json.gz \
  https://mcauleylab.ucsd.edu/public_datasets/data/steam/australian_user_reviews.json.gz \
  https://cseweb.ucsd.edu/~wckang/steam_games.json.gz \
  https://mcauleylab.ucsd.edu/public_datasets/data/steam/bundle_data.json.gz; do
  f=$(basename "$url"); [ -f "$f" ] || curl -fL -o "$f" "$url"
done
ls -la
