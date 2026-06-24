#!/usr/bin/env bash
# Download all raw data files required by Connoisseur Companion.
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "→ California Culinary Map (text)"
curl -fsSL -o "$DATA_DIR/California-Culinary-Map.txt" \
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/1r_mM6ZPYNxcFv65QkzubA/California-Culinary-Map.txt"

echo "→ Recipes (JSON)"
curl -fsSL -o "$DATA_DIR/Recipes.json" \
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/hpTjb6liKBLVHQK0UgMi5A/Recipes.json"

echo "→ Synthetic User Reviews (JSON)"
curl -fsSL -o "$DATA_DIR/Synthetic-User-Reviews.json" \
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/fQUs9wQ6aB6ts6fmkD2V2w/Synthetic-User-Reviews.json"

echo "→ Recipe images (ZIP, ~205 MB)"
curl -fsSL -o "$DATA_DIR/synthetic-recipe-images.zip" \
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5_Rr6ohviItzucyWk6nkrw/synthetic-recipe-images.zip"

echo "→ Extracting images..."
unzip -oq "$DATA_DIR/synthetic-recipe-images.zip" -d "$DATA_DIR"
rm "$DATA_DIR/synthetic-recipe-images.zip"

echo "✅ All data downloaded to $DATA_DIR"
