#!/usr/bin/env bash

set -e

DATASET_DIR="data/fashion_mnist"

mkdir -p "$DATASET_DIR"

cd "$DATASET_DIR"

download_and_extract() {
  local url="$1"
  local gz_file="$2"
  local raw_file="$3"

  if [ -f "$raw_file" ]; then
    echo "$raw_file already exists"
    return
  fi

  echo "Downloading $gz_file..."
  curl -L -o "$gz_file" "$url"

  echo "Extracting $gz_file..."
  gunzip -f "$gz_file"
}

BASE_URL="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com"

download_and_extract \
  "$BASE_URL/train-images-idx3-ubyte.gz" \
  "train-images-idx3-ubyte.gz" \
  "train-images-idx3-ubyte"

download_and_extract \
  "$BASE_URL/train-labels-idx1-ubyte.gz" \
  "train-labels-idx1-ubyte.gz" \
  "train-labels-idx1-ubyte"

download_and_extract \
  "$BASE_URL/t10k-images-idx3-ubyte.gz" \
  "t10k-images-idx3-ubyte.gz" \
  "t10k-images-idx3-ubyte"

download_and_extract \
  "$BASE_URL/t10k-labels-idx1-ubyte.gz" \
  "t10k-labels-idx1-ubyte.gz" \
  "t10k-labels-idx1-ubyte"

echo "Fashion-MNIST dataset is ready in $DATASET_DIR"