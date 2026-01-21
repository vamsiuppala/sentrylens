#!/bin/bash
# Download Eclipse AERI dataset

cd data/aeri

echo "Downloading AERI Problems CSV (1.5MB compressed)..."
curl -O https://download.eclipse.org/scava/aeri_stacktraces/problems_extract.csv.bz2

echo "Downloading AERI Incidents CSV (141MB compressed)..."
curl -O https://download.eclipse.org/scava/aeri_stacktraces/incidents_extract.csv.bz2

echo "Decompressing..."
bunzip2 -k problems_extract.csv.bz2
bunzip2 -k incidents_extract.csv.bz2

echo "Downloading AERI Problems FULL JSON (38MB compressed, 904MB raw)..."
curl -O https://download.eclipse.org/scava/aeri_stacktraces/problems_full.tar.bz2

echo "Extracting..."
tar -xjf problems_full.tar.bz2

echo "Done! Checking structure..."
ls -la