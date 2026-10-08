"""Utility script to generate data/development_query_manifest.json and .csv."""

import os
from src.benchmarks.dataset_manifest import ManifestManager
from src.benchmarks.csv_exporter import CSVExporter

def main():
    os.makedirs("data", exist_ok=True)
    manifest = ManifestManager.get_development_manifest()
    
    # 1. JSON
    ManifestManager.export_development_manifest_json("data/development_query_manifest.json")
    print(f"Exported {len(manifest)} queries to data/development_query_manifest.json")
    
    # 2. CSV
    CSVExporter.export_query_manifest(manifest, output_path="data/development_query_manifest.csv")
    print(f"Exported {len(manifest)} queries to data/development_query_manifest.csv")

if __name__ == "__main__":
    main()
