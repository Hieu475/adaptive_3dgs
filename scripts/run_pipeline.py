"""
Main entry point to run the reconstruction pipeline.
"""
import argparse
import yaml

def main():
    parser = argparse.ArgumentParser(description="Run Adaptive 3DGS Pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()

    print(f"Loading config from {args.config}")
    # TODO: Initialize and run pipeline
    
if __name__ == "__main__":
    main()
