"""
Utility to visualize Gaussian point cloud.
"""
import argparse

def main():
    parser = argparse.ArgumentParser(description="Visualize Gaussians")
    parser.add_argument("--path", type=str, required=True, help="Path to point cloud PLY")
    args = parser.parse_args()

    print(f"Visualizing {args.path}")
    # TODO: Load PLY and visualize
    
if __name__ == "__main__":
    main()
