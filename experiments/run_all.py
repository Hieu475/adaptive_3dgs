import subprocess
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def main():
    scripts = [
        'experiments/run_importance_validation.py',
        'experiments/run_component_independence.py', 
        'experiments/run_normalization_ablation.py',
    ]
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    for s in scripts:
        print(f'\n{"="*60}\nRunning {s}...\n{"="*60}')
        script_path = os.path.join(base_dir, s)
        subprocess.run([sys.executable, script_path] + sys.argv[1:], check=True, cwd=base_dir)

if __name__ == '__main__':
    main()
