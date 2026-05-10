"""Extract TensorBoard events and plot with matplotlib (using tbparse)."""

import os
import glob
from pathlib import Path

import matplotlib.pyplot as plt

try:
    from tbparse import SummaryReader
except ImportError:
    print("Installing tbparse...")
    os.system("pip install tbparse")
    from tbparse import SummaryReader


def main():
    """Find and plot all TensorBoard events."""
    Path('graficas').mkdir(exist_ok=True)
    
    # Find only the latest run directory
    all_runs = glob.glob('runs/carla_dqn_*', recursive=False)
    run_dirs = [max(all_runs, key=os.path.getctime)] if all_runs else []
    
    if not run_dirs:
        print("❌ No run directories found")
        return
    
    print(f"Found {len(run_dirs)} run(s)\n")
    
    for run_dir in run_dirs:
        print(f"Processing {run_dir}...")
        
        try:
            reader = SummaryReader(run_dir)
            df = reader.scalars
            
            if df.empty:
                print(f"  ⚠ No scalar data found")
                continue
            
            # Get unique tags (metrics)
            tags = df['tag'].unique()
            
            for tag in tags:
                # Filter data for this tag
                tag_data = df[df['tag'] == tag].sort_values('step')
                
                if len(tag_data) == 0:
                    continue
                
                # Create plot
                plt.figure(figsize=(10, 6))
                plt.plot(tag_data['step'], tag_data['value'], marker='o', linestyle='-', linewidth=2)
                plt.title(f'{tag}', fontsize=14, fontweight='bold')
                plt.xlabel('Step', fontsize=12)
                plt.ylabel('Value', fontsize=12)
                plt.grid(True, alpha=0.3)
                
                # Save figure
                run_name = Path(run_dir).name
                filename = f"{run_name}_{tag.replace('/', '_')}.png"
                filepath = os.path.join('graficas', filename)
                
                plt.tight_layout()
                plt.savefig(filepath, dpi=100)
                plt.close()
                
                print(f"  ✓ {tag}")
        
        except Exception as e:
            print(f"  ❌ Error: {e}\n")
    
    print("\n✓ All plots saved to 'graficas/' folder")


if __name__ == '__main__':
    main()
