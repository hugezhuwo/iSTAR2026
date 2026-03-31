import os
import shutil
import re

src_dir = '/home/bkai/vima_code'
dst_dir = '/home/bkai/nimingcode'

# List of important files to keep
files_to_copy = [
    'vima_policy.py',
    'nlir_decomposer.py',
    'vima_dataset.py',
    'vima_train_for_all.py',
    'test_all.py',
    'requirements.txt'
]

os.makedirs(dst_dir, exist_ok=True)

for f in files_to_copy:
    src_path = os.path.join(src_dir, f)
    dst_path = os.path.join(dst_dir, f)
    
    if not os.path.exists(src_path):
        continue
        
    with open(src_path, 'r', encoding='utf-8') as fin:
        content = fin.read()
        
    # Anonymize absolute paths
    content = re.sub(r'(/home/bkai/[^\s\'"]+)', r'/path/to/\1', content)
    content = re.sub(r'(/data/[^\s\'"]+)', r'/path/to/data', content)
    content = re.sub(r'(/path/to//home/bkai/.*?)', r'/path/to/workspace', content)
    
    # Anonymize wandb/logs/weights
    content = re.sub(r'wandb\.init\(.*?\)', 'wandb.init(project="anonymous_project")', content, flags=re.DOTALL)
    content = re.sub(r'torch\.load\([\'"].*?\.pth[\'"]\)', 'torch.load("/path/to/weights.pth")', content)
    content = re.sub(r'[\'"][^\'"]*\.pth[\'"]', '"/path/to/weights.pth"', content)
    content = re.sub(r'[\'"][^\'"]*\.pt[\'"]', '"/path/to/weights.pt"', content)

    with open(dst_path, 'w', encoding='utf-8') as fout:
        fout.write(content)

# We can create a README.md
readme_content = """# VIMA-based Policy Learning (Anonymous Repository)

This repository contains the core implementation of our paper.

## Structure
- `vima_policy.py`: Defines the visual-motor policy network.
- `nlir_decomposer.py`: The proposed Natural Language Instruction Decomposer network.
- `vima_dataset.py`: Dataset definition for training and loading trajectories.
- `vima_train_for_all.py`: The main training script with multi-task support.
- `test_all.py`: Evaluation and testing script.

## Setup
```bash
pip install -r requirements.txt
```

## Note
Certain exact paths, logging routines, configs and pre-trained weights have been replaced or removed to ensure double-blind review constraints.
"""

with open(os.path.join(dst_dir, 'README.md'), 'w') as f:
    f.write(readme_content)

print("Done processing files.")
