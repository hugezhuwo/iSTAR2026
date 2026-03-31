# VIMA-based Policy Learning (Anonymous Repository)

This repository contains the core implementation of our paper.

## Structure

```
.
├── config/              # Configuration files (hyperparameters, training settings)
├── data/                # Directory for placing datasets (empty in this repository)
├── subprompt/           # Directory for subprompt data or generation logic
├── VIMABench/           # Place the VIMABench environment code here
├── checkpoints/         # Directory for saving model weights
├── vima_policy.py       # Defines the visual-motor policy network
├── nlir_decomposer.py   # The proposed Natural Language Instruction Decomposer network
├── vima_dataset.py      # Dataset definition for training and loading trajectories
├── vima_train_for_all.py# The main training script with multi-task support
├── test_all.py          # Evaluation and testing script
└── requirements.txt     # Dependencies
```

## Setup
1. Clone the `VIMABench` repository or place it in the `VIMABench/` directory.
2. Place the dataset inside the `data/` directory.
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Note
Certain exact paths, logging routines, configs and pre-trained weights have been replaced or removed to ensure double-blind review constraints.
