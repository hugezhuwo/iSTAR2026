# iSTAR: In-Parameter Structured Task Reasoning for Vision-Language-Action Models

This repository contains the anonymous implementation of the paper:

**Differentiate-and-Inject: Enhancing VLAs via Functional Differentiation Induced by In-Parameter Structural Reasoning**

## Introduction

Vision-language-action (VLA) models have shown strong performance in robotic manipulation, but they often entangle task-level reasoning and low-level action generation within a single inference process. This coupling can limit reliability, especially in long-horizon tasks where successful execution depends on understanding object relations, subtask dependencies, and task progression.

We propose **iSTAR** (**in-parameter Structured TAsk Reasoning**), a framework that enhances VLA models by injecting task-level semantic structure directly into model parameters. Instead of relying on external planners, handcrafted decomposition, or prompt-based reasoning alone, iSTAR performs task-structured reasoning within the model family itself.

The key idea is to functionally differentiate the VLA into two coordinated roles:

- a **pre-action reasoning module** that extracts and organizes task-relevant semantic concepts;
- an **action-generating module** that executes conditioned on the resolved subtask semantics.

This design preserves end-to-end action generation while improving reliability, compositionality, and long-horizon generalization.

## Method Overview

iSTAR consists of four main components:

1. **Concept Extraction**  
   Extracts object-centric and action-centric concept representations from a pre-action VLA module.

2. **Dynamic Implicit Concept Graph**  
   Models object relevance, temporal ordering, and relational structure, and supports task-aware concept selection and structured reasoning over concepts.

3. **Subtask Prompt Projector**  
   Maps structure-aware internal concepts into subtask-level semantic embeddings or prompts.

4. **Action Generation**  
   Conditions the downstream VLA policy on the resolved task-level semantic commitment.

Overall, iSTAR decomposes a monolithic VLA inference process into differentiated semantic reasoning and action execution stages, while keeping both stages within the same parameter family and without introducing external planners.

## Repository Structure

```text
.
├── config/                 # Configuration files
├── data/                   # Dataset directory (not included)
├── subprompt/              # Subtask prompt data / generation logic
├── VIMABench/              # External benchmark environment (to be placed manually)
├── checkpoints/            # Saved model weights
├── vima_policy.py          # VLA / policy definition
├── nlir_decomposer.py      # Task-structure / semantic decomposition module
├── vima_dataset.py         # Dataset loader and trajectory processing
├── vima_train_for_all.py   # Main training script
├── test_all.py             # Evaluation script
└── requirements.txt        # Python dependencies
```

## Supported Setting

This anonymous release focuses on the **VIMA-based implementation** used in our experiments.

The full paper evaluates iSTAR across:

- **VIMA-Bench**
- **LIBERO**
- **Real-world UR3 manipulation tasks**

The current repository is organized around the VIMA-style policy learning pipeline and the corresponding task-structured reasoning components.

## Installation

### 1. Clone the repository

```bash
git clone <anonymous-repo-url>
cd iSTAR2026
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare external environments and datasets

- Place the benchmark environment code under `VIMABench/`
- Place datasets under `data/`

This repository does **not** include benchmark environments or raw datasets.

## Training

The main training entry is:

```bash
python vima_train_for_all.py
```

Training behavior is controlled by the configuration files in `config/`.

## Evaluation

To run evaluation:

```bash
python test_all.py
```

Please ensure that:

- the benchmark environment is correctly installed;
- the dataset is placed in `data/`;
- and the required checkpoints are available in `checkpoints/`.

## Notes for Anonymous Review

To preserve double-blind review constraints, several items have been removed or anonymized in this repository:

- exact dataset paths;
- private logging utilities;
- environment-specific scripts;
- selected pre-trained weights;
- and other non-essential infrastructure tied to the original experimental platform.

The released code is intended to expose the **core method implementation** while avoiding deanonymizing artifacts.

## ArXiv Version

The public preprint link will be added after the anonymous review process if appropriate.

## Citation

If the paper is accepted and this repository is de-anonymized, citation information will be added here.

## Acknowledgment

This repository is released for anonymous review only.
