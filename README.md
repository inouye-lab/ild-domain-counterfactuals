# ild_domaincf


Welcome! This is the official repository for the following paper:

>Kulinski, S., Zhou, Z., Bai, R., Kocaoglu, M., & Inouye, D. I. Towards Characterizing Domain Counterfactuals For Invertible Latent Causal Models. ICLR 2024.


This repository will be updated soon! Feel free to email the authors for code.



# General Guidance 

All experiments are conducted using wandb sweep. **TODO**: Add some instruction on how to run the experiments.

# Simulated Experiments

All corresponding code could be found in the `simulated` folder.

**TODO**: Add some instruction on how to run a single experiment.

To regenerate the figures in the paper:

### Step 1
Run all sweeps in the directory `simulated/configs`.

### Step 2 
Follow the instruction in the notebook `simulated/demo_results.ipynb` to regenerate the figures.

## Additional Experiments 

In the rebuttal, we add some additional experiments using Normalizing Flows and VAEs as G. The corresponding code could
be found in `simulated/flow` and `simulated/vae` respectively. Similarly, to regenerate the figures in the paper: (1) 
run the sweeps in the corresponding `configs` directory, and (2) follow the instruction in the notebook (share the same notebook
with other simulated experiments).

**TODO**: Clean up the code for these experiments.

# Image Experiments


## Validation

### Step 1
Go to directory `images/validation`, run
```
python run_prep_classifier.py
```
**TODO**: Fix path? Arguments?

### Step 2 
Run 

```angular2html
python run_test.py
```

# Notes