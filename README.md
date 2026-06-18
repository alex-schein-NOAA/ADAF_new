## ENVIRONMENT INSTALLATION ##

The provided ADAF_environment.yml file works to set up a conda environment that will run all the included scripts.
!! EXTREMELY IMPORTANT !!
If you're on an HPC login node without a GPU, the install will not see cuda and will default to CPU-only install.
To fix this, you MUST first get the CUDA version on a GPU compute node with "nvidia-smi" (should be 13.1 or similar) and then on the login node, run "export CONDA_OVERRIDE_CUDA=13.1" (or your CUDA version).
Again, do this BEFORE any conda installation!!

## MODEL TRAINING ##
To start a new training run, you need to use train_launcher_sbatch.sh in order to properly set up the environment. The variables in the file header are configured for use on Ursa; for your use they may need to be changed.
The train.py file is controlled by user-defined parameters, which are detailed in config/params_default.yaml and can be modified by either command-line arguments (useful if you only want to change a few arguments) or making a new parameters file and passing it in with the --config flag.
