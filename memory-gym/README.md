# Project2-Sehej-Jain-Memory-based Agent in Partially Observable Environments

The code has 2 folders:

1. Tensorflow code:
   This is what I started to work on before realizing that one of the environments to test over requires Gymnasium and TF does not natively supports it. I have included the code for the same in the folder.

   - `model_lib.py`: This file contains the main architecture of the GRU PPO model.
   - `model.py`: This file uses the model_lib to create the model, extract policy logits and the values from the model and finally return it to the PPO algorithm.

1. Stable Baselines 3:
   This is the working implementation of the code.
   - `model.py`: This file contains the code for the model architecture and the policy.
   - `model_vis_vec.py`: This is the file with the same model, but uses separate visual and vector observations and then concatenates them to pass it to the model. (This is referenced in the paper.)
   - `main.py`: This is the main file which contains the code for the PPO algorithm and the training loop. This file is set to use the Mortar Mayhem environment. To use the other environment, change the environment name in line 24.

My code uses the same hyperparameters as in the original paper. I plan to tune the hyperparameters in the next phase of the project.
