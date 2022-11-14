# SERA_PyTorch
A PyTorch compatible implementation of the SERA loss function [1]

Installation and Usage Instructions:

- Simply download SERA.py and import the *SERA* module into the script from which the forward and backward passes are called.
- Similarly to native PyTorch loss functions, *SERA* must be instantiated before the training loop 
  -    **loss_function = SERA.apply**
- Prior to the forward pass, the target and predicted tensors must be cloned, detached, and converted to numpy arrays 
  -  **true_np = true_tens.clone().detach().numpy()**
  -  **pred_np = pred_tens.clone().detach().numpy()**
- the forward pass is called by passing in the predicted and target arrays, as well as a set of 3 relevance values for the corresponding control points. These relevance values are a parameter of the function that can be manually tuned. 
  -  **loss = loss_function(pred_np, true_np, [1, 0, 1])**
- The backwards pass can be called via 
  -  **loss.backward()**


[1] Ribeiro, R. P. & Moniz, N. Imbalanced regression and extreme value prediction. Mach Learn 109, 1803–1835 (2020).
