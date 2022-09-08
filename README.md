# SERA_PyTorch
A PyTorch compatible implementation of the SERA loss function [1]

Installation and Usage Instructions:

- Simply download SERA.py and import the *SERA* module into the script from which the forward and backward passes are called.
- Similarly to native PyTorch loss functions, *SERA* must be instantiated before the training loop 
-   i.e **loss_function = SERA.apply**
- Prior to the forward loop, the target and predicted tensors must be cloned, detached, and converted to numpy arrays (i.e **pred_np = pred_tens.clone().detach().numpy()**)
- the forward pass can be called by passing in the predicted and target arrays, as well as a set of 3 relevance values for the corresponding control points (i.e **loss = loss_function(y_red, y_true, [1, 0, 1])**)
- The backwards pass can be called via **loss.backward()**





[1] Ribeiro, R. P. & Moniz, N. Imbalanced regression and extreme value prediction. Mach Learn 109, 1803–1835 (2020).
