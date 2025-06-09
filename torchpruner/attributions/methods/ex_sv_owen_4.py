

import itertools  
import numpy as np  
from math import factorial  
import torch  
from itertools import combinations
from ..attributions import _AttributionMetric  
  
class ExactHierarchicalShapleyAttributionMetric(_AttributionMetric):  
    """  
    Compute structured Shapley value attributions with feature grouping.  
    """  
      
    def __init__(self, *args, group_size=2, **kwargs):  
        super().__init__(*args, **kwargs)  
        self.group_size = group_size  
      
    def run(self, module, group_size=None, **kwargs):  
        module = super().run(module, **kwargs)  
        group_size = group_size if group_size is not None else self.group_size  
          
        with torch.no_grad():  
            activations = []  
            handle = module.register_forward_hook(self._forward_hook(activations))  
              
            # Run forward pass to get activations  
            for idx, (x, y) in enumerate(self.data_gen):  
                x, y = x.to(self.device), y.to(self.device)  
                _ = self.model(x)  
              
            handle.remove()  
              
            # Compute structured Shapley values for each sample  
            all_shapley = []  
            for activation in activations:  
                shapley_vals = self._compute_structured_shapley(  
                    activation, group_size  
                )  
                all_shapley.append(shapley_vals)  
              
            return self.aggregate_over_samples(np.array(all_shapley))  
      
    def _compute_structured_shapley(self, example, group_size):  
        d = example.shape[0] if len(example.shape) == 1 else example.shape[1]  
        num_groups = d // group_size  
        feature_indices = list(range(d))  
          
        # Group features  
        groups = [feature_indices[i*group_size:(i+1)*group_size]   
                 for i in range(num_groups)]  
        shapley_values = np.zeros(d)  
          
        for group in groups:  
            other_features = list(set(feature_indices) - set(group))  
              
            for x_j in group:  
                phi_j = 0.0  
                local_group = list(group)  
                local_group.remove(x_j)  
                  
                # Iterate over all R_j ⊆ other group features  
                for r_size in range(len(other_features) + 1):  
                    for R in combinations(other_features, r_size):  
                        R = list(R)  
                        w_r = (factorial(r_size) *   
                              factorial(len(other_features) - r_size) /   
                              factorial(len(other_features)))  
                          
                        # Iterate over all S_j ⊆ current group (except x_j)  
                        for s_size in range(len(local_group) + 1):  
                            for S in combinations(local_group, s_size):  
                                S = list(S)  
                                w_s = (factorial(s_size) *   
                                      factorial(len(local_group) - s_size) /   
                                      factorial(len(group)))  
                                  
                                full_set = R + S + [x_j]  
                                subset = R + S  
                                  
                                # Compute marginal contribution  
                                delta = self._evaluate_subset(example, full_set) - \  
                                       self._evaluate_subset(example, subset)  
                                phi_j += w_r * w_s * delta  
                  
                shapley_values[x_j] = phi_j  
          
        return shapley_values  
      
    def _evaluate_subset(self, example, indices):  
        """Evaluate model with only specified feature indices active."""  
        masked_example = np.zeros_like(example)  
        if len(example.shape) == 1:  
            masked_example[indices] = example[indices]  
        else:  
            masked_example[:, indices] = example[:, indices]  
          
        # Convert to tensor and evaluate  
        with torch.no_grad():  
            x_tensor = torch.tensor(masked_example).float().to(self.device)  
            if len(x_tensor.shape) == 1:  
                x_tensor = x_tensor.unsqueeze(0)  
              
            # Run through model and compute loss  
            output = self.model(x_tensor)  
            # Return a simple metric (you may need to adapt this)  
            return output.sum().item()  
      
    def _forward_hook(self, activations):  
        def _hook(module, _, output):  
            activations.append(output.detach().cpu().numpy())  
        return _hook
