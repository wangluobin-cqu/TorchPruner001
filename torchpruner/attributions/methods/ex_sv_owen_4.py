import torch  
import numpy as np  
from math import factorial  
from itertools import combinations  
from ..attributions import _AttributionMetric  
  
class ExactHierarchicalShapleyAttributionMetric(_AttributionMetric):  
    """  
    Compute structured Shapley value attributions with feature grouping.  
    """  
      
    def __init__(self, *args, group_size=3, **kwargs):  
        super().__init__(*args, **kwargs)  
        self.group_size = group_size  
      
    def run(self, module, **kwargs):  
        module = super().run(module, **kwargs)  
        return self.run_module_hierarchical_shapley(module)  
      
    def run_module_hierarchical_shapley(self, module):  
        """  
        Implementation of exact hierarchical Shapley value computation.  
        """  
        sv_results = []  
          
        with torch.no_grad():  
            # Register hook to capture activations  
            activations = []  
            def capture_hook(mod, inp, out):  
                activations.append(out.detach().clone())  
              
            handle = module.register_forward_hook(capture_hook)  
              
            for batch_idx, (x, y) in enumerate(self.data_gen):  
                x, y = x.to(self.device), y.to(self.device)  
                  
                # Get original activations to determine feature dimension  
                _ = self.model(x)  
                original_activations = activations[-1]  
                n_features = original_activations.shape[1]  
                batch_size = original_activations.shape[0]  
                  
                # Initialize Shapley values for this batch  
                batch_sv = np.zeros((batch_size, n_features))  
                  
                # Compute hierarchical Shapley values for each sample  
                for sample_idx in range(batch_size):  
                    sample_sv = self._compute_hierarchical_shapley_single(  
                        x[sample_idx:sample_idx+1],  
                        y[sample_idx:sample_idx+1],  
                        module, n_features  
                    )  
                    batch_sv[sample_idx] = sample_sv  
                  
                sv_results.append(batch_sv)  
              
            handle.remove()  
          
        # Concatenate all batches  
        all_sv = np.concatenate(sv_results, axis=0)  
        return self.aggregate_over_samples(all_sv)  
      
  def _compute_hierarchical_shapley_single(self, x_single, y_single, module, n_features):  
        """  
        Compute hierarchical Shapley values for a single sample using exact computation.  
        """  
        phi = np.zeros(n_features)  
          
        # Create feature groups  
        groups = []  
        for i in range(0, n_features, self.group_size):  
            groups.append(list(range(i, min(i + self.group_size, n_features))))  
          
        # Compute exact hierarchical Shapley values  
        for group in groups:  
            other_features = [g for g in groups if g != group]  
              
            for x_j in group:  
                phi_j = 0.0  
                local_group = [f for f in group if f != x_j]  
                  
                # Iterate over all R_j ⊆ other group features  
                for r_size in range(len(other_features) + 1):  
                    for R in combinations(other_features, r_size):  
                        R = list(R)  
                        expanded_R = [f for subgroup in R for f in subgroup] 
                        w_r = (factorial(r_size) *   
                              factorial(len(groups) - r_size-1) /   
                              factorial(len(groups))) if groups else 1.0  
                          
                        # Iterate over all S_j ⊆ current group (except x_j)  
                        for s_size in range(len(local_group) + 1):  
                            for S in combinations(local_group, s_size):  
                                S = list(S)  
                                w_s = (factorial(s_size) *   
                                      factorial(len(group) - s_size-1) /   
                                      factorial(len(group))) if group else 1.0  
                                  
                                full_set = expanded_R + S + [x_j]  
                                subset = expanded_R + S   
                                  
                                # Compute marginal contribution  
                                loss_full = self._evaluate_coalition(x_single, y_single, module, full_set, n_features)  
                                loss_subset = self._evaluate_coalition(x_single, y_single, module, subset, n_features)  
                                delta = loss_subset - loss_full  # Higher loss means lower importance  
                                phi_j += w_r * w_s * delta  
                  
                phi[x_j] = phi_j  
          
        return phi 
      
    def _evaluate_coalition(self, x_single, y_single, module, indices, n_features):  
        """  
        Evaluate the coalition value v(S) by masking features according to indices.  
        """  
        # Create coalition mask  
        coalition_mask = np.zeros(n_features)  
        coalition_mask[indices] = 1.0  
          
        def mask_hook(mod, inp, out):  
            masked_out = out.clone()  
            # Apply coalition mask  
            mask_tensor = torch.tensor(coalition_mask, dtype=torch.float32).to(self.device)  
            # Expand mask to match output dimensions  
            if len(masked_out.shape) > 2:  
                # For convolutional layers  
                mask_tensor = mask_tensor.view(1, -1, 1, 1)  
            else:  
                # For linear layers  
                mask_tensor = mask_tensor.view(1, -1)  
              
            return masked_out * mask_tensor  
          
        # Register hook and evaluate  
        handle = module.register_forward_hook(mask_hook)  
        try:  
            output = self.model(x_single)  
            loss = self.criterion(output, y_single, reduction="none")  
            return loss.item()  
        finally:  
            handle.remove()  
      
    def find_evaluation_module(self, module, find_best_evaluation_module=False):  
        # Hierarchical Shapley works directly on the target module  
        return module
