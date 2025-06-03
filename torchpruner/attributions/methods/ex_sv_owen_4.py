import torch  
import numpy as np  
from itertools import combinations  
from ..attributions import _AttributionMetric  
  
  
class ExactHierarchicalShapleyAttributionMetric(_AttributionMetric):  
    """  
    Compute attributions as exact hierarchical Shapley values.  
      
    This method implements the exact computation of the hierarchical Shapley formula:  
    Sh_j = Σ_{R_j ⊆ B\B_(j)} [|R_j|!(m-|R_j|-1)!/m!] *   
           Σ_{S_j ⊆ B_(j)\{x_j}} [|S_j|!(|B_(j)|-|S_j|-1)!/|B_(j)|!] *   
           [v(R_j ∪ S_j ∪ {x_j}) - v(R_j ∪ S_j)]  
    """  
  
    def __init__(self, *args, group_size=4, **kwargs):  
        super().__init__(*args, **kwargs)  
        self.group_size = group_size  
  
    def run(self, module, **kwargs):  
        module = super().run(module, **kwargs)  
        return self.run_module_exact_hierarchical(module)  
  
    def run_module_exact_hierarchical(self, module):  
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
                  
                # Compute exact hierarchical Shapley values for each sample  
                for sample_idx in range(batch_size):  
                    sample_sv = self._compute_exact_hierarchical_shapley_single(  
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
  
    def _compute_exact_hierarchical_shapley_single(self, x_single, y_single, module, n_features):  
        """  
        Compute exact hierarchical Shapley values for a single sample.  
        """  
        phi = np.zeros(n_features)  
          
        # Create feature groups  
        groups = self._create_groups(n_features)  
        m = len(groups)  # number of groups  
          
        # For each feature j  
        for j in range(n_features):  
            # Find which group contains feature j  
            group_j_idx = self._find_group_index(j, groups)  
            group_j = groups[group_j_idx]  
              
            shapley_value = 0.0  
              
            # First sum: over all subsets R_j of groups excluding group containing j  
            other_groups = [g for i, g in enumerate(groups) if i != group_j_idx]  
              
            # Enumerate all possible subsets of other groups  
            for r_size in range(len(other_groups) + 1):  
                for r_groups in combinations(other_groups, r_size):  
                    # Flatten selected groups to get R_j  
                    r_j = []  
                    for group in r_groups:  
                        r_j.extend(group)  
                      
                    # Calculate first weight: |R_j|!(m-|R_j|-1)!/m!  
                    weight_1 = self._calculate_factorial_weight(len(r_groups), m)  
                      
                    # Second sum: over all subsets S_j of group_j excluding j  
                    group_j_without_j = [f for f in group_j if f != j]  
                    group_j_size = len(group_j)  
                      
                    inner_sum = 0.0  
                    for s_size in range(len(group_j_without_j) + 1):  
                        for s_j in combinations(group_j_without_j, s_size):  
                            s_j = list(s_j)  
                              
                            # Calculate second weight: |S_j|!(|B_(j)|-|S_j|-1)!/|B_(j)|!  
                            weight_2 = self._calculate_factorial_weight(len(s_j), group_j_size)  
                              
                            # Calculate marginal contribution: v(R_j ∪ S_j ∪ {j}) - v(R_j ∪ S_j)  
                            coalition_without_j = r_j + s_j  
                            coalition_with_j = r_j + s_j + [j]  
                              
                            v_without_j = self._evaluate_coalition(x_single, y_single, module, coalition_without_j, n_features)  
                            v_with_j = self._evaluate_coalition(x_single, y_single, module, coalition_with_j, n_features)  
                              
                            marginal_contribution = v_with_j - v_without_j  
                            inner_sum += weight_2 * marginal_contribution  
                      
                    shapley_value += weight_1 * inner_sum  
              
            phi[j] = shapley_value  
          
        return phi  
  
    def _create_groups(self, n_features):  
        """Create feature groups of fixed size."""  
        groups = []  
        for i in range(0, n_features, self.group_size):  
            groups.append(list(range(i, min(i + self.group_size, n_features))))  
        return groups  
  
    def _find_group_index(self, feature_idx, groups):  
        """Find which group contains the given feature."""  
        for i, group in enumerate(groups):  
            if feature_idx in group:  
                return i  
        raise ValueError(f"Feature {feature_idx} not found in any group")  
  
    def _calculate_factorial_weight(self, subset_size, total_size):  
        """Calculate the factorial weight: |S|!(n-|S|-1)!/n!"""  
        if total_size == 0:  
            return 1.0  
          
        # Use log to avoid overflow for large factorials  
        import math  
        log_weight = (math.lgamma(subset_size + 1) +   
                     math.lgamma(total_size - subset_size) -   
                     math.lgamma(total_size + 1))  
        return math.exp(log_weight)  
  
    def _evaluate_coalition(self, x_single, y_single, module, coalition_features, n_features):  
        """  
        Evaluate the coalition value v(S) by masking features according to coalition.  
        """  
        # Create mask: 1 for features in coalition, 0 otherwise  
        coalition_mask = np.zeros(n_features)  
        for feature_idx in coalition_features:  
            coalition_mask[feature_idx] = 1  
          
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
        # Exact hierarchical Shapley works directly on the target module  
        return module
