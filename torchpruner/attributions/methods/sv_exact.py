import torch  
import numpy as np  
import itertools  
import scipy.special  
from ..attributions import _AttributionMetric  
  
  
class ExactShapleyAttributionMetric(_AttributionMetric):  
    """  
    Compute attributions as exact Shapley values using exhaustive enumeration.  
    Warning: This method has exponential complexity and should only be used   
    for small feature sets (typically < 20 features).  
    """  
  
    def __init__(self, *args, max_features=20, **kwargs):  
        super().__init__(*args, **kwargs)  
        self.max_features = max_features  
  
    def run(self, module, **kwargs):  
        module = super().run(module, **kwargs)  
          
        # Get the output dimension to check feasibility  
        with torch.no_grad():  
            for x, _ in self.data_gen:  
                x = x.to(self.device)  
                _ = self.model(x)  
                break  
          
        # Check if exact computation is feasible  
        n_features = module.weight.shape[0] if hasattr(module, 'weight') else None  
        if n_features and n_features > self.max_features:  
            raise ValueError(f"Exact Shapley computation not feasible for {n_features} features. "  
                           f"Maximum supported: {self.max_features}")  
          
        return self.run_module_exact(module)  
  
    def run_module_exact(self, module):  
        """  
        Implementation of exact Shapley value computation using exhaustive enumeration.  
        """  
        sv_results = []  
          
        with torch.no_grad():  
            # Register hook to capture activations  
            activations = []  
            def capture_hook(mod, inp, out):  
                activations.append(out.detach().clone())  
              
            handle = module.register_forward_hook(capture_hook)  
              
            # Run forward pass to get baseline activations  
            for batch_idx, (x, y) in enumerate(self.data_gen):  
                x, y = x.to(self.device), y.to(self.device)  
                  
                # Get original activations  
                _ = self.model(x)  
                original_activations = activations[-1]  
                n_features = original_activations.shape[1]  
                batch_size = original_activations.shape[0]  
                  
                # Initialize Shapley values for this batch  
                batch_sv = np.zeros((batch_size, n_features))  
                  
                # Compute exact Shapley values for each feature  
                for j in range(n_features):  
                    for sample_idx in range(batch_size):  
                        batch_sv[sample_idx, j] = self._compute_exact_shapley_single(  
                            x[sample_idx:sample_idx+1],   
                            y[sample_idx:sample_idx+1],   
                            module, j, n_features  
                        )  
                  
                sv_results.append(batch_sv)  
              
            handle.remove()  
          
        # Concatenate all batches  
        all_sv = np.concatenate(sv_results, axis=0)  
        return self.aggregate_over_samples(all_sv)  
  
    def _compute_exact_shapley_single(self, x_single, y_single, module, feature_j, n_features):  
        """  
        Compute exact Shapley value for a single feature j in a single sample.  
        """  
        def powerset(iterable):  
            s = list(iterable)  
            return itertools.chain.from_iterable(  
                itertools.combinations(s, r) for r in range(len(s) + 1)  
            )  
          
        shapley_value = 0.0  
          
        # Iterate over all possible coalitions S ⊆ N \ {j}  
        features_without_j = set(range(n_features)) - {feature_j}  
          
        for coalition in powerset(features_without_j):  
            coalition = list(coalition)  
            coalition_size = len(coalition)  
              
            # Compute weight: |S|!(n-|S|-1)!/n!  
            weight = (scipy.special.factorial(coalition_size) *   
                     scipy.special.factorial(n_features - coalition_size - 1) /   
                     scipy.special.factorial(n_features))  
              
            # Compute marginal contribution: v(S ∪ {j}) - v(S)  
            marginal_contribution = self._evaluate_coalition_difference(  
                x_single, y_single, module, coalition, feature_j  
            )  
              
            shapley_value += weight * marginal_contribution  
          
        return shapley_value  
  
    def _evaluate_coalition_difference(self, x_single, y_single, module, coalition, feature_j):  
        """  
        Compute v(S ∪ {j}) - v(S) where v is the model's loss function.  
        """  
        # Create masks for coalition S and S ∪ {j}  
        activations_s = []  
        activations_s_union_j = []  
          
        def mask_hook_s(mod, inp, out):  
            masked_out = out.clone()  
            # Zero out all features except those in coalition S  
            mask = torch.zeros_like(masked_out)  
            for idx in coalition:  
                mask[:, idx] = 1.0  
            activations_s.append(masked_out * mask)  
            return masked_out * mask  
          
        def mask_hook_s_union_j(mod, inp, out):  
            masked_out = out.clone()  
            # Zero out all features except those in coalition S ∪ {j}  
            mask = torch.zeros_like(masked_out)  
            for idx in coalition + [feature_j]:  
                mask[:, idx] = 1.0  
            activations_s_union_j.append(masked_out * mask)  
            return masked_out * mask  
          
        # Evaluate v(S)  
        handle_s = module.register_forward_hook(mask_hook_s)  
        output_s = self.model(x_single)  
        loss_s = self.criterion(output_s, y_single, reduction="none")  
        handle_s.remove()  
          
        # Evaluate v(S ∪ {j})  
        handle_s_union_j = module.register_forward_hook(mask_hook_s_union_j)  
        output_s_union_j = self.model(x_single)  
        loss_s_union_j = self.criterion(output_s_union_j, y_single, reduction="none")  
        handle_s_union_j.remove()  
          
        # Return marginal contribution  
        marginal = (loss_s - loss_s_union_j).item()  
        return marginal  
  
    def find_evaluation_module(self, module, find_best_evaluation_module=False):  
        # For exact computation, we need to work directly on the target module  
        return module
