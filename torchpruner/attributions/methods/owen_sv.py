import torch  
import numpy as np  
from ..attributions import _AttributionMetric  
  
  
class OwenShapleyAttributionMetric(_AttributionMetric):  
    """  
    Compute attributions as approximate Shapley values using Owen sampling.  
      
    This method uses stratified sampling with binomial distributions to   
    approximate Shapley values more efficiently than traditional Monte Carlo.  
    """  
  
    def __init__(self, *args, runs=2, q_splits=100, **kwargs):  
        super().__init__(*args, **kwargs)  
        self.runs = runs  
        self.q_splits = q_splits  
  
    def run(self, module, **kwargs):  
        module = super().run(module, **kwargs)  
        return self.run_module_owen(module)  
  
    def run_module_owen(self, module):  
        """  
        Implementation of Owen sampling for Shapley value computation.  
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
                  
                # Compute Owen Shapley values for each sample  
                for sample_idx in range(batch_size):  
                    sample_sv = self._compute_owen_shapley_single(  
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
  
    def _compute_owen_shapley_single(self, x_single, y_single, module, n_features):  
        """  
        Compute Owen Shapley values for a single sample using stratified sampling.  
        """  
        phi = np.zeros(n_features)  
          
        # Generate stratified samples using Owen's method  
        s = []  
        for _ in range(self.runs):  
            for q_num in range(self.q_splits + 1):  
                q = q_num / self.q_splits  
                # Generate binary mask using binomial distribution  
                mask = np.random.binomial(1, q, n_features)  
                s.append(mask)  
        s = np.array(s)  
          
        # Compute Shapley values for each feature  
        for j in range(n_features):  
            marginal_contributions = []  
              
            for mask in s:  
                # Create coalition S without feature j  
                coalition_s = mask.copy()  
                coalition_s[j] = 0  
                  
                # Create coalition S ∪ {j}  
                coalition_s_union_j = mask.copy()  
                coalition_s_union_j[j] = 1  
                  
                # Evaluate v(S) and v(S ∪ {j})  
                loss_s = self._evaluate_coalition(x_single, y_single, module, coalition_s)  
                loss_s_union_j = self._evaluate_coalition(x_single, y_single, module, coalition_s_union_j)  
                  
                # Compute marginal contribution  
                marginal_contribution = loss_s - loss_s_union_j 
                marginal_contributions.append(marginal_contribution)  
              
            # Average marginal contributions  
            phi[j] = np.mean(marginal_contributions)  
          
        return phi  
  
    def _evaluate_coalition(self, x_single, y_single, module, coalition_mask):  
        """  
        Evaluate the coalition value v(S) by masking features according to coalition_mask.  
        """  
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
        # Owen sampling works directly on the target module  
        return module
