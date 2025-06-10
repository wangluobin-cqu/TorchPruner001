import torch  
import numpy as np  
from ..attributions import _AttributionMetric  
  
  
class GroupedOwenShapleyAttributionMetric(_AttributionMetric):  
    """  
    Compute attributions as approximate Shapley values using grouped Owen sampling.  
      
    This method uses two-layer stratified sampling: first sampling feature groups,  
    then sampling individual features within selected groups.  
    """  
  
    def __init__(self, *args, runs=2, q_splits=100, group_size=4, q2_splits=50, **kwargs):  
        super().__init__(*args, **kwargs)  
        self.runs = runs  
        self.q_splits = q_splits  
        self.group_size = group_size  
        self.q2_splits = q2_splits  
  
    def run(self, module, **kwargs):  
        module = super().run(module, **kwargs)  
        return self.run_module_grouped_owen(module)  
  
    def run_module_grouped_owen(self, module):  
        """  
        Implementation of grouped Owen sampling for Shapley value computation.  
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
                  
                # Compute grouped Owen Shapley values for each sample  
                for sample_idx in range(batch_size):  
                    sample_sv = self._compute_grouped_owen_shapley_single(  
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
  
    def _compute_grouped_owen_shapley_single(self, x_single, y_single, module, n_features):  
    """  
    Compute grouped Owen Shapley values for a single sample using two-layer sampling.  
    """  
    phi = np.zeros(n_features)  

    # Create simple fixed-size groups
    groups = []  
    for i in range(0, n_features, self.group_size):  
        groups.append(list(range(i, min(i + self.group_size, n_features))))  

    # Find the group that contains x_j (specific feature to isolate)
    x_j_group = None
    for group in groups:
        if x_single[0, group].argmax() == x_single[0, group].shape[0]:  # This logic will depend on how x_j is defined
            x_j_group = group
            break

    # Generate two-layer stratified samples  
    s = []  
    for _ in range(self.runs):  
        for q1_num in range(self.q_splits + 1):  
            q1 = q1_num / self.q_splits  # First layer sampling probability  

            for q2_num in range(self.q2_splits + 1):  
                q2 = q2_num / self.q2_splits  # Second layer sampling probability  

                feature_mask = np.zeros(n_features)  

                # First layer: sample groups with probability q1 except for the group containing x_j
                for group in groups:
                    if group != x_j_group:  # Skip the group containing x_j here
                        if np.random.binomial(1, q1) == 1:  
                            # Group selected, second layer: sample features within group with probability q2  
                            for feature_idx in group:  
                                feature_mask[feature_idx] = np.random.binomial(1, q2)

                # Second layer: sample features within the group containing x_j with probability q2
                if x_j_group is not None:
                    for feature_idx in x_j_group:
                        feature_mask[feature_idx] = np.random.binomial(1, q2)

                s.append(feature_mask)

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
        # Grouped Owen sampling works directly on the target module  
        return module
