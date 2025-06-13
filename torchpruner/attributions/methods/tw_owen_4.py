import torch    
import numpy as np    
import logging  
from ..attributions import _AttributionMetric    
  
class GroupedOwenShapleyAttributionMetric(_AttributionMetric):    
    """    
    Compute attributions as approximate Shapley values using grouped Owen sampling.    
        
    This method uses two-layer stratified sampling: first sampling feature groups,    
    then sampling individual features within selected groups.    
    """    
  
    def __init__(self, *args, runs=2, q_splits=10, group_size=2, q2_splits=10, **kwargs):    
        super().__init__(*args, **kwargs)    
        self.runs = runs    
        self.q_splits = q_splits    
        self.group_size = group_size    
        self.q2_splits = q2_splits    
  
    def run(self, module, **kwargs):    
        module = super().run(module, **kwargs)  
        if hasattr(self.model, "forward_partial"):  
            return self.run_module_grouped_owen_with_partial_batch(module)  
        else:  
            logging.warning("Consider adding a 'forward_partial' method to your model to speed-up Grouped Owen Shapley values computation")  
            return self.run_module_grouped_owen_batch(module)  
  
    def run_module_grouped_owen_with_partial_batch(self, module):  
        """  
        批量处理版本的分组Owen采样，使用forward_partial优化  
        """  
        d = len(self.data_gen.dataset)  
        sv = None  
        sampling_patterns = None  
        c = 0  
  
        with torch.no_grad():  
            for idx, (x, y) in enumerate(self.data_gen):  
                x, y = x.to(self.device), y.to(self.device)  
                  
                # 获取到目标模块的激活  
                original_z, _ = self.run_forward_partial(x, to_module=module)  
                _, original_loss = self.run_forward_partial(original_z, y_true=y, from_module=module)  
                n_features = original_z.shape[1]  
                batch_size = original_z.shape[0]  
                  
                # 预生成所有采样模式（只在第一个批次生成）  
                if sampling_patterns is None:  
                    sampling_patterns = self._generate_sampling_patterns(n_features)  
                  
                if sv is None:  
                    sv = np.zeros((d, n_features))  
                  
                # 批量计算所有采样模式的Shapley值  
                batch_sv = self._compute_batch_shapley_with_partial(  
                    original_z, original_loss, y, module, sampling_patterns  
                )  
                  
                sv[c:c+batch_size] = batch_sv  
                c += batch_size  
  
            return self.aggregate_over_samples(sv)  
  
    def run_module_grouped_owen_batch(self, module):  
        """  
        批量处理版本的分组Owen采样，使用hook方式  
        """  
        d = len(self.data_gen.dataset)  
        sv = None  
        sampling_patterns = None  
        c = 0  
          
        with torch.no_grad():  
            for idx, (x, y) in enumerate(self.data_gen):  
                x, y = x.to(self.device), y.to(self.device)  
                  
                # 获取原始激活来确定特征维度  
                activations = []  
                def capture_hook(mod, inp, out):  
                    activations.append(out.detach().clone())  
                  
                handle = module.register_forward_hook(capture_hook)  
                _ = self.model(x)  
                handle.remove()  
                  
                original_activations = activations[0]  
                n_features = original_activations.shape[1]  
                batch_size = original_activations.shape[0]  
                  
                # 预生成所有采样模式  
                if sampling_patterns is None:  
                    sampling_patterns = self._generate_sampling_patterns(n_features)  
                  
                if sv is None:  
                    sv = np.zeros((d, n_features))  
                  
                # 批量计算Shapley值  
                batch_sv = self._compute_batch_shapley_with_hooks(  
                    x, y, module, sampling_patterns  
                )  
                  
                sv[c:c+batch_size] = batch_sv  
                c += batch_size  
  
            return self.aggregate_over_samples(sv)  
  
    def _generate_sampling_patterns(self, n_features):  
        """  
        预生成所有采样模式，避免在计算过程中重复生成随机数  
        """  
        groups = []  
        for i in range(0, n_features, self.group_size):  
            groups.append(list(range(i, min(i + self.group_size, n_features))))  
          
        patterns = []  
        for group_idx, group in enumerate(groups):  
            other_groups = [g for g_idx, g in enumerate(groups) if g_idx != group_idx]  
              
            for x_j in group:  
                feature_patterns = []  
                  
                for run in range(self.runs):  
                    for q1_num in range(self.q_splits):  
                        q1 = q1_num / self.q_splits  
                          
                        for q2_num in range(self.q2_splits):  
                            q2 = q2_num / self.q2_splits  
                              
                            # 预生成采样掩码  
                            feature_mask = np.zeros(n_features)  
                              
                            # 第一层：采样其他组  
                            for other_group in other_groups:  
                                if np.random.binomial(1, q1) == 1:  
                                    for idx in other_group:  
                                        feature_mask[idx] = 1  
                              
                            # 第二层：采样当前组内其他特征  
                            current_group_others = [f for f in group if f != x_j]  
                            for idx in current_group_others:  
                                if np.random.binomial(1, q2) == 1:  
                                    feature_mask[idx] = 1  
                              
                            # 生成两个联盟掩码  
                            coalition_s = feature_mask.copy()  
                            coalition_s[x_j] = 0  
                              
                            coalition_s_union_j = feature_mask.copy()  
                            coalition_s_union_j[x_j] = 1  
                              
                            feature_patterns.append((coalition_s, coalition_s_union_j))  
                  
                patterns.append((x_j, feature_patterns))  
          
        return patterns  
  
    def _compute_batch_shapley_with_partial(self, original_z, original_loss, y, module, sampling_patterns):  
        """  
        使用forward_partial的批量Shapley值计算  
        """  
        batch_size, n_features = original_z.shape[0], original_z.shape[1]  
        batch_sv = np.zeros((batch_size, n_features))  
          
        for feature_idx, feature_patterns in sampling_patterns:  
            marginal_contributions = []  
              
            # 批量处理所有采样模式  
            for coalition_s, coalition_s_union_j in feature_patterns:  
                # 应用掩码到激活  
                mask_s = torch.tensor(coalition_s, dtype=torch.float32).to(self.device)  
                mask_s_union_j = torch.tensor(coalition_s_union_j, dtype=torch.float32).to(self.device)  
                  
                if len(original_z.shape) > 2:  
                    mask_s = mask_s.view(1, -1, 1, 1)  
                    mask_s_union_j = mask_s_union_j.view(1, -1, 1, 1)  
                else:  
                    mask_s = mask_s.view(1, -1)  
                    mask_s_union_j = mask_s_union_j.view(1, -1)  
                  
                # 批量计算两个联盟的损失  
                masked_z_s = original_z * mask_s  
                masked_z_s_union_j = original_z * mask_s_union_j  
                  
                _, loss_s = self.run_forward_partial(masked_z_s, y_true=y, from_module=module)  
                _, loss_s_union_j = self.run_forward_partial(masked_z_s_union_j, y_true=y, from_module=module)  
                  
                # 计算边际贡献  
                marginal_contribution = (loss_s - loss_s_union_j).detach().cpu().numpy()  
                marginal_contributions.append(marginal_contribution)  
              
            # 聚合所有采样的边际贡献  
            if marginal_contributions:  
                avg_contribution = np.mean(marginal_contributions, axis=0)  
                batch_sv[:, feature_idx] = avg_contribution.squeeze()  
          
        return batch_sv  
  
    def _compute_batch_shapley_with_hooks(self, x, y, module, sampling_patterns):  
        """  
        使用hook方式的批量Shapley值计算  
        """  
        batch_size = x.shape[0]  
        n_features = len(sampling_patterns)  
        batch_sv = np.zeros((batch_size, n_features))  
          
        for feature_idx, feature_patterns in sampling_patterns:  
            marginal_contributions = []  
              
            for coalition_s, coalition_s_union_j in feature_patterns:  
                # 计算两个联盟的损失  
                loss_s = self._evaluate_coalition_batch(x, y, module, coalition_s)  
                loss_s_union_j = self._evaluate_coalition_batch(x, y, module, coalition_s_union_j)  
                  
                # 计算边际贡献  
                marginal_contribution = loss_s - loss_s_union_j  
                marginal_contributions.append(marginal_contribution)  
              
            # 聚合边际贡献  
            if marginal_contributions:  
                avg_contribution = np.mean(marginal_contributions, axis=0)  
                batch_sv[:, feature_idx] = avg_contribution  
          
        return batch_sv  
  
    def _evaluate_coalition_batch(self, x_batch, y_batch, module, coalition_mask):  
        """  
        批量评估联盟值  
        """  
        def mask_hook(mod, inp, out):  
            masked_out = out.clone()  
            mask_tensor = torch.tensor(coalition_mask, dtype=torch.float32).to(self.device)  
              
            if len(masked_out.shape) > 2:  
                mask_tensor = mask_tensor.view(1, -1, 1, 1)  
            else:  
                mask_tensor = mask_tensor.view(1, -1)  
              
            return masked_out * mask_tensor  
          
        handle = module.register_forward_hook(mask_hook)  
        try:  
            output = self.model(x_batch)  
            loss = self.criterion(output, y_batch, reduction="none")  
            return loss.detach().cpu().numpy()  
        finally:  
            handle.remove()  
  
    def find_evaluation_module(self, module, find_best_evaluation_module=False):  
        return module
