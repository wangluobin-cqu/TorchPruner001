

import itertools  
import numpy as np  
from math import factorial  
import torch  
from ..attributions import _AttributionMetric  
  
class StructuredShapleyAttributionMetric(_AttributionMetric):  
    """  
    计算结构化Shapley值归因，将特征分组后计算每个特征的贡献  
    """  
      
    def init(self, *args, group_size=2, **kwargs):  
        super().__init__(*args, **kwargs)  
        self.group_size = group_size  
      
    def run(self, module, group_size=None, **kwargs):  
        module = super().run(module, **kwargs)  
        group_size = group_size if group_size is not None else self.group_size  
          
        # 收集所有样本的归因值  
        all_attributions = []  
          
        with torch.no_grad():  
            for idx, (x, y) in enumerate(self.data_gen):  
                x, y = x.to(self.device), y.to(self.device)  
                  
                # 对每个样本计算结构化Shapley值  
                for sample_idx in range(x.shape[0]):  
                    sample_x = x[sample_idx:sample_idx+1]  
                    sample_y = y[sample_idx:sample_idx+1]  
                      
                    # 获取模块的激活值维度  
                    with torch.no_grad():  
                        temp_output = self._get_module_output(sample_x, module)  
                        d = temp_output.shape[1]  # 特征维度  
                      
                    # 计算结构化Shapley值  
                    shapley_values = self._compute_structured_shapley(  
                        sample_x, sample_y, module, d, group_size  
                    )  
                    all_attributions.append(shapley_values)  
          
        # 聚合所有样本的归因值  
        return self.aggregate_over_samples(np.array(all_attributions))  
      
    def _get_module_output(self, x, module):  
        """获取指定模块的输出"""  
        def hook_fn(module, input, output):  
            self._temp_output = output  
          
        handle = module.register_forward_hook(hook_fn)  
        try:  
            self.model(x)  
            return self._temp_output  
        finally:  
            handle.remove()  
            if hasattr(self, '_temp_output'):  
                delattr(self, '_temp_output')  
      
    def _compute_structured_shapley(self, x, y, module, d, group_size):  
        """计算结构化Shapley值"""  
        num_groups = d // group_size  
        feature_indices = list(range(d))  
          
        # 将特征分组  
        groups = [feature_indices[i*group_size:(i+1)*group_size] for i in range(num_groups)]  
        shapley_values = np.zeros(d)  
          
        for group in groups:  
            other_features = list(set(feature_indices) - set(group))  
              
            for x_j in group:  
                phi_j = 0.0  
                local_group = list(group)  
                local_group.remove(x_j)  
                  
                # 遍历所有 R_j ⊆ 其他组特征  
                for R in self._powerset(other_features):  
                    r_size = len(R)  
                    if len(other_features) == 0:  
                        w_r = 1.0  
                    else:  
                        w_r = factorial(r_size) * factorial(len(other_features) - r_size) / factorial(len(other_features))  
                      
                    # 遍历所有 S_j ⊆ 当前组（除 x_j）  
                    for S in self._powerset(local_group):  
                        s_size = len(S)  
                        if len(group) <= 1:  
                            w_s = 1.0  
                        else:  
                            w_s = factorial(s_size) * factorial(len(local_group) - s_size) / factorial(len(group) - 1)  
                          
                        full_set = list(R) + list(S) + [x_j]  
                        subset = list(R) + list(S)  
                          
                        # 计算损失差异  
                        delta = self._compute_loss_difference(x, y, module, full_set, subset)  
                        phi_j += w_r * w_s * delta  
                  
                shapley_values[x_j] = phi_j  
          
        return shapley_values  
      
    def _compute_loss_difference(self, x, y, module, full_set, subset):  
        """计算掩码后的损失差异"""  
        # 计算包含完整集合的损失  
        loss_full = self._compute_masked_loss(x, y, module, full_set)  
        # 计算包含子集的损失  
        loss_subset = self._compute_masked_loss(x, y, module, subset)  
          
        return loss_full - loss_subset  
      
    def _compute_masked_loss(self, x, y, module, active_indices):  
        """计算掩码激活后的损失"""  
        def mask_hook(module, input, output):  
            # 创建掩码，只保留active_indices中的激活  
            mask = torch.zeros_like(output)  
            if len(active_indices) > 0:  
                mask[:, active_indices] = 1.0  
            return output * mask  
          
        handle = module.register_forward_hook(mask_hook)  
        try:  
            with torch.no_grad():  
                pred = self.model(x)  
                loss = self.criterion(pred, y, reduction='none')  
                return loss.item()  
        finally:  
            handle.remove()  
      
    @staticmethod  
    def _powerset(s):  
        """返回集合s的所有子集"""  
        return list(itertools.chain.from_iterable(  
            itertools.combinations(s, r) for r in range(len(s)+1)  
        ))
