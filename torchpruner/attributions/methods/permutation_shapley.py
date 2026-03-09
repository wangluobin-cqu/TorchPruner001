import torch  
import numpy as np  
from ..attributions import _AttributionMetric  
  
class PermutationShapleyAttributionMetric(_AttributionMetric):  
    """  
    基于排列的Shapley值近似方法  
    通过随机排列顺序来计算Shapley值  
    """  
      
    def __init__(self, *args, sv_samples=5, **kwargs):  
        super().__init__(*args, **kwargs)  
        self.samples = sv_samples  
      
    def run(self, module, **kwargs):  
        module = super().run(module, **kwargs)  
          
        if hasattr(self.model, "forward_partial"):  
            result = self._run_with_partial(module)  
        else:  
            result = self._run_full(module)  
          
        return result  
      
    def _run_with_partial(self, module):  
        """使用partial forward的实现"""  
        d = len(self.data_gen.dataset)  
        sv = np.zeros((d, module.weight.shape[0]))  
          
        with torch.no_grad():  
            for idx, (x, y) in enumerate(self.data_gen):  
                x, y = x.to(self.device), y.to(self.device)  
                original_z, _ = self.run_forward_partial(x, to_module=module)  
                _, original_loss = self.run_forward_partial(original_z, y_true=y, from_module=module)  
                  
                n = original_z.shape[1]  
                  
                for j in range(self.samples):  
                    # 生成随机排列  
                    permutation = np.random.permutation(n)  
                    z = original_z.clone()  
                    current_loss = original_loss.clone()  
                      
                    # 按排列顺序逐个添加特征  
                    for i, feature_idx in enumerate(permutation):  
                        # 恢复当前特征  
                        temp_z = z.clone()  
                        temp_z[:, feature_idx] = original_z[:, feature_idx]  
                          
                        _, new_loss = self.run_forward_partial(temp_z, y_true=y, from_module=module)  
                          
                        # 计算边际贡献  
                        marginal = (new_loss - current_loss) / self.samples  
                        sv[idx, feature_idx] += marginal.squeeze().detach().cpu().numpy()  
                          
                        current_loss = new_loss  
                        z[:, feature_idx] = original_z[:, feature_idx]  
          
        return self.aggregate_over_samples(sv)  
      
    def _run_full(self, module):  
        """完整前向传播实现"""  
        # 实现类似逻辑但不使用partial forward  
        pass
