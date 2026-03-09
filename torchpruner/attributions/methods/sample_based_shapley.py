import torch  
import numpy as np  
from ..attributions import _AttributionMetric  
  
class SampleBasedShapleyAttributionMetric(_AttributionMetric):  
    """  
    基于采样的Shapley值近似方法  
    使用随机采样来近似计算Shapley值  
    """  
      
    def __init__(self, *args, sv_samples=10, **kwargs):  
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
        """使用partial forward的快速实现"""  
        d = len(self.data_gen.dataset)  
        sv = np.zeros((d, module.weight.shape[0]))  
          
        with torch.no_grad():  
            for idx, (x, y) in enumerate(self.data_gen):  
                x, y = x.to(self.device), y.to(self.device)  
                original_z, _ = self.run_forward_partial(x, to_module=module)  
                _, original_loss = self.run_forward_partial(original_z, y_true=y, from_module=module)  
                  
                n = original_z.shape[1]  
                  
                for j in range(self.samples):  
                    # 随机采样子集  
                    subset_size = np.random.randint(1, n)  
                    subset = np.random.choice(n, subset_size, replace=False)  
                      
                    z = original_z.clone()  
                    z.index_fill_(1, torch.tensor(subset).long().to(self.device), 0.0)  
                    _, new_loss = self.run_forward_partial(z, y_true=y, from_module=module)  
                      
                    # 计算边际贡献  
                    marginal_contribution = (new_loss - original_loss) / self.samples  
                    sv[idx, subset] += marginal_contribution.squeeze().detach().cpu().numpy()  
          
        return self.aggregate_over_samples(sv)  
      
    def _run_full(self, module):  
        """完整前向传播实现"""  
        # 实现类似逻辑但不使用partial forward  
        pass
