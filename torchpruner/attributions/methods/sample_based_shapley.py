import torch
import numpy as np
from ..attributions import _AttributionMetric


class SampleBasedShapleyAttributionMetric(_AttributionMetric):
    """
    通过随机采样 coalition 来近似 Shapley 值。
    对每个单元 i，采样若干个不含 i 的子集 S，
    用 v(S ∪ {i}) - v(S) 的平均值近似其贡献。
    """

    def __init__(self, *args, sv_samples=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples = sv_samples
        self.mask_indices = []

    def run(self, module, sv_samples=None, **kwargs):
        module = super().run(module, **kwargs)
        sv_samples = sv_samples if sv_samples is not None else self.samples

        if hasattr(self.model, "forward_partial"):
            return self._run_with_partial(module, sv_samples)
        else:
            return self._run_full(module, sv_samples)

    def _run_with_partial(self, module, sv_samples):
        d = len(self.data_gen.dataset)
        sv = None
        c = 0

        with torch.no_grad():
            for x, y in self.data_gen:
                x, y = x.to(self.device), y.to(self.device)

                original_z, _ = self.run_forward_partial(x, to_module=module)
                n_units = original_z.shape[1]
                batch_size = original_z.shape[0]

                if sv is None:
                    sv = np.zeros((d, n_units), dtype=np.float32)

                # 对每个单元单独估计边际贡献
                for i in range(n_units):
                    contrib = torch.zeros(batch_size, device=self.device)

                    candidate_indices = [j for j in range(n_units) if j != i]

                    for _ in range(sv_samples):
                        subset_size = np.random.randint(0, n_units)  # 可以取空集
                        if subset_size > len(candidate_indices):
                            subset_size = len(candidate_indices)

                        if subset_size == 0:
                            subset = []
                        else:
                            subset = np.random.choice(
                                candidate_indices, subset_size, replace=False
                            ).tolist()

                        # v(S ∪ {i})：保留 i，其余不在 coalition 中的置零
                        keep_with_i = set(subset + [i])
                        mask_with_i = [j for j in range(n_units) if j not in keep_with_i]

                        z_with_i = original_z.clone()
                        if len(mask_with_i) > 0:
                            z_with_i.index_fill_(
                                1,
                                torch.tensor(mask_with_i, device=self.device),
                                0.0
                            )
                        _, loss_with_i = self.run_forward_partial(
                            z_with_i, y_true=y, from_module=module
                        )

                        # v(S)：不包含 i
                        keep_without_i = set(subset)
                        mask_without_i = [j for j in range(n_units) if j not in keep_without_i]

                        z_without_i = original_z.clone()
                        if len(mask_without_i) > 0:
                            z_without_i.index_fill_(
                                1,
                                torch.tensor(mask_without_i, device=self.device),
                                0.0
                            )
                        _, loss_without_i = self.run_forward_partial(
                            z_without_i, y_true=y, from_module=module
                        )

                        contrib += (loss_with_i - loss_without_i).view(-1)

                    sv[c:c+batch_size, i] = (contrib / sv_samples).detach().cpu().numpy()

                c += batch_size

        return self.aggregate_over_samples(sv)

    def _run_full(self, module, sv_samples):
        raise NotImplementedError(
            "SampleBasedShapleyAttributionMetric currently requires model.forward_partial()"
        )
