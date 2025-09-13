import numpy as np
import torch

class ApproxAttribution():
    def __init__(self,model):
        self.model = model
        self.device = next(self.model.parameters()).device

    def grad_approx(self,h,X,target=0):
        ret = []
        for elem in X:
            for index, value in np.ndenumerate(elem):
                elem = elem.to(self.device)
                new = elem.clone()
                new[index] += h
                res = np.abs(((self.model(new) - self.model(elem))/h).detach().cpu().numpy())
                ret.append(res)
        ret = np.array(ret)
        if ret.ndim == 1:
            return torch.tensor(ret)
        return torch.tensor(ret[:,0,target])

    def int_grad_approx(self,h,X,baseline = 0,riemann_step = 50,target=0):
        if not torch.is_tensor(X):
            X = torch.stack(X)
        ret = []
        if not torch.is_tensor(baseline):
            baseline = torch.tensor(baseline, dtype=X.dtype, device=X.device)
        if baseline.shape == torch.Size([]):
            baseline = torch.full_like(X[0],baseline, dtype=torch.float)
        for elem in X:
            for index, value in np.ndenumerate(elem):
                elem = elem.to(self.device)
                new = elem.clone()
                new[index] += h
                attr = 0
                for i in range(1,riemann_step+1):
                    scaled_new = baseline[index] + (i / riemann_step) * (new - baseline[index])
                    scaled_elem = baseline[index] + (i / riemann_step) * (elem - baseline[index])
                    attr+= ((self.model(scaled_new) - self.model(scaled_elem))/h).detach().cpu()



                val = (attr/riemann_step) * (value-baseline[index])

                ret.append(val)
        ret = np.array(ret)
        if ret.ndim != 3:
            return torch.tensor(ret)
        return torch.tensor(ret[:,0,target])

    def grad_x_i_approx(self,h,X,target=0):
        ret = []
        for elem in X:
            for index, value in np.ndenumerate(elem):
                elem = elem.to(self.device)
                new = elem.clone()
                new[index] += h

                ret.append(value * ((self.model(new) - self.model(elem))/h).detach().cpu().numpy())
        ret = np.array(ret)
        if ret.ndim == 1:
            return torch.tensor(ret)
        return torch.tensor(ret[:,0,target])


def pearson_correlation(X,Y):
    res = []
    for x,y in zip(X,Y):
        res.append(np.corrcoef(x,y)[0,1])
    return res