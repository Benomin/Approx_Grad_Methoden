import numpy as np

class ApproxAttribution():
    def __init__(self,model):
        self.model = model
        self.device = next(self.model.parameters()).device

    def grad_approx(self,h,X,target=0):
        print("Grad Approx")
        ret = []
        for elem in X:
            for index, value in np.ndenumerate(elem):
                elem = elem.to(self.device)
                new = elem.clone()
                new[index] += h
                res = np.abs(((self.model(new) - self.model(elem))/h).detach().cpu().numpy())
                #test
                ret.append(res)
                #print(res)
        return np.array(ret)[:,0,target]

    def int_grad_approx(self,h,X,baseline = 0,riemann_step = 50,target=0):
        ret = []
        for elem in X:
            for index, value in np.ndenumerate(elem):
                elem = elem.to(self.device)
                new = elem.clone()
                new[index] += h
                attr = 0
                for i in range(1,riemann_step+1):
                    scaled_new  = (baseline + i/riemann_step) * (new - baseline)
                    scaled_elem = (baseline + i/riemann_step) * (elem - baseline)
                    attr+= ((self.model(scaled_new) - self.model(scaled_elem))/h).detach().cpu().numpy()


                val = (attr/riemann_step) * (value-baseline)
                if np.array([np.isnan(j) for j in self.model(scaled_new).detach().cpu().numpy()]).any():
                    print("Error")
                    print((self.model(scaled_new)))


                ret.append(val)


        return np.array(ret)[:,0,target]

    def grad_x_i_approx(self,h,X,target=0):
        ret = []
        for elem in X:
            for index, value in np.ndenumerate(elem):
                elem = elem.to(self.device)
                new = elem.clone()
                new[index] += h



                ret.append(-1 * value * ((self.model(new) - self.model(elem))/h).detach().cpu().numpy())
        return np.array(ret)[:,0,target]


def pearson_correlation(X,Y):
    res = []
    for x,y in zip(X,Y):
        res.append(np.corrcoef(x,y)[0,1])
    return res