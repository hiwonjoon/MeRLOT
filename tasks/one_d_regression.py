import gin
import numpy as np

@gin.configurable(module=f'{__name__}')
class Dataset(object):
    ranges = {
        'linear':{
            'all': {
                'A': [(-3,3)],
                'b': [(-3,3)],
            },
            'training': {
                'A': [(-2,-1),(1,2)],
                'b': [(-2,-1),(1,2)],
            },
            'int': {
                'A': [(-1,1)],
                'b': [(-1,1)],
            },
            'ext': {
                'A': [(-3,-2),(2,3)],
                'b': [(-3,-2),(2,3)],
            }
        },
        'quadratic': {
            'all': {
                'A': [(-3,3)],
                'b': [(-3,3)],
            },
            'training': {
                'A': [(-2,-1),(1,2)],
                'b': [(-2,-1),(1,2)],
            },
            'int': {
                'A': [(-1,1)],
                'b': [(-1,1)],
            },
            'ext': {
                'A': [(-3,-2),(2,3)],
                'b': [(-3,-2),(2,3)],
            }
        }
    }

    def __init__(
        self,
        task_random_seed,
        dtype,
        c_interval_size,
        c_len_range,
        q_len_range
    ):
        assert dtype in ['all','training','int','ext']
        self.dtype = dtype

        self.seed = task_random_seed
        self.reset_seed()

        self.c_interval_size = c_interval_size
        self.c_len_range = c_len_range
        self.q_len_range = q_len_range

        self.x_dim = 1
        self.y_dim = 1

    def _pick_constant(self,type):
        ret = {}
        for key,ranges in self.ranges[type][self.dtype].items():
            probs = np.array([(high-low) for low,high in ranges])
            probs = probs / np.sum(probs)

            r = self.rg.choice(len(probs),p=probs)
            c = self.rg.uniform(*ranges[r])

            ret[key] = c
        return ret

    def _gen_linear(self,x_shift):
        C = self._pick_constant('linear')
        A,b = C['A'], C['b']
        return lambda x: A * (x-x_shift) + b

    def _gen_quadratic(self,x_shift):
        C = self._pick_constant('quadratic')
        A,b = C['A'], C['b']

        return lambda x: A * (x-x_shift)**2 + b

    def _compose(self,sep,f1,f2):
        return lambda x: np.where(x < sep,f1(x),f2(x))

    def _add_noise(self,f):
        def foo(x,*args):
            ret = f(x,*args)
            ret += self.rg.normal(scale=0.1,size=ret.shape)
            return ret
        return foo

    def reset_seed(self):
        self.rg = np.random.RandomState(self.seed)

    def gen_task(self,c_len=None,q_len=None,get_range=False):
        c_len = self.rg.random_integers(*self.c_len_range) if c_len is None else c_len
        q_len = self.rg.random_integers(*self.q_len_range) if q_len is None else q_len

        def _build_recursive(center_pts):
            left = self.rg.choice([self._gen_linear,self._gen_quadratic])(x_shift=center_pts)
            if center_pts > 4.:
                return left

            right = _build_recursive(center_pts + 2)
            return self._compose(center_pts + 1, left, right)

        init_r = self.rg.uniform(-5,-4)
        fn = _build_recursive(init_r)
        ob_fn = self._add_noise(fn)

        x = np.linspace(-5,5,200)
        y = fn(x)

        if self.c_interval_size == 0:
            l,u = -5, 5
        else:
            l = self.rg.uniform(-5,5-self.c_interval_size)
            u = l + self.c_interval_size

        train_x = self.rg.uniform(l,u,c_len)
        train_y = ob_fn(train_x)

        test_x = self.rg.uniform(l,u,q_len)
        test_y = fn(test_x)

        if get_range:
            return (fn,x,y,l,u,init_r),train_x,train_y,test_x,test_y
        else:
            return (fn,x,y,l,u),train_x,train_y,test_x,test_y

    def batch(self,batch_size):
        while True:
            batch = []

            c_len = self.rg.random_integers(*self.c_len_range)
            q_len = self.rg.random_integers(*self.q_len_range)

            for _ in range(batch_size):
                (_,_,_,_,_),cx,cy,qx,qy = self.gen_task(c_len,q_len)
                batch.append((cx,cy,qx,qy))

            yield (np.array(e)[:,:,None].astype(np.float32) for e in zip(*batch))

if __name__ == "__main__":
    import matplotlib
    from mpl_toolkits.mplot3d import Axes3D
    matplotlib.use('module://imgcat')
    from matplotlib import pyplot as plt

    ds = Dataset(0,'training',4,(7,15),(5,10))
    while True:
        (task,x,y,l,u),train_x,train_y,test_x,test_y = ds.gen_task(7,10)

        fig,ax = plt.subplots()
        ax.plot(x,y,'-',color='red')
        ax.plot(train_x,train_y,'+',color='blue')
        ax.plot(test_x,test_y,'*',color='green')

        fig.show()
        input()
        plt.close(fig)

