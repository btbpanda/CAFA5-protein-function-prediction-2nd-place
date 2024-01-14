import cupy as cp
import numpy as np
import tqdm


class LogRegMultilabel:

    def __init__(self, alpha=0.001, max_iter=20, lr=1, tol=1e-5, output_batch=100, intercept_scaling=1):

        self.alpha = alpha
        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self.output_batch = output_batch
        self.intercept_scaling = intercept_scaling
        self.weights = None

    def fit(self, X, y):

        nrows, ncols = X.shape

        mempool = cp.cuda.MemoryPool()
        with cp.cuda.using_allocator(allocator=mempool.malloc):

            # prepare features
            X_ = np.ones((nrows, ncols + 1), dtype=np.float32)
            X[:, 0] *= self.intercept_scaling
            X_[:, 1:] = X
            X = cp.asarray(X_)
            ncols += 1

            # prepare targets
            if len(y.shape) == 1:
                y = y[:, np.newaxis]

            weights = []

            for i in tqdm.tqdm(list(range(0, y.shape[1], self.output_batch))):

                y_ = cp.asarray(y[:, i: i + self.output_batch], dtype=np.float32)
                _, nout = y_.shape
                mask = ~cp.isnan(y_)

                w = cp.zeros((ncols, nout), dtype=np.float32)
                flg = False

                for _ in range(self.max_iter):

                    pred = cp.dot(X, w)  # + self.i
                    p = 1 / (1 + np.exp(-pred))

                    # use float64 to prevent overflow in dot - impossible
                    grad = cp.empty((ncols, nout), dtype=cp.float32)
                    cp.dot(X.T, cp.where(mask, p - y_, 0), out=grad)
                    grad = grad / mask.sum(axis=0).astype(np.float32) + self.alpha * w
                    # loop for hess to prevent OOM
                    delta = cp.zeros((ncols, nout), dtype=np.float32)

                    for k in range(nout):
                        hess = cp.empty((ncols, ncols), dtype=cp.float32)

                        idx = cp.nonzero(mask[:, k])[0]
                        p_sl = p[idx][:, [k]]
                        X_sl = X[idx]
                        cp.dot((X_sl * p_sl * (1 - p_sl)).T, X_sl, out=hess)
                        hess = hess / idx.shape[0] + cp.diag(cp.ones(ncols, dtype=np.float32) * self.alpha)
                        delta[:, k] = cp.dot(cp.linalg.inv(hess), grad[:, k])

                    delta_norm = (delta ** 2).sum()

                    if delta_norm < self.tol:
                        flg = True

                    w -= self.lr * delta

                    if flg:
                        break
                else:
                    print(f'No convergence, delta norm = {delta_norm}')

                weights.append(w.get())

            del X, y_, grad, hess, delta, w, pred, p, idx, p_sl

        mempool.free_all_blocks()
        self.weights = np.concatenate(weights, axis=1)

        return self

    def predict(self, X, batch_size=10000):

        mempool = cp.cuda.MemoryPool()
        res = np.empty((X.shape[0], self.weights.shape[1]), dtype=np.float32)

        with cp.cuda.using_allocator(allocator=mempool.malloc):
            interc, w = cp.asarray(self.weights[0]) * self.intercept_scaling, cp.asarray(self.weights[1:])

            for i in tqdm.tqdm(list(range(0, X.shape[0], batch_size))):
                batch = cp.asarray(X[i: i + batch_size], dtype=np.float32)
                pred = cp.dot(batch, w) + interc
                pred = 1 / (1 + cp.exp(-pred))

                res[i: i + batch_size] = pred.get()

            del pred, batch, interc, w

        mempool.free_all_blocks()

        return res
