from copy import copy

import cupy as cp
from py_boost import Callback
from py_boost.gpu.losses import BCELoss, BCEMetric


class BCEWithNaNLoss(BCELoss):

    def base_score(self, y_true):
        # Replace .mean with nanmean function to calc base score
        means = cp.nanmean(y_true, axis=0)
        means = cp.where(cp.isnan(means), 0, means)
        means = cp.clip(means, self.clip_value, 1 - self.clip_value)

        return cp.log(means / (1 - means))

    def get_grad_hess(self, y_true, y_pred):
        # first, get nan mask for y_true
        mask = cp.isnan(y_true)
        # then, compute loss with any values at nan places just to prevent the exception
        grad, hess = super().get_grad_hess(cp.where(mask, 0, y_true), y_pred)
        # invert mask
        mask = (~mask).astype(cp.float32)
        # multiply grad and hess on inverted mask
        # now grad and hess eq. 0 on NaN points
        # that actually means that prediction on that place should not be updated
        grad = grad * mask
        hess = hess * mask

        return grad, hess


class BCEwithNaNMetric(BCEMetric):

    def __call__(self, y_true, y_pred, sample_weight=None):
        mask = ~cp.isnan(y_true)

        err = super().error(cp.where(mask, y_true, 0), y_pred)
        err = err * mask

        if sample_weight is not None:
            err = err * sample_weight
            mask = mask * sample_weight

        return float(err.sum() / mask.sum())


class WarmStart(Callback):

    def __init__(self, model):
        model.to_cpu()
        self.model = copy(model)
        self.model.postprocess_fn = lambda x: x

    def before_train(self, build_info):
        build_info['model'].base_score = cp.asarray(self.model.base_score)

        train = build_info['data']['train']
        train['ensemble'] = cp.asarray(self.model.predict(train['features_cpu']))

        valid = build_info['data']['valid']
        valid['ensemble'] = [cp.asarray(self.model.predict(x)) for x in valid['features_cpu']]

        self.model.to_cpu()

        return

    def after_train(self, build_info):
        build_info['model'].models = self.model.models + build_info['model'].models
        # update the actual iteration
        build_info['num_iter'] = build_info['num_iter'] + len(self.model.models)
        # update the actual best round
        early_stop = build_info['model'].callbacks.callbacks[-1]
        early_stop.best_round = early_stop.best_round + len(self.model.models)

        # not to store old trees multiple times
        self.model = None

        return
