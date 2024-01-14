import os

import torch


# some usefull tools for training and using networks

class SWA:
    """
    Implements a version of stochastic weights averaging
    It saves n the best epochs states and then average it in final model.
    n - n checkpoints to save
    path - folder to save checkpoints
    weighted - weight checkpoints by score
    rewrite - rewrite checkpoints in folder if exists or raise error
    use_apex - use apex mixed precision. Should be True if trained with apex
    """

    def __init__(self, n, path='temp_swa',  rewrite=False, use_apex=False):

        self.store_n = n
        self.scores = []
        self.file = path + '/checkpoint{0}_{1}.pth'
        os.makedirs(path, exist_ok=rewrite)
        # self.weighted = weighted
        self.use_apex = use_apex

        for f in [x for x in os.listdir(path) if x[:10] == 'checkpoint']:

            if os.path.exists(path + '/' + f):
                os.remove(path + '/' + f)

    def add_checkpoint(self, model, score=1, ):

        for n in range(self.store_n):
            if len(self.scores) == n or score >= self.scores[n]:
                break
        else:
            return None

        for i in range(len(self.scores), n, -1):
            os.rename(self.file.format(i - 1, self.scores[i - 1]), self.file.format(i, self.scores[i - 1]))

        if len(self.scores) > 0 and os.path.exists(self.file.format(self.store_n, self.scores[-1])):
            os.remove(self.file.format(self.store_n, self.scores[-1]))

        self.scores.insert(n, score)

        if len(self.scores) > self.store_n:
            self.scores.pop()

        torch.save(model.state_dict(), self.file.format(n, score))

        return None

    def set_weights(self, model, n=10, weighted=True, **apex_params):

        n = min(n, len(self.scores))

        for k, score in enumerate(self.scores):

            if k == n:
                break

            w = score / sum(self.scores[:n]) if weighted else 1 / n

            new_state = torch.load(self.file.format(k, score))
            # upd new state with weights
            for i in new_state.keys():
                new_state[i] = new_state[i] * w

            if k == 0:
                state_dict = new_state
            else:
                # upd state
                for i in state_dict.keys():
                    state_dict[i] += new_state[i]

        model.load_state_dict(state_dict)

        return model

    def set_weight(self, model, n=0, **apex_params):

        model.load_state_dict(torch.load(self.file.format(n, self.scores[n])))

        return model
