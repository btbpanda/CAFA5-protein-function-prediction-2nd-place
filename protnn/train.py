import torch
import tqdm
from torch import nn


def train(model, swa, train_dl, val_dl, evaluator, n_ep=20, lr=1e-3, clip_grad=1, weight_decay=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    scores = []
    for n in range(n_ep):

        model.train()
        for batch in tqdm.tqdm(train_dl):
            opt.zero_grad()

            output = model(batch)
            loss = loss_fn(output, batch['y'])
            loss.backward()

            if clip_grad is not None:
                nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_grad)
            opt.step()

        score = evaluator(model, val_dl)
        swa.add_checkpoint(model, score=score)
        print(f'Epoch {n}: CAFA5 score {score}')
        scores.append(score)

    return model, swa, scores
