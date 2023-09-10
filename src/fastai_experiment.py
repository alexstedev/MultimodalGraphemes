import pandas as pd
import numpy as np
from functools import partial
from torch import nn
import torch.nn.functional as F
import torch

from fastai.callback.progress import ProgressCallback
from fastai.callback.schedule import fit_one_cycle

from fastai.data.block import CategoryBlock
from fastai.data.transforms import ColReader, Normalize, RandomSplitter

from fastai.learner import Metric, Learner

from fastai.torch_core import Module

from fastai.vision.augment import aug_transforms
from fastai.vision.core import PILImageBW
from fastai.vision.data import ImageBlock
from fastai.vision.learner import create_body, create_head

train = pd.read_csv('bengaliai/train.csv')
test = pd.read_csv('bengaliai/test.csv')
class_map = pd.read_csv('bengaliai/class_map.csv')

train.head()

graph_vocab = train['grapheme_root'].unique()
vowel_vocab = train['vowel_diacritic'].unique()
const_vocab = train['consonant_diacritic'].unique()

blocks = (ImageBlock(cls=PILImageBW),
          CategoryBlock(vocab=graph_vocab),
          CategoryBlock(vocab=vowel_vocab),
          CategoryBlock(vocab=const_vocab))

getters = [
           ColReader('image_id', pref='images/', suff='.png'),
           ColReader('grapheme_root'),
           ColReader('vowel_diacritic'),
           ColReader('consonant_diacritic')
]

batch_tfms = [*aug_transforms(do_flip=False, size=128),
              Normalize.from_stats(mean=0.0692, std=0.2051)]

bengel = DataBlock(blocks=blocks,
                   getters = getters,
                   splitter=RandomSplitter(),
                   batch_tfms=batch_tfms,
                   n_inp=1)

bs=128

dls  = bengel.dataloaders(train.sample(1000), bs=bs)
dls.show_batch(max_n=1, figsize=(3,3))


# CUSTOM HEADS
n = train[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].nunique(); print(n)

dls.c

body = create_body(resnet34, pretrained=True)

l = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2),
                    padding=(3,3), bias=False)
l.weight = nn.Parameter(l.weight.sum(dim=1, keepdim=True))

body[0] = l


class MultiModel(Module):
  "A three-headed model given a `body` and `n` output features"
  def __init__(self, body:nn.Sequential, n:L):
    nf = num_features_model(nn.Sequential(*body.children())) * (2)
    self.body = body
    self.grapheme = create_head(nf, n[0])
    self.vowel = create_head(nf, n[1])
    self.consonant = create_head(nf, n[2])

  def forward(self, x):
    y = self.body(x)
    graph = self.grapheme(y)
    vowel = self.vowel(y)
    const = self.consonant(y)
    return [graph, vowel, const]

net = MultiModel(body, dls.c)

"""## Training

We're going to want a custom loss function here. We'll base it on Miguel Pinto's notebook [here](https://www.kaggle.com/mnpinto/bengali-ai-fastai-starter-lb0-9598).
"""

from sklearn.metrics import recall_score

class CombinationLoss(Module):
    "Cross Entropy Loss on multiple targets"
    def __init__(self, func=F.cross_entropy, weights=[2, 1, 1]):
        self.func, self.w = func, weights

    def forward(self, xs, *ys, reduction='mean'):
        for i, w, x, y in zip(range(len(xs)), self.w, xs, ys):
            if i == 0: loss = w*self.func(x, y, reduction=reduction)
            else: loss += w*self.func(x, y, reduction=reduction)
        return loss

class RecallPartial(Metric):
    "Stores predictions and targets on CPU in accumulate to perform final calculations with `func`."
    def __init__(self, a=0, **kwargs):
        self.func = partial(recall_score, average='macro', zero_division=0)
        self.a = a

    def reset(self): self.targs,self.preds = [],[]

    def accumulate(self, learn):
        pred = learn.pred[self.a].argmax(dim=-1)
        targ = learn.y[self.a]
        pred,targ = to_detach(pred),to_detach(targ)
        pred,targ = flatten_check(pred,targ)
        self.preds.append(pred)
        self.targs.append(targ)

    @property
    def value(self):
        if len(self.preds) == 0: return
        preds,targs = torch.cat(self.preds),torch.cat(self.targs)
        return self.func(targs, preds)

    @property
    def name(self): return train.columns[self.a+1]

class RecallCombine(Metric):
    def accumulate(self, learn):
        scores = [learn.metrics[i].value for i in range(3)]
        self.combine = np.average(scores, weights=[2,1,1])

    @property
    def value(self):
        return self.combine

learn = Learner(dls, net, loss_func=CombinationLoss(),
                metrics=[RecallPartial(a=i) for i in range(len(dls.c))] + [RecallCombine()],
                )

learn.fit_one_cycle(10, 1e-3)

