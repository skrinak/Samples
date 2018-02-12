#
# Smallest possible cats v dogs
#
from fastai.conv_learner import *
PATH = '../Data/dogscats'
sz=224; bs=64

#############################################################################
tfms = tfms_from_model(resnet50, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=bs)
learn = ConvLearner.pretrained(resnet50, data)
# %time learn.fit(1e-2, 3, cycle_len=1)
learn.fit(1e-2, 3, cycle_len=1)

#############################################################################
learn.unfreeze()
learn.bn_freeze(True)
# %time learn.fit([1e-5, 1e-4, 1e-2], 1, cycle_len=1)
learn.fit([1e-5, 1e-4, 1e-2], 1, cycle_len=1)

#############################################################################
# Test time augomentation 
# %time log_preds, y = learn.TTA()
log_preds, y = learn.TTA()
metrics.log_loss(y, np.exp(log_preds)), accuracy(log_preds, y)


