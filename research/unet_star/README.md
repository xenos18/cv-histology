# U-Net star development

U-net star is the unofficial name of the improvement for U-Net, which is to repeat the following steps:

1. training the model on the training data
2. update the training data using model prediction (if model confidence >= alpha -> update the pixel class, otherwise leave the original one)

This system did not produce meaningful results on a cursory review, so the notebooks with it are not in a proper state. Nevertheless, it is published here to allow for retesting and refinement.

N.B.: this README was created by translate russian README using DeepL Translate, this version can contain some inaccuracies.