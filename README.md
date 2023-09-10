<h1 align="center">
    Multimodal Learning for Handwritten Grapheme Classification <br>
</h1>

<div align = "justify">

**Description:** This repository is a project that constituted a late submission to the [BengaliAI Handwritten Grapheme Classification Challenge](https://www.kaggle.com/c/bengaliai-cv19/overview) on Kaggle.

---

</div>

# Competition description

## Task formulation

You’re given the image of a handwritten Bengali grapheme and are challenged to separately classify three constituent elements in the image: grapheme root, vowel diacritics, and consonant diacritics.

# Approach

## What worked

* Model: ResNet34 with three heads
* Head configuration: Mish -> Conv2D -> BatchNorm -> Pooling layer -> Linear
* Train with Cutmix (alpha=1) and Mixup (alpha=4) augmentations for more than 100 epochs, as these augmentations require a big number of epochs to converge.
* Pooling layer: 0.5 * (AveragePooling + MaxPooling)
* Dataset: 5-fold of uncropped images generated via stratified split using multilabel stratification from [iterative_stratification](https://pypi.org/project/iterative-stratification/).
* Weights: 7-grapheme, 1-consonant, 2-vowel
* AdamW and OneCycleWithWarmUp

## What didn't work

* SEResNext
* RAdam and Over9000 optimizer
* 3 different models instead of a single model with 3 heads
* Postprocessing

## References

I've gotten references for several tried techniques from multiple notebooks, but the main inspirations for the overall attempt:
* [Phalanx's 3rd place solution](https://www.kaggle.com/competitions/bengaliai-cv19/discussion/135982)
* [Dieter's 5th place solution](https://www.kaggle.com/competitions/bengaliai-cv19/discussion/136129)
* [Cyr1ll's 41st place solution](https://www.kaggle.com/competitions/bengaliai-cv19/discussion/136084)
* [WalkWithFastAI (initial experiment)](https://walkwithfastai.com/Multimodal_Head_and_Kaggle)
