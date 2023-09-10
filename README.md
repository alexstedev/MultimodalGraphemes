<h1 align="center">
    Multimodal Learning for Handwritten Grapheme Classification <br>
</h1>

<div align="center">

    <a href="https://www.kaggle.com/c/bengaliai-cv19/overview">![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)</a>

</div>

<div align = "justify">

**Description:** This repository is a project that constituted a late submission to the [BengaliAI Handwritten Grapheme Classification Challenge](https://www.kaggle.com/c/bengaliai-cv19/overview) on Kaggle.

---

</div>

# Competition description

## Task formulation

You’re given the image of a handwritten Bengali grapheme and are challenged to separately classify three constituent elements in the image: grapheme root, vowel diacritics, and consonant diacritics.


## Data and metric 

* ~200000 single channel images
* Class distribution: 
    * Grapheme root: 168 classes
    * Vowel diacritic: 11 classes
    * Consonant diacritic: 7 classes
    * 168 * 11 * 77 ~ 13k different grapheme variations
* Metric: weighted sum of recall for each component with weights:
  * 2 - Grapheme root
  * 1 - Vowel diacritic
  * 1 - Consonant diacritic

# Approach

## Final solution

* Model: ResNet34 with three heads
* Head configuration: Mish -> Conv2D -> BatchNorm -> Pooling layer -> Linear
* Train with Cutmix (alpha=1) and Mixup (alpha=4) augmentations for more than 100 epochs, as these augmentations require a big number of epochs to converge.
* Pooling layer: 0.5 * (AveragePooling + MaxPooling)
* Dataset: 5-fold of uncropped images generated via stratified split using multilabel stratification from [iterative_stratification](https://pypi.org/project/iterative-stratification/).
* Weights: 7-grapheme, 1-consonant, 2-vowel
* AdamW and OneCycleWithWarmUp

## References

I've gotten references for several tried techniques from multiple notebooks, but the main inspirations for the overall attempt:
* [Phalanx's 3rd place solution](https://www.kaggle.com/competitions/bengaliai-cv19/discussion/135982)
* [Dieter's 5th place solution](https://www.kaggle.com/competitions/bengaliai-cv19/discussion/136129)
