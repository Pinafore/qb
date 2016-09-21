## Classifier: Gender

This classifier is responsible for determining if a question is describing a male, female, or
non-person. The scores are produced by talking all the runs of sentences in the train/dev folds

### Scores

Fold | Score
-----|------
train|{{ train_score }}
dev|{{ dev_score }}

### Confusion Matrices
![]({{ unnormalized_confusion_plot }}){width=50%} ![]({{ normalized_confusion_plot }}){width=50%}
