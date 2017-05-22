## Guesser: {{ guesser_name }}

In the following report the `dev` fold refers to `guesserdev`, not the literal `dev` fold

### Dataset Statistics

Total number of answer classes in all folds: {{ n_answers_all_folds }}

Quiz Bowl dataset with min_class_examples={{ min_class_examples }}
(This means an answer must have at least n={{ min_class_examples }} training examples to be included
in the answer set)

* Number of All Train Questions: {{ n_total_train_questions }}
* Number of Train Min {{ min_class_examples }} Questions: {{ n_train_questions }}
* Number of Dev Questions: {{ n_dev_questions }}

* Number of All Train Answers: {{ n_total_train_answers }}
* Number of Train Min {{ min_answer_examples }} Answers: {{ n_train_answers }}
* Number of Dev Questions: {{ n_dev_answers }}

#### Answerable Classes

##### Using all train answers/questions

* Number of answers in common between train/dev: {{ all_n_common_train_dev }}
* Percent of answers in common between train/dev: {{ all_p_common_train_dev }}

##### Using all min_class_examples={{ min_class_examples }} answers/questions

* Number of answers in common between train/dev: {{ min_n_common_train_dev }}
* Percent of answers in common between train/dev: {{ min_p_common_train_dev }}

#### Answerable Questions

##### Using all train answers/questions

* Number of answerable train questions: {{ all_n_answerable_train }}
* Number of train questions: {{ n_train_questions }}
* Percent of answerable train questions: {{ all_p_answerable_train }}

* Number of answerable dev questions: {{ all_n_answerable_dev }}
* Number of dev questions: {{ n_dev_questions }}
* Percent of answerable dev questions: {{ all_p_answerable_dev }}

##### Using min_class_examples={{ min_class_examples }} answers/questions

* Number of answerable train questions: {{ min_n_answerable_train }}
* Number of train questions: {{ n_train_questions }}
* Percent of answerable train questions: {{ min_p_answerable_train }}

* Number of answerable dev questions: {{ min_n_answerable_dev }}
* Number of dev questions: {{ n_dev_questions }}
* Percent of answerable dev questions: {{ min_p_answerable_dev }}

### Parameters

{{ guesser_params | pprint }}

### Accuracy

#### Dev Fold

Position | Accuracy
---------|---------
start|{{ dev_accuracy['start'] }}
25%|{{ dev_accuracy['p_25'] }}
50%|{{ dev_accuracy['p_50'] }}
75%|{{ dev_accuracy['p_75'] }}
end|{{ dev_accuracy['end'] }}

![Dev Accuracy]({{ dev_accuracy_plot }}){width=60%}
\ 


### Recall

#### Dev Fold

![Dev Recall]({{ dev_recall_plot }}){width=60%}
\ 


### Dev: Number correct by Number of Training Examples


![Dev Correct by N Training Examples]({{ dev_correct_by_count_plot }}){width=60%}
\


### Number of Dev Examples vs Training Examples


![Number of Dev Examples by Training Examples]({{ n_train_vs_dev_plot }}){width=60%}
\
