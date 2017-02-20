## Guesser: {{ guesser_name }}

### Parameters

{{ guesser_params }}

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

#### Test Fold

Position | Accuracy
---------|---------
start|{{ test_accuracy['start'] }}
25%|{{ test_accuracy['p_25'] }}
50%|{{ test_accuracy['p_50'] }}
75%|{{ test_accuracy['p_75'] }}
end|{{ test_accuracy['end'] }}

![Test Accuracy]({{ test_accuracy_plot }}){width=60%}
\ 

### Recall

#### Dev Fold

![Dev Recall]({{ dev_recall_plot }}){width=60%}
\ 

### Test Fold

![Test Recall]({{ test_recall_plot }}){width=60%}
\ 
