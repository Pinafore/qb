## VW Audit Regressor: Feature Weights

This report contains output from running VW's audit regressor as shown below (real file paths are different):

```
vw --compressed -t -d vw_input.tar.gz -i vw_model --audit_regressor regressor_audit.txt
```

### Top Features

![]({{ feature_importance_plot }}){width=50%}

### Top 100 Weights

```
{{ top_features }}
```
