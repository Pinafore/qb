# End-of-pipeline Report

This report contains two end-of-pipeline analysis of the qanta guesser and
buzzer. The first is a buzzer performance analysis; the second is the
performance of guesser and buzzer plotted against time axis.

## End-of-pipeline

- buzz: how frequent does the the buzzer buzz
- choose_best: did the buzzer choose the best guesser (earliest correct)
- choose_hopeful: did the buzzer choose a hopeful guesser
- rush: did the buzzer rush (w.r.t to all guessers)
- late: did the buzzer buzz too late (w.r.t to all guessers)
- not_buzzing_when_shouldnt: did buzzer choose not to buzz when the question is not hopeful
- reward: the average reward over the dataset (without opponent)
- hopeful: is the question hopeful (w.r.t to all guessers)
- correct: how many correct guessers
- choose_guesser: the guesser chosen by the buzzer
- best_guesser: the best guesser (earliest correct)

### Configs

{% for name, config in hype_configs['dev'] %}
- Config {{name}}
    - n_layers: {{ config.n_layers }}
    - n_hidden: {{ config.n_hidden }}
    - batch_norm: {{ config.batch_norm }}
    - neg_weight: {{ config.neg_weight }}
{% endfor %}

### Buzzing Too Early or Too Late
{% for fold, plot in rush_late_plot.items() %}
![{{fold}} rush & late]({{ plot }}){width=100%}
{% endfor %}

## Histogram

The numbers we report in this section are the accuracy of the guessers at
different positions and the buzzing behaviour of the buzzer. Ideally the overall
buzzing behaviour should match the accuracy curve, and the buzzer's selection of
guessers should reflect the difference in performance of different guessers.

### Lines

In this following figures, same line style indicates same guesser.

{% for fold, plot in his_lines.items() %}
![{{fold}} histogram]({{ plot }}){width=100%}
{% endfor %}

### Stacked Aread Charts

{% for fold, plot in his_stacked.items() %}
![{{fold}} histogram]({{ plot }}){width=100%}
{% endfor %}
