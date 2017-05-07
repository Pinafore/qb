## Guesser Comparison Report

This report runs a comparison between all enabled guessers. It in general terms generates the following:

* With `N` guessers enabled, compute accuracy through different positions in the question broken down by how many guessers got it correct.
* Sample `K` questions at random for each number of correct guessers and show the question text along with which guessers got it correct and which got it wrong. This encourages looking at micro examples of how the system behaves
* For each possible pair of guessers, compute more detailed statistics on how their guesses differ

### Accuracy by Number of Guessers Correct

![Accuracy by N Correct Guessers]({{ dev_accuracy_by_n_correct_plot }}){width=70%}

### Sampled Questions
{% for n_correct, samples in sampled_questions_by_correct.items() %}
#### N={{ n_correct }} Guessers Correct Samples
{% for sample in samples %}
##### Text:

> {{ sample[0] }}

Qnum, Sent, Token: {{ sample[1] }}

Answer: {{ sample[2] }}

N Training Questions: {{ sample[3] }}

Correct Guessers: {{ sample[4] }}

Wrong Guessers: {{ sample[5] }}
{% endfor %}
{% endfor %}
