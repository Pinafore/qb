jq '. | map({question: .question, answer: .answer, model: "es"})' es_final_adversarial_questions.json > es_adv.json
jq '. | map({question: .question, answer: .answer, model: "rnn"})' rnn_final_adversarial_questions.json > rnn_adv.json
jq '.' es_adv.json rnn_adv.json > trick.expo.json
