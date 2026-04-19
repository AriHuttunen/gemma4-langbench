# Belebele Evaluation Summary


4 models × 4 languages × 900 questions = 3,600 prompts per model.


## Accuracy by Model and Language

| Model | eng_Latn | est_Latn | fin_Latn | swe_Latn |
|-------|----------|----------|----------|----------|
| anthropic_claude-haiku-4.5 | 82.4% | 70.8% | 76.1% | 81.2% |
| anthropic_claude-sonnet-4.6 | 95.1% | 90.1% | 92.0% | 89.4% |
| google_gemma-4-26b-a4b-it | 96.7% | 89.1% | 92.6% | 92.6% |
| local_Gemma-4_E4B_Q8_0_think | 96.0% | 87.9% | 92.2% | 92.6% |

## Top 20 Hardest Questions (across all models × languages)

n_wrong_total is out of 16 (4 models × 4 languages).

| # | q_no | n_wrong | eng | est | fin | swe | Question | Link |
|---|------|---------|-----|-----|-----|-----|----------|------|
| 1 | 2 | 16 | 4 | 4 | 4 | 4 | How long have humans been magnifying objects using lenses? | [link](https://en.wikibooks.org/wiki/High_School_Earth_Science/Telescopes) |
| 2 | 1 | 16 | 4 | 4 | 4 | 4 | What can be found at The Giza Plateau? | [link](https://en.wikibooks.org/wiki/The_Seven_Wonders_of_the_World/The_Great_Pyramid) |
| 3 | 2 | 16 | 4 | 4 | 4 | 4 | Based on the information given in the passage, what was not mentioned  | [link](https://en.wikinews.org/wiki/Beverly_Hall,_indicted_public_school_superintendent,_dies_aged_68) |
| 4 | 1 | 16 | 4 | 4 | 4 | 4 | While visiting the location described in the passage, which of the fol | [link](https://en.wikivoyage.org/wiki/Auschwitz-Birkenau) |
| 5 | 2 | 16 | 4 | 4 | 4 | 4 | Which statement does not accurately describe auxiliary languages? | [link](https://en.wikivoyage.org/wiki/Auxiliary_languages) |
| 6 | 1 | 16 | 4 | 4 | 4 | 4 | According to the passage, which scenario would be ideal for a traveler | [link](https://en.wikivoyage.org/wiki/Thimphu) |
| 7 | 2 | 16 | 4 | 4 | 4 | 4 | Which of the following facts about Timbuktu is true? | [link](https://en.wikivoyage.org/wiki/Timbuktu) |
| 8 | 1 | 15 | 4 | 3 | 4 | 4 | According to the passage, what was iron used for first? | [link](https://en.wikibooks.org/wiki/History_of_Rail_Transport/Early_Rail_Transportation) |
| 9 | 1 | 15 | 3 | 4 | 4 | 4 | Which of the following is not an accurate fact about the Hangeul alpha | [link](https://en.wikibooks.org/wiki/Wikijunior:Languages/Korean) |
| 10 | 2 | 15 | 4 | 4 | 3 | 4 | Which of the following is not a reason for the multitude of air accide | [link](https://en.wikinews.org/wiki/Iranian_passenger_jet%27s_wheel_catches_fire) |
| 11 | 1 | 15 | 3 | 4 | 4 | 4 | When did Alonso end his race? | [link](https://en.wikinews.org/wiki/Jenson_Button_wins_2006_Hungarian_Grand_Prix) |
| 12 | 2 | 15 | 4 | 4 | 4 | 3 | According the passage, where are swimming conditions likely to be the  | [link](https://en.wikivoyage.org/wiki/Auckland) |
| 13 | 2 | 15 | 4 | 4 | 4 | 3 | According to the passage, what might help backcountry skiers who want  | [link](https://en.wikivoyage.org/wiki/Cross_country_skiing) |
| 14 | 2 | 13 | 4 | 2 | 4 | 3 | According to the passage, which of the following aspects of asynchrono | [link](https://en.wikibooks.org/wiki/Blended_Learning_in_K-12/Synchronous_and_asynchronous_communication_methods) |
| 15 | 1 | 13 | 2 | 3 | 4 | 4 | Which of the following was not cited by Albert Muchanga as something t | [link](https://en.wikinews.org/wiki/Benin,_Nigeria_join_African_Union_continental_free_trade_bloc) |
| 16 | 1 | 13 | 3 | 3 | 3 | 4 | Based on the passage, which of the following should a price-conscious  | [link](https://en.wikivoyage.org/wiki/Aggregators) |
| 17 | 2 | 12 | 3 | 3 | 3 | 3 | Based on the information in the passage, when might a zoom lens be pre | [link](https://en.wikibooks.org/wiki/Modern_Photography/Lenses) |
| 18 | 1 | 12 | 3 | 3 | 2 | 4 | According to the passage, what quality of the Giza pyramids wasn’t lik | [link](https://en.wikibooks.org/wiki/Wikijunior:World_Heritage_Sites/Pyramids_of_Giza) |
| 19 | 1 | 12 | 3 | 4 | 4 | 1 | According to the passage, what information is known following the bomb | [link](https://en.wikinews.org/wiki/Bomb_blasts_kill_several_in_Iran) |
| 20 | 2 | 12 | 4 | 3 | 4 | 1 | According to Versace, when did the strongest winds begin? | [link](https://en.wikinews.org/wiki/Hungary%27s_St_Stephen%27s_Day_hit_by_storm:_3_left_dead) |

## Language Flip: English-correct → Non-English wrong

Questions where the model answered English correctly but failed in ≥1 of est/fin/swe.

| Model | Flip count |
|-------|------------|
| anthropic_claude-haiku-4.5 | 335 |
| anthropic_claude-sonnet-4.6 | 118 |
| google_gemma-4-26b-a4b-it | 120 |
| local_Gemma-4_E4B_Q8_0_think | 133 |

## Model Disagreement (unique outlier per language)

Questions where exactly 1 or 3 of 4 models got it right — one model stands out.

| Language | Disagreement count |
|----------|--------------------|
| eng_Latn | 154 |
| est_Latn | 265 |
| fin_Latn | 198 |
| swe_Latn | 142 |

## Note: Unparseable Responses

Counted as wrong in all accuracy figures above (matches eval_state.json).

| Model | Unparseable count |
|-------|-------------------|
| anthropic_claude-haiku-4.5 | 695 |
| anthropic_claude-sonnet-4.6 | 194 |
| google_gemma-4-26b-a4b-it | 40 |
| local_Gemma-4_E4B_Q8_0_think | 4 |

## Output Files

- `accuracy.csv` — per-model per-language accuracy (verified against eval_state)
- `hardest_questions.csv` — all 900 questions sorted by failure count across models×langs
- `language_flip.csv` — questions eng-correct but non-eng-wrong, per model
- `model_disagreement.csv` — per-question per-language which models got it right/wrong
