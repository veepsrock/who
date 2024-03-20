# MLFlow Experiments
The original taxonomy we had in production had 18 child labels, which aggregated up to 5 parent themes. Please refer to [this report for reference](https://datarockfound.atlassian.net/wiki/spaces/~115018218/pages/1705443335/WHO+Theme+Classification+Report) and a high level overview of performance and decisions made.

## Taxonomy Experiment
The objective of this experiment was to test the performance of different taxonomy types. We tested different versions of a simplified taxonomy to see how it compares to predicting at the child level vs. the parent level. The only experiment variable is the taxonomy type. 

### Results
Through this experimentation, we discovered that the original production taxonomy outperformed predicting the themes at the parent level. We did make changes to breakout vaccines as its own category.

Here are the parameters. The results of each run can be found [here](taxonomy_experiment_runs.csv)
- simplified: simplified taxonomy
```
theme_dict=
{
    "conspiracy and corruption": ["bioweapon", "conspiracy", "corruption", "media-bias", "medical-exploitation"],
    "illness and stigma": ["stigmatization", "case-reporting", "symptoms-severity", "variants"],
    "intervention and capacity":["capacity"],
    "alternative treatment and prevention": ["alternative-cures", "religious-practices"],
    "treatment and medicine": ["prevention-collective", "prevention-individual", "treatment","vaccine-efficacy", "vaccine-side-effects"]
}
```
- simplified_vax_and: simplified taxonomy + breaking out vaccines as its own category, using "and" 
```
theme_dict_vax=
{
    "conspiracy and corruption": ["bioweapon", "conspiracy", "corruption", "media-bias", "medical-exploitation"],
    "illness and stigma": ["stigmatization", "case-reporting", "symptoms-severity", "variants"],
    "intervention and capacity":["capacity"],
    "alternative treatment and prevention": ["alternative-cures", "religious-practices"],
    "treatment and medicine": ["prevention-collective", "prevention-individual", "treatment"],
    "vaccines": ["vaccine-efficacy", "vaccine-side-effects"]
}
```
- simplified_vax_new: simplified taxonomy, using "-" instead of "and"
```
theme_dict = {
    "conspiracy-corruption": ["bioweapon, conspiracy, corruption, media-bias, medical-exploitation"],
    "illness-cause": ["stigmatization, case-reporting, symptoms-severity, variants"],
    "intervention-capacity":["public health capacity"],
    "prevention-treatment-alternative": ["alternative cures, religious practices"],
    "prevention-treatment-approved": ["prevention, collective prevention, individual treatment, vaccine efficacy, vaccine side effects"]
}
```
- child_labels: using original production taxonomy with child labels
```
child_labels = ["alternative cures, religious practices", "bioweapon, conspiracy, corruption, media-bias, medical-exploitation", "stigmatization, case-reporting, symptoms-severity, variants", "public health capacity", "prevention, collective prevention, individual treatment", "vaccine efficacy, vaccine side effects"]

theme_dict = {
    "conspiracy-corruption": ["bioweapon, conspiracy, corruption, media-bias, medical-exploitation"],
    "illness-cause": ["stigmatization, case-reporting, symptoms-severity, variants"],
    "intervention-capacity":["public health capacity"],
    "prevention-treatment-alternative": ["alternative cures, religious practices"],
    "prevention-treatment-approved": ["prevention, collective prevention, individual treatment, vaccine efficacy, vaccine side effects"]
}
```
- child_labels_vax: using original production taxonomy with child labels, breaking out vaccines as its own category
```
child_labels_vax = ["alternative cures, religious practices", "bioweapon, conspiracy, corruption, media-bias, medical-exploitation", "stigmatization, case-reporting, symptoms-severity, variants", "public health capacity", "prevention, collective prevention, individual treatment", "vaccine efficacy, vaccine side effects"]

theme_dict_vax = {
    "conspiracy-corruption": ["bioweapon, conspiracy, corruption, media-bias, medical-exploitation"],
    "illness-cause": ["stigmatization, case-reporting, symptoms-severity, variants"],
    "intervention-capacity":["public health capacity"],
    "prevention-treatment-alternative": ["alternative cures, religious practices"],
    "prevention-treatment-approved": ["prevention, collective prevention, individual treatment"],
    "vaccines": ["vaccine efficacy, vaccine side effects"]
}
```
- new_key_words: adding more key words to original child labels
```
key_words = ["herbal remedies, home remedies, healers, religious practices, religious beliefs", "conspiracy, corruption, bioweapon, exploitation, exhortion, profiteering, fake news, bias", "case reporting, disease symptoms, disease transmission, disease severity, disease lethality, disease variants, stigmatization","hospital capacity, government capacity, interventions restrictions, lockdowns", "disease treatment, medicine, vaccines, vaccine safety, vaccine side effects, vaccine efficacy"]

theme_dict_key = {
    "prevention-treatment-alternative": "herbal remedies, home remedies, healers, religious practices, religious beliefs",
    "conspiracy-corruption": "conspiracy, corruption, bioweapon, exploitation, exhortion, profiteering, fake news, bias",
    "illness-cause": "case reporting, disease symptoms, disease transmission, disease severity, disease lethality, disease variants, stigmatization",
    "intervention-capacity":"hospital capacity, government capacity, interventions restrictions, lockdowns",
    "prevention-treatment-approved": "disease treatment, medicine, vaccines, vaccine safety, vaccine side effects, vaccine efficacy"
}
```
- new_key_words_vax: adding more key words to original child labels, and making vaccine its own category
```
key_words_vax = ["herbal remedies, home remedies, healers, religious practices, religious beliefs", "conspiracy, corruption, bioweapon, exploitation, exhortion, profiteering, fake news, bias", "case reporting, disease symptoms, disease transmission, disease severity, disease lethality, disease variants, stigmatization","hospital capacity, government capacity, interventions restrictions, lockdowns", "disease treatment, medicine", "vaccines, vaccine safety, vaccine side effects, vaccine efficacy"]

theme_dict_key_vax = {
    "prevention-treatment-alternative": "herbal remedies, home remedies, healers, religious practices, religious beliefs",
    "conspiracy-corruption": "conspiracy, corruption, bioweapon, exploitation, exhortion, profiteering, fake news, bias",
    "illness-cause": "case reporting, disease symptoms, disease transmission, disease severity, disease lethality, disease variants, stigmatization",
    "intervention-capacity":"hospital capacity, government capacity, interventions restrictions, lockdowns",
    "prevention-treatment-approved": "disease treatment, medicine",
    "vaccines": "vaccines, vaccine safety, vaccine side effects, vaccine efficacy"
}
```
- nlp_words: using a natural language sentence to describe the themes
```
theme_dict_nlp = {
    "prevention-treatment-alternative": "Discussion of herbal remedies, alternative cures, religious practices, healers, and other home cures for disease prevention and treatment",
    "conspiracy-corruption": "Discussion of conspiracy, corruption, exploitation, profiteering, fake news, bias and other expressions of distrust in public health",
    "illness-cause": "Discussion of disease outbreaks, symptions, severity, tranmissions, variants and stigmatization",
    "intervention-capacity":"Discussion of hospital capacity, government capacity, interventions, restrictions, lockdowns, and general ability of the public health system to meet needs",
    "prevention-treatment-approved": "Discussion of medical treatment for diseases including medicine and procedures proscribed by a medical professional. Discussion of vaccines including vaccine safet, side effects, and efficacy"
}
```
## Zero Shot Threshold Experiment
The objective of this experiment was to test the performance of different thresholds for confidence levels. The original taxonomy we had a 0.8 confidence threshold. The two experiment variables are the taxonomy type (parent or child), and threshold (0.8 and above)

### Results
Through this experimentation, we discovered that while precision may increase with a higher threshold, recall declines affecting the overall F1 scores. We decided to keep the threshold at the original 0.8 level.

Here are the parameters. The results of each run can be found [here](zero_shot_threshold.csv)
- zero_shot_parent: model predicting at the parent level themes
- zero_shot_child: model predicting at the child level themes
- threshold: 0.8 and above


## Zero Shot Scores Experiment
The objective of this experiment was to test the performance of the number of labels used to predict. The original taxonomy we had a 0.8 confidence threshold, and generated the parent themes using all the labels. The two experiment variables are the taxonomy type (parent or child), and number of labels (all or an entered number)

### Results
Through this experimentation, we discovered that the model performed best with taking the top two labels.

Here are the parameters. The results of each run can be found [here](zero_shot_threshold.csv)
- zero_shot_parent: model predicting at the parent level themes
- zero_shot_child: model predicting at the child level themes
- number_of_levels: all or 2