# Transformers

Transformers are powerful attention-based models that have great power for generalization.
Googles T5 is a popular and very recent Text-to-Text transformer.
A cool blog post about it can be found on Googles [AI.blog](!https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html).
T5 is supposedly able to handle all kinds of NLP tasks like summarization, question answering, and text classification.
The [paper](!https://arxiv.org/abs/1910.10683) sheds some light on what they actually wanted to achieve.

In our exercise we want to answer the question if we can use a pre-trained transformer model to summarize abstracts of our theses so they come close to their original titles?

We will try this task two ways.
Both should summarize the abstract to a thesis title.
First of all you will need to download a German T5 model from this [link](!https://huggingface.co/ml6team/mt5-small-german-finetune-mlsum)

## Task 1: Out of the box
Let's see if we can put the to use in our thesis example from the previous exercises.
Follow the [instructions](!https://huggingface.co/ozcangundes/mt5-small-turkish-summarization) for a Turkish summarization model
on [Hugging Face](!https://huggingface.co/). The steps should work the same for German.
Implement some summarization and check qualitatively if you're content with the results.
**A new theses.tsv file** will be provided in the **/res** folder.

## Task 2: Fine tune it
The big strength of those huge transformer model is that they can be fine tuned on specific tasks. 
So lets try that for our ***abstract to thesis title*** summarization. 

Implement transfer learning of the model you used in the previous task. 
and fine tune the model on our dataset.(input abstracts -> output titles) 
After fine tuning try out the summarization and check qualitatively if you're content with the results.

### Hint!
* Be mindful to clean the data! 
* Remove non German theses titles. 
* lower case stuff
* tokenize properly
* remove very short titles, you can evaluate by a metric, e.g. len(abstract_words) / len(thesis title words)

