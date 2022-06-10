# Transformers


## BERT

To get you warmed up and familiar with some of the libararies, we start out easy with a BERT tutorial from J. Alammar. 
The tutorial builds a simple sentiment analysis model based on pretrained BERT models with the [HuggingFace Library](huggingface.co/). 
It will get you familiarized with the libary and make the next exercise a bit easier. 
The tutorial has nice graphics and visualizations and will increase your general understanding of transformers and especially the BERT model even more. 

[Link to the tutorial](https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)
## wav2vec 2.0 for keyword recognition
This exercise is a bit more advanced and you will be mostly on your own.
There are a couple of options you will have to think about and decide which implementation path you want to follow.

For this exercise please use the [speech-commands-dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) from google.
You can also use the [HuggingFace api](https://huggingface.co/datasets/speech_commands) to get the data or use [torchaudio](https://pytorch.org/audio/stable/_modules/torchaudio/datasets/speechcommands.html).

There are a couple of options, that will lead to differnt performance on this problem. They vary in complexity as well as performance. 

Choose one the options that suits you best or the one that you think might yield the best performance.
1. What model will you use? ```BASE vs. LARGE``` and what pretrained weights ```ASR vs BASE```, ```XLSR53 vs ENGLISH```?
1. HuggingFace or ```torchaudio.pipelines```?
1. Use a simple neural classification head?
3. Extract features and use them with some downstream classifier (e.g. SVM, Naive Bayes etc.)
    1. what pooling strategy will you use? (mean/ statistical ...)
    2. convolutional head?
    3. RNN?
    3. Should you use a dimeninsionality reduction method?
1. Or use CTC loss and a greedy decoder? (closed vocab!)


What implementation do you think would work best in a real-world scenario?