## XPersona Dataset
XPersona dataset is an extension of the persona-chat [dataset](https://arxiv.org/pdf/2003.07568.pdf)(https://www.aclweb.org/anthology/P18-1205/).  Specifically, we extend the [ConvAI2](http://convai.io) to other six languages: Chinese, French, Indonesian, Italian, Korean, and Japanese. Since the test set of ConvAI2 is hidden, we split the original validation set into a new validation set and test sets.

## Dataset Format
The data is a list of dialogues formated as following:
```bash
data = [dialogue1, dialogue2, dialogue3...]
```
```bash
dialogue = {"persona":[sentence1, sentence2...], "dialogue": [[user_utterence1, response1], [user_utterence2, response2]...]}
```