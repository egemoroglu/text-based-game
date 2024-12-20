TEST BASED GAME WITH NATURAL LANGUAGE PROCESSING

This is a graduation project in Gazi University Technology Faculty Computer Engineering Department. In this procject, an 
adventure game is created where players can command in text. Basically, the game understands the user input and generates 
a story. There is an event pool for the story telling and at the beginning of the game a random story is picked and GPT generates
a story. Then, the story continues based on the player input.

TECHNOLOSIES:

1. GPT API for creating dynamic story: GPT is used  to create dynamic story and every time player plays the game, the story telling is changed. Although the story is the same, story telling changes from game to game.

2. Natural Language Processing: Different NLP algorithms and technologies are implemented and test during development.

3. AWS Sagemaker: To perform the machine learning sagemaker service used. Once the NLP algorithm is implemented and tested in local machine with small sized datasets, the NLP algorithm deployed to sagemaker and tested with large datasets.

NLP ALGORITHMS AND MODELS

1. Bert Model: At the beginning of the project, pretrained bert-base-uncased model used. However, due lack of resources this bert model was too big to train or fine tune.

2. Distibert Model: Once the bert model was to large and could not be train within the resources available, a smaller model was implemented that was pretrained with 66 million dataset, distilbert-base-uncased. This model fine tuned with sentiment140 dataset where there are 1.6 million tweets. 

3. Support Vector Machine (SVM): To examine different algorithms and their accuracy, SVM was implemented and trained during development proccess. Unfortunately, the SVM was not accurate enough be used.

4. 1D Convolutional Neural Network (CNN 1D): CNN 1D algorithm was also implemented and tested during development and the results were compared against the original implementation and SVM results. 

Recommadations: If you are training the model in local machines and you do not have enough resources like GPU, try using libraries like linearSVC that only uses CPU for training purposes. If you have enough GPU or working on a cloud provider like Google cloud or AWS sagemaker, consider using libraries like SDGClassifier to utilize the GPU for faster training.