# Detection and Classification of Acoustic Scenes and Acoustic Events
This project proposes the study of neural networks combined with different signal processing and deep learning methods to evaluate their performance when detecting and classifying both sound scenes and events. Evaluations of the outlined techniques were conducted on two DCASE2019 datasets. After a deep experimentation process, results show that sound classification improves up to 7\% in Micro-AUPRC when combining them together, along with the designed neural network. 

## Requirements
- Docker
- Nvidia-docker
- docker-compose

## Set up
```
Clone repo

make run (first time it can take a while)

make dev
```

## How to use it

`cd code/configs/`

Add the desired config files with the parameters that you want to try.

`cd ..`

In the Makefile add a name to your batch of trainings.

`make _nametraining_`

Training of the different experiments will start automatically.

## Show logs (tensorboard)
```
docker exec -it tb bash

If you want to share a link so that everyone can check your experiments you should uninstall tensorboard from pip and reinstall it. And then execute:

tensorboard dev upload --logdir yourpath

You'll be given a link to observe your logs

If you don't care just type:

tensorboard --logdir yourpath

And then go to localhost:6006 in your pc. Make sure to add the ports mapping in the docker-compose otherwise you won't see anything.
```
