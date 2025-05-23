# Viualization Interface that enables interactive Reinforcement Learning from Human Feedback
A visualization tool for exploring the behavior space and providing preferences for RLHF.


## INSTALLATION
### Backend
Build docker image from Dockerfile
```
$ docker build -t interactive-rlhf .
```

Create container from the docker image named interactive-rlhf
```
$ docker run -it -p 5000:5000 --name interactive-rlhf interactive-rlhf
```


For later uses when the container is already created:
```
$ docker start interactive-rlhf
```

Run this command in the terminal to get a bash sessions for the container.
```
$ docker exec -it interactive-rlhf /bin/bash
```
In the container execute this:
```
$ xvfb-run -s "-screen 0 1400x900x24" python3 pairwise_comparison.py
```
or this for the groupwise comparison:
```
$ xvfb-run -s "-screen 0 1400x900x24" python3 pairwise_group_comparison.py
```

### Frontend
Run this command in another terminal:
```
$ git clone https://github.com/jankomp/interactive_rlhf_gui.git
$ cd interactive_rlhf_gui
$ yarn install
$ yarn serve
```