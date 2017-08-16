Provided dockerfiles (all images use ubuntu 16.04):

* **python3** - python3, vizdoom with dependecies and X11 support. By default runs python [basic example](examples/python/basic.py).
* **tensorflow** - python3, vizdoom with dependecies, tensorflow and X11 support. By default runs [tensorflow learning example](examples/python/learning_tensorflow.py).
* **torch** - torch, vizdoom with dependencies and X11 support. By default run lua [basic example](examples/lua/basic.py). TODO

# Building
```bash
# Uses Dockerfile in python3 directory to create image tagged vizdoom
docker build python3 -t vizdoom 


# Uses Dockerfile in tensorflow directory to create image tagged vizdoom_tf
docker build tensorflow -t vizdoom_tf 

```
# Running (requires building beforehand)

```
# '--net=host -e DISPLAY=${DISPLAY} --privileged' required for GUI
# Run basic example in container named 'basic' 
docker run  -ti --net=host -e DISPLAY=${DISPLAY} --privileged --rm --name basic vizdoom

# Run tensorflow learning example in container named 'vizdoom_tf'
nvidia-docker run  -ti --privileged --net=host -e DISPLAY=${DISPLAY} --rm --name vizdoom_tf vizdoom_tf

# Run bash in interactive mode
docker run  -ti --privileged --net=host -e DISPLAY=${DISPLAY} --rm --name basic vizdoom bash

```
