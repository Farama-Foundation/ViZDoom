Provided dockerfiles (all images use ubuntu 16.04):

* **python3** - python3, vizdoom with dependecies and X11 support. By default runs [basic example](examples/python/basic.py).
* **tensorflow** - python3, vizdoom with dependecies, tensorflow and X11 support. By default runs [tensorflow learning example](examples/python/learning_tensorflow.py).
* **torch** - TODO

# Building
```bash
# Uses Dockerfile in python3 directory to create image tagged vizdoom
docker build python3 -t vizdoom 


# Uses Dockerfile in tensorflow directory to create image tagged vizdoom_tf
docker build tensorflow -t vizdoom_tf 

```
# Running (requires building beforehand)

```
# Run basic example in container named 'basic' ('--net=host -e DISPLAY=${DISPLAY'} is needed for X11) 
docker run  -t --net=host -e DISPLAY=${DISPLAY} --rm --name basic vizdoom

# Run tensorflow learning example in container named 'vizdoom_tf' ('--net=host -e DISPLAY=${DISPLAY'} is needed for X11)
docker run  -t --net=host -e DISPLAY=${DISPLAY} --rm --name vizdoom_tf vizdoom_tf

```
