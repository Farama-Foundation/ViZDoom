> For the examples to work properly you need to install ViZDoom [system-wide with pip](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md) or create a link to dir with ViZDoom pip_package or set doom_game_path and vizdoom_path manually or in config files.

# Examples

## [audio_buffer.py](https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/audio_buffer.py)
Demonstrates how to use the audio buffer.

## [basic.py](https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/basic.py)
Demonstrates how to use the most basic features of the environment. It configures the engine, and makes the agent perform random actions. It also prints the current state and the reward earned with every action.

## [buffers.py](https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/buffers.py)
Demonstrates all the visual buffers (screen buffer, depth buffer, labels buffer, map buffer) available in ViZDoom environment.

## [cig_multiplayer.py](https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/cig_multiplayer.py), [cig_multiplayer_host.py](https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/cig_multiplayer_host.py)
Demonstrates how to configure and play the multiplayer game for CIG competition.

## [cig_multiplayer_bots.py](https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/cig_multiplayer_bots.py)
Demonstrates how to play with bots to simulate the multiplayer game. Helpful for developing an AI agent for CIG competition.

## [cig_singleplayer.py](https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/cig_singleplayer.py), [cig_multiplayer_host.py](https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/cig_singleplayer_host.py)
Demonstrates how to configure and play the singleplayer game for CIG competition.

## [delta_buttons.py](https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/delta_buttons.py)
Shows how delta buttons work (they may take values other than 0 and 1 and can be used for precise movement).

## [format.py](https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/format.py)
Presents different formats of the screen buffer. [OpenCV](http://opencv.org/) is used to display the images.

## [fps.py](https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/fps.py)
Tests the performance of the environment in frames per second. It should give you some idea how fast the framework works on your hardware.

## [labels.py](https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/labels.py)
Contains an example of how to implement basic Q-learning on the interface within PyTorch.

## [learning_pytorch.py](https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/learning_pytorch.py), [learning_pytorch_test.py](https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/learning_pytorch_test.py)
Contains an example of how to implement basic Q-learning on the interface within PyTorch.

## [learning_theano.py](https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/learning_theano.py)
Contains an example of how to implement basic Q-learning on the interface within Theano.

## [learning_tensorflow.py](https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/learning_tensorflow.py)
Contains an example of how to implement basic Q-learning on the interface within Tensorflow.

## [multiple_instances.py](https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/multiple_instances.py), [multiple_instances_advance.py](https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/multiple_instances_advance.py)
Contains an example of how to create a "server" and have multiple agents playing on the server at once. Combine with bots.py and learning_x.py to train agents against some AI

## [objects_and_sectors.py](https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/multiple_instances.py)
Demonstrates how to access list of actors/objects and map geometry of the scenario.

## [pyoblige.py](https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/pyoblige.py) (level generator)
Contains an example of how to combine [PyOblige](https://github.com/mwydmuch/PyOblige) with ViZDoom.

## [record_episodes.py](https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/record_episodes.py)
Demonstrates how to record an episode and replay it.

## [record_multiplayer.py](https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/record_multiplayer.py)
Demonstrates how to record the multiplayer game and replay it.

## [scenarios.py](https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/scenarios.py)
Presents different scenarios that come with ViZDoom environment.

## [seed.py](https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/seed.py)
Shows how to run deterministic episodes by setting the seed. After setting the seed every episode will look the same (if the agent behaves deterministically).

## [shaping.py](https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/shaping.py)
Demonstrates how to make use of the game variables to implement [shaping](https://en.wikipedia.org/wiki/Shaping_(psychology)) using health_guided.wad scenario.

## [spectator.py](https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/spectator.py)
Shows how to use the *SPECTATOR* mode in which YOU play Doom and AI is the spectator (intended for apprenticeship learning).

## [ticrate.py](https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/ticrate.py)
Demonstrates how to use different ticrates in ASYNC mode.
