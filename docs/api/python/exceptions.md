# Exceptions

ViZDoom defines several exceptions that can be thrown by its API.
Most of the exceptions contain more information in the exception's message.

```{eval-rst}
.. autoexception:: vizdoom.FileDoesNotExistException
```
Means that file specified as part of a configuration does not exist.

```{eval-rst}
.. autoexception:: vizdoom.MessageQueueException
```

Means that communication with ViZDoom's instance failed. Usually, means a problem with permissions or system configuration.

```{eval-rst}
.. autoexception:: vizdoom.SharedMemoryException
```

Means that allocation/reading of shared memory failed. Usually, means a problem with permissions or system configuration.

```{eval-rst}
.. autoexception:: vizdoom.ViZDoomErrorException
```

Means that an error in the ViZDoom engine occurred.

```{eval-rst}
.. autoexception:: vizdoom.ViZDoomIsNotRunningException
```

Means that called method cannot be used when ViZDoom instance is not running.

```{eval-rst}
.. autoexception:: vizdoom.ViZDoomUnexpectedExitException
```

Means that ViZDoom's instance was closed/terminated/killed from the outside.

```{eval-rst}
.. autoexception:: vizdoom.ViZDoomNoOpenALSoundException
```

Means that sound was requested but could not be initialized. Usually means that no sound device is available or that the sound backend is not properly configured. Added in 1.3.0.
