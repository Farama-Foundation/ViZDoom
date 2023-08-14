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
.. autoexception:: vizdoom.SignalException
```

Means that a signal was cached by ViZDoom's instance.

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
