# Exceptions

ViZDoom defines several exceptions that can be thrown by its API:
Most of the exceptions contain more information in "what()" message.
The declarations of all the enums can be found in the `include/ViZDoomExceptions.h` header file.


* `FileDoesNotExistException` - means that file specified as part of a configuration does not exist.

* `MessageQueueException/SharedMemoryException` - means that communication with ViZDoom's instance failed. Usually, means a problem with permissions or system configuration.

* `ViZDoomErrorException` - means that an error in the ViZDoom engine occurred.

* `ViZDoomIsNotRunningException` - means that called method cannot be used when ViZDoom instance is not running.

* `ViZDoomUnexpectedExitException` - means that ViZDoom's instance was closed/terminated/killed from the outside.

* `ViZDoomNoOpenALSoundException` - means that sound was requested but could not be initialized. Usually means that no sound device is available or that the sound backend is not properly configured. Added in 1.3.0.
