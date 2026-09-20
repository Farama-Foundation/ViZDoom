// Wraps an SDL mutex object. (A critical section is a Windows synchronization
// object similar to a mutex but optimized for access by threads belonging to
// only one process, hence the class name.)

#ifndef CRITSEC_H
#define CRITSEC_H

//VIZDOOM_CODE
#include <SDL3/SDL.h>
#include "i_system.h"

class FCriticalSection
{
public:
	FCriticalSection()
	{
		CritSec = SDL_CreateMutex();
		if (CritSec == NULL)
		{
			I_FatalError("Failed to create a critical section mutex.");
		}
	}
	~FCriticalSection()
	{
		if (CritSec != NULL)
		{
			SDL_DestroyMutex(CritSec);
		}
	}
	//VIZDOOM_CODE
	void Enter()
	{
		SDL_LockMutex(CritSec);
	}
	//VIZDOOM_CODE
	void Leave()
	{
		SDL_UnlockMutex(CritSec);
	}
private:
	SDL_Mutex *CritSec; //VIZDOOM_CODE
};

#endif
