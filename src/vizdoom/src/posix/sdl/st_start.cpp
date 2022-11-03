/*
** st_start.cpp
** Handles the startup screen.
**
**---------------------------------------------------------------------------
** Copyright 2006-2007 Randy Heit
** All rights reserved.
**
** Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions
** are met:
**
** 1. Redistributions of source code must retain the above copyright
**    notice, this list of conditions and the following disclaimer.
** 2. Redistributions in binary form must reproduce the above copyright
**    notice, this list of conditions and the following disclaimer in the
**    documentation and/or other materials provided with the distribution.
** 3. The name of the author may not be used to endorse or promote products
**    derived from this software without specific prior written permission.
**
** THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
** IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
** OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
** IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
** INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
** NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
** THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**---------------------------------------------------------------------------
**
*/

// HEADER FILES ------------------------------------------------------------

#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <termios.h>

#include "st_start.h"
#include "doomdef.h"
#include "i_system.h"
#include "c_cvars.h"

//VIZDOOM_CODE
#include "viz_main.h"
#include "viz_system.h"
EXTERN_CVAR(Int, viz_connect_timeout)


// MACROS ------------------------------------------------------------------

// TYPES -------------------------------------------------------------------

class FTTYStartupScreen : public FStartupScreen
{
	public:
		FTTYStartupScreen(int max_progress);
		~FTTYStartupScreen();

		void Progress();
		void NetInit(const char *message, int num_players);
		void NetProgress(int count);
		void NetMessage(const char *format, ...);	// cover for printf
		void NetDone();
		bool NetLoop(bool (*timer_callback)(void *), void *userdata);
	protected:
		bool DidNetInit;
		int NetMaxPos, NetCurPos;
		const char *TheNetMessage;
		termios OldTermIOS;
};

// EXTERNAL FUNCTION PROTOTYPES --------------------------------------------

// PUBLIC FUNCTION PROTOTYPES ----------------------------------------------

void I_ShutdownJoysticks();

// PRIVATE FUNCTION PROTOTYPES ---------------------------------------------

static void DeleteStartupScreen();

// EXTERNAL DATA DECLARATIONS ----------------------------------------------

// PUBLIC DATA DEFINITIONS -------------------------------------------------

FStartupScreen *StartScreen;

CUSTOM_CVAR(Int, showendoom, 0, CVAR_ARCHIVE|CVAR_GLOBALCONFIG)
{
	if (self < 0) self = 0;
	else if (self > 2) self=2;
}

// PRIVATE DATA DEFINITIONS ------------------------------------------------

static const char SpinnyProgressChars[4] = { '|', '/', '-', '\\' };

// CODE --------------------------------------------------------------------

//==========================================================================
//
// FStartupScreen :: CreateInstance
//
// Initializes the startup screen for the detected game.
// Sets the size of the progress bar and displays the startup screen.
//
//==========================================================================

FStartupScreen *FStartupScreen::CreateInstance(int max_progress)
{
	atterm(DeleteStartupScreen);
	return new FTTYStartupScreen(max_progress);
}

//===========================================================================
//
// DeleteStartupScreen
//
// Makes sure the startup screen has been deleted before quitting.
//
//===========================================================================

void DeleteStartupScreen()
{
	if (StartScreen != NULL)
	{
		delete StartScreen;
		StartScreen = NULL;
	}
}

//===========================================================================
//
// FTTYStartupScreen Constructor
//
// Sets the size of the progress bar and displays the startup screen.
//
//===========================================================================

FTTYStartupScreen::FTTYStartupScreen(int max_progress)
	: FStartupScreen(max_progress)
{
	DidNetInit = false;
	NetMaxPos = 0;
	NetCurPos = 0;
	TheNetMessage = NULL;
}

//===========================================================================
//
// FTTYStartupScreen Destructor
//
// Called just before entering graphics mode to deconstruct the startup
// screen.
//
//===========================================================================

FTTYStartupScreen::~FTTYStartupScreen()
{
	NetDone();	// Just in case it wasn't called yet and needs to be.
}

//===========================================================================
//
// FTTYStartupScreen :: Progress
//
// If there was a progress bar, this would move it. But the basic TTY
// startup screen doesn't have one, so this function does nothing.
//
//===========================================================================

void FTTYStartupScreen::Progress()
{
}

//===========================================================================
//
// FTTYStartupScreen :: NetInit
//
// Sets stdin for unbuffered I/O, displays the given message, and shows
// a progress meter.
//
//===========================================================================

void FTTYStartupScreen::NetInit(const char *message, int numplayers)
{
	if (!DidNetInit)
	{
		termios rawtermios;

		fprintf (stderr, "Press 'Q' to abort network game synchronization.");
        fprintf (stderr, "\nNetwork game synchronization timeout: %ds.", (unsigned int)*viz_connect_timeout);
        // Set stdin to raw mode so we can get keypresses in ST_CheckNetAbort()
		// immediately without waiting for an EOL.
		tcgetattr (STDIN_FILENO, &OldTermIOS);
		rawtermios = OldTermIOS;
		rawtermios.c_lflag &= ~(ICANON | ECHO);
		tcsetattr (STDIN_FILENO, TCSANOW, &rawtermios);
		DidNetInit = true;
	}
	if (numplayers == 1)
	{
		// Status message without any real progress info.
		fprintf (stderr, "\n%s.", message);
	}
	else
	{
		fprintf (stderr, "\n%s: ", message);
	}
	fflush (stderr);
	TheNetMessage = message;
	NetMaxPos = numplayers;
	NetCurPos = 0;
	NetProgress(1);		// You always know about yourself
}

//===========================================================================
//
// FTTYStartupScreen :: NetDone
//
// Restores the old stdin tty settings.
//
//===========================================================================

void FTTYStartupScreen::NetDone()
{
	// Restore stdin settings
	if (DidNetInit)
	{
		tcsetattr (STDIN_FILENO, TCSANOW, &OldTermIOS);
		printf ("\n");
		DidNetInit = false;
	}
}

//===========================================================================
//
// FTTYStartupScreen :: NetMessage
//
// Call this between NetInit() and NetDone() instead of Printf() to
// display messages, because the progress meter is mixed in the same output
// stream as normal messages.
//
//===========================================================================

void FTTYStartupScreen::NetMessage(const char *format, ...)
{
	FString str;
	va_list argptr;
	
	va_start (argptr, format);
	str.VFormat (format, argptr);
	va_end (argptr);
	fprintf (stderr, "\r%-40s\n", str.GetChars());
}

//===========================================================================
//
// FTTYStartupScreen :: NetProgress
//
// Sets the network progress meter. If count is 0, it gets bumped by 1.
// Otherwise, it is set to count.
//
//===========================================================================

void FTTYStartupScreen::NetProgress(int count)
{
	int i;

	if (count == 0)
	{
		NetCurPos++;
	}
	else if (count > 0)
	{
		NetCurPos = count;
	}
	if (NetMaxPos == 0)
	{
		// Spinny-type progress meter, because we're a guest waiting for the host.
		fprintf (stderr, "\r%s: %c", TheNetMessage, SpinnyProgressChars[NetCurPos & 3]);
		fflush (stderr);
	}
	else if (NetMaxPos > 1)
	{
		// Dotty-type progress meter.
		fprintf (stderr, "\r%s: ", TheNetMessage);
		for (i = 0; i < NetCurPos; ++i)
		{
			fputc ('.', stderr);
		}
		fprintf (stderr, "%*c[%2d/%2d]", NetMaxPos + 1 - NetCurPos, ' ', NetCurPos, NetMaxPos);
		fflush (stderr);
	}
}

//===========================================================================
//
// FTTYStartupScreen :: NetLoop
//
// The timer_callback function is called at least two times per second
// and passed the userdata value. It should return true to stop the loop and
// return control to the caller or false to continue the loop.
//
// ST_NetLoop will return true if the loop was halted by the callback and
// false if the loop was halted because the user wants to abort the
// network synchronization.
//
//===========================================================================

bool FTTYStartupScreen::NetLoop(bool (*timer_callback)(void *), void *userdata)
{
	fd_set rfds;
	struct timeval tv;
	int retval;
	char k;

	unsigned int loopEnterTime = I_MSTime();

	for (;;)
	{
        usleep(100 * 1000);
        VIZ_InterruptionPoint();

        if((unsigned int)*viz_connect_timeout * 1000 < I_MSTime() - loopEnterTime) {
            fprintf (stderr, "\nTimeout, network game synchronization aborted.");
            return false;
        }

        if (timer_callback (userdata))		{
            fputc ('Timer callback...\n', stderr);
            return true;
		}
	}
}

void ST_Endoom()
{
	I_ShutdownJoysticks();
	exit(0);
}
