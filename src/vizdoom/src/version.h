/*
** version.h
**
**---------------------------------------------------------------------------
** Copyright 1998-2007 Randy Heit
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

#ifndef __VERSION_H__
#define __VERSION_H__

#include "viz_version.h"

const char *GetGitDescription();
const char *GetGitHash();
const char *GetGitTime();
const char *GetVersionString();

/** Lots of different version numbers **/
//VIZDOOM_CODE
#define VERSIONSTR VIZ_VERSION_STR " (ZDOOM 2.8.1+)"

// The version as seen in the Windows resource
#define RC_FILEVERSION 2,8,1,0
#define RC_PRODUCTVERSION 2,8,1,0
#define RC_PRODUCTVERSION2 "2.8.1+"

// Version identifier for network games.
// Bump it every time you do a release unless you're certain you
// didn't change anything that will affect sync.
#define NETGAMEVERSION 231

// Version stored in the ini's [LastRun] section.
// Bump it if you made some configuration change that you want to
// be able to migrate in FGameConfigFile::DoGlobalSetup().
#define LASTRUNVERSION "211"

// Protocol version used in demos.
// Bump it if you change existing DEM_ commands or add new ones.
// Otherwise, it should be safe to leave it alone.
#define DEMOGAMEVERSION 0x21D

// Minimum demo version we can play.
// Bump it whenever you change or remove existing DEM_ commands.
#define MINDEMOVERSION 0x21C

// SAVEVER is the version of the information stored in level snapshots.
// Note that SAVEVER is not directly comparable to VERSION.
// SAVESIG should match SAVEVER.

// MINSAVEVER is the minimum level snapshot version that can be loaded.
#define MINSAVEVER	3100

// Use 4500 as the base git save version, since it's higher than the
// SVN revision ever got.
#define SAVEVER 4531

#define SAVEVERSTRINGIFY2(x) #x
#define SAVEVERSTRINGIFY(x) SAVEVERSTRINGIFY2(x)
#define SAVESIG "ZDOOMSAVE" SAVEVERSTRINGIFY(SAVEVER)

//VIZDOOM_CODE
// This is so that derivates can use the same savegame versions without worrying about engine compatibility
#define GAMESIG "VIZDOOM"
#define BASEWAD "vizdoom.pk3"

//VIZDOOM_CODE
// More stuff that needs to be different for derivatives.
#define GAMENAME "ViZDoom"
#define GAMENAMELOWERCASE "vizdoom"
#define FORUM_URL "https://groups.google.com/forum/#!forum/vizdoom"
#define BUGS_FORUM_URL	"https://github.com/mwydmuch/ViZDoom/issues"

//VIZDOOM_CODE
//#if defined(__APPLE__) || defined(_WIN32)
//#define GAME_DIR GAMENAME
//#else
//#define GAME_DIR ".config/" GAMENAMELOWERCASE
//#endif
#define GAME_DIR "_" GAMENAMELOWERCASE

// The maximum length of one save game description for the menus.
#define SAVESTRINGSIZE		24

#endif //__VERSION_H__
