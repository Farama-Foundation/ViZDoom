/*
 Copyright (C) 2016 by Wojciech Jaśkowski, Michał Kempka, Grzegorz Runc, Jakub Toczek, Marek Wydmuch

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
*/

#ifndef __VIZ_MAIN_H__
#define __VIZ_MAIN_H__

#include "viz_defines.h"

extern int vizTime;
extern bool vizNextTic;
extern bool vizUpdate;
extern int vizLastUpdate;
extern int vizNodesRecv[VIZ_MAX_PLAYERS];

void VIZ_Init();

void VIZ_AsyncStartTic();

void VIZ_Tic();

void VIZ_Update();

void VIZ_CVARsInit();

void VIZ_CVARsUpdate();

void VIZ_Close();

void VIZ_IgnoreNextDoomError();

void VIZ_DoomError(const char *error);

void VIZ_Error(const char *func, const char *error, ...);

void VIZ_DebugMsg(int level, const char *func, const char *msg, ...);

void VIZ_InterruptionPoint();

void VIZ_Sleep(unsigned int ms);

#endif
