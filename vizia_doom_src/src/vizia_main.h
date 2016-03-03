#ifndef __VIZIA_MAIN_H__
#define __VIZIA_MAIN_H__

extern int vizia_time;

void Vizia_Init();

void Vizia_AsyncStartTic();

void Vizia_Tic();

void Vizia_Update();

bool Vizia_IsPaused();

void Vizia_Close();

#endif
