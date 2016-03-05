#ifndef __VIZDOOM_MAIN_H__
#define __VIZDOOM_MAIN_H__

extern int vizdoom_time;

void ViZDoom_Init();

void ViZDoom_AsyncStartTic();

void ViZDoom_Tic();

void ViZDoom_Update();

bool ViZDoom_IsPaused();

void ViZDoom_Close();

#endif
