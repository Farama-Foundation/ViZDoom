#ifndef __VIZDOOM_SCREEN_H__
#define __VIZDOOM_SCREEN_H__

#include <stddef.h>

extern unsigned int vizdoomScreenWidth;
extern unsigned int vizdoomScreenHeight;
extern size_t vizdoomScreenPitch;
extern size_t vizdoomScreenSize;

void ViZDoom_ScreenInit();

void ViZDoom_ScreenUpdate();

void ViZDoom_ScreenClose();

#endif
