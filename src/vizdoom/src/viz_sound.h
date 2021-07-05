

#ifndef VIZDOOM_VIZ_SOUND_H
#define VIZDOOM_VIZ_SOUND_H

#include <cstddef>

#include "basictypes.h"
#include "s_sound.h"

// this will allow for up to N consecutive audio tics to be extracted
// increase this is you want frameskip > 4
#define MAX_SOUND_FRAMES_TO_STORE   4

int VIZ_SoundSamplesPerTic();
int VIZ_SoundSizePerTicBytes();
int VIZ_SoundBufferSizeBytes();

#endif //VIZDOOM_VIZ_SOUND_H