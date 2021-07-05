#include "sounddef.h"

#include "viz_sound.h"
#include "viz_labels.h"


EXTERN_CVAR (Int, samp_fre)


int VIZ_SoundSamplesPerTic() {
    return *samp_fre / TICRATE;
}

int VIZ_SoundSizePerTicBytes() {
    return SOUND_NUM_CHANNELS * sizeof(short) * VIZ_SoundSamplesPerTic();
}

int VIZ_SoundBufferSizeBytes() {
    return VIZ_SoundSizePerTicBytes() * MAX_SOUND_FRAMES_TO_STORE;
}