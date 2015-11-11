#include "vizia_shared_memory.h"

#include "doomdef.h"
#include "doomstat.h"
#include "v_video.h"

bip::shared_memory_object *viziaSM;
size_t viziaSMSize;

void Vizia_SMInit(){
    printf("VIZIA SM INIT\n");

    bip::shared_memory_object::remove(VIZIA_SM_NAME);

    viziaSM = new bip::shared_memory_object(bip::open_or_create, VIZIA_SM_NAME, bip::read_write);
    Vizia_SMSetSize(screen->GetWidth(), screen->GetHeight());

    printf ("SM size: %zu\n", viziaSMSize);
}

void Vizia_SMSetSize(int scr_w, int src_h){
    viziaSMSize = sizeof(ViziaInputStruct) + sizeof(ViziaGameVarsStruct) + (sizeof(BYTE) * scr_w * src_h);
    viziaSM->truncate(viziaSMSize);
}

size_t Vizia_SMGetInputRegionBeginning(){
    return 0;
}

size_t Vizia_SMGetGameVarsRegionBeginning(){
    return sizeof(ViziaInputStruct);
}

size_t Vizia_SMGetScreenRegionBeginning(){
    return sizeof(ViziaInputStruct) + sizeof(ViziaGameVarsStruct);
}

void Vizia_SMClose(){
    bip::shared_memory_object::remove(VIZIA_SM_NAME);
}