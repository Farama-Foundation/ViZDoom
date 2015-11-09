#include "vizia_shared_memory.h"

#include "doomdef.h"
#include "doomstat.h"
#include "v_video.h"

shared_memory_object *viziaSM;

void Vizia_SMInit(){
    shared_memory_object::remove(VIZIA_SM_NAME);

    viziaSM = new shared_memory_object(open_or_create, VIZIA_SM_NAME, read_write);
    Vizia_SMSetSize(screen->GetWidth(), screen->GetHeight());
}

int Vizia_SMSetSize(int scr_w, int src_h){
    viziaSM->truncate(sizeof(ViziaInputSMStruct) + sizeof(ViziaGameVarsSMStruct) + sizeof(BYTE) * scr_w * src_h);
}

size_t Vizia_SMGetInputRegionBeginning(){
    return 0;
}

size_t Vizia_SMGetGameVarsRegionBeginning(){
    return sizeof(ViziaInputSMStruct);
}

size_t Vizia_SMGetScreenRegionBeginning(){
    return sizeof(ViziaInputSMStruct) + sizeof(ViziaGameVarsSMStruct);
}

void Vizia_SMClose(){
    shared_memory_object::remove(VIZIA_SM_NAME);
}