#include "vizdoom_shared_memory.h"
#include "vizdoom_message_queue.h"
#include "vizdoom_defines.h"

#include "doomstat.h"
#include "v_video.h"

bip::shared_memory_object vizdoomSM;
size_t vizdoomSMSize;
char * vizdoomSMName;

void ViZDoom_SMInit(const char * id){

	vizdoomSMName = new char[strlen(VIZDOOM_SM_NAME_BASE) + strlen(id)];
	strcpy(vizdoomSMName, VIZDOOM_SM_NAME_BASE);
	strcat(vizdoomSMName, id);

    try {
        bip::shared_memory_object::remove(vizdoomSMName);
        vizdoomSM = bip::shared_memory_object(bip::open_or_create, vizdoomSMName, bip::read_write);

        vizdoomSMSize = sizeof(ViZDoomInputStruct) + sizeof(ViZDoomGameVarsStruct) +
                      (sizeof(BYTE) * screen->GetWidth() * screen->GetHeight() * 4);
        vizdoomSM.truncate(vizdoomSMSize);
    }
    catch(bip::interprocess_exception &ex){
        printf("ViZDoom_SMInit: Error creating shared memory");
        ViZDoom_MQSend(VIZDOOM_MSG_CODE_DOOM_ERROR);
        exit(1);
    }
}

void ViZDoom_SMClose(){
    bip::shared_memory_object::remove(vizdoomSMName);
	delete[] vizdoomSMName;
}