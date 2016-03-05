#include "vizdoom_message_queue.h"
#include "vizdoom_input.h"
#include "vizdoom_game.h"
#include "vizdoom_main.h"
#include "vizdoom_defines.h"

#include "doomtype.h"
#include "c_cvars.h"

EXTERN_CVAR (Bool, vizdoom_async)

#include <string.h>

bip::message_queue *vizdoomMQController;
bip::message_queue *vizdoomMQDoom;
char * vizdoomMQControllerName;
char * vizdoomMQDoomName;

void ViZDoom_MQInit(const char * id){

    vizdoomMQControllerName = strcat(strdup(VIZDOOM_MQ_NAME_CTR_BASE), id);
    vizdoomMQDoomName = strcat(strdup(VIZDOOM_MQ_NAME_DOOM_BASE), id);

    try{
        vizdoomMQController = new bip::message_queue(bip::open_only, vizdoomMQControllerName);//, VIZDOOM_MQ_MAX_MSG_NUM, VIZDOOM_MQ_MAX_MSG_SIZE);
        vizdoomMQDoom = new bip::message_queue(bip::open_only, vizdoomMQDoomName);//, VIZDOOM_MQ_MAX_MSG_NUM, VIZDOOM_MQ_MAX_MSG_SIZE);
    }
    catch(bip::interprocess_exception &ex){
        Printf("ViZDoom_MQInit: Error creating message queues");
        ViZDoom_MQSend(VIZDOOM_MSG_CODE_DOOM_ERROR);
        exit(1);
    }
}

void ViZDoom_MQSend(uint8_t code){
    ViziaMessageSignalStruct msg;
    msg.code = code;
    vizdoomMQController->send(&msg, sizeof(ViziaMessageSignalStruct), 0);
}

void ViZDoom_MQSend(uint8_t code, const char * command){
    ViziaMessageCommandStruct msg;
    msg.code = code;
    strncpy(msg.command, command, VIZDOOM_MQ_MAX_CMD_LEN);
    vizdoomMQController->send(&msg, sizeof(ViziaMessageCommandStruct), 0);
}

void ViZDoom_MQRecv(void *msg, size_t &size, unsigned int &priority){
    vizdoomMQDoom->receive(msg, sizeof(ViziaMessageCommandStruct), size, priority);
}

bool ViZDoom_MQTryRecv(void *msg, size_t &size, unsigned int &priority){
    return vizdoomMQDoom->try_receive(msg, sizeof(ViziaMessageCommandStruct), size, priority);
}

void ViZDoom_MQTic(){

    ViziaMessageCommandStruct msg;

    unsigned int priority;
    bip::message_queue::size_type recv_size;

    do {
        if(!*vizdoom_async) ViZDoom_MQRecv(&msg, recv_size, priority);
        else{
            bool isMsg = ViZDoom_MQTryRecv(&msg, recv_size, priority);
            if(!isMsg) break;
        }
        switch(msg.code){
            case VIZDOOM_MSG_CODE_TIC :
                vizdoomNextTic = true;
                break;

            case VIZDOOM_MSG_CODE_UPDATE:
                ViZDoom_Update();
                ViZDoom_GameVarsTic();
                ViZDoom_MQSend(VIZDOOM_MSG_CODE_DOOM_DONE);
                break;

            case VIZDOOM_MSG_CODE_TIC_N_UPDATE:
                vizdoomUpdate = true;
                vizdoomNextTic = true;
                break;

            case VIZDOOM_MSG_CODE_COMMAND :
                ViZDoom_Command(strdup(msg.command));
                break;

            case VIZDOOM_MSG_CODE_CLOSE :
            case VIZDOOM_MSG_CODE_ERROR:
                exit(0);

            default : break;
        }
    }while(!vizdoomNextTic);
}

void ViZDoom_MQClose(){
    //bip::message_queue::remove(vizdoomMQControllerName);
    //bip::message_queue::remove(vizdoomMQDoomName);
}