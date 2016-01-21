#include "vizia_message_queue.h"
#include "vizia_input.h"
#include "vizia_game.h"
#include "vizia_main.h"

#include "doomtype.h"
#include "c_cvars.h"

EXTERN_CVAR (Bool, vizia_async)

#include <string.h>

bip::message_queue *viziaMQController;
bip::message_queue *viziaMQDoom;
char * viziaMQControllerName;
char * viziaMQDoomName;

void Vizia_MQInit(const char * id){

    viziaMQControllerName = strcat(strdup(VIZIA_MQ_NAME_CTR_BASE), id);
    viziaMQDoomName = strcat(strdup(VIZIA_MQ_NAME_DOOM_BASE), id);

    try{
        viziaMQController = new bip::message_queue(bip::open_only, viziaMQControllerName);//, VIZIA_MQ_MAX_MSG_NUM, VIZIA_MQ_MAX_MSG_SIZE);
        viziaMQDoom = new bip::message_queue(bip::open_only, viziaMQDoomName);//, VIZIA_MQ_MAX_MSG_NUM, VIZIA_MQ_MAX_MSG_SIZE);
    }
    catch(bip::interprocess_exception &ex){
        Printf("Vizia_MQInit: Error creating message queues");
        Vizia_MQSend(VIZIA_MSG_CODE_DOOM_ERROR);
        exit(1);
    }
}

void Vizia_MQSend(uint8_t code){
    ViziaMessageSignalStruct msg;
    msg.code = code;
    viziaMQController->send(&msg, sizeof(ViziaMessageSignalStruct), 0);
    //return viziaMQ->try_send(&msg, sizeof(ViziaMessageSignalStruct), 0);
}

void Vizia_MQSend(uint8_t code, const char * command){
    ViziaMessageCommandStruct msg;
    msg.code = code;
    strncpy(msg.command, command, VIZIA_MQ_MAX_CMD_LEN);
    viziaMQController->send(&msg, sizeof(ViziaMessageCommandStruct), 0);
}

void Vizia_MQRecv(void *msg, unsigned long &size, unsigned int &priority){
    viziaMQDoom->receive(msg, sizeof(ViziaMessageCommandStruct), size, priority);
}

bool Vizia_MQTryRecv(void *msg, unsigned long &size, unsigned int &priority){
    return viziaMQDoom->try_receive(msg, sizeof(ViziaMessageCommandStruct), size, priority);
}

void Vizia_MQTic(){

    if(!*vizia_async) Vizia_MQSend(VIZIA_MSG_CODE_DOOM_DONE);

    ViziaMessageCommandStruct msg;

    unsigned int priority;
    bip::message_queue::size_type recv_size;

    bool nextTic = false;
    do {
        if(!*vizia_async) Vizia_MQRecv(&msg, recv_size, priority);
        else{
            bool isMsg = Vizia_MQTryRecv(&msg, recv_size, priority);
            if(!isMsg) break;
        }
        switch(msg.code){
            case VIZIA_MSG_CODE_TIC :
                if(*vizia_async){
                    Vizia_GameVarsTic();
                    Vizia_MQSend(VIZIA_MSG_CODE_DOOM_DONE);
                }
                nextTic = true;
                break;

            case VIZIA_MSG_CODE_UPDATE:
                Vizia_Update();
                if(*vizia_async){
                    Vizia_GameVarsTic();
                }
                Vizia_MQSend(VIZIA_MSG_CODE_DOOM_DONE);
                break;

            case VIZIA_MSG_CODE_TIC_N_UPDATE:
                vizia_update = true;
                nextTic = true;
                if(*vizia_async){
                    Vizia_Update();
                    Vizia_GameVarsTic();
                    Vizia_MQSend(VIZIA_MSG_CODE_DOOM_DONE);
                }
                break;

            case VIZIA_MSG_CODE_COMMAND :
                Vizia_Command(strdup(msg.command));
                break;

            case VIZIA_MSG_CODE_CLOSE :
            case VIZIA_MSG_CODE_ERROR:
                exit(0);

            default : break;
        }
    }while(!nextTic);
}

void Vizia_MQClose(){
    //bip::message_queue::remove(viziaMQControllerName);
    //bip::message_queue::remove(viziaMQDoomName);
}