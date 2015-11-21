#include "vizia_message_queue.h"

#include "info.h"
#include "c_bind.h"
#include "c_console.h"
#include "c_dispatch.h"
#include "g_game.h"
#include "g_level.h"

#include <string.h>

bip::message_queue *viziaMQController;
bip::message_queue *viziaMQDoom;
char * viziaMQControllerName;
char * viziaMQDoomName;

void Vizia_Command(char * command){
    AddCommandString(command);
}

void Vizia_MQInit(const char * id){
    //bip::message_queue::remove(VIZIA_MQ_NAME);
    //viziaMQ = new bip::message_queue(bip::open_or_create, VIZIA_MQ_NAME, VIZIA_MQ_MAX_MSG_NUM, VIZIA_MQ_MAX_MSG_SIZE);
    viziaMQControllerName = strcat(strdup(VIZIA_MQ_NAME_CTR_BASE), id);
    viziaMQDoomName = strcat(strdup(VIZIA_MQ_NAME_DOOM_BASE), id);

    viziaMQController = new bip::message_queue(bip::open_or_create, viziaMQControllerName, VIZIA_MQ_MAX_MSG_NUM, VIZIA_MQ_MAX_MSG_SIZE);
    viziaMQDoom = new bip::message_queue(bip::open_or_create, viziaMQDoomName, VIZIA_MQ_MAX_MSG_NUM, VIZIA_MQ_MAX_MSG_SIZE);
}

void Vizia_MQSend(uint8_t code){
    ViziaMessageSignalStruct msg;
    msg.code = code;
    viziaMQController->send(&msg, sizeof(ViziaMessageSignalStruct), 0);
    //return viziaMQ->try_send(&msg, sizeof(ViziaMessageSignalStruct), 0);
}

bool Vizia_MQTrySend(uint8_t code){
    ViziaMessageSignalStruct msg;
    msg.code = code;
    return viziaMQController->try_send(&msg, sizeof(ViziaMessageSignalStruct), 0);
}

void Vizia_MQSend(uint8_t code, const char * command){
    ViziaMessageCommandStruct msg;
    msg.code = code;
    strncpy(msg.command, command, VIZIA_MQ_MAX_CMD_LEN);
    viziaMQController->send(&msg, sizeof(ViziaMessageCommandStruct), 0);
}

bool Vizia_MQTrySend(uint8_t code, const char * command){
    ViziaMessageCommandStruct msg;
    msg.code = code;
    strncpy(msg.command, command, VIZIA_MQ_MAX_CMD_LEN);
    return viziaMQController->try_send(&msg, sizeof(ViziaMessageCommandStruct), 0);
}

void Vizia_MQRecv(void *msg, unsigned long &size, unsigned int &priority){
    viziaMQDoom->receive(&msg, sizeof(ViziaMessageCommandStruct), size, priority);
}

bool Vizia_MQTryRecv(void *msg, unsigned long &size, unsigned int &priority){
    return viziaMQDoom->try_receive(&msg, sizeof(ViziaMessageCommandStruct), size, priority);
}

void Vizia_MQTic(){

    Vizia_MQSend(VIZIA_MSG_CODE_DOOM_TIC);

    ViziaMessageCommandStruct msg;

    unsigned int priority;
    bip::message_queue::size_type recvd_size;

    bool nextTic = false;
    do {
        viziaMQDoom->receive(&msg, sizeof(ViziaMessageCommandStruct), recvd_size, priority);
        switch(msg.code){
            case VIZIA_MSG_CODE_READY :
            case VIZIA_MSG_CODE_TIC :
                nextTic = true;
                break;
            case VIZIA_MSG_CODE_COMMAND :
                Vizia_Command(strdup(msg.command));
                break;
            case VIZIA_MSG_CODE_CLOSE :
                Vizia_Command(strdup("exit"));
                break;
            default : break;
        }
    }while(!nextTic);

}

void Vizia_MQClose(){
    //bip::message_queue::remove(viziaMQControllerName);
    bip::message_queue::remove(viziaMQDoomName);
}