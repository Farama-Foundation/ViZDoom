#include "vizia_message_queue.h"
#include "vizia_input.h"
#include "vizia_main.h"

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
        printf("Vizia_MQInit: Error creating message queues");
        Vizia_MQSend(VIZIA_MSG_CODE_DOOM_ERROR);
        Vizia_Command(strdup("exit"));
    }
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

    Vizia_MQSend(VIZIA_MSG_CODE_DOOM_DONE);

    ViziaMessageCommandStruct msg;

    unsigned int priority;
    bip::message_queue::size_type recvd_size;

    bool nextTic = false;
    do {
        viziaMQDoom->receive(&msg, sizeof(ViziaMessageCommandStruct), recvd_size, priority);
        switch(msg.code){
            case VIZIA_MSG_CODE_TIC :
                nextTic = true;
                break;
            case VIZIA_MSG_CODE_UPDATE:
                Vizia_Update();
                Vizia_MQSend(VIZIA_MSG_CODE_DOOM_DONE);
                break;
            case VIZIA_MSG_CODE_TIC_N_UPDATE:
                vizia_update = true;
                nextTic = true;
                break;
            case VIZIA_MSG_CODE_COMMAND :
                Vizia_Command(strdup(msg.command));
                break;
            case VIZIA_MSG_CODE_CLOSE :
            case VIZIA_MSG_CODE_ERROR:
                Vizia_Command(strdup("exit"));
                break;
            default : break;
        }
    }while(!nextTic);
}

void Vizia_MQClose(){
    //bip::message_queue::remove(viziaMQControllerName);
    //bip::message_queue::remove(viziaMQDoomName);
}