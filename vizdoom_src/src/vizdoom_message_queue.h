/*
 Copyright (C) 2016 by Wojciech Jaśkowski, Michał Kempka, Grzegorz Runc, Jakub Toczek, Marek Wydmuch

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
*/

#ifndef __VIZDOOM_MESSAGE_QUEUE_H__
#define __VIZDOOM_MESSAGE_QUEUE_H__

#include <boost/interprocess/ipc/message_queue.hpp>

namespace bip = boost::interprocess;

extern bip::message_queue *vizdoomMQSend;
extern bip::message_queue *vizdoomMQRecv;

#define VIZDOOM_MQ_NAME_CTR_BASE "ViZDoomMQCtr"
#define VIZDOOM_MQ_NAME_DOOM_BASE "ViZDoomMQDoom"
#define VIZDOOM_MQ_MAX_MSG_NUM 64
#define VIZDOOM_MQ_MAX_MSG_SIZE sizeof(ViziaMessageCommandStruct)
#define VIZDOOM_MQ_MAX_CMD_LEN 128

#define VIZDOOM_MSG_CODE_DOOM_DONE 11
#define VIZDOOM_MSG_CODE_DOOM_CLOSE 12
#define VIZDOOM_MSG_CODE_DOOM_ERROR 13

#define VIZDOOM_MSG_CODE_TIC 21
#define VIZDOOM_MSG_CODE_UPDATE 22
#define VIZDOOM_MSG_CODE_TIC_N_UPDATE 23
#define VIZDOOM_MSG_CODE_COMMAND 24
#define VIZDOOM_MSG_CODE_CLOSE 25
#define VIZDOOM_MSG_CODE_ERROR 26

struct ViziaMessageSignalStruct{
    uint8_t code;
};

struct ViziaMessageCommandStruct{
    uint8_t code;
    char command[VIZDOOM_MQ_MAX_CMD_LEN];
};

void ViZDoom_MQInit(const char * id);

void ViZDoom_MQSend(uint8_t code);
void ViZDoom_MQSend(uint8_t code, const char * command);
void ViZDoom_MQRecv(void *msg, size_t &size, unsigned int &priority);
bool ViZDoom_MQTryRecv(void *msg, size_t &size, unsigned int &priority);

void ViZDoom_MQTic();

void ViZDoom_MQClose();

#endif
