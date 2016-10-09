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

#ifndef __VIZ_MESSAGE_QUEUE_H__
#define __VIZ_MESSAGE_QUEUE_H__

#include <boost/interprocess/ipc/message_queue.hpp>

namespace bip = boost::interprocess;

extern bip::message_queue *vizMQController;
extern bip::message_queue *vizMQDoom;

#define VIZ_MQ_NAME_CTR_BASE "ViZDoomMQCtr"
#define VIZ_MQ_NAME_DOOM_BASE "ViZDoomMQDoom"
#define VIZ_MQ_MAX_MSG_NUM 64
#define VIZ_MQ_MAX_MSG_SIZE sizeof(VIZMessageCommand)
#define VIZ_MQ_MAX_CMD_LEN 128

#define VIZ_MSG_CODE_DOOM_DONE 11
#define VIZ_MSG_CODE_DOOM_CLOSE 12
#define VIZ_MSG_CODE_DOOM_ERROR 13

#define VIZ_MSG_CODE_TIC 21
#define VIZ_MSG_CODE_UPDATE 22
#define VIZ_MSG_CODE_TIC_AND_UPDATE 23
#define VIZ_MSG_CODE_COMMAND 24
#define VIZ_MSG_CODE_CLOSE 25
#define VIZ_MSG_CODE_ERROR 26


struct VIZMessage{
    uint8_t code;
    char command[VIZ_MQ_MAX_CMD_LEN];
};

void VIZ_MQInit(const char * id);

void VIZ_MQSend(uint8_t code, const char * command = nullptr);
void VIZ_MQReceive(void *msg);
bool VIZ_MQTryReceive(void *msg);

void VIZ_MQTic();

void VIZ_MQClose();

#endif
