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

#include "ViZDoomMessageQueue.h"
#include "ViZDoomExceptions.h"

namespace vizdoom {

    MessageQueue::MessageQueue(std::string name) : name(name) {
        this->init();
    }

    MessageQueue::~MessageQueue() {
        this->close();
    }

    void MessageQueue::init() {
        try {
            bip::message_queue::remove(this->name.c_str());
            this->mq = new bip::message_queue(bip::open_or_create, this->name.c_str(), MQ_MAX_MSG_NUM, sizeof(Message));
        }
        catch (...) { // bip::interprocess_exception
            throw MessageQueueException("Failed to create message queues.");
        }
    }

    void MessageQueue::close() {
        bip::message_queue::remove(this->name.c_str());
        if (this->mq) {
            delete this->mq;
            this->mq = nullptr;
        }
    }

    void MessageQueue::send(uint8_t code, const char *command) {
        Message msg;
        msg.code = code;
        if (command) strncpy(msg.command, command, MQ_MAX_CMD_LEN);

        try {
            this->mq->send(&msg, sizeof(Message), 0);
        }
        catch (...) { // bip::interprocess_exception
            throw MessageQueueException("Failed to send message.");
        }
    }

    Message MessageQueue::receive() {
        Message msg;

        unsigned int priority;
        size_t size;

        try {
            this->mq->receive(&msg, sizeof(Message), size, priority);
        }
        catch (...) { // bip::interprocess_exception
            throw MessageQueueException("Failed to receive message.");
        }

        return msg;
    }

}