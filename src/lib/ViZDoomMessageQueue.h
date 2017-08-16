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

#ifndef __VIZDOOM_MESSAGEQUEUE_H__
#define __VIZDOOM_MESSAGEQUEUE_H__

#include <cstdint>

#include <boost/interprocess/ipc/message_queue.hpp>

namespace vizdoom {

    namespace b         = boost;
    namespace bip       = boost::interprocess;

    /* Message queues' settings */
    #define MQ_MAX_MSG_NUM      64
    #define MQ_MAX_MSG_SIZE     sizeof(Message)
    #define MQ_MAX_CMD_LEN      128

    /* Message struct */
    struct Message {
        uint8_t code;
        char command[MQ_MAX_CMD_LEN];
    };

    class MessageQueue {

    public:
        MessageQueue(std::string name);
        ~MessageQueue();

        void init();
        void close();

        void send(uint8_t code, const char *command = nullptr);
        Message receive();

        //void receive(Message  *msg);
        //bool tryReceive(Message  *msg);

    private:
        bip::message_queue *mq;
        std::string name;
    };
}

#endif
