#ifndef __VIZDOOM_EXCEPTIONS_H__
#define __VIZDOOM_EXCEPTIONS_H__

#include <exception>
#include <string>
#include <cstring>

namespace vizdoom{

    /* Warnings TO DO:
     * When config after init
     * When run time stuff before init
     * When skill level > 5 < 1
     * When value in action > MaxValue
     * When action shorter or longer
     * Wrong path
     */

    class Exception : public std::exception {
    public:
        virtual const char* what() const throw(){ return "Unknown exception."; }
    };

    class SharedMemoryException : public Exception {
    public:
        const char* what() const throw(){ return "Shared memory error."; }
    };

    class MessageQueueException : public Exception {
    public:
        const char* what() const throw(){ return "Message queue error."; }
    };

    class DoomErrorException : public Exception {
    public:
        const char* what() const throw(){ return "Controlled Doom instance raported error."; }
    };

    class DoomUnexpectedExitException : public Exception {
    public:
        const char* what() const throw(){ return "Controlled Doom instance exited unexpectedly."; }
    };

    class DoomIsNotRunningException : public Exception {
    public:
        const char* what() const throw(){ return "Controlled Doom instance is not running or not ready."; }
    };

    class PathDoesNotExistsException : public Exception {
    public:
        PathDoesNotExistsException(std::string path){
            this->path = path;
        }
        ~PathDoesNotExistsException() throw(){}
        const char* what() const throw(){
            std::string what = std::string("Path \"") + this->path + "\" does not exists.";
            return strdup(what.c_str());
        }

    private:
        std::string path;
    };
}

#endif