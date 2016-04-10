#ifndef __VIZDOOM_EXCEPTIONS_H__
#define __VIZDOOM_EXCEPTIONS_H__

#include <exception>
#include <string>
#include <cstring>

namespace vizdoom{

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

    class ViZDoomErrorException : public Exception {
    public:
        const char* what() const throw(){ return "Controlled ViZDoom instance reported error."; }
    };

    class ViZDoomMismatchedVersionException : public Exception {
    public:
        const char* what() const throw(){ return "Controlled ViZDoom version does not match API version."; }
    };

    class ViZDoomUnexpectedExitException : public Exception {
    public:
        const char* what() const throw(){ return "Controlled ViZDoom instance exited unexpectedly."; }
    };

    class ViZDoomIsNotRunningException : public Exception {
    public:
        const char* what() const throw(){ return "Controlled ViZDoom instance is not running or not ready."; }
    };

    class PathDoesNotExistException : public Exception {
    public:
        PathDoesNotExistException(std::string path){
            this->path = path;
        }
        ~PathDoesNotExistException() throw(){}
        const char* what() const throw(){
            std::string what = std::string("Path \"") + this->path + "\" does not exists.";
            return strdup(what.c_str());
        }

    private:
        std::string path;
    };
}

#endif