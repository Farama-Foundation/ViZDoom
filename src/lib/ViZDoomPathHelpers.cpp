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

#include "ViZDoomPathHelpers.h"
#include "ViZDoomExceptions.h"

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>

#if defined (OS_LINUX) || defined (OS_OSX)
	#include <dlfcn.h>
	const int objectMarker = 0;
#elif defined(OS_WIN)
    #include <Windows.h>
    EXTERN_C IMAGE_DOS_HEADER __ImageBase;
#endif

namespace vizdoom {

    namespace b     = boost;
    namespace bfs   = boost::filesystem;

    std::string fileExtension(std::string filePath) {
        bfs::path path(filePath);
        return path.extension().string();
    }

    bool hasExtension(std::string filePath) {
        bfs::path path(filePath);
        return path.has_extension();
    }

    bool fileExists(std::string filePath) {
        bfs::path path(filePath);
        return bfs::is_regular_file(path);

        //std::ifstream file(filePath);
        //bool exist = file.good();
        //file.close();
        //return exist;
    }

    std::string relativePath(std::string relativePath, std::string basePath) {
        bfs::path outPath(basePath);
        outPath.remove_filename();
        outPath /= relativePath;

        //return outPath.string();

        bfs::path normalizedPath;
        for (auto i = outPath.begin(); i != outPath.end(); ++i) {
            if (*i == "..") {
                // /a/b/.. is not necessarily /a if b is a symbolic link
                // /a/b/../.. is not /a/b/.. under most circumstances
                // We can end up with ..s in our result because of symbolic links
                if (boost::filesystem::is_symlink(normalizedPath)) normalizedPath /= *i;
                
                // Otherwise it should be safe to resolve the parent
                else if (normalizedPath.filename() == ".." || normalizedPath.filename() == "") normalizedPath /= *i;
                else normalizedPath = normalizedPath.parent_path();
            } else if (*i != ".") normalizedPath /= *i;
        }

        return normalizedPath.string();
    }

    std::string checkFile(std::string filePath, std::string expectedExt) {
        if (!fileExists(filePath)) {
            if (!expectedExt.length() || hasExtension(filePath)) throw FileDoesNotExistException(filePath);
            if (!fileExists(filePath + "." + expectedExt))
                throw FileDoesNotExistException(filePath + "(." + expectedExt + ")");
            filePath += "." + expectedExt;
        }

        return filePath;
    }

    std::string prepareFilePathArg(std::string filePath) {
        b::erase_all(filePath, "\n");
        b::erase_all(filePath, "\r");

        return filePath;
    }

    std::string prepareFilePathCmd(std::string filePath) {
        filePath = prepareFilePathArg(filePath);
        if (b::find_first(filePath, " ") && filePath[0] != '\"' && filePath[filePath.length() - 1] != '\"')
            filePath = std::string("\"") + filePath + "\"";

        return filePath;
    }

    std::string prepareExeFilePath(std::string filePath) {
        filePath = prepareFilePathArg(filePath);

        #ifdef OS_WIN
            return checkFile(filePath, "exe");
        #else

        return checkFile(filePath);
#endif
    }

    std::string prepareWadFilePath(std::string filePath) {
        filePath = prepareFilePathArg(filePath);
        return checkFile(filePath, "wad");
    }

    std::string prepareLmpFilePath(std::string filePath) {
        filePath = checkFile(filePath, "lmp");
        return prepareFilePathCmd(filePath);
    }

    std::string initializeThisSharedObjectPath() {
        #if defined (OS_LINUX) || defined (OS_OSX)
            Dl_info dlInfo;
            dladdr(&objectMarker, &dlInfo);
            std::string thisObjectPath = std::string(dlInfo.dli_fname);
        #elif defined(OS_WIN)
            char path[MAX_PATH];
            HMODULE hm = NULL;

            if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, 
                (LPCSTR)&initializeThisSharedObjectPath, &hm)){
                HMODULE hm = (HINSTANCE)&__ImageBase;
            }

            GetModuleFileNameA(hm, path, sizeof(path));
            std::string thisObjectPath = std::string(path);
        #else
			std::string thisObjectPath = ".";
        #endif

        thisObjectPath = boost::filesystem::absolute(thisObjectPath).parent_path().generic_string();
        return thisObjectPath;
    }

    std::string THIS_SHARED_OBJECT_PATH = initializeThisSharedObjectPath();

    std::string getThisSharedObjectPath() {
        return THIS_SHARED_OBJECT_PATH;
    }

}
