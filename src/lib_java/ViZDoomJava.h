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

#ifndef __VIZDOOM_JAVA_H__
#define __VIZDOOM_JAVA_H__

#include "ViZDoom.h"

#include <jni.h>

#include <cstdlib>
#include <cstring>
#include <functional>
#include <type_traits>

using namespace vizdoom;

/* Exceptions rethrowing as Java exceptions */
/*--------------------------------------------------------------------------------------------------------------------*/

void throwAsJavaException(JNIEnv *env) {
    try {
        throw;
    }
    catch(FileDoesNotExistException& e){
        jclass ex = env->FindClass("vizdoom/FileDoesNotExistException");
        if(ex) env->ThrowNew(ex, e.what());
    }
    catch(MessageQueueException& e){
        jclass ex = env->FindClass("vizdoom/MessageQueueException");
        if(ex) env->ThrowNew(ex, e.what());
    }
    catch(SharedMemoryException& e){
        jclass ex = env->FindClass("vizdoom/SharedMemoryException");
        if(ex) env->ThrowNew(ex, e.what());
    }
    catch(SignalException& e){
        jclass ex = env->FindClass("vizdoom/SignalException");
        if(ex) env->ThrowNew(ex, e.what());
    }
    catch(ViZDoomErrorException& e){
        jclass ex = env->FindClass("vizdoom/ViZDoomErrorException");
        if(ex) env->ThrowNew(ex, e.what());
    }
    catch(ViZDoomIsNotRunningException& e){
        jclass ex = env->FindClass("vizdoom/ViZDoomIsNotRunningException");
        if(ex) env->ThrowNew(ex, e.what());
    }
    catch(ViZDoomUnexpectedExitException& e){
        jclass ex = env->FindClass("vizdoom/ViZDoomUnexpectedExitException");
        if(ex) env->ThrowNew(ex, e.what());
    }
    catch(const std::exception& e) {
        /* unknown exception */
        jclass jc = env->FindClass("java/lang/Error");
        if(jc) env->ThrowNew (jc, e.what());
    }
    catch(...) {
        /* unidentified exception */
        jclass jc = env->FindClass("java/lang/Error");
        if(jc) env->ThrowNew (jc, "Unidentified exception");
    }
}


/* C++ helpers to simplify Java binding */
/*--------------------------------------------------------------------------------------------------------------------*/

// Gets object instance from Java
template<class T>
T* getObjectFromJava(JNIEnv *jEnv, jobject jObj){
    jclass classT = jEnv->GetObjectClass(jObj);
    jfieldID classId = jEnv->GetFieldID(classT, "internalPtr", "J");

    if (classId == nullptr)
        return nullptr;
    else
        return (T*)jEnv->GetLongField(jObj, classId);
}

// Types converters

// From C++ to Java
jboolean castTojboolean(bool val) { return (jboolean)val; }
jboolean castTojboolean(JNIEnv *jEnv, bool val) { return (jboolean)val; }

jint castTojint(int val) { return (jint)val; }
jint castTojint(JNIEnv *jEnv, int val) { return (jint)val; }

jdouble castTojdouble(double val) { return (jdouble)val; }
jdouble castTojdouble(JNIEnv *jEnv, double val) { return (jdouble)val; }

jstring castTojstring(JNIEnv *jEnv, std::string val){
    return jEnv->NewStringUTF(val.c_str());
}

template<class T>
jbyteArray castTojbyteArray(JNIEnv *jEnv, std::vector<T>& val) {
    jbyteArray jVal = jEnv->NewByteArray(val.size());
    jbyte *jValArr = jEnv->GetByteArrayElements(jVal, NULL);
    if(sizeof(jbyte) == sizeof(T)) std::memcpy(jValArr, val.data(), val.size());
    else for (int i = 0; i < val.size(); ++i) jValArr[i] = (jint)val[i];
    jEnv->ReleaseByteArrayElements(jVal, jValArr, NULL);
    return jVal;
}

template<class T>
jintArray castTojintArray(JNIEnv *jEnv, std::vector<T>& val) {
    jintArray jVal = jEnv->NewIntArray(val.size());
    jint *jValArr = jEnv->GetIntArrayElements(jVal, NULL);
    if(sizeof(jint) == sizeof(T)) std::memcpy(jValArr, val.data(), val.size());
    else for (int i = 0; i < val.size(); ++i) jValArr[i] = (jint)val[i];
    jEnv->ReleaseIntArrayElements(jVal, jValArr, NULL);
    return jVal;
}

template<class T>
jdoubleArray castTojdoubleArray(JNIEnv *jEnv, std::vector<T>& val) {
    jdoubleArray jVal = jEnv->NewDoubleArray(val.size());
    jdouble *jValArr = jEnv->GetDoubleArrayElements(jVal, NULL);
    if(sizeof(jint) == sizeof(T)) std::memcpy(jValArr, val.data(), val.size());
    else for (int i = 0; i < val.size(); ++i) jValArr[i] = (jdouble)val[i];
    jEnv->ReleaseDoubleArrayElements(jVal, jValArr, NULL);
    return jVal;
}

// From Java to C++
bool jbooleanCast(jint jVal){ return (bool)jVal; }
bool jbooleanCast(JNIEnv *jEnv, jint jVal){ return (bool)jVal; }

int jintCast(jint jVal){ return (int)jVal; }
int jintCast(JNIEnv *jEnv, jint jVal){ return (int)jVal; }

double jdoubleCast(jdouble jVal){ return (double)jVal; }
double jdoubleCast(JNIEnv *jEnv, jdouble jVal){ return (double)jVal; }

std::string jstringCast(JNIEnv *jEnv, jstring jVal){
    const char *val = jEnv->GetStringUTFChars(jVal, NULL);
    return std::string(val);
}

std::vector<int> jintArrayCast(JNIEnv *jEnv, jintArray jVal){
    int jValLen = jEnv->GetArrayLength(jVal);
    jint *jValArr = jEnv->GetIntArrayElements(jVal, NULL);
    std::vector<int> val;
    for (int i=0; i<jValLen; ++i) val.push_back((int)jValArr[i]);
    jEnv->ReleaseIntArrayElements(jVal, jValArr, NULL);
    return val;
}

template<class T>
T jobjectCastToEnum(JNIEnv *jEnv, const char* jClassName, jobject jEnum) {
    jclass jClass = jEnv->FindClass(jClassName);
    if(jClass == 0) return static_cast<T>(0);
    jmethodID jClassId = jEnv->GetMethodID(jClass, "ordinal", "()I");
    if (jClassId == 0) return static_cast<T>(0);
    jint jEnumVal = jEnv->CallIntMethod(jEnum, jClassId);
    auto enumVal = static_cast<T>(jEnumVal);
    jEnv->DeleteLocalRef(jClass);
    return enumVal;
}

// Constructs Java object instance
// T - object type, A1 - args types
template<class T, class... A>
void constructJavaObject(JNIEnv *jEnv, jobject jObj, A... args){
    jclass classT = jEnv->GetObjectClass(jObj);
    jfieldID classId = jEnv->GetFieldID(classT, "internalPtr", "J");
    if (classId == NULL) return;
    T *obj = new T(args...);
    jEnv->SetLongField(jObj, classId, (jlong)obj);
}

// Calls object method from Java
// O - object type, R - return type, A1, A2 - args types
template<class O, class R, class... A1, class... A2>
R callObjMethod(JNIEnv *jEnv, jobject jObj, R(O::*func)(A1...), A2... args){
    try {
        O *obj = getObjectFromJava<O>(jEnv, jObj);
        auto objFunc = std::bind(func, obj, std::forward<A2>(args)...);
        return objFunc();
    }
    catch(...){
        throwAsJavaException(jEnv);
    }
}

// It feels so bad to use so many macros...

// JNI function name macros
#define _JNI_FUNC_NAME(jrt, p, c, s) JNIEXPORT jrt JNICALL Java_ ## p ## _ ## c ## _ ## s
#define JNI_FUNC_NAME(jrt, p, c, s) _JNI_FUNC_NAME(jrt, p, c, s)

// Select macros
#define _EXPAND( a ) a
#define _CAT( a, b ) a ## b
#define _SELECT( name, num ) _CAT( name ## _, num )

#define _GET_COUNT( _1, _2, _3, _4, _5, _6, count, ... ) count
#define _VA_SIZE( ... ) _EXPAND( _GET_COUNT( __VA_ARGS__, 6, 5, 4, 3, 2, 1 ) )

#define _VA_SELECT( name, ... ) _EXPAND( _SELECT( name, _VA_SIZE(__VA_ARGS__) )(__VA_ARGS__) )


/* JNI_EXPORT(...) macro - generates declaration of JNI function */
/*--------------------------------------------------------------------------------------------------------------------*/

#define JNI_EXPORT(...) _VA_SELECT(JNI_EXPORT, __VA_ARGS__)

#define JNI_EXPORT_2(jrt, s) \
JNI_FUNC_NAME(jrt, JAVA_PACKAGE, JAVA_CLASS, s) (JNIEnv *jEnv, jobject jObj)

#define JNI_EXPORT_3(jrt, s, ja1t) \
JNI_FUNC_NAME(jrt, JAVA_PACKAGE, JAVA_CLASS, s) (JNIEnv *jEnv, jobject jObj, ja1t jarg1)

#define JNI_EXPORT_4(jrt, s, ja1t, ja2t) \
JNI_FUNC_NAME(jrt, JAVA_PACKAGE, JAVA_CLASS, s) (JNIEnv *jEnv, jobject jObj, ja1t jarg1, ja2t jarg2)

#define JNI_EXPORT_5(jrt, s, ja1t, ja2t, ja3t) \
JNI_FUNC_NAME(jrt, JAVA_PACKAGE, JAVA_CLASS, s) (JNIEnv *jEnv, jobject jObj, ja1t jarg1, ja2t jarg2, ja3t jarg3)


/* JNI_METHOD(...) macro - generates definition of JNI function that call object method */
/*--------------------------------------------------------------------------------------------------------------------*/

#define JNI_METHOD(jrt, ...) JNI_METHOD_ ## jrt (jrt, __VA_ARGS__)
#define JNI_METHOD_void(...) JNI_METHOD_VOID(__VA_ARGS__)
#define JNI_METHOD_jboolean(...) JNI_METHOD_RETT(__VA_ARGS__)
#define JNI_METHOD_jint(...) JNI_METHOD_RETT(__VA_ARGS__)
#define JNI_METHOD_jdouble(...) JNI_METHOD_RETT(__VA_ARGS__)
#define JNI_METHOD_jstring(...) JNI_METHOD_RETT(__VA_ARGS__)
#define JNI_METHOD_jintArray(...) JNI_METHOD_RETT(__VA_ARGS__)
#define JNI_METHOD_jobject(...) JNI_METHOD_RETT(__VA_ARGS__)
#define JNI_METHOD_jobjectArray(...) JNI_METHOD_RETT(__VA_ARGS__)

#define JNI_METHOD_VOID(...) _VA_SELECT(JNI_METHOD_VOID, __VA_ARGS__)
#define JNI_METHOD_RETT(...) _VA_SELECT(JNI_METHOD_RETT, __VA_ARGS__)

#define JNI_METHOD_VOID_3(jrt, s, f) \
JNI_EXPORT_2(jrt, s) { callObjMethod(jEnv, jObj, &CPP_CLASS::f); }

#define JNI_METHOD_RETT_3(jrt, s, f) \
JNI_EXPORT_2(jrt, s) { return castTo ## jrt(jEnv, callObjMethod(jEnv, jObj, &CPP_CLASS::f)); }

#define JNI_METHOD_VOID_4(jrt, s, f, ja1t) \
JNI_EXPORT_3(jrt, s, ja1t) { \
    auto arg1 = ja1t ## Cast(jEnv, jarg1); \
    callObjMethod(jEnv, jObj, &CPP_CLASS::f, arg1); }

#define JNI_METHOD_RETT_4(jrt, s, f, ja1t) \
JNI_EXPORT_3(jrt, s, ja1t) { \
    auto arg1 = ja1t ## Cast(jEnv, jarg1); \
    return castTo ## jrt(jEnv, callObjMethod(jEnv, jObj, &CPP_CLASS::f, arg1)); }

#define JNI_METHOD_VOID_5(jrt, s, f, ja1t, ja2t) \
JNI_EXPORT_4(jrt, s, ja1t, ja2t) { \
    auto arg1 = ja1t ## Cast(jEnv, jarg1); \
    auto arg2 = ja2t ## Cast(jEnv, jarg2); \
    callObjMethod(jEnv, jObj, &CPP_CLASS::f, arg1, arg2); }

#define JNI_METHOD_RETT_5(jrt, s, f, ja1t, ja2t) \
JNI_EXPORT_4(jrt, s, ja1t, ja2t) { \
    auto arg1 = ja1t ## Cast(jEnv, jarg1); \
    auto arg2 = ja2t ## Cast(jEnv, jarg2); \
    return castTo ## jrt(jEnv, callObjMethod(jEnv, jObj, &CPP_CLASS::f, arg1, arg2)); }

#define JNI_METHOD_VOID_6(jrt, s, f, ja1t, ja2t, ja3t) \
JNI_EXPORT_5(jrt, s, ja1t, ja2t, ja3t) { \
    auto arg1 = ja1t ## Cast(jEnv, jarg1); \
    auto arg2 = ja2t ## Cast(jEnv, jarg2); \
    auto arg3 = ja3t ## Cast(jEnv, jarg3); \
    callObjMethod(jEnv, jObj, &CPP_CLASS::f, arg1, arg2, arg3); }

#define JNI_METHOD_RETT_6(jrt, s, f, ja1t, ja2t, ja3t) \
JNI_EXPORT_5(jrt, s, ja1t, ja2t, ja3t) { \
    auto arg1 = ja1t ## Cast(jEnv, jarg1); \
    auto arg2 = ja2t ## Cast(jEnv, jarg2); \
    auto arg3 = ja3t ## Cast(jEnv, jarg3); \
    return castTo ## jrt(jEnv, callObjMethod(jEnv, jObj, &CPP_CLASS::f, arg1, arg2, arg3)); }

#endif