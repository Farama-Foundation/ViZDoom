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

#include "ViZDoom.h"
#include "ViZDoomGameJava.h"

#include <jni.h>

#include <functional>
#include <type_traits>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/type_traits.hpp>

using namespace vizdoom;
namespace b = boost;


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
jintArray castTojintArray(JNIEnv *jEnv, std::vector<T>& val) {
    jintArray jVal = jEnv->NewIntArray(val.size());
    jint *jValArr = jEnv->GetIntArrayElements(jVal, NULL);
    for (int i=0; i < val.size(); ++i) jValArr[i] = (jint)val[i];
    jEnv->ReleaseIntArrayElements(jVal, jValArr, NULL);
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

#define JAVA_PACKAGE vizdoom
#define JAVA_CLASS DoomGame
#define CPP_CLASS DoomGameJava

// It feels so bad to use so many macros...

// Functions declaration
#define _JNI_FUNC_NAME(jrt, p, c, s) JNIEXPORT jrt JNICALL Java_ ## p ## _ ## c ## _ ## s
#define JNI_FUNC_NAME(jrt, p, c, s) _JNI_FUNC_NAME(jrt, p, c, s)

#define JNI_EXPORT_0_ARG(jrt, s) \
JNI_FUNC_NAME(jrt, JAVA_PACKAGE, JAVA_CLASS, s) (JNIEnv *jEnv, jobject jObj)

#define JNI_EXPORT_1_ARG(jrt, s, ja1t) \
JNI_FUNC_NAME(jrt, JAVA_PACKAGE, JAVA_CLASS, s) (JNIEnv *jEnv, jobject jObj, ja1t jarg1)

#define JNI_EXPORT_2_ARG(jrt, s, ja1t, ja2t) \
JNI_FUNC_NAME(jrt, JAVA_PACKAGE, JAVA_CLASS, s) (JNIEnv *jEnv, jobject jObj, ja1t jarg1, ja2t jarg2)

#define JNI_EXPORT_3_ARG(jrt, s, ja1t, ja2t, ja3t) \
JNI_FUNC_NAME(jrt, JAVA_PACKAGE, JAVA_CLASS, s) (JNIEnv *jEnv, jobject jObj, ja1t jarg1, ja2t jarg2, ja3t jarg3)

// Full functions
#define JNI_METHOD_VOID_0_ARG(jrt, s, c, f) \
JNI_EXPORT_0_ARG(jrt, s) { callObjMethod(jEnv, jObj, &c::f); }

#define JNI_METHOD_RETT_0_ARG(jrt, s, c, f) \
JNI_EXPORT_0_ARG(jrt, s) { return castTo ## jrt(jEnv, callObjMethod(jEnv, jObj, &c::f)); }

#define JNI_METHOD_VOID_1_ARG(jrt, s, c, f, ja1t) \
JNI_EXPORT_1_ARG(jrt, s, ja1t) { \
    auto arg1 = ja1t ## Cast(jEnv, jarg1); \
    callObjMethod(jEnv, jObj, &c::f, arg1); }

#define JNI_METHOD_RETT_1_ARG(jrt, s, c, f, ja1t) \
JNI_EXPORT_1_ARG(jrt, s, ja1t) { \
    auto arg1 = ja1t ## Cast(jEnv, jarg1); \
    return castTo ## jrt(jEnv, callObjMethod(jEnv, jObj, &c::f, arg1)); }

#define JNI_METHOD_VOID_2_ARG(jrt, s, c, f, ja1t, ja2t) \
JNI_EXPORT_2_ARG(jrt, s, ja1t, ja2t) { \
    auto arg1 = ja1t ## Cast(jEnv, jarg1); \
    auto arg2 = ja2t ## Cast(jEnv, jarg2); \
    callObjMethod(jEnv, jObj, &c::f, arg1, arg2); }

#define JNI_METHOD_RETT_2_ARG(jrt, s, c, f, ja1t, ja2t) \
JNI_EXPORT_2_ARG(jrt, s, ja1t, ja2t) { \
    auto arg1 = ja1t ## Cast(jEnv, jarg1); \
    auto arg2 = ja2t ## Cast(jEnv, jarg2); \
    return castTo ## jrt(jEnv, callObjMethod(jEnv, jObj, &c::f, arg1, arg2)); }

#define JNI_METHOD_VOID_3_ARG(jrt, s, c, f, ja1t, ja2t, ja3t) \
JNI_EXPORT_3_ARG(jrt, s, ja1t, ja2t, ja3t) { \
    auto arg1 = ja1t ## Cast(jEnv, jarg1); \
    auto arg2 = ja2t ## Cast(jEnv, jarg2); \
    auto arg3 = ja3t ## Cast(jEnv, jarg3); \
    callObjMethod(jEnv, jObj, &c::f, arg1, arg2, arg3); }

#define JNI_METHOD_RETT_3_ARG(jrt, s, c, f, ja1t, ja2t, ja3t) \
JNI_EXPORT_3_ARG(jrt, s, ja1t, ja2t, ja3t) { \
    auto arg1 = ja1t ## Cast(jEnv, jarg1); \
    auto arg2 = ja2t ## Cast(jEnv, jarg2); \
    auto arg3 = ja3t ## Cast(jEnv, jarg3); \
    return castTo ## jrt(jEnv, callObjMethod(jEnv, jObj, &c::f, arg1, arg2, arg3)); }


/* Binding methods */
/*--------------------------------------------------------------------------------------------------------------------*/

// Constructor
JNI_EXPORT_0_ARG(void, DoomGameNative) {
    constructJavaObject<DoomGameJava>(jEnv, jObj);
}

JNI_METHOD_RETT_1_ARG(jboolean, loadConfig, DoomGameJava, loadConfig, jstring)
JNI_METHOD_RETT_0_ARG(jboolean, init, DoomGameJava, init)
JNI_METHOD_VOID_0_ARG(void, close, DoomGameJava, close)
JNI_METHOD_VOID_0_ARG(void, newEpisode__, DoomGameJava, newEpisode_)
JNI_METHOD_VOID_1_ARG(void, newEpisode__Ljava_lang_String, DoomGameJava, newEpisode_str, jstring)
JNI_METHOD_VOID_1_ARG(void, replayEpisode__Ljava_lang_String_2, DoomGameJava, replayEpisode_str, jstring)
JNI_METHOD_VOID_2_ARG(void, replayEpisode__Ljava_lang_String_2I, DoomGameJava, replayEpisode_str_int, jstring, jint)
JNI_METHOD_RETT_0_ARG(jboolean, isRunning, DoomGameJava, isRunning)
JNI_METHOD_VOID_1_ARG(void, setAction, DoomGameJava, setAction, jintArray)
JNI_METHOD_VOID_0_ARG(void, advanceAction__, DoomGameJava, advanceAction_)
JNI_METHOD_VOID_1_ARG(void, advanceAction__I, DoomGameJava, advanceAction_int, jint)
JNI_METHOD_VOID_2_ARG(void, advanceAction__IZ, DoomGameJava, advanceAction_int_bool, jint, jboolean)
JNI_METHOD_VOID_3_ARG(void, advanceAction__IZZ, DoomGameJava, advanceAction_int_bool_bool, jint, jboolean, jboolean)
JNI_METHOD_RETT_1_ARG(jdouble, makeAction___3I, DoomGameJava, makeAction_vec, jintArray)
JNI_METHOD_RETT_2_ARG(jdouble, makeAction___3II, DoomGameJava, makeAction_vec_int, jintArray, jint)

JNI_EXPORT_0_ARG(jobject, getState){
    auto state = callObjMethod(jEnv, jObj, &DoomGameJava::getState);
    if (state == nullptr) return NULL;

    jclass jStateClass = jEnv->FindClass("vizdoom/GameState");
    if (jStateClass == 0) return NULL;

    jintArray jGameVariables = castTojintArray(jEnv, state->gameVariables);
    jintArray jScreenBuffer = state->screenBuffer != nullptr ? castTojintArray(jEnv, *state->screenBuffer) : 0;
    jintArray jDepthBuffer = state->depthBuffer != nullptr ? castTojintArray(jEnv, *state->depthBuffer) : 0;
    jintArray jLabelsBuffer = state->labelsBuffer != nullptr ? castTojintArray(jEnv, *state->labelsBuffer) : 0;
    jintArray jAutomapBuffer = state->automapBuffer != nullptr ? castTojintArray(jEnv, *state->automapBuffer) : 0;

    jclass jLabelClass = jEnv->FindClass("vizdoom/Label");
    if (jLabelClass == 0) return NULL;
    jobjectArray jLabels = jEnv->NewObjectArray(state->labels.size(), jLabelClass, NULL);
    jmethodID jLabelConstructor = jEnv->GetMethodID(jLabelClass, "<init>", "(ILjava/lang/StringI)V");
    if (jLabelConstructor == 0) return NULL;

    for(size_t i = 0; i < state->labels.size(); ++i){
        jobject jLabel = jEnv->NewObject(jLabelClass, jLabelConstructor, (jint)state->labels[i].objectId,
                                         castTojstring(jEnv, state->labels[i].objectName), (jint)state->labels[i].value);
        jEnv->SetObjectArrayElement(jLabels, i, jLabel);
    }

    jmethodID jStateConstructor = jEnv->GetMethodID(jStateClass, "<init>", "(I[I[I[I[I[I[Lvizdoom/Label)V");
    if (jStateConstructor == 0) return NULL;
    jobject jState = jEnv->NewObject(jStateClass, jStateConstructor, (jint)state->number,
        jGameVariables, jScreenBuffer, jDepthBuffer, jLabelsBuffer, jAutomapBuffer, jLabels);

    return jState;
}

JNI_EXPORT_0_ARG(jintArray, getLastAction){
    auto lastAction = callObjMethod(jEnv, jObj, &DoomGameJava::getLastAction);
    return castTojintArray(jEnv, lastAction);
}

JNI_METHOD_RETT_0_ARG(jboolean, isNewEpisode, DoomGameJava, isNewEpisode)
JNI_METHOD_RETT_0_ARG(jboolean, isEpisodeFinished, DoomGameJava, isEpisodeFinished)
JNI_METHOD_RETT_0_ARG(jboolean, isPlayerDead, DoomGameJava, isPlayerDead)
JNI_METHOD_VOID_0_ARG(void, respawnPlayer, DoomGameJava, respawnPlayer)

JNI_EXPORT_1_ARG(void, addAvailableButton__Lvizdoom_Button_2, jobject){
    auto arg1 = jobjectCastToEnum<Button>(jEnv, "vizdoom/Button", jarg1);
    callObjMethod(jEnv, jObj, &DoomGameJava::addAvailableButton_btn, arg1);
}

JNI_EXPORT_2_ARG(void, addAvailableButton__Lvizdoom_Button_2I, jobject, jint){
    auto arg1 = jobjectCastToEnum<Button>(jEnv, "vizdoom/Button", jarg1);
    auto arg2 = jintCast(jarg2);
    callObjMethod(jEnv, jObj, &DoomGameJava::addAvailableButton_btn_int, arg1, arg2);
}

JNI_METHOD_VOID_0_ARG(void, clearAvailableButtons, DoomGameJava, clearAvailableButtons)
JNI_METHOD_RETT_0_ARG(jint, getAvailableButtonsSize, DoomGameJava, getAvailableButtonsSize)

JNI_EXPORT_2_ARG(void, setButtonMaxValue, jobject, jint){
    auto arg1 = jobjectCastToEnum<Button>(jEnv, "vizdoom/Button", jarg1);
    auto arg2 = jintCast(jarg2);
    callObjMethod(jEnv, jObj, &DoomGameJava::setButtonMaxValue, arg1, arg2);
}

JNI_EXPORT_1_ARG(jint, getButtonMaxValue, jobject){
    auto arg1 = jobjectCastToEnum<Button>(jEnv, "vizdoom/Button", jarg1);
    return castTojint(callObjMethod(jEnv, jObj, &DoomGameJava::getButtonMaxValue, arg1));
}

JNI_EXPORT_1_ARG(void, addAvailableGameVariable, jobject){
    auto arg1 = jobjectCastToEnum<GameVariable>(jEnv, "vizdoom/GameVariable", jarg1);
    callObjMethod(jEnv, jObj, &DoomGameJava::addAvailableGameVariable, arg1);
}

JNI_EXPORT_1_ARG(jint, getGameVariable, jobject){
    auto arg1 = jobjectCastToEnum<GameVariable>(jEnv, "vizdoom/GameVariable", jarg1);
    callObjMethod(jEnv, jObj, &DoomGameJava::getGameVariable, arg1);
}

JNI_METHOD_VOID_0_ARG(void, clearAvailableGameVariables, DoomGameJava, clearAvailableGameVariables)
JNI_METHOD_RETT_0_ARG(jint, getAvailableGameVariablesSize, DoomGameJava, getAvailableGameVariablesSize)
JNI_METHOD_VOID_1_ARG(void, addGameArgs, DoomGameJava, addGameArgs, jstring)
JNI_METHOD_VOID_0_ARG(void, clearGameArgs, DoomGameJava, clearGameArgs)
JNI_METHOD_VOID_1_ARG(void, sendGameCommand, DoomGameJava, sendGameCommand, jstring)
JNI_METHOD_RETT_0_ARG(jint, getModeNative, DoomGameJava, getMode)

JNI_EXPORT_1_ARG(void, setMode, jobject){
    auto arg1 = jobjectCastToEnum<Mode>(jEnv, "vizdoom/Mode", jarg1);
    callObjMethod(jEnv, jObj, &DoomGameJava::setMode, arg1);
}

JNI_METHOD_RETT_0_ARG(jint, getTicrate, DoomGameJava, getTicrate)
JNI_METHOD_VOID_1_ARG(void, setTicrate, DoomGameJava, setTicrate, jint)
JNI_METHOD_RETT_0_ARG(jdouble, getLivingReward, DoomGameJava, getLivingReward)
JNI_METHOD_VOID_1_ARG(void, setLivingReward, DoomGameJava, setLivingReward, jdouble)
JNI_METHOD_RETT_0_ARG(jdouble, getDeathPenalty, DoomGameJava, getDeathPenalty)
JNI_METHOD_VOID_1_ARG(void, setDeathPenalty, DoomGameJava, setDeathPenalty, jdouble)
JNI_METHOD_RETT_0_ARG(jdouble, getLastReward, DoomGameJava, getLastReward)
JNI_METHOD_RETT_0_ARG(jdouble, getTotalReward, DoomGameJava, getTotalReward)
JNI_METHOD_VOID_1_ARG(void, setViZDoomPath, DoomGameJava, setViZDoomPath, jstring)
JNI_METHOD_VOID_1_ARG(void, setDoomGamePath, DoomGameJava, setDoomGamePath, jstring)
JNI_METHOD_VOID_1_ARG(void, setDoomScenarioPath, DoomGameJava, setDoomScenarioPath, jstring)
JNI_METHOD_VOID_1_ARG(void, setDoomMap, DoomGameJava, setDoomMap, jstring)
JNI_METHOD_VOID_1_ARG(void, setDoomSkill, DoomGameJava, setDoomSkill, jint)
JNI_METHOD_VOID_1_ARG(void, setDoomConfigPath, DoomGameJava, setDoomConfigPath, jstring)
JNI_METHOD_RETT_0_ARG(jint, getSeed, DoomGameJava, getSeed)
JNI_METHOD_VOID_1_ARG(void, setSeed, DoomGameJava, setSeed, jint)
JNI_METHOD_RETT_0_ARG(jint, getEpisodeStartTime, DoomGameJava, getEpisodeStartTime)
JNI_METHOD_VOID_1_ARG(void, setEpisodeStartTime, DoomGameJava, setEpisodeTimeout, jint)
JNI_METHOD_RETT_0_ARG(jint, getEpisodeTimeout, DoomGameJava, getEpisodeTimeout)
JNI_METHOD_VOID_1_ARG(void, setEpisodeTimeout, DoomGameJava, setEpisodeTimeout, jint)
JNI_METHOD_RETT_0_ARG(jint, getEpisodeTime, DoomGameJava, getEpisodeTime)

JNI_EXPORT_1_ARG(void, setScreenResolution, jobject){
    auto arg1 = jobjectCastToEnum<ScreenResolution>(jEnv, "vizdoom/ScreenResolution", jarg1);
    callObjMethod(jEnv, jObj, &DoomGameJava::setScreenResolution, arg1);
}

JNI_EXPORT_1_ARG(void, setScreenFormat, jobject){
    auto arg1 = jobjectCastToEnum<ScreenFormat>(jEnv, "vizdoom/ScreenFormat", jarg1);
    callObjMethod(jEnv, jObj, &DoomGameJava::setScreenFormat, arg1);
}

JNI_METHOD_VOID_1_ARG(void, setRenderHud, DoomGameJava, setRenderHud, jboolean)
JNI_METHOD_VOID_1_ARG(void, setRenderMinimalHud, DoomGameJava, setRenderMinimalHud, jboolean)
JNI_METHOD_VOID_1_ARG(void, setRenderWeapon, DoomGameJava, setRenderWeapon, jboolean)
JNI_METHOD_VOID_1_ARG(void, setRenderCrosshair, DoomGameJava, setRenderCrosshair, jboolean)
JNI_METHOD_VOID_1_ARG(void, setRenderDecals, DoomGameJava, setRenderDecals, jboolean)
JNI_METHOD_VOID_1_ARG(void, setRenderParticles, DoomGameJava, setRenderParticles, jboolean)
JNI_METHOD_VOID_1_ARG(void, setRenderEffectsSprites, DoomGameJava, setRenderEffectsSprites, jboolean)
JNI_METHOD_VOID_1_ARG(void, setRenderMessages, DoomGameJava, setRenderMessages, jboolean)
JNI_METHOD_VOID_1_ARG(void, setWindowVisible, DoomGameJava, setWindowVisible, jboolean)
JNI_METHOD_VOID_1_ARG(void, setConsoleEnabled, DoomGameJava, setConsoleEnabled, jboolean)
JNI_METHOD_VOID_1_ARG(void, setSoundEnabled, DoomGameJava, setSoundEnabled, jboolean)
JNI_METHOD_RETT_0_ARG(jint, getScreenWidth, DoomGameJava, getScreenWidth)
JNI_METHOD_RETT_0_ARG(jint, getScreenHeight, DoomGameJava, getScreenHeight)
JNI_METHOD_RETT_0_ARG(jint, getScreenChannels, DoomGameJava, getScreenChannels)
JNI_METHOD_RETT_0_ARG(jint, getScreenPitch, DoomGameJava, getScreenPitch)
JNI_METHOD_RETT_0_ARG(jint, getScreenSize, DoomGameJava, getScreenSize)
JNI_METHOD_RETT_0_ARG(jint, getScreenFormatNative, DoomGameJava, getScreenFormat)

JNI_EXPORT_2_ARG(jdouble, doomTicsToMs, jdouble, jint){
    return (jdouble) doomTicsToMs(jarg1, jarg2);
}

JNI_EXPORT_2_ARG(jdouble, msToDoomTics, jdouble, jint){
    return (jdouble) msToDoomTics(jarg1, jarg2);
}

JNI_EXPORT_2_ARG(jdouble, doomTicsToSec, jdouble, jint){
    return (jdouble) doomTicsToSec(jarg1, jarg2);
}

JNI_EXPORT_2_ARG(jdouble, secToDoomTics, jdouble, jint){
    return (jdouble) secToDoomTics(jarg1, jarg2);
}

JNI_EXPORT_1_ARG(jdouble, doomFixedToDouble, jint){
    return (jdouble) doomFixedToDouble(jarg1);
}

JNI_EXPORT_1_ARG(jboolean, isBinaryButton, jobject){
    auto arg1 = jobjectCastToEnum<Button>(jEnv, "vizdoom/Button", jarg1);
    return (jboolean) isBinaryButton(arg1);
}

JNI_EXPORT_1_ARG(jboolean, isDeltaButton, jobject){
    auto arg1 = jobjectCastToEnum<Button>(jEnv, "vizdoom/Button", jarg1);
    return (jboolean) isDeltaButton(arg1);
}
