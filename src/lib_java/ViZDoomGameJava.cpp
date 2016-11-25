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

#include "ViZDoomGameJava.h"

#include <iostream>

JNI_EXPORT(void, DoomGameNative) {
    constructJavaObject<DoomGameJava>(jEnv, jObj);
}

JNI_METHOD(jboolean, loadConfig, loadConfig, jstring)
JNI_METHOD(jboolean, init, init)
JNI_METHOD(void, close, close)
JNI_METHOD(void, newEpisode__, newEpisode_)
JNI_METHOD(void, newEpisode__Ljava_lang_String, newEpisode_str, jstring)
JNI_METHOD(void, replayEpisode__Ljava_lang_String_2, replayEpisode_str, jstring)
JNI_METHOD(void, replayEpisode__Ljava_lang_String_2I, replayEpisode_str_int, jstring, jint)
JNI_METHOD(jboolean, isRunning, isRunning)
JNI_METHOD(void, setAction, setAction, jintArray)
JNI_METHOD(void, advanceAction__, advanceAction_)
JNI_METHOD(void, advanceAction__I, advanceAction_int, jint)
JNI_METHOD(void, advanceAction__IZ, advanceAction_int_bool, jint, jboolean)
JNI_METHOD(jdouble, makeAction___3I, makeAction_vec, jintArray)
JNI_METHOD(jdouble, makeAction___3II, makeAction_vec_int, jintArray, jint)

JNI_EXPORT(jobject, getState){
    auto state = callObjMethod(jEnv, jObj, &DoomGameJava::getState);
    if (state == nullptr) return NULL;

    jclass jStateClass = jEnv->FindClass("vizdoom/GameState");
    if (jStateClass == 0) return NULL;

    jdoubleArray jGameVariables = castTojdoubleArray(jEnv, state->gameVariables);
    jbyteArray jScreenBuffer = state->screenBuffer != nullptr ? castTojbyteArray(jEnv, *state->screenBuffer) : NULL;
    jbyteArray jDepthBuffer = state->depthBuffer != nullptr ? castTojbyteArray(jEnv, *state->depthBuffer) : NULL;
    jbyteArray jLabelsBuffer = state->labelsBuffer != nullptr ? castTojbyteArray(jEnv, *state->labelsBuffer) : NULL;
    jbyteArray jAutomapBuffer = state->automapBuffer != nullptr ? castTojbyteArray(jEnv, *state->automapBuffer) : NULL;

    jclass jLabelClass = jEnv->FindClass("vizdoom/Label");
    if (jLabelClass == 0) return NULL;
    jobjectArray jLabels = jEnv->NewObjectArray(state->labels.size(), jLabelClass, NULL);
    jmethodID jLabelConstructor = jEnv->GetMethodID(jLabelClass, "<init>", "(ILjava/lang/String;BDDD)V");
    if (jLabelConstructor == 0) return NULL;

    for(size_t i = 0; i < state->labels.size(); ++i){
        jobject jLabel = jEnv->NewObject(jLabelClass, jLabelConstructor,
                                        (jint)state->labels[i].objectId, castTojstring(jEnv, state->labels[i].objectName),
                                        (jint)state->labels[i].value, (jdouble)state->labels[i].objectPositionX,
                                        (jdouble)state->labels[i].objectPositionY, (jdouble)state->labels[i].objectPositionZ);
        jEnv->SetObjectArrayElement(jLabels, i, jLabel);
    }

    jmethodID jStateConstructor = jEnv->GetMethodID(jStateClass, "<init>", "(I[D[B[B[B[B[Lvizdoom/Label;)V");
    if (jStateConstructor == 0) return NULL;
    jobject jState = jEnv->NewObject(jStateClass, jStateConstructor, (jint)state->number,
                                jGameVariables, jScreenBuffer, jDepthBuffer, jLabelsBuffer, jAutomapBuffer, jLabels);

    return jState;
}

JNI_EXPORT(jintArray, getLastAction){
    auto lastAction = callObjMethod(jEnv, jObj, &DoomGameJava::getLastAction);
    return castTojintArray(jEnv, lastAction);
}

JNI_METHOD(jboolean, isNewEpisode, isNewEpisode)
JNI_METHOD(jboolean, isEpisodeFinished, isEpisodeFinished)
JNI_METHOD(jboolean, isPlayerDead, isPlayerDead)
JNI_METHOD(void, respawnPlayer, respawnPlayer)

JNI_EXPORT(jobjectArray, getAvailableButtons){
    auto enums = callObjMethod(jEnv, jObj, &DoomGameJava::getAvailableButtons);
    jclass jEnumClass = jEnv->FindClass("vizdoom/Button");
    if (jEnumClass == 0) return NULL;
    jobjectArray jEnums = jEnv->NewObjectArray(enums.size(), jEnumClass, NULL);
    for(size_t i = 0; i < enums.size(); ++i){
        std::string enumName = buttonToString(enums[i]);
        jfieldID jEnumID = jEnv->GetStaticFieldID(jEnumClass, enumName.c_str(), "Lvizdoom/Button;");
        jobject jEnum = jEnv->GetStaticObjectField(jEnumClass, jEnumID);
        jEnv->SetObjectArrayElement(jEnums, i, jEnum);
    }

    return jEnums;
}

JNI_EXPORT(void, setAvailableButtons, jobjectArray){
    std::vector<Button> arg1;
    int jarg1Len = jEnv->GetArrayLength(jarg1);
    for(int i = 0; i < jarg1Len; ++i){
        jobject jEnum = (jobject)jEnv->GetObjectArrayElement(jarg1, i);
        arg1.push_back(jobjectCastToEnum<Button>(jEnv, "vizdoom/Button", jEnum));
    }
    callObjMethod(jEnv, jObj, &DoomGameJava::setAvailableButtons, arg1);
}

JNI_EXPORT(void, addAvailableButton__Lvizdoom_Button_2, jobject){
    auto arg1 = jobjectCastToEnum<Button>(jEnv, "vizdoom/Button", jarg1);
    callObjMethod(jEnv, jObj, &DoomGameJava::addAvailableButton_btn, arg1);
}

JNI_EXPORT(void, addAvailableButton__Lvizdoom_Button_2I, jobject, jint){
    auto arg1 = jobjectCastToEnum<Button>(jEnv, "vizdoom/Button", jarg1);
    auto arg2 = jintCast(jarg2);
    callObjMethod(jEnv, jObj, &DoomGameJava::addAvailableButton_btn_int, arg1, arg2);
}

JNI_METHOD(void, clearAvailableButtons, clearAvailableButtons)
JNI_METHOD(jint, getAvailableButtonsSize, getAvailableButtonsSize)

JNI_EXPORT(void, setButtonMaxValue, jobject, jint){
    auto arg1 = jobjectCastToEnum<Button>(jEnv, "vizdoom/Button", jarg1);
    auto arg2 = jintCast(jarg2);
    callObjMethod(jEnv, jObj, &DoomGameJava::setButtonMaxValue, arg1, arg2);
}

JNI_EXPORT(jint, getButtonMaxValue, jobject){
    auto arg1 = jobjectCastToEnum<Button>(jEnv, "vizdoom/Button", jarg1);
    return castTojint(callObjMethod(jEnv, jObj, &DoomGameJava::getButtonMaxValue, arg1));
}

JNI_EXPORT(jobjectArray, getAvailableGameVariables){
    auto enums = callObjMethod(jEnv, jObj, &DoomGameJava::getAvailableGameVariables);
    jclass jEnumClass = jEnv->FindClass("vizdoom/GameVariable");
    if (jEnumClass == 0) return NULL;
    jobjectArray jEnums = jEnv->NewObjectArray(enums.size(), jEnumClass, NULL);
    for(size_t i = 0; i < enums.size(); ++i){
        std::string enumName = gameVariableToString(enums[i]);
        jfieldID jEnumID = jEnv->GetStaticFieldID(jEnumClass, enumName.c_str(), "Lvizdoom/GameVariable;");
        jobject jEnum = jEnv->GetStaticObjectField(jEnumClass, jEnumID);
        jEnv->SetObjectArrayElement(jEnums, i, jEnum);
    }

    return jEnums;
}

JNI_EXPORT(void, setAvailableGameVariables, jobjectArray){
    std::vector<GameVariable> arg1;
    int jarg1Len = jEnv->GetArrayLength(jarg1);
    for(int i = 0; i < jarg1Len; ++i){
        jobject jEnum = (jobject)jEnv->GetObjectArrayElement(jarg1, i);
        arg1.push_back(jobjectCastToEnum<GameVariable>(jEnv, "vizdoom/GameVariable", jEnum));
    }
    callObjMethod(jEnv, jObj, &DoomGameJava::setAvailableGameVariables, arg1);
}

JNI_EXPORT(void, addAvailableGameVariable, jobject){
    auto arg1 = jobjectCastToEnum<GameVariable>(jEnv, "vizdoom/GameVariable", jarg1);
    callObjMethod(jEnv, jObj, &DoomGameJava::addAvailableGameVariable, arg1);
}

JNI_EXPORT(jdouble, getGameVariable, jobject){
    auto arg1 = jobjectCastToEnum<GameVariable>(jEnv, "vizdoom/GameVariable", jarg1);
    return castTojdouble(callObjMethod(jEnv, jObj, &DoomGameJava::getGameVariable, arg1));
}

JNI_METHOD(void, clearAvailableGameVariables, clearAvailableGameVariables)
JNI_METHOD(jint, getAvailableGameVariablesSize, getAvailableGameVariablesSize)
JNI_METHOD(void, addGameArgs, addGameArgs, jstring)
JNI_METHOD(void, clearGameArgs, clearGameArgs)
JNI_METHOD(void, sendGameCommand, sendGameCommand, jstring)
JNI_METHOD(jint, getModeNative, getMode)

JNI_EXPORT(void, setMode, jobject){
    auto arg1 = jobjectCastToEnum<Mode>(jEnv, "vizdoom/Mode", jarg1);
    callObjMethod(jEnv, jObj, &DoomGameJava::setMode, arg1);
}

JNI_METHOD(jint, getTicrate, getTicrate)
JNI_METHOD(void, setTicrate, setTicrate, jint)
JNI_METHOD(jdouble, getLivingReward, getLivingReward)
JNI_METHOD(void, setLivingReward, setLivingReward, jdouble)
JNI_METHOD(jdouble, getDeathPenalty, getDeathPenalty)
JNI_METHOD(void, setDeathPenalty, setDeathPenalty, jdouble)
JNI_METHOD(jdouble, getLastReward, getLastReward)
JNI_METHOD(jdouble, getTotalReward, getTotalReward)
JNI_METHOD(void, setViZDoomPath, setViZDoomPath, jstring)
JNI_METHOD(void, setDoomGamePath, setDoomGamePath, jstring)
JNI_METHOD(void, setDoomScenarioPath, setDoomScenarioPath, jstring)
JNI_METHOD(void, setDoomMap, setDoomMap, jstring)
JNI_METHOD(void, setDoomSkill, setDoomSkill, jint)
JNI_METHOD(void, setDoomConfigPath, setDoomConfigPath, jstring)
JNI_METHOD(jint, getSeed, getSeed)
JNI_METHOD(void, setSeed, setSeed, jint)
JNI_METHOD(jint, getEpisodeStartTime, getEpisodeStartTime)
JNI_METHOD(void, setEpisodeStartTime, setEpisodeStartTime, jint)
JNI_METHOD(jint, getEpisodeTimeout, getEpisodeTimeout)
JNI_METHOD(void, setEpisodeTimeout, setEpisodeTimeout, jint)
JNI_METHOD(jint, getEpisodeTime, getEpisodeTime)

JNI_EXPORT(void, setScreenResolution, jobject){
    auto arg1 = jobjectCastToEnum<ScreenResolution>(jEnv, "vizdoom/ScreenResolution", jarg1);
    callObjMethod(jEnv, jObj, &DoomGameJava::setScreenResolution, arg1);
}

JNI_EXPORT(void, setScreenFormat, jobject){
    auto arg1 = jobjectCastToEnum<ScreenFormat>(jEnv, "vizdoom/ScreenFormat", jarg1);
    callObjMethod(jEnv, jObj, &DoomGameJava::setScreenFormat, arg1);
}

JNI_METHOD(jboolean, isDepthBufferEnabled, isDepthBufferEnabled);
JNI_METHOD(void, setDepthBufferEnabled, setDepthBufferEnabled, jboolean);

JNI_METHOD(jboolean, isLabelsBufferEnabled, isLabelsBufferEnabled);
JNI_METHOD(void, setLabelsBufferEnabled, setLabelsBufferEnabled, jboolean);

JNI_METHOD(jboolean, isAutomapBufferEnabled, isAutomapBufferEnabled);
JNI_METHOD(void, setAutomapBufferEnabled, setAutomapBufferEnabled, jboolean);

JNI_EXPORT(void, setAutomapMode, jobject){
    auto arg1 = jobjectCastToEnum<AutomapMode>(jEnv, "vizdoom/AutomapMode", jarg1);
    callObjMethod(jEnv, jObj, &DoomGameJava::setAutomapMode, arg1);
}

JNI_METHOD(void, setAutomapRotate, setAutomapRotate, jboolean);
JNI_METHOD(void, setAutomapRenderTextures, setAutomapRenderTextures, jboolean);

JNI_METHOD(void, setRenderHud, setRenderHud, jboolean)
JNI_METHOD(void, setRenderMinimalHud, setRenderMinimalHud, jboolean)
JNI_METHOD(void, setRenderWeapon, setRenderWeapon, jboolean)
JNI_METHOD(void, setRenderCrosshair, setRenderCrosshair, jboolean)
JNI_METHOD(void, setRenderDecals, setRenderDecals, jboolean)
JNI_METHOD(void, setRenderParticles, setRenderParticles, jboolean)
JNI_METHOD(void, setRenderEffectsSprites, setRenderEffectsSprites, jboolean)
JNI_METHOD(void, setRenderMessages, setRenderMessages, jboolean)
JNI_METHOD(void, setRenderCorpses, setRenderCorpses, jboolean)
JNI_METHOD(void, setWindowVisible, setWindowVisible, jboolean)
JNI_METHOD(void, setConsoleEnabled, setConsoleEnabled, jboolean)
JNI_METHOD(void, setSoundEnabled, setSoundEnabled, jboolean)
JNI_METHOD(jint, getScreenWidth, getScreenWidth)
JNI_METHOD(jint, getScreenHeight, getScreenHeight)
JNI_METHOD(jint, getScreenChannels, getScreenChannels)
JNI_METHOD(jint, getScreenPitch, getScreenPitch)
JNI_METHOD(jint, getScreenSize, getScreenSize)
JNI_METHOD(jint, getScreenFormatNative, getScreenFormat)

JNI_EXPORT(jdouble, doomTicsToMs, jdouble, jint){
    return (jdouble) doomTicsToMs(jarg1, jarg2);
}

JNI_EXPORT(jdouble, msToDoomTics, jdouble, jint){
    return (jdouble) msToDoomTics(jarg1, jarg2);
}

JNI_EXPORT(jdouble, doomTicsToSec, jdouble, jint){
    return (jdouble) doomTicsToSec(jarg1, jarg2);
}

JNI_EXPORT(jdouble, secToDoomTics, jdouble, jint){
    return (jdouble) secToDoomTics(jarg1, jarg2);
}

JNI_EXPORT(jdouble, doomFixedToDouble, jint){
    return (jdouble) doomFixedToDouble(jarg1);
}

JNI_EXPORT(jboolean, isBinaryButton, jobject){
    auto arg1 = jobjectCastToEnum<Button>(jEnv, "vizdoom/Button", jarg1);
    return (jboolean) isBinaryButton(arg1);
}

JNI_EXPORT(jboolean, isDeltaButton, jobject){
    auto arg1 = jobjectCastToEnum<Button>(jEnv, "vizdoom/Button", jarg1);
    return (jboolean) isDeltaButton(arg1);
}
