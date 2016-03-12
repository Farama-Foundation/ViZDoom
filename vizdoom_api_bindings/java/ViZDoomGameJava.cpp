#include "ViZDoomGameJava.h"
#include "ViZDoomGame.h"
#include "ViZDoomDefines.h"
#include "ViZDoomExceptions.h"
#include "ViZDoomUtilities.h"

#include <jni.h>

using namespace vizdoom;

DoomGame* GetObject(JNIEnv *env, jobject obj){
	jclass thisClass = env->GetObjectClass(obj);
	jfieldID fidNumber = env->GetFieldID(thisClass, "internalPtr", "J");
	   if (NULL == fidNumber) return NULL;
	jlong number = env->GetLongField(obj, fidNumber);
	DoomGame *ret;
	ret =(class DoomGame*)number;
	return ret;
}
/*
 * Class:     ViZDoomGameJava
 * Method:    DoomTics2Ms
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_ViZDoomGameJava_DoomTicsToMs
  (JNIEnv *env, jobject obj, jint time){
	int ret=DoomTicsToMs(time);
	return ret;
	}

/*
 * Class:     ViZDoomGameJava
 * Method:    Ms2DoomTics
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_ViZDoomGameJava_MsToDoomTics
  (JNIEnv * env, jobject obj, jint time){
	int ret=MsToDoomTics(time);
	return ret;
}

/*
 * Class:     ViZDoomGameJava
 * Method:    DoomFixedToDouble
 * Signature: (I)D
 */
JNIEXPORT jdouble JNICALL Java_ViZDoomGameJava_DoomFixedToDouble
  (JNIEnv * env, jobject obj, jint time){
	double ret=DoomFixedToDouble(time);
	return ret;
	}

/*
 * Class:     ViZDoomGameJava
 * Method:    isBinaryButton
 * Signature: (Lenums/Button)Z
 */
JNIEXPORT jboolean JNICALL Java_ViZDoomGameJava_isBinaryButton
  (JNIEnv *env, jobject obj, jobject enumVal){
	jclass jclassEnum = env->FindClass("enums/Button");
	if(jclassEnum != 0)
    	{	
        	jmethodID ordinal_ID = env->GetMethodID(jclassEnum, "ordinal", "()I");
		if (ordinal_ID == 0){
			return 0;
		}
		jint value = env->CallIntMethod(enumVal, ordinal_ID);
		Button ret=static_cast<Button>(value);
		bool retval = isBinaryButton(ret);
	// Delete local references created		
        	env->DeleteLocalRef(jclassEnum);
		return retval;
    	}
}

/*
 * Class:     ViZDoomGameJava
 * Method:    isDeltaButton
 * Signature: (Lenums/Button)Z
 */
JNIEXPORT jboolean JNICALL Java_ViZDoomGameJava_isDeltaButton
    (JNIEnv *env, jobject obj, jobject enumVal){
	jclass jclassEnum = env->FindClass("enums/Button");
	if(jclassEnum != 0)
    	{	
        	jmethodID ordinal_ID = env->GetMethodID(jclassEnum, "ordinal", "()I");
		if (ordinal_ID == 0){
			return 0;
		}
		jint value = env->CallIntMethod(enumVal, ordinal_ID);
		Button ret=static_cast<Button>(value);
		bool retval = isDeltaButton(ret);
	// Delete local references created		
        	env->DeleteLocalRef(jclassEnum);
		return retval;
    	}

}

/*
 * Class:     ViZDoomGameJava
 * Method:    DoomGame
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_DoomGame
  (JNIEnv *env, jobject obj){
	jclass thisClass = env->GetObjectClass(obj);
	jfieldID fidNumber = env->GetFieldID(thisClass, "internalPtr", "J");
	   if (NULL == fidNumber) return;
	DoomGame *game=new DoomGame();
	env->SetLongField(obj,fidNumber, (jlong) game );
}


/*
 * Class:     ViZDoomGameJava
 * Method:    loadConfig
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL Java_ViZDoomGameJava_loadConfig
  (JNIEnv *env, jobject obj, jstring path){
	try{
		DoomGame* game=GetObject(env,obj);
	 	char * path2;
	    	path2 = const_cast<char*>(env->GetStringUTFChars(path , NULL )) ;	
		bool ret = game->loadConfig(path2);
		return (jboolean) ret;
	}
	catch (std::exception& e)
  	{
    		std::cout << "C++ unknown exception"<< '\n';
		return 0;
  	}

}

/*
 * Class:     ViZDoomGameJava
 * Method:    init
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_ViZDoomGameJava_init
 (JNIEnv *env, jobject obj){
	DoomGame* game=GetObject(env,obj);
	try
  	{
	bool ret=game->init();
	return (jboolean) ret;
	
	}
  	catch (DoomUnexpectedExitException& e)
  	{	

 		jclass DoomUnexpectedExitException = env->FindClass("errors/DoomUnexpectedExitException");
        	env->ThrowNew(DoomUnexpectedExitException, e.what());
		return 0;
  	}
	catch (PathDoesNotExistException& e)
	{
		jclass PathDoesNotExistException = env->FindClass("errors/PathDoesNotExistException");
			env->ThrowNew(PathDoesNotExistException, e.what());
		return 0;
	}
	catch (SharedMemoryException& e)
  	{	

 		jclass SharedMemoryException = env->FindClass("errors/SharedMemoryException");
        	env->ThrowNew(SharedMemoryException, e.what());
		return 0;
  	}
	catch (MessageQueueException& e)
  	{	

 		jclass MessageQueueException = env->FindClass("errors/MessageQueueException");
        	env->ThrowNew(MessageQueueException, e.what());
		return 0;
  	}
	catch (DoomErrorException& e)
  	{	

 		jclass DoomErrorException = env->FindClass("errors/DoomErrorException");
        	env->ThrowNew(DoomErrorException, e.what());
		return 0;
  	}
	catch (DoomIsNotRunningException& e)
  	{	

 		jclass DoomIsNotRunningException = env->FindClass("errors/DoomIsNotRunningException");
        	env->ThrowNew(DoomIsNotRunningException, e.what());
		return 0;
  	}
	catch (Exception& e)
  	{	

 		jclass Exception = env->FindClass("errors/Exception");
        	env->ThrowNew(Exception, e.what());
		return 0;
  	}
	catch (std::exception& e)
  	{
    		std::cout << "C++ unknown exception"<< '\n';
		return 0;
  	}
}

/*
 * Class:     ViZDoomGameJava
 * Method:    close
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_close
  (JNIEnv *env, jobject obj){
	DoomGame* game=GetObject(env,obj);
	game->close();
}

/*
 * Class:     ViZDoomGameJava
 * Method:    newEpisode
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_newEpisode
    (JNIEnv *env, jobject obj){
	try{
	DoomGame* game=GetObject(env,obj);
	game->newEpisode();
	}
	
	catch (DoomIsNotRunningException& e)
  	{	

 		jclass DoomIsNotRunningException = env->FindClass("errors/DoomIsNotRunningException");
        	env->ThrowNew(DoomIsNotRunningException, e.what());
  	}
	catch (std::exception& e)
  	{
    		std::cout << "C++ unknown exception"<< '\n';
  	}
}

/*
 * Class:     ViZDoomGameJava
 * Method:    isRunning
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_ViZDoomGameJava_isRunning
   (JNIEnv *env, jobject obj){
	DoomGame* game=GetObject(env,obj);
	bool ret=game->isRunning();
	return (jboolean)ret;
}

/*
 * Class:     ViZDoomGameJava
 * Method:    setNextAction
 * Signature: (Ljava/util/ArrayList;)V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_setAction
  (JNIEnv *env, jobject obj, jintArray ourarray)
{
	try {
	DoomGame* game=GetObject(env,obj);
	int NumElts = env->GetArrayLength(ourarray);
	jint *oarr = env->GetIntArrayElements(ourarray, NULL);
	std::vector<int> ourvector;
	for (int i=0;i<NumElts;i++){
		ourvector.push_back(oarr[i]);
	}
	game->setAction(ourvector);
	}
	catch (DoomIsNotRunningException& e)
  	{	

 		jclass DoomIsNotRunningException = env->FindClass("errors/DoomIsNotRunningException");
        	env->ThrowNew(DoomIsNotRunningException, e.what());
  	}
	catch (std::exception& e)
  	{
    		std::cout << "C++ unknown exception"<< '\n';
  	}
}

/*
 * Class:     ViZDoomGameJava
 * Method:    advanceAction
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_advanceAction__
  (JNIEnv *env, jobject obj){
	try{	
	DoomGame* game=GetObject(env,obj);
	game->advanceAction();
	}
	catch (DoomIsNotRunningException& e)
	{
 		jclass DoomIsNotRunningException = env->FindClass("errors/DoomIsNotRunningException");
        	env->ThrowNew(DoomIsNotRunningException, e.what());
  	}
	catch (DoomUnexpectedExitException& e)
	{
		jclass DoomUnexpectedExitException = env->FindClass("errors/DoomUnexpectedExitException");
			env->ThrowNew(DoomUnexpectedExitException, e.what());
	}
	catch (Exception& e)
  	{	

 		jclass Exception = env->FindClass("errors/Exception");
        	env->ThrowNew(Exception, e.what());

  	}
	catch (std::exception& e)
  	{
    		std::cout << "C++ unknown exception"<< '\n';

  	}
}

/*
 * Class:     ViZDoomGameJava
 * Method:    advanceAction
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_advanceAction__I
  (JNIEnv *env, jobject obj, jint int1){
	try{
	DoomGame* game=GetObject(env,obj);
	game->advanceAction(int1);
	}
	catch (DoomIsNotRunningException& e)
  	{	

 		jclass DoomIsNotRunningException = env->FindClass("errors/DoomIsNotRunningException");
        	env->ThrowNew(DoomIsNotRunningException, e.what());
  	}
	catch (DoomUnexpectedExitException& e)
	{
		jclass DoomUnexpectedExitException = env->FindClass("errors/DoomUnexpectedExitException");
			env->ThrowNew(DoomUnexpectedExitException, e.what());
	}
	catch (Exception& e)
  	{	

 		jclass Exception = env->FindClass("errors/Exception");
        	env->ThrowNew(Exception, e.what());

  	}
	catch (std::exception& e)
  	{
    		std::cout << "C++ unknown exception"<< '\n';

  	}
}

/*
 * Class:     ViZDoomGameJava
 * Method:    advanceAction
 * Signature: (IZZ)V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_advanceAction__IZZ
  (JNIEnv *env, jobject obj, jint int1, jboolean bol1, jboolean bol2){
	try{
	DoomGame* game=GetObject(env,obj);
	game->advanceAction(int1, bol1,bol2);
	}
	catch (DoomIsNotRunningException& e)
  	{	

 		jclass DoomIsNotRunningException = env->FindClass("errors/DoomIsNotRunningException");
        	env->ThrowNew(DoomIsNotRunningException, e.what());
  	}
	catch (DoomUnexpectedExitException& e)
	{
		jclass DoomUnexpectedExitException = env->FindClass("errors/DoomUnexpectedExitException");
			env->ThrowNew(DoomUnexpectedExitException, e.what());
	}
	catch (Exception& e)
  	{	

 		jclass Exception = env->FindClass("errors/Exception");
        	env->ThrowNew(Exception, e.what());

  	}
	catch (std::exception& e)
  	{
    		std::cout << "C++ unknown exception"<< '\n';

  	}
}


/*
 * Class:     ViZDoomGameJava
 * Method:    makeAction
 * Signature: ([I)F
 */
JNIEXPORT jdouble JNICALL Java_ViZDoomGameJava_makeAction___3I
  (JNIEnv *env, jobject obj, jintArray ourarray)
{
	try {
	DoomGame* game=GetObject(env,obj);
	int NumElts = env->GetArrayLength(ourarray);
	jint *oarr = env->GetIntArrayElements(ourarray, NULL);
	std::vector<int> ourvector;
	for (int i=0;i<NumElts;i++){
		ourvector.push_back(oarr[i]);
	}
	double ret = game->makeAction(ourvector);
	return ret;	
	}
	catch (DoomIsNotRunningException& e)
  	{	

 		jclass DoomIsNotRunningException = env->FindClass("errors/DoomIsNotRunningException");
        	env->ThrowNew(DoomIsNotRunningException, e.what());
		return 0;
  	}
	catch (DoomUnexpectedExitException& e)
	{
		jclass DoomUnexpectedExitException = env->FindClass("errors/DoomUnexpectedExitException");
			env->ThrowNew(DoomUnexpectedExitException, e.what());
	}
	catch (std::exception& e)
  	{
    		std::cout << "C++ unknown exception"<< '\n';
		return 0;
  	}
}



/*
 * Class:     ViZDoomGameJava
 * Method:    makeAction
 * Signature: ([II)F
 */
JNIEXPORT jdouble JNICALL Java_ViZDoomGameJava_makeAction___3II
  (JNIEnv *env, jobject obj, jintArray ourarray, jint integ)
{
	try {
	DoomGame* game=GetObject(env,obj);
	int NumElts = env->GetArrayLength(ourarray);
	jint *oarr = env->GetIntArrayElements(ourarray, NULL);
	std::vector<int> ourvector;
	for (int i=0;i<NumElts;i++){
		ourvector.push_back(oarr[i]);
	}
	double ret = game->makeAction(ourvector, integ);
	return ret;
	}
	catch (DoomIsNotRunningException& e)
  	{	

 		jclass DoomIsNotRunningException = env->FindClass("errors/DoomIsNotRunningException");
        	env->ThrowNew(DoomIsNotRunningException, e.what());
		return 0;
  	}
	catch (DoomUnexpectedExitException& e)
	{
		jclass DoomUnexpectedExitException = env->FindClass("errors/DoomUnexpectedExitException");
			env->ThrowNew(DoomUnexpectedExitException, e.what());
	}
	catch (std::exception& e)
  	{
    		std::cout << "C++ unknown exception"<< '\n';
		return 0;
  	}
}

/*
 * Class:     ViZDoomGameJava
 * Method:    getState
 * Signature: ()LGameState;
 */
JNIEXPORT jobject JNICALL Java_ViZDoomGameJava_getState
  (JNIEnv *env, jobject obj){
	DoomGame* game=GetObject(env,obj);
	jclass state = env->FindClass("GameState");	
	int rozmiar=game->getScreenSize();	
	std::vector<int> ourvector;
	GameState statec=game->getState();
	
	ourvector=statec.gameVariables;
	jintArray jbuffer = env->NewIntArray(ourvector.size());
	jint *oarr = env->GetIntArrayElements(jbuffer, NULL);
	
	for (int i=0;i<ourvector.size();i++){
		oarr[i]=ourvector[i];
		
	}
	env->ReleaseIntArrayElements(jbuffer, oarr, NULL);

	uint8_t *pointer;
	pointer=statec.imageBuffer;
	jintArray jbuffer2 = env->NewIntArray(rozmiar);
	oarr = env->GetIntArrayElements(jbuffer2, NULL);

	for (int i=0;i<rozmiar;i++){
		oarr[i]=(int)*(pointer+i);

	}
	env->ReleaseIntArrayElements(jbuffer2, oarr, NULL);	

	jmethodID constructor = env->GetMethodID(state, "<init>", "(I[I[I)V");
	jobject result = env->NewObject(state, constructor, statec.number,jbuffer, jbuffer2);
	return result;


}

/*
 * Class:     ViZDoomGameJava
 * Method:    getLastAction
 */
JNIEXPORT jintArray JNICALL Java_ViZDoomGameJava_getLastAction
  (JNIEnv *env, jobject obj){
	std::vector<int> ourvector;
	DoomGame* game=GetObject(env,obj);
	ourvector = game->getLastAction();
	jintArray bob=env->NewIntArray(ourvector.size());
	jint *oarr = env->GetIntArrayElements(bob, NULL);
	for (int i=0;i<ourvector.size();i++){
	oarr[i]=ourvector.at(i);
	}
 	env->ReleaseIntArrayElements(bob, oarr, NULL);
	return bob;
	
	
}

/*
 * Class:     ViZDoomGameJava
 * Method:    isNewEpisode
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_ViZDoomGameJava_isNewEpisode
  (JNIEnv *env, jobject obj){
	try{
		DoomGame* game=GetObject(env,obj);
		bool ret=game->isNewEpisode();
		return (jboolean)ret;
	}
	catch (DoomIsNotRunningException& e)
  	{	

 		jclass DoomIsNotRunningException = env->FindClass("errors/DoomIsNotRunningException");
        	env->ThrowNew(DoomIsNotRunningException, e.what());
		return 0;
  	}
	catch (Exception& e)
  	{	

 		jclass Exception = env->FindClass("errors/Exception");
        	env->ThrowNew(Exception, e.what());
		return 0;
  	}
	catch (std::exception& e)
  	{
    		std::cout << "C++ unknown exception"<< '\n';
		return 0;
  	}
}

/*
 * Class:     ViZDoomGameJava
 * Method:    isEpisodeFinished
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_ViZDoomGameJava_isEpisodeFinished
  (JNIEnv *env, jobject obj){
	try{
		DoomGame* game=GetObject(env,obj);
		bool ret=game->isEpisodeFinished();
		return (jboolean)ret;
	}
	catch (DoomIsNotRunningException& e)
  	{	

 		jclass DoomIsNotRunningException = env->FindClass("errors/DoomIsNotRunningException");
        	env->ThrowNew(DoomIsNotRunningException, e.what());
		return 0;
  	}
	catch (Exception& e)
  	{	

 		jclass Exception = env->FindClass("errors/Exception");
        	env->ThrowNew(Exception, e.what());
		return 0;
  	}
	catch (std::exception& e)
  	{
    		std::cout << "C++ unknown exception"<< '\n';
		return 0;
  	}
}

/*
 * Class:     ViZDoomGameJava
 * Method:    isPlayerDead
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_ViZDoomGameJava_isPlayerDead
  (JNIEnv *env, jobject obj){
	try{
		DoomGame* game=GetObject(env,obj);
		bool ret=game->isPlayerDead();
		return (jboolean)ret;
	}
	catch (DoomIsNotRunningException& e)
  	{	

 		jclass DoomIsNotRunningException = env->FindClass("errors/DoomIsNotRunningException");
        	env->ThrowNew(DoomIsNotRunningException, e.what());
		return 0;
  	}
	catch (Exception& e)
  	{	

 		jclass Exception = env->FindClass("errors/Exception");
        	env->ThrowNew(Exception, e.what());
		return 0;
  	}
	catch (std::exception& e)
  	{
    		std::cout << "C++ unknown exception"<< '\n';
		return 0;
  	}
}

/*
 * Class:     ViZDoomGameJava
 * Method:    respawnPlayer
 * Signature: ()Z
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_respawnPlayer
  (JNIEnv *env, jobject obj){
		DoomGame* game=GetObject(env,obj);
		game->respawnPlayer();

}
/*
 * Class:     ViZDoomGameJava
 * Method:    addAvailableButton
 * Signature: (Lenums/Button;)V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_addAvailableButton__Lenums_Button_2
  (JNIEnv *env, jobject obj, jobject enumVal){
	DoomGame* game=GetObject(env,obj);
	jclass jclassEnum = env->FindClass("enums/Button");
	if(jclassEnum != 0)
    	{	
        	jmethodID ordinal_ID = env->GetMethodID(jclassEnum, "ordinal", "()I");
		if (ordinal_ID == 0){
			return;
		}
		jint value = env->CallIntMethod(enumVal, ordinal_ID);
		Button ret=static_cast<Button>(value);
		game->addAvailableButton(ret);
	// Delete local references created		
        	env->DeleteLocalRef(jclassEnum);
    	}

}

/*
 * Class:     ViZDoomGameJava
 * Method:    addAvailableButton
 * Signature: (Lenums/Button;I)V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_addAvailableButton__Lenums_Button_2I
  (JNIEnv *env, jobject obj, jobject enumVal, jint intval){
	DoomGame* game=GetObject(env,obj);
	jclass jclassEnum = env->FindClass("enums/Button");
	if(jclassEnum != 0)
    	{	
        	jmethodID ordinal_ID = env->GetMethodID(jclassEnum, "ordinal", "()I");
		if (ordinal_ID == 0){
			return;
		}
		jint value = env->CallIntMethod(enumVal, ordinal_ID);
		Button ret=static_cast<Button>(value);
		game->addAvailableButton(ret, intval);
	// Delete local references created		
        	env->DeleteLocalRef(jclassEnum);
    	}
}


/*
 * Class:     ViZDoomGameJava
 * Method:    clearAvailableButtons
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_clearAvailableButtons 
  (JNIEnv *env, jobject obj){
	DoomGame* game=GetObject(env,obj);
	game->clearAvailableButtons();
};

/*
 * Class:     ViZDoomGameJava
 * Method:    getAvailableButtonsSize
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_ViZDoomGameJava_getAvailableButtonsSize
  (JNIEnv *env, jobject obj){
	DoomGame* game=GetObject(env,obj);
	int ret=game->getAvailableButtonsSize();
	return ret;
}

/*
 * Class:     ViZDoomGameJava
 * Method:    setButtonMaxValue
 * Signature: (Lenums/Button;I)V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_setButtonMaxValue
  (JNIEnv *env, jobject obj, jobject enumVal, jint intval){
	DoomGame* game=GetObject(env,obj);
	jclass jclassEnum = env->FindClass("enums/Button");
	if(jclassEnum != 0)
    	{	
        	jmethodID ordinal_ID = env->GetMethodID(jclassEnum, "ordinal", "()I");
		if (ordinal_ID == 0){
			return;
		}
		jint value = env->CallIntMethod(enumVal, ordinal_ID);
		Button ret=static_cast<Button>(value);
		game->setButtonMaxValue(ret, intval);
	// Delete local references created		
        	env->DeleteLocalRef(jclassEnum);
    	}
}
/*
 * Class:     ViZDoomGameJava
 * Method:    getButtonMaxValue
 * Signature: (Lenums/Button)V
 */
JNIEXPORT jint JNICALL Java_ViZDoomGameJava_getButtonMaxValue
  (JNIEnv *env, jobject obj, jobject enumVal){
DoomGame* game=GetObject(env,obj);
	jclass jclassEnum = env->FindClass("enums/Button");
	if(jclassEnum != 0)
    	{	
        	jmethodID ordinal_ID = env->GetMethodID(jclassEnum, "ordinal", "()I");
		if (ordinal_ID == 0){
			return 0;
		}
		jint value = env->CallIntMethod(enumVal, ordinal_ID);
		Button ret=static_cast<Button>(value);
		int retval = game->getButtonMaxValue(ret);
	// Delete local references created		
        	env->DeleteLocalRef(jclassEnum);
		return retval;
    	}
}


/*
 * Class:     ViZDoomGameJava
 * Method:    addAvailableGameVariable
 * Signature: (Lenums/GameVariable;)V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_addAvailableGameVariable
  (JNIEnv *env, jobject obj, jobject enumVal){
	DoomGame* game=GetObject(env,obj);
	jclass jclassEnum = env->FindClass("enums/GameVariable");
	if(jclassEnum != 0)
    	{	
        	jmethodID ordinal_ID = env->GetMethodID(jclassEnum, "ordinal", "()I");
		if (ordinal_ID == 0){
			return;
		}
		jint value = env->CallIntMethod(enumVal, ordinal_ID);
		GameVariable ret=static_cast<GameVariable>(value);
		game->addAvailableGameVariable(ret);
	// Delete local references created		
        	env->DeleteLocalRef(jclassEnum);
    	}

}

/*
 * Class:     ViZDoomGameJava
 * Method:    clearAvailableGameVariables
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_clearAvailableGameVariables
  (JNIEnv *env, jobject obj){
	DoomGame* game=GetObject(env,obj);
	game->clearAvailableGameVariables();

}

/*
 * Class:     ViZDoomGameJava
 * Method:    getAvailableGameVariablesSize
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_ViZDoomGameJava_getAvailableGameVariablesSize
  (JNIEnv *env, jobject obj){
	DoomGame* game=GetObject(env,obj);
	int ret=game->getAvailableGameVariablesSize();
	return ret;
}

/*
 * Class:     ViZDoomGameJava
 * Method:    addGameArgs
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_addGameArgs
  (JNIEnv *env, jobject obj , jstring str){
	DoomGame* game=GetObject(env,obj);
	char * str2;
    	str2 = const_cast<char*>(env->GetStringUTFChars(str , NULL )) ;
	game->addGameArgs(str2);
}

/*
 * Class:     ViZDoomGameJava
 * Method:    clearCustomGameArgs
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_clearGameArgs
  (JNIEnv *env, jobject obj){
	DoomGame* game=GetObject(env,obj);
	game->clearGameArgs();
}

/*
 * Class:     ViZDoomGameJava
 * Method:    sendGameCommand
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_sendGameCommand //TODO wywala jvm
  (JNIEnv *env, jobject obj, jstring str){
	try{	
		DoomGame* game=GetObject(env,obj);
		char * str2;
	    	str2 = const_cast<char*>(env->GetStringUTFChars(str , NULL )) ;	
		game->sendGameCommand(str2);
		
	}
	catch (DoomIsNotRunningException& e)
  	{	
 		jclass DoomIsNotRunningException = env->FindClass("errors/DoomIsNotRunningException");
        	env->ThrowNew(DoomIsNotRunningException, e.what());

  	}
	catch (Exception& e)
  	{	
 		jclass Exception = env->FindClass("errors/Exception");
        	env->ThrowNew(Exception, e.what());

  	}
	catch (...)
  	{
    		std::cout << "C++ unknown exception"<<std::endl;

  	}
}

/*
 * Class:     ViZDoomGameJava
 * Method:    getGameScreen
 * Signature: ()Ljava/util/ArrayList;
 */
JNIEXPORT jintArray JNICALL Java_ViZDoomGameJava_getGameScreen
  (JNIEnv *env, jobject obj){
	DoomGame* game=GetObject(env,obj);
	int rozmiar=game->getScreenSize();	
	std::vector<int> ourvector;

	uint8_t *pointer;
	pointer=game->getGameScreen();
	jintArray jbuffer = env->NewIntArray(rozmiar);
	jint *oarr;
	oarr = env->GetIntArrayElements(jbuffer, NULL);

	for (int i=0;i<rozmiar;i++){
		oarr[i]=(int)*(pointer+i);

	}
	env->ReleaseIntArrayElements(jbuffer, oarr, NULL);
	return jbuffer;




}

/*
 * Class:     ViZDoomGameJava
 * Method:    getMode
 * Signature: ()Lenums/Mode;
 */
JNIEXPORT jint JNICALL Java_ViZDoomGameJava_getMod
  (JNIEnv *env, jobject obj){
	DoomGame* game=GetObject(env,obj);
	Mode mode=game->getMode();
	return (jint) mode;
}

/*
 * Class:     ViZDoomGameJava
 * Method:    setGMode
 * Signature: (Lenums/Mode;)V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_setMode
  (JNIEnv *env, jobject obj, jobject enumVal){
	DoomGame* game=GetObject(env,obj);
	jclass jclassEnum = env->FindClass("enums/Mode");
	
	if(jclassEnum != 0)
    	{	
        	jmethodID ordinal_ID = env->GetMethodID(jclassEnum, "ordinal", "()I");
		if (ordinal_ID == 0){
			return;
		}
		jint value = env->CallIntMethod(enumVal, ordinal_ID);
		Mode ret=static_cast<Mode>(value);
		game->setMode(ret);
	// Delete local references created		
        	env->DeleteLocalRef(jclassEnum);
    	}
}

/*
 * Class:     ViZDoomGameJava
 * Method:    getGameVariable
 * Signature: (Lenums/GameVariable;)I
 */
JNIEXPORT jint JNICALL Java_ViZDoomGameJava_getGameVariable
  (JNIEnv *env, jobject obj, jobject enumVal){
	DoomGame* game=GetObject(env,obj);
	jclass jclassEnum = env->FindClass("enums/GameVariable");
	
	if(jclassEnum != 0)
    	{	
        	jmethodID ordinal_ID = env->GetMethodID(jclassEnum, "ordinal", "()I");
		if (ordinal_ID == 0){
			return -1;
		}
		jint value = env->CallIntMethod(enumVal, ordinal_ID);
		GameVariable ret=static_cast<GameVariable>(value);
		int retint=game->getGameVariable(ret);
	// Delete local references created		
        	env->DeleteLocalRef(jclassEnum);
		return retint;
    	}

}

/*
 * Class:     ViZDoomGameJava
 * Method:    getLivingReward
 * Signature: ()F
 */
JNIEXPORT jdouble JNICALL Java_ViZDoomGameJava_getLivingReward
 (JNIEnv *env, jobject obj){
	DoomGame* game=GetObject(env,obj);
	double ret = game->getLivingReward();
	return ret;
}


/*
 * Class:     ViZDoomGameJava
 * Method:    setLivingReward
 * Signature: (F)V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_setLivingReward
(JNIEnv *env, jobject obj, jdouble rew){
	DoomGame* game=GetObject(env,obj);
	game->setLivingReward(rew);
}

/*
 * Class:     ViZDoomGameJava
 * Method:    getDeathPenalty
 * Signature: ()F
 */
JNIEXPORT jdouble JNICALL Java_ViZDoomGameJava_getDeathPenalty
 (JNIEnv *env, jobject obj){
	DoomGame* game=GetObject(env,obj);
	double ret = game->getDeathPenalty();
	return ret;
}

/*
 * Class:     ViZDoomGameJava
 * Method:    setDeathPenalty
 * Signature: (F)V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_setDeathPenalty
  (JNIEnv *env, jobject obj, jdouble rew){
	DoomGame* game=GetObject(env,obj);
	game->setDeathPenalty(rew);
}

/*
 * Class:     ViZDoomGameJava
 * Method:    getLastReward
 * Signature: ()F
 */
JNIEXPORT jdouble JNICALL Java_ViZDoomGameJava_getLastReward
  (JNIEnv *env, jobject obj){
	DoomGame* game=GetObject(env,obj);
	double ret=game->getLastReward();
	return ret;

}

/*
 * Class:     ViZDoomGameJava
 * Method:    getSummaryReward
 * Signature: ()F
 */
JNIEXPORT jdouble JNICALL Java_ViZDoomGameJava_getSummaryReward
  (JNIEnv *env, jobject obj){
	DoomGame* game=GetObject(env,obj);
	double ret=game->getSummaryReward();
	return ret;

}

/*
 * Class:     ViZDoomGameJava
 * Method:    setDoomEnginePath
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_setDoomEnginePath
 (JNIEnv *env, jobject obj, jstring path){
	try{
		DoomGame* game=GetObject(env,obj);
	 	char * path2;
	    	path2 = const_cast<char*>(env->GetStringUTFChars(path , NULL )) ;	
		game->setDoomEnginePath(path2);
	}
	catch (...)
  	{
    		std::cout << "C++ unknown exception"<<std::endl;

  	}
}
/*
 * Class:     ViZDoomGameJava
 * Method:    setDoomGamePath
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_setDoomGamePath
  (JNIEnv *env, jobject obj, jstring path){
	try{	
		DoomGame* game=GetObject(env,obj);
		char * path2;
	    	path2 = const_cast<char*>(env->GetStringUTFChars(path , NULL )) ;
		game->setDoomGamePath(path2);
	}
	catch (...)
  	{
    		std::cout << "C++ unknown exception"<<std::endl;

  	}
}

/*
 * Class:     ViZDoomGameJava
 * Method:    setDoomScenarioPath
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_setDoomScenarioPath
 (JNIEnv *env, jobject obj, jstring path){
	try{
		DoomGame* game=GetObject(env,obj);
		char * path2;
	    	path2 = const_cast<char*>(env->GetStringUTFChars(path , NULL )) ;
		game->setDoomScenarioPath(path2);
	}
	catch (...)
  	{
    		std::cout << "C++ unknown exception"<<std::endl;

  	}
}

/*
 * Class:     ViZDoomGameJava
 * Method:    setDoomMap
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_setDoomMap
  (JNIEnv *env, jobject obj, jstring map){
	DoomGame* game=GetObject(env,obj);
	char * path;
    	path = const_cast<char*>(env->GetStringUTFChars(map , NULL )) ;
	game->setDoomMap(path);

}


/*
 * Class:     ViZDoomGameJava
 * Method:    setDoomSkill
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_setDoomSkill
  (JNIEnv *env, jobject obj, jint skill){
	DoomGame* game=GetObject(env,obj);
	game->setDoomSkill(skill);
}


/*
 * Class:     ViZDoomGameJava
 * Method:    setDoomConfigPath
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_setDoomConfigPath
(JNIEnv *env, jobject obj, jstring path){
	DoomGame* game=GetObject(env,obj);
 	char * path2;
    	path2 = const_cast<char*>(env->GetStringUTFChars(path , NULL )) ;	
	game->setDoomConfigPath(path2);
}

/*
 * Class:     ViZDoomGameJava
 * Method:    getSeed
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_ViZDoomGameJava_getSeed
  (JNIEnv *env, jobject obj){
	DoomGame* game=GetObject(env,obj);
	int ret = game->getSeed();
	return (jint)ret;
}

/*
 * Class:     ViZDoomGameJava
 * Method:    setSeed
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_setSeed
  (JNIEnv *env, jobject obj, jint seed){
	DoomGame* game=GetObject(env,obj);
	game->setSeed(seed);
}


/*
 * Class:     ViZDoomGameJava
 * Method:    getEpisodeStartTime
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_ViZDoomGameJava_getEpisodeStartTime
  (JNIEnv *env, jobject obj){
	DoomGame* game=GetObject(env,obj);
	int ret = game->getEpisodeStartTime();
	return (jint) ret;
}

/*
 * Class:     ViZDoomGameJava
 * Method:    setEpisodeStartTime
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_setEpisodeStartTime
  (JNIEnv *env, jobject obj, jint tics){
	DoomGame* game=GetObject(env,obj);
	game->setEpisodeTimeout(tics);

}

/*
 * Class:     ViZDoomGameJava
 * Method:    getEpisodeTimeout
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_ViZDoomGameJava_getEpisodeTimeout
    (JNIEnv *env, jobject obj){
	DoomGame* game=GetObject(env,obj);
	int ret = game->getEpisodeTimeout();
	return ret;
}
/*
 * Class:     ViZDoomGameJava
 * Method:    setEpisodeTimeout
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_setEpisodeTimeout
  (JNIEnv *env, jobject obj, jint tics){
	DoomGame* game=GetObject(env,obj);
	game->setEpisodeTimeout(tics);

}

/*
 * Class:     ViZDoomGameJava
 * Method:    getEpisodeTime
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_ViZDoomGameJava_getEpisodeTime
    (JNIEnv *env, jobject obj){
	DoomGame* game=GetObject(env,obj);
	int ret = game->getEpisodeTime();
	return ret;
}

/*
 * Class:     ViZDoomGameJava
 * Method:    setScreenResolution
 * Signature: (II)V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_setScreenResolution
 (JNIEnv *env, jobject obj, jobject enumVal){
	DoomGame* game=GetObject(env,obj);
	jclass jclassEnum = env->FindClass("enums/ScreenResolution");
	
	if(jclassEnum != 0)
    	{	
        	jmethodID ordinal_ID = env->GetMethodID(jclassEnum, "ordinal", "()I");
		if (ordinal_ID == 0){
			return;
		}
		jint value = env->CallIntMethod(enumVal, ordinal_ID);
		ScreenResolution ret=static_cast<ScreenResolution>(value);
		game->setScreenResolution(ret);
	// Delete local references created		
        	env->DeleteLocalRef(jclassEnum);
    	}

}
	


/*
 * Class:     ViZDoomGameJava
 * Method:    setScreenFormat
 * Signature: (Lenums/ScreenFormat;)V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_setScreenFormat
  (JNIEnv *env, jobject obj, jobject enumVal){
	DoomGame* game=GetObject(env,obj);
	jclass jclassEnum = env->FindClass("enums/ScreenFormat");
	
	if(jclassEnum != 0)
    	{	
        	jmethodID ordinal_ID = env->GetMethodID(jclassEnum, "ordinal", "()I");
		if (ordinal_ID == 0){
			return;
		}
		jint value = env->CallIntMethod(enumVal, ordinal_ID);
		ScreenFormat ret=static_cast<ScreenFormat>(value);
		game->setScreenFormat(ret);
	// Delete local references created		
        	env->DeleteLocalRef(jclassEnum);
    	}

}

/*
 * Class:     ViZDoomGameJava
 * Method:    setRenderHud
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_setRenderHud
  (JNIEnv *env, jobject obj, jboolean bol){
	DoomGame* game=GetObject(env,obj);
	game->setRenderHud(bol);
}

/*
 * Class:     ViZDoomGameJava
 * Method:    setRenderWeapon
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_setRenderWeapon
  (JNIEnv *env, jobject obj, jboolean bol){
	DoomGame* game=GetObject(env,obj);
	game->setRenderWeapon(bol);

}

/*
 * Class:     ViZDoomGameJava
 * Method:    setRenderCrosshair
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_setRenderCrosshair
  (JNIEnv *env , jobject obj, jboolean bol){
	DoomGame* game=GetObject(env,obj);
	game->setRenderCrosshair(bol);
}

/*
 * Class:     ViZDoomGameJava
 * Method:    setRenderDecals
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_setRenderDecals
 (JNIEnv *env, jobject obj, jboolean bol){
	DoomGame* game=GetObject(env,obj);
	game->setRenderDecals(bol);
}

/*
 * Class:     ViZDoomGameJava
 * Method:    setRenderParticles
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_setRenderParticles
  (JNIEnv *env, jobject obj, jboolean bol){
	DoomGame* game=GetObject(env,obj);
	game->setRenderParticles(bol);

}

/*
 * Class:     ViZDoomGameJava
 * Method:    setVisibleWindow
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_setWindowVisible
  (JNIEnv *env, jobject obj, jboolean bol){
	DoomGame* game=GetObject(env,obj);
	game->setWindowVisible(bol);
}

/*
 * Class:     ViZDoomGameJava
 * Method:    setDisabledConsole
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL Java_ViZDoomGameJava_setConsoleEnabled
  (JNIEnv *env, jobject obj, jboolean bol){
	DoomGame* game=GetObject(env,obj);
	game->setConsoleEnabled(bol);

}

/*
 * Class:     ViZDoomGameJava
 * Method:    getScreenWidth
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_ViZDoomGameJava_getScreenWidth
  (JNIEnv *env, jobject obj){
	DoomGame* game=GetObject(env,obj);
	int ret;
	ret=game->getScreenWidth();
	return (jint)ret;
}

/*
 * Class:     ViZDoomGameJava
 * Method:    getScreenHeight
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_ViZDoomGameJava_getScreenHeight
  (JNIEnv *env, jobject obj){
	DoomGame* game=GetObject(env,obj);
	int ret;
	ret=game->getScreenHeight();
	return (jint)ret;
}
/*
 * Class:     ViZDoomGameJava
 * Method:    getScreenChannels
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_ViZDoomGameJava_getScreenChannels
   (JNIEnv *env, jobject obj){
	DoomGame* game=GetObject(env,obj);
	int ret;
	ret=game->getScreenChannels();
	return (jint)ret;

}

/*
 * Class:     ViZDoomGameJava
 * Method:    getScreenPitch
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_ViZDoomGameJava_getScreenPitch
  (JNIEnv *env, jobject obj){
	DoomGame* game=GetObject(env,obj);
	int ret;
	ret=game->getScreenPitch();
	return (jint)ret;

}

/*
 * Class:     ViZDoomGameJava
 * Method:    getScreenSize
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_ViZDoomGameJava_getScreenSize
  (JNIEnv *env, jobject obj){
	DoomGame* game=GetObject(env,obj);
	int ret;
	ret=game->getScreenSize();
	return (jint)ret;


}

/*
 * Class:     ViZDoomGameJava
 * Method:    getScreenFormat
 * Signature: ()Lenums/ScreenFormat;
 */
JNIEXPORT jint JNICALL Java_ViZDoomGameJava_getScreenForma
  (JNIEnv *env, jobject obj){
	DoomGame* game=GetObject(env,obj);
	ScreenFormat ret = game->getScreenFormat();
	return (jint) ret;
}


