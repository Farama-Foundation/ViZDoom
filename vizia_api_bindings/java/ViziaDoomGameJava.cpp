#include <jni.h>
#include "ViziaDoomGameJava.h"
#include "../../vizia_api_src/ViziaDoomGame.h"
#include <unistd.h>
#include "../../vizia_api_src/ViziaDefines.h"

Vizia::DoomGame* GetObject(JNIEnv *env, jobject obj){
	jclass thisClass = env->GetObjectClass(obj);
	jfieldID fidNumber = env->GetFieldID(thisClass, "internalPtr", "J");
	   if (NULL == fidNumber) return NULL;
	jlong number = env->GetLongField(obj, fidNumber);
	Vizia::DoomGame *ret;
	ret =(class Vizia::DoomGame*)number;
	return ret;
}
/*
 * Class:     ViziaDoomGameJava
 * Method:    DoomTics2Ms
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_ViziaDoomGameJava_DoomTics2Ms
  (JNIEnv *env, jobject obj, jint time){
	int ret=Vizia::DoomTics2Ms(time);
	return ret;
	}

/*
 * Class:     ViziaDoomGameJava
 * Method:    Ms2DoomTics
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_ViziaDoomGameJava_Ms2DoomTics
  (JNIEnv * env, jobject obj, jint time){
	int ret=Vizia::Ms2DoomTics(time);
	return ret;
}

/*
 * Class:     ViziaDoomGameJava
 * Method:    DoomFixedToFloat
 * Signature: (I)F
 */
JNIEXPORT jdouble JNICALL Java_ViziaDoomGameJava_DoomFixedToDouble
  (JNIEnv * env, jobject obj, jint time){
	double ret=Vizia::DoomFixedToDouble(time);
	return ret;
	}

/*
 * Class:     ViziaDoomGameJava
 * Method:    DoomGame
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_DoomGame
  (JNIEnv *env, jobject obj){
	jclass thisClass = env->GetObjectClass(obj);
	jfieldID fidNumber = env->GetFieldID(thisClass, "internalPtr", "J");
	   if (NULL == fidNumber) return;
	Vizia::DoomGame *game=new Vizia::DoomGame();
	env->SetLongField(obj,fidNumber, (jlong) game );
}


/*
 * Class:     ViziaDoomGameJava
 * Method:    loadConfig
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL Java_ViziaDoomGameJava_loadConfig
  (JNIEnv *env, jobject obj, jstring path){
	Vizia::DoomGame* game=GetObject(env,obj);
 	char * path2;
    	path2 = const_cast<char*>(env->GetStringUTFChars(path , NULL )) ;	
	bool ret = game->loadConfig(path2);
	return (jboolean) ret;

}

/*
 * Class:     ViziaDoomGameJava
 * Method:    init
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_ViziaDoomGameJava_init
 (JNIEnv *env, jobject obj){
	Vizia::DoomGame* game=GetObject(env,obj);
	try
  	{
	bool ret=game->init();
	return (jboolean) ret;
	
	}
  	catch (Vizia::DoomUnexpectedExitException& e)
  	{	

 		jclass DoomUnexpectedExitException = env->FindClass("errors/DoomUnexpectedExitException");
        	env->ThrowNew(DoomUnexpectedExitException, e.what());
		return 0;
  	}
	catch (Vizia::SharedMemoryException& e)
  	{	

 		jclass SharedMemoryException = env->FindClass("errors/SharedMemoryException");
        	env->ThrowNew(SharedMemoryException, e.what());
		return 0;
  	}
	catch (Vizia::MessageQueueException& e)
  	{	

 		jclass MessageQueueException = env->FindClass("errors/MessageQueueException");
        	env->ThrowNew(MessageQueueException, e.what());
		return 0;
  	}
	catch (Vizia::DoomErrorException& e)
  	{	

 		jclass DoomErrorException = env->FindClass("errors/DoomErrorException");
        	env->ThrowNew(DoomErrorException, e.what());
		return 0;
  	}
	catch (Vizia::DoomIsNotRunningException& e)
  	{	

 		jclass DoomIsNotRunningException = env->FindClass("errors/DoomIsNotRunningException");
        	env->ThrowNew(DoomIsNotRunningException, e.what());
		return 0;
  	}
	catch (Vizia::Exception& e)
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
 * Class:     ViziaDoomGameJava
 * Method:    close
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_close
  (JNIEnv *env, jobject obj){
	Vizia::DoomGame* game=GetObject(env,obj);
	game->close();
}

/*
 * Class:     ViziaDoomGameJava
 * Method:    newEpisode
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_newEpisode
    (JNIEnv *env, jobject obj){
	try{
	Vizia::DoomGame* game=GetObject(env,obj);
	game->newEpisode();
	}
	
	catch (Vizia::DoomIsNotRunningException& e)
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
 * Class:     ViziaDoomGameJava
 * Method:    isRunning
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_ViziaDoomGameJava_isRunning
   (JNIEnv *env, jobject obj){
	Vizia::DoomGame* game=GetObject(env,obj);
	bool ret=game->isRunning();
	return (jboolean)ret;
}

/*
 * Class:     ViziaDoomGameJava
 * Method:    setNextAction
 * Signature: (Ljava/util/ArrayList;)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_setAction
  (JNIEnv *env, jobject obj, jintArray ourarray)
{
	try {
	Vizia::DoomGame* game=GetObject(env,obj);
	int NumElts = env->GetArrayLength(ourarray);
	jint *oarr = env->GetIntArrayElements(ourarray, NULL);
	std::vector<int> ourvector;
	for (int i=0;i<NumElts;i++){
		ourvector.push_back(oarr[i]);
	}
	game->setAction(ourvector);
	}
	catch (Vizia::DoomIsNotRunningException& e)
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
 * Class:     ViziaDoomGameJava
 * Method:    advanceAction
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_advanceAction__
  (JNIEnv *env, jobject obj){
	try{	
	Vizia::DoomGame* game=GetObject(env,obj);
	game->advanceAction();
	}
	catch (Vizia::DoomIsNotRunningException& e)
  	{	

 		jclass DoomIsNotRunningException = env->FindClass("errors/DoomIsNotRunningException");
        	env->ThrowNew(DoomIsNotRunningException, e.what());
  	}
	catch (Vizia::Exception& e)
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
 * Class:     ViziaDoomGameJava
 * Method:    advanceAction
 * Signature: (ZZI)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_advanceAction__ZZI
  (JNIEnv *env, jobject obj, jboolean bol1, jboolean bol2, jint int1){
	try{
	Vizia::DoomGame* game=GetObject(env,obj);
	game->advanceAction(bol1,bol2,int1);
	}
	catch (Vizia::DoomIsNotRunningException& e)
  	{	

 		jclass DoomIsNotRunningException = env->FindClass("errors/DoomIsNotRunningException");
        	env->ThrowNew(DoomIsNotRunningException, e.what());
  	}
	catch (Vizia::Exception& e)
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
 * Class:     ViziaDoomGameJava
 * Method:    makeAction
 * Signature: ([I)F
 */
JNIEXPORT jdouble JNICALL Java_ViziaDoomGameJava_makeAction___3I
  (JNIEnv *env, jobject obj, jintArray ourarray)
{
	try {
	Vizia::DoomGame* game=GetObject(env,obj);
	int NumElts = env->GetArrayLength(ourarray);
	jint *oarr = env->GetIntArrayElements(ourarray, NULL);
	std::vector<int> ourvector;
	for (int i=0;i<NumElts;i++){
		ourvector.push_back(oarr[i]);
	}
	double ret = game->makeAction(ourvector);
	return ret;	
	}
	catch (Vizia::DoomIsNotRunningException& e)
  	{	

 		jclass DoomIsNotRunningException = env->FindClass("errors/DoomIsNotRunningException");
        	env->ThrowNew(DoomIsNotRunningException, e.what());
		return 0;
  	}
	catch (std::exception& e)
  	{
    		std::cout << "C++ unknown exception"<< '\n';
		return 0;
  	}
}



/*
 * Class:     ViziaDoomGameJava
 * Method:    makeAction
 * Signature: ([II)F
 */
JNIEXPORT jdouble JNICALL Java_ViziaDoomGameJava_makeAction___3II
  (JNIEnv *env, jobject obj, jintArray ourarray, jint integ)
{
	try {
	Vizia::DoomGame* game=GetObject(env,obj);
	int NumElts = env->GetArrayLength(ourarray);
	jint *oarr = env->GetIntArrayElements(ourarray, NULL);
	std::vector<int> ourvector;
	for (int i=0;i<NumElts;i++){
		ourvector.push_back(oarr[i]);
	}
	double ret = game->makeAction(ourvector, integ);
	return ret;
	}
	catch (Vizia::DoomIsNotRunningException& e)
  	{	

 		jclass DoomIsNotRunningException = env->FindClass("errors/DoomIsNotRunningException");
        	env->ThrowNew(DoomIsNotRunningException, e.what());
		return 0;
  	}
	catch (std::exception& e)
  	{
    		std::cout << "C++ unknown exception"<< '\n';
		return 0;
  	}
}

/*
 * Class:     ViziaDoomGameJava
 * Method:    getState
 * Signature: ()LState;
 */
JNIEXPORT jobject JNICALL Java_ViziaDoomGameJava_getState
  (JNIEnv *env, jobject obj){
	Vizia::DoomGame* game=GetObject(env,obj);
	jclass state = env->FindClass("State");	
	int rozmiar=game->getScreenSize();	
	std::vector<int> ourvector;
	Vizia::DoomGame::State statec=game->getState();
	
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
 * Class:     ViziaDoomGameJava
 * Method:    getLastAction
 */
JNIEXPORT jintArray JNICALL Java_ViziaDoomGameJava_getLastAction
  (JNIEnv *env, jobject obj){
	std::vector<int> ourvector;
	Vizia::DoomGame* game=GetObject(env,obj);
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
 * Class:     ViziaDoomGameJava
 * Method:    isNewEpisode
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_ViziaDoomGameJava_isNewEpisode
  (JNIEnv *env, jobject obj){
	try{
		Vizia::DoomGame* game=GetObject(env,obj);
		bool ret=game->isNewEpisode();
		return (jboolean)ret;
	}
	catch (Vizia::DoomIsNotRunningException& e)
  	{	

 		jclass DoomIsNotRunningException = env->FindClass("errors/DoomIsNotRunningException");
        	env->ThrowNew(DoomIsNotRunningException, e.what());
		return 0;
  	}
	catch (Vizia::Exception& e)
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
 * Class:     ViziaDoomGameJava
 * Method:    isEpisodeFinished
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_ViziaDoomGameJava_isEpisodeFinished
  (JNIEnv *env, jobject obj){
	try{
		Vizia::DoomGame* game=GetObject(env,obj);
		bool ret=game->isEpisodeFinished();
		return (jboolean)ret;
	}
	catch (Vizia::DoomIsNotRunningException& e)
  	{	

 		jclass DoomIsNotRunningException = env->FindClass("errors/DoomIsNotRunningException");
        	env->ThrowNew(DoomIsNotRunningException, e.what());
		return 0;
  	}
	catch (Vizia::Exception& e)
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
 * Class:     ViziaDoomGameJava
 * Method:    addAvailableButton
 * Signature: (Lenums/Button;)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_addAvailableButton__Lenums_Button_2
  (JNIEnv *env, jobject obj, jobject enumVal){
	Vizia::DoomGame* game=GetObject(env,obj);
	jclass jclassEnum = env->FindClass("enums/Button");
	if(jclassEnum != 0)
    	{	
        	jmethodID ordinal_ID = env->GetMethodID(jclassEnum, "ordinal", "()I");
		if (ordinal_ID == 0){
			return;
		}
		jint value = env->CallIntMethod(enumVal, ordinal_ID);
		Vizia::Button ret=static_cast<Vizia::Button>(value);
		game->addAvailableButton(ret);
	// Delete local references created		
        	env->DeleteLocalRef(jclassEnum);
    	}

}

/*
 * Class:     ViziaDoomGameJava
 * Method:    addAvailableButton
 * Signature: (Lenums/Button;I)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_addAvailableButton__Lenums_Button_2I
  (JNIEnv *env, jobject obj, jobject enumVal, jint intval){
	Vizia::DoomGame* game=GetObject(env,obj);
	jclass jclassEnum = env->FindClass("enums/Button");
	if(jclassEnum != 0)
    	{	
        	jmethodID ordinal_ID = env->GetMethodID(jclassEnum, "ordinal", "()I");
		if (ordinal_ID == 0){
			return;
		}
		jint value = env->CallIntMethod(enumVal, ordinal_ID);
		Vizia::Button ret=static_cast<Vizia::Button>(value);
		game->addAvailableButton(ret, intval);
	// Delete local references created		
        	env->DeleteLocalRef(jclassEnum);
    	}
}


/*
 * Class:     ViziaDoomGameJava
 * Method:    clearAvailableButtons
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_clearAvailableButtons 
  (JNIEnv *env, jobject obj){
	Vizia::DoomGame* game=GetObject(env,obj);
	game->clearAvailableButtons();
};

/*
 * Class:     ViziaDoomGameJava
 * Method:    getAvailableButtonsSize
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_ViziaDoomGameJava_getAvailableButtonsSize
  (JNIEnv *env, jobject obj){
	Vizia::DoomGame* game=GetObject(env,obj);
	int ret=game->getAvailableButtonsSize();
	return ret;
}

/*
 * Class:     ViziaDoomGameJava
 * Method:    setButtonMaxValue
 * Signature: (Lenums/Button;I)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_setButtonMaxValue
  (JNIEnv *env, jobject obj, jobject enumVal, jint intval){
	Vizia::DoomGame* game=GetObject(env,obj);
	jclass jclassEnum = env->FindClass("enums/Button");
	if(jclassEnum != 0)
    	{	
        	jmethodID ordinal_ID = env->GetMethodID(jclassEnum, "ordinal", "()I");
		if (ordinal_ID == 0){
			return;
		}
		jint value = env->CallIntMethod(enumVal, ordinal_ID);
		Vizia::Button ret=static_cast<Vizia::Button>(value);
		game->setButtonMaxValue(ret, intval);
	// Delete local references created		
        	env->DeleteLocalRef(jclassEnum);
    	}
}

/*
 * Class:     ViziaDoomGameJava
 * Method:    addAvailableGameVariable
 * Signature: (Lenums/GameVariable;)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_addAvailableGameVariable
  (JNIEnv *env, jobject obj, jobject enumVal){
	Vizia::DoomGame* game=GetObject(env,obj);
	jclass jclassEnum = env->FindClass("enums/GameVar");
	if(jclassEnum != 0)
    	{	
        	jmethodID ordinal_ID = env->GetMethodID(jclassEnum, "ordinal", "()I");
		if (ordinal_ID == 0){
			return;
		}
		jint value = env->CallIntMethod(enumVal, ordinal_ID);
		Vizia::GameVariable ret=static_cast<Vizia::GameVariable>(value);
		game->addAvailableGameVariable(ret);
	// Delete local references created		
        	env->DeleteLocalRef(jclassEnum);
    	}

}

/*
 * Class:     ViziaDoomGameJava
 * Method:    clearAvailableGameVariables
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_clearAvailableGameVariables
  (JNIEnv *env, jobject obj){
	Vizia::DoomGame* game=GetObject(env,obj);
	game->clearAvailableGameVariables();

}

/*
 * Class:     ViziaDoomGameJava
 * Method:    getAvailableGameVariablesSize
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_ViziaDoomGameJava_getAvailableGameVariablesSize
  (JNIEnv *env, jobject obj){
	Vizia::DoomGame* game=GetObject(env,obj);
	int ret=game->getAvailableGameVariablesSize();
	return ret;
}

/*
 * Class:     ViziaDoomGameJava
 * Method:    addCustomGameArg
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_addCustomGameArg
  (JNIEnv *env, jobject obj , jstring str){
	Vizia::DoomGame* game=GetObject(env,obj);
	char * str2;
    	str2 = const_cast<char*>(env->GetStringUTFChars(str , NULL )) ;
	game->addCustomGameArg(str2);
}

/*
 * Class:     ViziaDoomGameJava
 * Method:    clearCustomGameArgs
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_clearCustomGameArgs
  (JNIEnv *env, jobject obj){
	Vizia::DoomGame* game=GetObject(env,obj);
	game->clearCustomGameArgs();
}

/*
 * Class:     ViziaDoomGameJava
 * Method:    sendGameCommand
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_sendGameCommand //TODO wywala jvm
  (JNIEnv *env, jobject obj, jstring str){
	try{	
		Vizia::DoomGame* game=GetObject(env,obj);
		char * str2;
	    	str2 = const_cast<char*>(env->GetStringUTFChars(str , NULL )) ;	
		game->sendGameCommand(str2);
		
	}
	catch (Vizia::DoomIsNotRunningException& e)
  	{	
		std::cout<<"To tu 1"<<std::endl;
 		jclass DoomIsNotRunningException = env->FindClass("errors/DoomIsNotRunningException");
        	env->ThrowNew(DoomIsNotRunningException, e.what());

  	}
	catch (Vizia::Exception& e)
  	{	
		std::cout<<"To tu 2"<<std::endl;
 		jclass Exception = env->FindClass("errors/Exception");
        	env->ThrowNew(Exception, e.what());

  	}
	catch (...)
  	{
    		std::cout << "C++ unknown exception"<<std::endl;

  	}
}

/*
 * Class:     ViziaDoomGameJava
 * Method:    getGameScreen
 * Signature: ()Ljava/util/ArrayList;
 */
JNIEXPORT jintArray JNICALL Java_ViziaDoomGameJava_getGameScreen
  (JNIEnv *env, jobject obj){
	Vizia::DoomGame* game=GetObject(env,obj);
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
 * Class:     ViziaDoomGameJava
 * Method:    getMode
 * Signature: ()Lenums/Mode;
 */
JNIEXPORT jint JNICALL Java_ViziaDoomGameJava_getMod
  (JNIEnv *env, jobject obj){
	Vizia::DoomGame* game=GetObject(env,obj);
	Vizia::Mode mode=game->getMode();
	return (jint) mode;
}

/*
 * Class:     ViziaDoomGameJava
 * Method:    setGMode
 * Signature: (Lenums/Mode;)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_setMode
  (JNIEnv *env, jobject obj, jobject enumVal){
	Vizia::DoomGame* game=GetObject(env,obj);
	jclass jclassEnum = env->FindClass("enums/Mode");
	
	if(jclassEnum != 0)
    	{	
        	jmethodID ordinal_ID = env->GetMethodID(jclassEnum, "ordinal", "()I");
		if (ordinal_ID == 0){
			return;
		}
		jint value = env->CallIntMethod(enumVal, ordinal_ID);
		Vizia::Mode ret=static_cast<Vizia::Mode>(value);
		game->setMode(ret);
	// Delete local references created		
        	env->DeleteLocalRef(jclassEnum);
    	}
}

/*
 * Class:     ViziaDoomGameJava
 * Method:    getGameVariable
 * Signature: (Lenums/GameVariable;)I
 */
JNIEXPORT jint JNICALL Java_ViziaDoomGameJava_getGameVar
  (JNIEnv *env, jobject obj, jobject enumVal){
	Vizia::DoomGame* game=GetObject(env,obj);
	jclass jclassEnum = env->FindClass("enums/GameVariable");
	
	if(jclassEnum != 0)
    	{	
        	jmethodID ordinal_ID = env->GetMethodID(jclassEnum, "ordinal", "()I");
		if (ordinal_ID == 0){
			return -1;
		}
		jint value = env->CallIntMethod(enumVal, ordinal_ID);
		Vizia::GameVariable ret=static_cast<Vizia::GameVariable>(value);
		int retint=game->getGameVariable(ret);
	// Delete local references created		
        	env->DeleteLocalRef(jclassEnum);
		return retint;
    	}

}

/*
 * Class:     ViziaDoomGameJava
 * Method:    getLivingReward
 * Signature: ()F
 */
JNIEXPORT jfloat JNICALL Java_ViziaDoomGameJava_getLivingReward
 (JNIEnv *env, jobject obj){
	Vizia::DoomGame* game=GetObject(env,obj);
	float ret = game->getLivingReward();
	return ret;
}


/*
 * Class:     ViziaDoomGameJava
 * Method:    setLivingReward
 * Signature: (F)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_setLivingReward
(JNIEnv *env, jobject obj, jfloat rew){
	Vizia::DoomGame* game=GetObject(env,obj);
	game->setLivingReward(rew);
}

/*
 * Class:     ViziaDoomGameJava
 * Method:    getDeathPenalty
 * Signature: ()F
 */
JNIEXPORT jfloat JNICALL Java_ViziaDoomGameJava_getDeathPenalty
 (JNIEnv *env, jobject obj){
	Vizia::DoomGame* game=GetObject(env,obj);
	float ret = game->getDeathPenalty();
	return ret;
}

/*
 * Class:     ViziaDoomGameJava
 * Method:    setDeathPenalty
 * Signature: (F)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_setDeathPenalty
  (JNIEnv *env, jobject obj, jfloat rew){
	Vizia::DoomGame* game=GetObject(env,obj);
	game->setDeathPenalty(rew);
}

/*
 * Class:     ViziaDoomGameJava
 * Method:    getLastReward
 * Signature: ()F
 */
JNIEXPORT jfloat JNICALL Java_ViziaDoomGameJava_getLastReward
  (JNIEnv *env, jobject obj){
	Vizia::DoomGame* game=GetObject(env,obj);
	float ret=game->getLastReward();
	return ret;

}

/*
 * Class:     ViziaDoomGameJava
 * Method:    getSummaryReward
 * Signature: ()F
 */
JNIEXPORT jfloat JNICALL Java_ViziaDoomGameJava_getSummaryReward
  (JNIEnv *env, jobject obj){
	Vizia::DoomGame* game=GetObject(env,obj);
	float ret=game->getSummaryReward();
	return ret;

}

/*
 * Class:     ViziaDoomGameJava
 * Method:    setDoomGamePath
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_setDoomGamePath
 (JNIEnv *env, jobject obj, jstring path){
	Vizia::DoomGame* game=GetObject(env,obj);
 	char * path2;
    	path2 = const_cast<char*>(env->GetStringUTFChars(path , NULL )) ;	
	game->setDoomGamePath(path2);
}

/*
 * Class:     ViziaDoomGameJava
 * Method:    setDoomIwadPath
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_setDoomIwadPath
  (JNIEnv *env, jobject obj, jstring path){
	Vizia::DoomGame* game=GetObject(env,obj);
	char * path2;
    	path2 = const_cast<char*>(env->GetStringUTFChars(path , NULL )) ;
	game->setDoomIwadPath(path2);
}

/*
 * Class:     ViziaDoomGameJava
 * Method:    setDoomFilePath
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_setDoomFilePath
 (JNIEnv *env, jobject obj, jstring path){
	Vizia::DoomGame* game=GetObject(env,obj);
	char * path2;
    	path2 = const_cast<char*>(env->GetStringUTFChars(path , NULL )) ;
	game->setDoomFilePath(path2);
}

/*
 * Class:     ViziaDoomGameJava
 * Method:    setDoomMap
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_setDoomMap
  (JNIEnv *env, jobject obj, jstring map){
	Vizia::DoomGame* game=GetObject(env,obj);
	char * path;
    	path = const_cast<char*>(env->GetStringUTFChars(map , NULL )) ;
	game->setDoomMap(path);

}


/*
 * Class:     ViziaDoomGameJava
 * Method:    setDoomSkill
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_setDoomSkill
  (JNIEnv *env, jobject obj, jint skill){
	Vizia::DoomGame* game=GetObject(env,obj);
	game->setDoomSkill(skill);
}


/*
 * Class:     ViziaDoomGameJava
 * Method:    setDoomConfigPath
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_setDoomConfigPath
(JNIEnv *env, jobject obj, jstring path){
	Vizia::DoomGame* game=GetObject(env,obj);
 	char * path2;
    	path2 = const_cast<char*>(env->GetStringUTFChars(path , NULL )) ;	
	game->setDoomConfigPath(path2);
}

/*
 * Class:     ViziaDoomGameJava
 * Method:    getSeed
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_ViziaDoomGameJava_getSeed
  (JNIEnv *env, jobject obj){
	Vizia::DoomGame* game=GetObject(env,obj);
	int ret = game->getSeed();
	return (jint)ret;
}

/*
 * Class:     ViziaDoomGameJava
 * Method:    setSeed
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_setSeed
  (JNIEnv *env, jobject obj, jint seed){
	Vizia::DoomGame* game=GetObject(env,obj);
	game->setSeed(seed);
}

/*
 * Class:     ViziaDoomGameJava
 * Method:    setAutoNewEpisode
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_setAutoNewEpisode
  (JNIEnv * env, jobject obj, jboolean bol){
	Vizia::DoomGame* game=GetObject(env,obj);
	game->setAutoNewEpisode(bol);

}

/*
 * Class:     ViziaDoomGameJava
 * Method:    setNewEpisodeOnTimeout
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_setNewEpisodeOnTimeout
  (JNIEnv *env, jobject obj, jboolean bol){
	Vizia::DoomGame* game=GetObject(env,obj);
	game->setNewEpisodeOnTimeout(bol);

}

/*
 * Class:     ViziaDoomGameJava
 * Method:    setNewEpisodeOnPlayerDeath
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_setNewEpisodeOnPlayerDeath
  (JNIEnv *env, jobject obj , jboolean bol){
	Vizia::DoomGame* game=GetObject(env,obj);
	game->setNewEpisodeOnPlayerDeath(bol);

}

/*
 * Class:     ViziaDoomGameJava
 * Method:    setNewEpisodeOnMapEnd
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_setNewEpisodeOnMapEnd
  (JNIEnv *env, jobject obj, jboolean bol){
	Vizia::DoomGame* game=GetObject(env,obj);
	game->setNewEpisodeOnMapEnd(bol);

}

/*
 * Class:     ViziaDoomGameJava
 * Method:    getEpisodeStartTime
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_ViziaDoomGameJava_getEpisodeStartTime
  (JNIEnv *env, jobject obj){
	Vizia::DoomGame* game=GetObject(env,obj);
	int ret = game->getEpisodeStartTime();
	return (jint) ret;
}

/*
 * Class:     ViziaDoomGameJava
 * Method:    setEpisodeStartTime
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_setEpisodeStartTime
  (JNIEnv *env, jobject obj, jint tics){
	Vizia::DoomGame* game=GetObject(env,obj);
	game->setEpisodeTimeout(tics);

}

/*
 * Class:     ViziaDoomGameJava
 * Method:    getEpisodeTimeout
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_ViziaDoomGameJava_getEpisodeTimeout
    (JNIEnv *env, jobject obj){
	Vizia::DoomGame* game=GetObject(env,obj);
	int ret = game->getEpisodeTimeout();
	return ret;
}
/*
 * Class:     ViziaDoomGameJava
 * Method:    setEpisodeTimeout
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_setEpisodeTimeout
  (JNIEnv *env, jobject obj, jint tics){
	Vizia::DoomGame* game=GetObject(env,obj);
	game->setEpisodeTimeout(tics);

}

/*
 * Class:     ViziaDoomGameJava
 * Method:    setScreenResolution
 * Signature: (II)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_setScreenResolution
 (JNIEnv *env, jobject obj, jobject enumVal){
	Vizia::DoomGame* game=GetObject(env,obj);
	jclass jclassEnum = env->FindClass("enums/ScreenResolution");
	
	if(jclassEnum != 0)
    	{	
        	jmethodID ordinal_ID = env->GetMethodID(jclassEnum, "ordinal", "()I");
		if (ordinal_ID == 0){
			return;
		}
		jint value = env->CallIntMethod(enumVal, ordinal_ID);
		Vizia::ScreenResolution ret=static_cast<Vizia::ScreenResolution>(value);
		game->setScreenResolution(ret);
	// Delete local references created		
        	env->DeleteLocalRef(jclassEnum);
    	}

}
	


/*
 * Class:     ViziaDoomGameJava
 * Method:    setScreenFormat
 * Signature: (Lenums/ScreenFormat;)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_setScreenFormat
  (JNIEnv *env, jobject obj, jobject enumVal){
	Vizia::DoomGame* game=GetObject(env,obj);
	jclass jclassEnum = env->FindClass("enums/ScreenFormat");
	
	if(jclassEnum != 0)
    	{	
        	jmethodID ordinal_ID = env->GetMethodID(jclassEnum, "ordinal", "()I");
		if (ordinal_ID == 0){
			return;
		}
		jint value = env->CallIntMethod(enumVal, ordinal_ID);
		Vizia::ScreenFormat ret=static_cast<Vizia::ScreenFormat>(value);
		game->setScreenFormat(ret);
	// Delete local references created		
        	env->DeleteLocalRef(jclassEnum);
    	}

}

/*
 * Class:     ViziaDoomGameJava
 * Method:    setRenderHud
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_setRenderHud
  (JNIEnv *env, jobject obj, jboolean bol){
	Vizia::DoomGame* game=GetObject(env,obj);
	game->setRenderHud(bol);
}

/*
 * Class:     ViziaDoomGameJava
 * Method:    setRenderWeapon
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_setRenderWeapon
  (JNIEnv *env, jobject obj, jboolean bol){
	Vizia::DoomGame* game=GetObject(env,obj);
	game->setRenderWeapon(bol);

}

/*
 * Class:     ViziaDoomGameJava
 * Method:    setRenderCrosshair
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_setRenderCrosshair
  (JNIEnv *env , jobject obj, jboolean bol){
	Vizia::DoomGame* game=GetObject(env,obj);
	game->setRenderCrosshair(bol);
}

/*
 * Class:     ViziaDoomGameJava
 * Method:    setRenderDecals
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_setRenderDecals
 (JNIEnv *env, jobject obj, jboolean bol){
	Vizia::DoomGame* game=GetObject(env,obj);
	game->setRenderDecals(bol);
}

/*
 * Class:     ViziaDoomGameJava
 * Method:    setRenderParticles
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_setRenderParticles
  (JNIEnv *env, jobject obj, jboolean bol){
	Vizia::DoomGame* game=GetObject(env,obj);
	game->setRenderParticles(bol);

}

/*
 * Class:     ViziaDoomGameJava
 * Method:    setVisibleWindow
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_setWindowVisible
  (JNIEnv *env, jobject obj, jboolean bol){
	Vizia::DoomGame* game=GetObject(env,obj);
	game->setWindowVisible(bol);
}

/*
 * Class:     ViziaDoomGameJava
 * Method:    setDisabledConsole
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL Java_ViziaDoomGameJava_setConsoleEnabled
  (JNIEnv *env, jobject obj, jboolean bol){
	Vizia::DoomGame* game=GetObject(env,obj);
	game->setConsoleEnabled(bol);

}

/*
 * Class:     ViziaDoomGameJava
 * Method:    getScreenChannels
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_ViziaDoomGameJava_getScreenChannels
   (JNIEnv *env, jobject obj){
	Vizia::DoomGame* game=GetObject(env,obj);
	int ret;
	ret=game->getScreenChannels();
	return (jint)ret;

}

/*
 * Class:     ViziaDoomGameJava
 * Method:    getScreenPitch
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_ViziaDoomGameJava_getScreenPitch
  (JNIEnv *env, jobject obj){
	Vizia::DoomGame* game=GetObject(env,obj);
	int ret;
	ret=game->getScreenPitch();
	return (jint)ret;

}

/*
 * Class:     ViziaDoomGameJava
 * Method:    getScreenSize
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_ViziaDoomGameJava_getScreenSize
  (JNIEnv *env, jobject obj){
	Vizia::DoomGame* game=GetObject(env,obj);
	int ret;
	ret=game->getScreenSize();
	return (jint)ret;


}

/*
 * Class:     ViziaDoomGameJava
 * Method:    getScreenFormat
 * Signature: ()Lenums/ScreenFormat;
 */
JNIEXPORT jint JNICALL Java_ViziaDoomGameJava_getScreenForma
  (JNIEnv *env, jobject obj){
	Vizia::DoomGame* game=GetObject(env,obj);
	Vizia::ScreenFormat ret = game->getScreenFormat();
	return (jint) ret;
}


