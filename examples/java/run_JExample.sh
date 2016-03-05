#!/bin/bash
#javac -classpath ".:../../vizdoom_api_bindings/java" JavaExample.java
#java -Djava.library.path="../../vizdoom_api_bindings/java:../../bin/java" -classpath ".:../../vizdoom_api_bindings/java" JavaExample

#javac -classpath ".:../../vizdoom_api_bindings/java" Seed.java
#java -Djava.library.path="../../vizdoom_api_bindings/java:../../bin/java" -classpath ".:../../vizdoom_api_bindings/java" Seed

javac -classpath ".:../../vizdoom_api_bindings/java" Spectator.java
java -Djava.library.path="../../vizdoom_api_bindings/java:../../bin/java" -classpath ".:../../vizdoom_api_bindings/java" Spectator
