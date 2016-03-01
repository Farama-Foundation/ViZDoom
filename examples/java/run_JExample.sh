#!/bin/bash
#javac -classpath ".:../../vizia_api_bindings/java" JavaExample.java
#java -Djava.library.path="../../vizia_api_bindings/java:../../bin/java" -classpath ".:../../vizia_api_bindings/java" JavaExample

#javac -classpath ".:../../vizia_api_bindings/java" Seed.java
#java -Djava.library.path="../../vizia_api_bindings/java:../../bin/java" -classpath ".:../../vizia_api_bindings/java" Seed

javac -classpath ".:../../vizia_api_bindings/java" Spectator.java
java -Djava.library.path="../../vizia_api_bindings/java:../../bin/java" -classpath ".:../../vizia_api_bindings/java" Spectator
