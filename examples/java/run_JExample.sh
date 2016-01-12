#!/bin/bash
javac -classpath ".:../../vizia_api_bindings/java" JavaExample.java
java -Djava.library.path="../../vizia_api_bindings/java" -classpath ".:../../vizia_api_bindings/java" JavaExample
