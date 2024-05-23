java -Xms32768M \
    -classpath "earleyx/earleyx_fast.jar:lib/" parser.Main \
    -in test.txt  \
    -grammar grammars/earleyx/normal/earleyx_pcfg_0.2.grammar \
    -out earleyx/results \
    -verbose 1 \
    -thread 1


