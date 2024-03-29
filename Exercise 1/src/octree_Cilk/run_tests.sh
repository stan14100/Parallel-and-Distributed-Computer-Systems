# A simple bash script that for running experiments
# Note: To run the script make sure it you have execution rights 
# (use: chmod u+x run_tests.sh to give execution rights) 
!/bin/bash

NAME=$(hostname)
DATE=$(date "+%Y-%m-%d-%H:%M:%S")
FILE_PREF=$NAME-$DATE-test-tree

echo $NAME
echo $DATE

make clean; make
# run cube experiments
for N in 1000000 2000000 4000000 9000000 18000000 33000000 ; do \
    for T in 2 4 8 16 32 64 128 256 512 1024 2048 ; do \
            echo cube N=$N && ./test_octree $N 0 128 5 18 $T >> $FILE_PREF-cube.txt ; \
    done ; \
done ;
# run octant experiments
for N in 1000000 2000000 4000000 9000000 18000000 33000000 ; do \
    for T in 2 4 8 16 32 64 128 256 512 1024 2048 ; do \
           echo Plummer N=$N && ./test_octree $N 1 128 5 18 $T >> $FILE_PREF-plummer.txt ; \
    done ; \
done ;
