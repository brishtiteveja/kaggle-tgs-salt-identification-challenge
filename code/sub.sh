#! /usr/bin/sh

qsub -q wwtung -n -l nodes=1:ppn=20 run.sh
