#!/bin/bash
for i in rhf uhf ghf; do
    for n in 2 3; do
        python bootstrap_MPn.py $i $n
    python bootstrap_CCD.py $i
    python bootstrap_CCSD.py $i
    python bootstrap_QCISD.py $i
    python bootstrap_CC2.py $i
    python bootstrap_DCD.py $i
    python bootstrap_DCSD.py $i
    python bootstrap_DFCCD.py $i
    python bootstrap_DFCCSD.py $i
done
