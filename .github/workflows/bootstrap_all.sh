#!/bin/bash
for i in rhf uhf ghf; do
    for n in 2 3; do
        python -W ignore bootstrap_MPn.py $i $n
    done
    python -W ignore bootstrap_CCD.py $i
    python -W ignore bootstrap_CCSD.py $i
    python -W ignore bootstrap_QCISD.py $i
    python -W ignore bootstrap_CC2.py $i
    python -W ignore bootstrap_DCD.py $i
    python -W ignore bootstrap_DCSD.py $i
done
for i in rhf uhf; do
    python -W ignore bootstrap_DFCCD.py $i
    python -W ignore bootstrap_DFCCSD.py $i
done
