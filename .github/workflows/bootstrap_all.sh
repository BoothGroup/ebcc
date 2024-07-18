#!/bin/bash
for method in MPn CCD CCSD QCISD CC2 DCD DCSD; do
    for i in rhf uhf ghf; do
        for n in 2 3; do
            echo "Bootstrapping $method ($i) (order=$n)"
            python -W ignore bootstrap_${method}.py $i $n
        done
        echo "Bootstrapping $method ($i)"
        python -W ignore bootstrap_${method}.py $i
    done
done
for method in DFCCD DFCCSD; do
    for i in rhf uhf; do
        echo "Bootstrapping $method ($i)"
        python -W ignore bootstrap_${method}.py $i
    done
done
