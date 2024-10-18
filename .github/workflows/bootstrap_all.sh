#!/bin/bash

for method in MPn; do
    for n in 2 3; do
        for i in rhf uhf ghf; do
            echo "Bootstrapping $method ($i) (order=$n)"
            python -W ignore bootstrap_${method}.py $i $n &
        done
    done
done

for method in CCD CCSD QCISD CC2 DCD DCSD; do
    for i in rhf uhf ghf; do
        echo "Bootstrapping $method ($i)"
        python -W ignore bootstrap_${method}.py $i &
    done
    for i in rhf uhf; do
        echo "Bootstrapping DF-$method ($i)"
        python -W ignore bootstrap_DF${method}.py $i &
    done
done

for method in CCSDwtwp; do
    for i in rhf ghf; do
        echo "Bootstrapping $method ($i)"
        python -W ignore bootstrap_${method}.py $i &
    done
done

for method in CC3 CCSDT; do
    for i in rhf uhf ghf; do
        echo "Bootstrapping $method ($i)"
        python -W ignore bootstrap_${method}.py $i &
    done
done

wait
