#!/bin/bash

for F in tests/[!_]*.py ; do
	[[ -f $F ]] || continue
	F=${F%.py}
	F=${F/\//.}
	echo running test: ${F}
	python -m ${F}
	echo
	echo
done
