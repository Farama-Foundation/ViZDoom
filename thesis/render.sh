#!/bin/bash

#
# This will work, although a proper Makefile or latexmk is recommended.
# latexmk -pvc -pdf thesis-master-english.tex
#

for type in bachelor-english; do
        pdflatex thesis-$type.tex
        bibtex   thesis-$type
        pdflatex thesis-$type.tex
        pdflatex thesis-$type.tex
done

rm -f *.aux *.bak *.log *.blg *.bbl *.toc *.out

