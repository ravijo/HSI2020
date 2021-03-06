#!/bin/bash
# compile main.tex to produce main.pdf and
# remove temporary files generated during compilation
# Usage: sh compile.sh
# Author: Ravi Joshi
# Date: 4 April 2020
# -------------------------------------------------

# declare all temporary files
temp_file_extensions="aux bbl blg log out lof toc bcf run.xml fdb_latexmk fls"

# function
compile_tex()
{
  file_name=$1

  echo "Running pdflatex. Please wait..."
  pdflatex "${file_name}.tex"

  echo "Running bibtex. Please wait..."
  bibtex "${file_name}"

  echo "Running pdflatex again. Please wait..."
  pdflatex "${file_name}.tex"

  echo "Running pdflatex again. Please wait..."
  pdflatex "${file_name}.tex"

  echo "Removing temporary files. Please wait..."
  for temp_file_extension in $temp_file_extensions
  do
    # src: https://askubuntu.com/a/377442/569883
    find . -name "*.${temp_file_extension}" -type f -delete
  done

  echo "Finished. Generated file is ${file_name}.pdf"
}

compile_tex main
