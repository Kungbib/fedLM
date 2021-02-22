echo "Input file: ${1}"
echo "Language: ${2}"
echo "Output file: ${3}"

# python3 segmenter.py ${1} ${2} > ${3}

perl split-sentences.perl -l sv_no_da -p nonbreaking_prefix.sv_no_da < $1 | sed "s/^<P>$//" > $3
