
mkdir spieces
F=$1
D=$2
wiki="${F}/wiki/wiki"
oscar="${F}/oscar/oscar"

# bash sentencepiece.sh $out $size $files

# vocab for every lang for oscar for wiki for both for all langs

for size in 30000 50000;
do
    for lang in sv no da;
    do
        for file in $wiki $oscar;
        do
            if [ $lang = "no" ]
            then
                file="$file.$lang.ss $file.nn.ss" 
            else
                file="$file.$lang.ss"
            fi
            out="$D/vocab.$lang.${file##*/}.$size"
            bash sentencepiece.sh $out $size $file
        done

        out="$D/vocab.$lang.oscar+wiki.$size"
        if [ ${lang} = "no" ]
        then
            files="$wiki.$lang.ss $wiki.nn.ss $oscar.$lang.ss $oscar.nn.ss" 
        else
            files="$wiki.$lang.ss $oscar.$lang.ss"
        fi
        bash sentencepiece.sh $out $size $files
    done


    for file in $wiki $oscar;
    do
        out="$D/vocab.sv+no+da.${file##*/}.$size"
        bash sentencepiece.sh $out $size $file.sv.ss $file.no.ss $file.nn.ss $file.da.ss
    done
done
