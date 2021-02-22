
mkdir spieces
D="spieces"
wiki="../../wiki/wiki.ss"
oscar="../../oscar/oscar.ss"

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
                file="$file.$lang $file.nn" 
            else
                file="$file.$lang"
            fi
            out="$D/vocab.$lang.${file##*/}.$size"
            bash sentencepiece.sh $out $size $file
        done

        out="$D/vocab.$lang.oscar+wiki.$size"
        if [ ${lang} = "no" ]
        then
            files="$wiki.$lang $wiki.nn $oscar.$lang $oscar.nn" 
        else
            files="$wiki.$lang $oscar.$lang"
        fi
        bash sentencepiece.sh $out $size $files
    done


    for file in $wiki $oscar;
    do
        out="$D/vocab.sv+no+da.${file##*/}.$size"
        bash sentencepiece.sh $out $size $file.sv $file.no $file.nn $file.da
    done
done
