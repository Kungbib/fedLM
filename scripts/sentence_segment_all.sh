#! /usr/bin/bash

# opus100
# opus100 is already one sentence per line
# for l in nn no da sv;
# do
#     bash sentence_segment.sh ../opus100/opus100.train.$l $l ../opus100/opus100.train.ss.$l;
#     bash sentence_segment.sh ../opus100/opus100.validation.$l $l ../opus100/opus100.validation.ss.$l;
#     bash sentence_segment.sh ../opus100/opus100.test.$l $l ../opus100/opus100.test.ss.$l;
# done

# wiki
for l in nn no da sv;
do
    bash sentence_segment.sh ../wiki/${l}wiki/${l}wiki.txt $l ../wiki/wiki.${l}.ss;
done


# oscar
for l in nn no da sv;
do
    bash sentence_segment.sh ../oscar/oscar.train.$l $l ../oscar/oscar.$l.ss;
done
