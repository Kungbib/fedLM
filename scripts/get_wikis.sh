mkdir wiki
cd wiki
# Get the dumps
echo "Downloading dumps"
wget https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/nnwiki/20210201/nnwiki-20210201-pages-articles.xml.bz2
wget https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/nowiki/20210201/nowiki-20210201-pages-articles.xml.bz2
wget https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/dawiki/20210201/dawiki-20210201-pages-articles.xml.bz2
wget https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/svwiki/20210201/svwiki-20210201-pages-articles.xml.bz2

# move them into their respective directories
mkdir nnwiki nowiki dawiki svwiki
mv nnwiki-* nnwiki/.
mv nowiki-* nowiki/.
mv dawiki-* dawiki/.
mv svwiki-* svwiki/.

# run wikiextractor
echo "Run WikiExtractor"
mkdir nnwiki/we nowiki/we dawiki/we svwiki/we
python -m wikiextractor.WikiExtractor --json --processes 15 -o nnwiki/we nnwiki/nnwiki-20210201-pages-articles.xml.bz2 
python -m wikiextractor.WikiExtractor --json --processes 15 -o nowiki/we nowiki/nowiki-20210201-pages-articles.xml.bz2 
python -m wikiextractor.WikiExtractor --json --processes 15 -o dawiki/we dawiki/dawiki-20210201-pages-articles.xml.bz2 
python -m wikiextractor.WikiExtractor --json --processes 15 -o svwiki/we svwiki/svwiki-20210201-pages-articles.xml.bz2 

# create each one big jsonl file
cat nnwiki/we/*/* > nnwiki/we/all.jsonl
cat nowiki/we/*/* > nowiki/we/all.jsonl
cat dawiki/we/*/* > dawiki/we/all.jsonl
cat svwiki/we/*/* > svwiki/we/all.jsonl

# extra the texts from the json files
echo "Run Json Extractor"
python src/wiki_json_to_txt.py nnwiki/we/all.jsonl wiki.nn # nnwiki/nnwiki.txt
python src/wiki_json_to_txt.py nowiki/we/all.jsonl wiki.no # nowiki/nowiki.txt
python src/wiki_json_to_txt.py dawiki/we/all.jsonl wiki.da # dawiki/dawiki.txt
python src/wiki_json_to_txt.py svwiki/we/all.jsonl wiki.sv # svwiki/svwiki.txt

# remove unnecessary files
# for l in nn no da sv;
# do
#     rm -rf ${l}wiki/we ${l}wiki/*bz2
# done

cd -
