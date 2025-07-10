cd data
wget -O elasticsearch-8.15.0.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.15.0-linux-x86_64.tar.gz  # download Elasticsearch
tar zxvf elasticsearch-8.15.0.tar.gz
rm elasticsearch-8.15.0.tar.gz 
cd elasticsearch-8.15.0

nohup bin/elasticsearch &  # run Elasticsearch in background
echo "Elasticsearch is starting in the background. Waiting..."
sleep 60  # wait 60 seconds for Elasticsearch to start

cd ../..
echo "Trying to build index"
python prep_elastic.py --data_path data/dpr/psgs_w100.tsv --index_name wiki  # build index
