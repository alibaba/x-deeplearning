echo "Preparing imgs data..."
mkdir -p imgs
python randimg.py
echo "Preparing sample data..."
python randtxt.py > data.txt
echo "Preparing the job config..."
sed "s%\${pwd}%`pwd`%g" config.json.template > config.json
echo "Run the job..."
xdl_submit.py --config config.json
