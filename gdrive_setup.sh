#!/bin/bash

cd ../
mkdir tmp_gdrive
cd tmp_gdrive
wget -O drive https://drive.google.com/uc?id=0B3X9GlR6Embnb095MGxEYmJhY2c
sudo install drive /usr/local/bin/drive
cd ..
rm -rf tmp_gdrive


