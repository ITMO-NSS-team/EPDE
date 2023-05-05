#!/bin/bash
echo "Testing $1 via pytest"
dir=$(dirname $(pwd))
dir_2=$(dirname $dir)
echo $dir $dir_2
echo "Kek!"
if [ -e $1 ]
then
python -m pytest $1 "--path" $dir_2
else
echo "The script is missing"
fi
