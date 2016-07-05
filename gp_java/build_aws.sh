gradle clean
gradle shadowJar
mkdir ./build/jeff-dissertation
cp ./build/libs/gp_java-1.0-all.jar ./build/jeff-dissertation
echo "host: aws" > ./build/jeff-dissertation/.dissertation_jheaton
echo "project: /home/ec2-users/dissertation" >> ./build/jeff-dissertation/.dissertation_jheaton
tar -pczf ./build/jeff-dissertation.tar.gz ./build/jeff-dissertation/*
