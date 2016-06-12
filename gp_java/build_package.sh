gradle shadowJar
rm -R ./build/jeff-dissertation/
mkdir ./build/jeff-dissertation
cp ./build/libs/gp_java-1.0-all.jar ./build/jeff-dissertation
cp ~/.dissertation_jheaton ./build/jeff-dissertation/.dissertation_jheaton
mkdir ./build/jeff-dissertation/dissertation
cp -R ~/temp/dissertation/* ./build/jeff-dissertation/dissertation
