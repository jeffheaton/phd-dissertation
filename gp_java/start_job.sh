sudo yum -y update
sudo yum -y install sharutils
sudo yum -y install mutt
mkdir dissertation
echo "host: aws" > ~/.dissertation_jheaton
echo "project: /home/ec2-user/dissertation" >> ~/.dissertation_jheaton
java -Xmx50G -jar gp_java-1.0-all.jar experiment-1to5 > ~/dissertation/progress.log
