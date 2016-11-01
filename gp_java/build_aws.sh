IP="54.167.206.153"
gradle clean
gradle shadowJar
sftp -i ~/jeffheaton.pem ec2-user@$IP:. <<< $'put ./build/libs/gp_java-1.0-all.jar'
sftp -i ~/jeffheaton.pem ec2-user@$IP:. <<< $'put ./start_job.sh'
ssh -i ~/jeffheaton.pem ec2-user@$IP "chmod +x ./start_job.sh;./start_job.sh" &
