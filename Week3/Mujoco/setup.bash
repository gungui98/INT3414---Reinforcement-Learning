sudo apt install python3.5
sudo apt install pip3
sudo apt get git
pip3 install gym==0.7.4,Mujoco_py==0.5.7,tensorflow,pickle
cd ~/
mkdir .mujoco
cd .mujoco
wget --header="Accept: text/html" --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:21.0) Gecko/20100101 Firefox/21.0" https://www.roboti.us/download/mjpro131_linux.zip
unzip mjpro131_linux.zip