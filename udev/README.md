cd /etc/udev/rules.d
# download to here
sudo udevadm control --reload-rules
sudo udevadm trigger

# quest
sudo apt install adb

wget https://raw.githubusercontent.com/M0Rf30/android-udev-rules/master/51-android.rules
sudo cp 51-android.rules /etc/udev/rules.d/
sudo chmod a+r /etc/udev/rules.d/51-android.rules

sudo service udev restart #reconnect device

sudo udevadm control --reload-rules
sudo udevadm trigger


# (need connect quest)
adb reverse tcp:8080  tcp:8080 #网页
adb reverse tcp:7880  tcp:7880 #本地lvekit
