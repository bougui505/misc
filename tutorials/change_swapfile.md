1. Disable the swap file and delete it (not really needed as you will overwrite it)
```
sudo swapoff /swapfile
sudo rm  /swapfile
```

2. Create a new swap file of the desired size. With thanks to user Hackinet, you can create a 4 GB swap file with the command
```
sudo fallocate -l 4G /swapfile
```
In this command, adjust 4G to the size you want.

3. Assign it read/write permissions for root only (not strictly needed, but it tightens security)
```
sudo chmod 600 /swapfile
```

4. Format the file as swap:
```
sudo mkswap /swapfile
```

5. The file will be activated on the next reboot. If you want to activate it for the current session:
```
sudo swapon /swapfile
```
