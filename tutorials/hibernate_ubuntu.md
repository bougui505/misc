# Setting hibernate on Ubuntu 20.04.5 LTS

See: https://askubuntu.com/a/1075516

First of all **disable the secure boot on bios** if enabled.

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

See: https://askubuntu.com/a/1321773

6. Install dependencies:
```
sudo apt install pm-utils hibernate uswsusp
```

7. Find your UUID and swap offset:
```
findmnt -no UUID -T /swapfile && sudo swap-offset /swapfile
```

8. Edit `/etc/default/grub` and replace the string:
```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"
```
with your UUID and offset:
```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash resume=UUID=371b1a95-d91b-49f8-aa4a-da51cbf780b2 resume_offset=23888916"
```

9. Update GRUB:
```
sudo update-grub
```

10. Test your hibernation:
```
sudo systemctl hibernate
```
