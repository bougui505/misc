## Install fail2ban

Update and install Fail2Ban by typing the following commands:

	$> sudo apt-get update
	$> sudo apt-get install fail2ban

Edit SSH Fail2Ban configurations. Open up the "/etc/fail2ban/jail.local" file with the following command (jail.local file should be empty):

	$> sudo nano /etc/fail2ban/jail.local

	[ssh]
	enabled = true
	port = ssh
	filter = sshd
	logpath = /var/log/auth.log
	bantime = 900
	banaction = iptables-allports
	findtime = 900
	maxretry = 3

Restart Fail2Ban with the following command:

	$> sudo service fail2ban restart


Check IPTables list with the following command to see all your banned IP Addresses:

	$> sudo iptables -L -n --line


If you need to unban an IP Address use this command. Change the number to the line you want to remove:

	$> sudo iptables -D fail2ban-ssh 1

List banned ip:

  $> sudo fail2ban-client banned
