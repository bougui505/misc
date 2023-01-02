
See: https://www.gnu.org/software/ddrescue/manual/ddrescue_manual.html#Optical-media

```
ddrescue -n -b2048 /dev/dvd dvdimage mapfile
ddrescue -d -r1 -b2048 /dev/dvd dvdimage mapfile
```

See: https://askubuntu.com/a/1209008

```
ddrescue -d -R -r1 -b2048 /dev/dvd dvdimage mapfile
```

skip a 1MB chunk of disk each time there's an unrecoverable disk read error
(https://unix.stackexchange.com/a/665498)
ddrescue -d -r1 -b2048 /dev/dvd dvdimage mapfile

Also try to read the dvd with vlc and then pause. While keeping VLC opened try to recover the dvd with `ddrescue`
