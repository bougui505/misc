# npcli
Feed stdin data to a numpy array (default variable name is `A`) and apply arbitrary numpy operation on it and print the result on stdout.
```
$ np -h

usage: np [-h] [--nopipe] [-d DELIMITER] cmd

Using python and numpy from the Shell

positional arguments:
  cmd                   Command to run

optional arguments:
  -h, --help            show this help message and exit
  --nopipe              Not reading from pipe
  -d DELIMITER, --delimiter DELIMITER
                        Delimiter to use
```
```
$ paste =(seq 10) =(seq 11 20) | np 'print(A)'

1 11
2 12
3 13
4 14
5 15
6 16
7 17
8 18
9 19
10 20
```
```
$ paste =(seq 10) =(seq 11 20) | np 'mu=A.mean(axis=0);print(mu)'

5.5 15.5
```
```
$ paste =(seq 10) =(seq 11 20) | np 'mu=A.mean(axis=1);print(mu)'

6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
```
The python `print` command has been overwritten to print results as a shell friendly format. Therefore to print two variables you have to invoke the \'print\' command for each:
```
$ paste =(seq 10) =(seq 11 20) | np 'print(A.min());print(A.max())'


```
Change the delimiter with the -d option
```
$ paste -d ',' =(seq 10) =(seq 11 20) | np -d',' 'mu=A.mean(axis=0);print(mu)'

5.5,15.5
```
