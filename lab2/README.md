# ECE 358 Computer Networks, Group 95, Lab 2

This lab was written by Jonathan Gutschon and Daniel Robson.

This is our submission for the ECE358 Computer Networks Lab. It implements a discrete event simulator to evaluate the performance of LANs using CSMA/CD.

## Running the script

To run questions 1 and 2, enter the following command:

```python
python3 ./main.py
```

To run the graph verification, enter the following command:

```python
python3 ./main.py -q 0
```

To run a specific set of questions, enter the number(s) as an argument:

```python
python3 ./main.py -q <number(s)>

# e.g. question 1 and 2
python3 ./main.py -q 1 2
```

Optionally specify a simulation time using the `-t` argument:

```python
python3 ./main.py -t 1000
```

To print the values in csv format, enter the following command:

```python
python3 ./main.py --print-csv
```
