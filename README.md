# Beth-project

## Data analysis
The Beth datset represents more than 8 milions events collected over 23 honeypots, only nearly 1 milion of it will be used on this project.
Data are already divided into training, valadating and testing dataset (60% / 20% /20%).


### Features
Each of this dataset has this features:
 - timestamp: time in seconds since system  boot (float)
 - processId: id of the process spawning this log (integer)
 - threadId: id of the thread (integer)
 - parentProcessId: paren process id (integer)
 - userId: login integer id (integer)
 - mountNamespace: Set mounting restrictions this process log (integer)
 - processName: command executed (string)
 - hostName: host server (string)
 - eventId: id of the event generating this log (integer)
 - eventName: name of the event (string)
 - stackAddresses: memory values relevant to the process (list of integer)
 - argsNum: number of arguments (integer)
 - returnValue: value returned from this event log (integer)
 - args: arguments passed to this process (list of dictionaries)
 - sus: label (0/1) for suspicious activity (integer)
 - evil: label (0/1) for evil activity (integer)