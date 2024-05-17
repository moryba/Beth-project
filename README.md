# Beth-project

## Introduction 
The BETH dataset addresses a critical need in cybersecurity research: the availability of real-world, labeled data for anomaly detection. Unlike synthetic datasets, BETH captures genuine host activity and attacks, making it a valuable resource for developing robust machine learning models. 

Moreover, The BETH dataset fills a critical gap in cybersecurity research by providing real-world, labeled data. Its scale, diversity, and structured heterogeneity make it an invaluable resource for advancing anomaly detection techniques and enhancing the robustness of machine learning models in the cybersecurity domain.

**Size and Composition of the dataset:**
- BETH comprises over eight million data points collected from 23 hosts.
- Each host records both benign activity (normal behavior) and, at most, one attack.
- The dataset is diverse, reflecting various types of network traffic and system events.
  
**Structured Heterogeneity of the dataset:**
- BETH’s features are highly structured but heterogeneous.
- This diversity mirrors the complexity of real-world cybersecurity data.
- Features include network traffic statistics, system logs, and process-level information.
  
**Scale and realism of the BETH dataset:**
- BETH is one of the largest publicly available cybersecurity datasets.
- It captures contemporary host behavior, including modern attacks.
- Researchers can use BETH to study the impact of scale on anomaly detection algorithms.

**Behavioral Diversity:**
- The dataset covers a wide range of activities, from routine tasks to malicious actions.
- Hosts exhibit different patterns, making BETH suitable for behavioral analysis.

**Robustness Benchmarking:**
- BETH enables evaluating the robustness of machine learning models.
- Researchers can assess how well their algorithms generalize to unseen attacks.
- It serves as a benchmark for novel anomaly detection techniques.

## Data analysis
The Beth datset represents more than 8 milions events collected over 23 honeypots, only nearly 1 milion of it will be used on this project.
Data are already divided into training, valadating and testing dataset (60% / 20% /20%).


### Features
Each of this dataset has those features:
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

### Training dataset
**Shape:** With 763144 rows, Training subset represents 67% of the data.
**timestamp**
**Process id:**
**ThreadId**
**Parent Process Id**
**User Id**
![userid class distribution](pics/train_userid_plot.png "User ID")
**Mount Name space**
![MountNameSpace distribution](pics/train_mountnamespace_plot.png "Mount name space")
**Process Name**
**Host Name**
![HostName distribution](pics/train_hostname_plot.png "Host name")
**Events**
![EventId distribution](pics/train_eventid_plot.png "Event id")
![EventName distribution](pics/train_eventname_plot.png "Event name")
**Stack Addresses**

**ArgsNum**
**ReturnValue**
**Args**
**Suspicious:**
![Suspicious class distribution](pics/train_suspicious_plot.png "Suspicious")
**Evil**
### Validation dataset

### Testing dataset


## Data preparation
### Numerical data transformation
### StackAddresses
### Args
### Ordinal encoding
### Scaling
### Smote:
Dealing with unbalanced data can be tricky, most of the machine learning model will give good results for big classes and poor performance on the minority althought, as it is our case, minority class is more important.
To balance that, we tried to use Smote library combined as it is adviced with randoom undersampling for the majority class.
SMOTE (Synthetic Minority Oversampling TEchnique) works by interpolating new instances along line segments joining existing minority class instances.
