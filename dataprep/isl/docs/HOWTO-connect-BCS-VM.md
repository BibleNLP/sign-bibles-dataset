# Connect to BCS VM

### VM Details
- Name: BCS-CHA-KAVITHASHARE
- OS: Ubuntu 24.04.1 LTS
- Storage: 24 GB only
- Cores: 20 vCPUs
- RAM: 64 GB
- Mounted storage:
    - At /mnt/share
    - 4 TB

### Data Shared by ISL team
- Dictionary videos
    - 3197 mp4s
    - File names indicate gloss
    - 2.5 hours
    - Stored at /mnt/share/original
    - Size on disk: 142 GB
- 4 Gosples
    - 1128 mp4s
    - File names indicate verse ranges (pattern changes through out the set).
    - 19.5 hours of data
    - Stored at /mnt/share/matthew, /mnt/share/mark, /mnt/share/luke, /mnt/share/john
    - Size on disk: 1723 GB

## How to access


- Log in to RDS,from BCS laptop via Remminna, choose BCS-RDS02.
- Need the BCS domain user name and password.
- Also need ADSelfService App setup on mobile to approve push notifivation.
- Once inside the RDS machine(a windows system), teke the app "MobaXterm"
- On the left side pane, servers allotted to you will be listed.
- Select "BCS-CHA-KAVITHASHARE".
- You will get ssh access to the VM.

> sudo access is given to user "kavitha" without requring the password

> htop, is installed for resource monitoring

> nohup, should be used for long running scripts as the terminal will be lost in the next RDS session.

## Data tranfer options

- For small size data tranfer, there is folder navigation on left pane and "upload"/"download" options to the RDS machine. On RDS machine you have internet browser access and can use emails, gdrive, onedrive, etc.

- For code transfer, I recommend using git from the ssh terminal

Other tools
- scp
- wget
- aws s3



