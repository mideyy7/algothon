import os
os.environ["CMI_USER"] = "testUser"
os.environ["CMI_PASS"] = "testPass"

from main import IMCBot
bot = IMCBot("http://ec2-52-49-69-152.eu-west-1.compute.amazonaws.com/", "testUser", "testPass")
# Just check if it initializes and if start fails. Wait, auth will fail.
