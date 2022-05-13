import os
import sys
       

fileName = sys.argv[1]

print("Configuration file {}.yml".format(fileName))


ymlFile = open("configurations/{}.yml".format(fileName)) 
fileSave = 'results/{}'.format(fileName)    
          

# Create simulation director if not existing
if not os.path.isdir(fileSave):
    print("Did not find %s/, creating directory" % fileSave)
                   
    os.system("mkdir %s" % fileSave)

    os.system("mkdir %s/data_set" % fileSave)

    os.system("mkdir %s/FEM_results" % fileSave)

    os.system("mkdir %s/plots" % fileSave)
else:
    print("Result directory alreade existing\n")
    
exit(0)
