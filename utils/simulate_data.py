import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


def simulate_data(mu1,sigma1,mu2,sigma2):
    cont = True
#* * (2.1) Sample number of Class-1 and Class-2 and Region of Interests are chosen by user.
    prompt = 'Select the number of class 1 graphs: '
    C1 = int(input(prompt))
    while C1=="":
        prompt = 'Please choose a number: '
        C1 = int(input(prompt))
    while (C1 < 10):
        prompt = 'Please choose a number >9: '
        C1 = int(input(prompt))

    prompt = 'Select the number of class 2 graphs: '
    C2 = int(input(prompt))
    while C2=="":
        prompt = 'Please choose a number: '
        C2 = int(input(prompt))
    while (C2 < 10):
        prompt = 'Please choose a number >9: '
        C2 = int(input(prompt))
    prompt = 'Select the number of nodes (i.e., ROIS for brain graphs): '
    m = int(input(prompt))
    while (m==""):
        prompt = 'Please choose a number >20: '
        m = int(input(prompt))

    while ((m < 21)):
        prompt = 'Please choose a number >20: '
        m = int(input(prompt))

# * *

# * * (2.2) Matrixes are created as chosen numbers by user before which has random numbers.(not network atlasses yet.)

    N = C1 + C2 #total sample number
    dataC1 = np.random.normal(mu1, sigma1, [C1, m, m])
    dataC2 = np.random.normal(mu2, sigma2, [C2, m, m])
    data1 = np.append(dataC1,dataC2,axis=0) #main array which include random number matrixes of both classes

# * *

# * * (2.3) Matrixes with Random numbers are converted to Connectivity Matrixes
    for i in range(N):

        data1[i, :, :] = data1[i, :, :] - np.diag(np.diag(data1[i,:,:])) #Removing diagonal elements of each matrixes
        data1[i, :,:] = (data1[i, :,:] + data1[i, :,:].transpose())/2 #Converting each matrixes symetric connectivity matrixes
        t = np.triu(data1[i,:,:]) #Taking upper triangular part of each converted matrixes
        x=t[np.triu_indices(m,1)] #Creating 1xK matrixes which K is connectivity number of upper triangular part.
        x1=x.transpose()
# * *

# * * (2.4) All edited datas are added new arrays and these arrays are added in main array which called Data.
        if cont:
            Featurematrix = np.empty((0, x1.shape[0]), int)
            cont=False
        Featurematrix = np.append(Featurematrix, np.array([x1]), axis=0)
        #Every loop step , random matrixes fixed and updated.
    #Labels are created and added into Label array as one by one. So they correspond with datas which they belong.
    Label1 = np.ones((C1, 1),dtype=np.int)
    Label2 = np.ones((C2, 1),dtype=np.int)*0
    Label=np.append(Label1,Label2,axis=0).flatten()
    #Feature Matrix-->NxC
    #Label-->Nx1
    return Featurematrix,Label


