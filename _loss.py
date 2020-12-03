import torch

def custom_loss_1(output, target):

    #### base point is the first value of target variable ####

    outsgn = torch.sign(output - target[0]*output.shape[1])
    trgsgn = torch.sign(target - target[0]*output.shape[1])
    binary = torch.add((outsgn != trgsgn),1)
    se = (output - target)**2
    loss = torch.mean(se * binary)
    
    return loss


def custom_loss_2(output, target) :

    #### base point is the previous value of target variable ####

    zero = torch.zeros(1,output.shape[1]).to('cuda')
    binary = torch.sign(output[1:] - target[:-1]) != torch.sign(target[1:]- target[:-1])
    binary = torch.cat((zero,binary))
    binary = torch.add(binary,1)
    se = (output - target) ** 2
    loss = torch.mean(se * binary)

    return loss
