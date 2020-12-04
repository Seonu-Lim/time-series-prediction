import torch

class CustomLoss(object) :

    #### user can custom weight ####

    def __init__(self,w1,w2) :

        self.w1 = w1
        self.w2 = w2
        self.eps = 1e-7

    def custom_loss_1(self, output, target):

        ### baseline is the first value of target variable ###

        outsgn = torch.sign(output - target[0]*output.shape[1])
        trgsgn = torch.sign(target - target[0]*output.shape[1])
        binary = (outsgn != trgsgn)

        w1 = torch.ones(*binary.shape) * self.w1
        w2 = torch.ones(*binary.shape) * self.w2
        w1 = w1.to('cuda')
        w2 = w2.to('cuda')

        binary = torch.where( binary < self.eps, w1, w2)
        se = (output - target)**2
        loss = torch.mean(se * binary)

        return loss

    def custom_loss_2(self,output,target):

        ### baseline is the previous value of target variable ###

        zero = torch.zeros(1,output.shape[1]).to('cuda')
        binary = torch.sign(output[1:] - target[:-1]) != torch.sign(target[1:]- target[:-1])
        binary = torch.cat((zero,binary))
        w1 = torch.ones(*binary.shape) * self.w1
        w2 = torch.ones(*binary.shape) * self.w2
        w1 = w1.to('cuda')
        w2 = w2.to('cuda')

        binary = torch.where( binary < self.eps, w1, w2)
        se = (output - target) ** 2
        loss = torch.mean(se * binary)

        return loss
