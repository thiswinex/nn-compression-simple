from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import numpy as np

layerdict = []

class ADMM():
    def __init__ (self, model, update_interval=10, rho=1e-3, mu=10, tau_incr=1.5, tau_decr=1.5):
        """Base class for optimize method ADMM.

        Args:
            model (nn.Module): model needs to be optimize
            update_interval (int, optional): ADMM update interval. Defaults to 10 epoch.
        """
        super(ADMM, self).__init__()
        self.model = model

        self.keys = []
        self.W = []
        self.U = []
        self.Z = []
        self.preW = []
        self.storeW = []
        self.r = []
        self.s = []

        # initialize primal variable W / dual variable Z / residual U
        # modify here if you want to customize the layers you want to operate
        for key, value in model.named_parameters():
            if "weight" in key:
                self.keys.append(key)
                self.W.append(value)
                self.preW.append(value.clone())
                self.storeW.append(value.clone())
                self.Z.append(value.data.clone())
                self.U.append(value.data.clone().zero_())
                self.r.append(value.data.clone().zero_())
                self.s.append(value.data.clone().zero_())
        
        self.rho = torch.zeros((len(self.W)),dtype=torch.float).fill_(rho) # rho parameter (admm loss)
        self.update_interval = update_interval
        self.mu = mu
        self.tau_incr = tau_incr
        self.tau_decr = tau_decr


    def update(self, epoch):
        raise NotImplementedError
            

    def grad_update(self):
        """ Update gradient. It serves the same purpose with function "loss_update()" below, so just use one of them instead of two.
            Example:
            >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            >>> optimizer.zero_grad()
            >>> loss = loss_fn(model(input), target)
            >>> loss.backward()
            >>> admm.grad_update()
            >>> optimizer.step()
        """
        for i in range(len(self.W)):
            grad = self.W[i].data - (self.Z[i]) + (self.U[i])
            grad = grad * self.rho[i]
            self.W[i].grad.data += grad


    def loss_update(self, loss):
        """Update loss. It serves the same purpose with function "grad_update()" above, so just use one of them instead of two.
            Example:
            >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            >>> optimizer.zero_grad()
            >>> loss = loss_fn(model(input), target)
            >>> admm.loss_update(loss)
            >>> loss.backward()
            >>> optimizer.step()

        Args:
            loss (nn.Tensor): model loss
        """
        for i in range(len(self.W)):
            loss += self.rho[i] / 2 * torch.norm((self.W[i] - self.Z[i] + self.U[i]), 2)


    def print_info(self, epoch):
        """Print infomation to ensure that the optimization process is carried out correctly.
        """
        if epoch % self.update_interval !=0:
            return
        print('\n' + '-' * 30)
        for i in range(len(self.W)):
            print(self.keys[i])
            print('W val:',self.W[i].data.view(1,-1))
            print('Z val:',self.Z[i].data.view(1,-1))
            print('U val:',self.U[i].data.view(1,-1))
            print('2-norm of W and Z:',torch.norm(self.r[i].view(1,-1)))
            print('L2 norm of prime residual:',torch.norm(self.s[i].view(1,-1)))
            print('rho val:',self.rho[i])
        print('\n' + '-' * 30)


    def apply_projW(self):
        """Apply projection to the model (using Z to replace W). Use "restoreW()" to restore W from being projected.
        """
        for i in range(len(self.W)):
            self.storeW[i] = self.W[i].clone()
            self.W[i].data.copy_(self.Z[i])

    def restoreW(self):
        """Restore W from being projected.
        """
        for i in range(len(self.W)):
            self.W[i].data.copy_(self.storeW[i])


class ADMM_pruning(ADMM):
    def __init__ (self, model, update_interval=3, l1=False, prune_type=0, rho=1e-3, phi=1e-5, mu=10, tau_incr=1.5, tau_decr=1.5):
        """Implements ADMM pruning method.
        Args:
            model (nn.Module): model needs to be optimize
            update_interval (int, optional): ADMM update interval. Defaults to 10 epoch.
            l1 (bool, optional): Using l1 norm to sparsify the model instead of specifying pruning rate. Defaults to False.
            prune_type (int, optional): Type of pruning. 0 for normal pruning, 1 for channel pruning and 2 for filter pruning. Defaults to 0.
            phi (float, optional): Regularization parameter of l1 norm. Defaults to 1e-5.
        """

        super(ADMM_pruning, self).__init__(model, update_interval, rho, mu, tau_incr, tau_decr)

        self.mask = []
        self.l1 = l1
        self.prune_percent = []
        self.prune_type = prune_type

        # init mask and pruning rate.
        for i in range(len(self.W)):
            self.mask.append(self.W[i].clone().zero_())
            self.prune_percent.append(None if self.l1 else 50)
        
        self.phi = torch.zeros((len(self.W)),dtype=torch.float).fill_(phi) # phi parameter (l1 loss)

        # init Z
        for i in range(len(self.W)):
            self.Z[i], self.mask[i] = self.pruning_Z_update(i)


    def update(self, epoch):
        """ ADMM update function.
            Example:
            >>> for epoch in range(1, args.epochs + 1):
            >>>     train(args)
            >>>     test(args)
            >>>     admm.update(epoch)
        """
        if epoch % self.update_interval != 0:
            return
        for i in range(len(self.W)):
            # update Z
            self.Z[i], self.mask[i] = self.pruning_Z_update(i, percent=self.prune_percent[i], prune_type=self.prune_type)

            # update U
            self.U[i] += self.W[i] - self.Z[i]

            # admm-update end
            
            # update rho
            self.r[i] = self.W[i].data - self.Z[i]
            self.s[i] = self.rho[i] * (self.W[i].data - self.preW[i].data)
            # F-norm
            flatten_r = self.r[i].view(1,-1)
            flatten_s = self.s[i].view(1,-1)
            norm_r = torch.sqrt(torch.sum(torch.mul(flatten_r, flatten_r))) 
            norm_s = torch.sqrt(torch.sum(torch.mul(flatten_s, flatten_s))) 
            if norm_r > self.mu * norm_s:
                # set upper bound of rho
                if self.rho[i] < 0.4:
                    self.rho[i] = self.rho[i] * self.tau_incr
                    # the scaled dual variable u = (1/ρ)y must also be rescaled
                    # after updating ρ
                    self.U[i] = self.U[i] / self.tau_incr
            elif norm_s > self.mu * norm_r:
                self.rho[i] = self.rho[i] / self.tau_decr
                # the scaled dual variable u = (1/ρ)y must also be rescaled
                # after updating ρ
                self.U[i] = self.U[i] * self.tau_decr
            
            self.preW[i] = self.W[i].clone()
        self.print_info(epoch)


    def pruning_Z_update(self, i): 
        """The opertaion that projects Z to a pruning set.

        Args:
            i (int): layer(tensor) index

        Returns:
            Z: Z after projection
            mask: a mask for masking gradient during finetuning.
        """
        # 0 for normal pruning / 1 for channel pruning / 2 for filter pruning
        # if percent = None: it means using L1 norm to determine the pruning ratio
        prune_percent = self.percent[i]
        prune_type = self.prune_type
        if percent is None:
            delta = self.phi[i] / self.rho[i]
            Z = self.W[i] + self.U[i]
            Z_new = Z.detach()
            if (Z > delta).sum() != 0:
                Z_new[Z > delta] = Z[Z > delta] - delta
            if (Z < -delta).sum() != 0:
                Z_new[Z < -delta] = Z[Z < -delta] + delta
            if (abs(Z) <= delta).sum() != 0:
                mask = (abs(Z) <= delta)
                Z_new[mask] = 0
        else:
            Z = self.W[i] + self.U[i]
            Z_new = Z.detach()

            if len(Z.shape) != 4:   # if not conv, use normal pruning
                prune_type = 0
            if prune_type == 0:
                percent = int(percent * Z.numel()) // 100
                Z_abs = torch.abs(Z)
                thres, _ = torch.kthvalue(Z_abs.view(-1), percent)
                mask = (Z_abs < thres)
                Z_new[mask] = 0
            elif prune_type == 1: # channel pruning
                Z_abs = torch.abs(Z)
                l1_norm = torch.sum(Z_abs, dim=(0,2,3))
                percent = int(percent * l1_norm.numel()) // 100
                thres, _ = torch.kthvalue(l1_norm, percent)
                mask = (l1_norm < thres).view(1, -1, 1, 1).broadcast_to(Z.shape)
                Z_new[mask] = 0
            elif prune_type == 2: # filter pruning
                Z_abs = torch.abs(Z)
                l1_norm = torch.sum(Z_abs, dim=(1,2,3))
                percent = int(percent * l1_norm.numel()) // 100
                thres, _ = torch.kthvalue(l1_norm, percent)
                mask = (l1_norm < thres).view(1, -1, 1, 1).broadcast_to(Z.shape)
                Z_new[mask] = 0
            else:
                raise NotImplementedError(f'Unknown pruning type.')
        return Z_new, mask

            
    def grad_mask(self):
        """ mask gradient during finetuning            
            Example:
            >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            >>> optimizer.zero_grad()
            >>> loss = loss_fn(model(input), target)
            >>> loss.backward()
            >>> if epoch > finetune_epoch:
            >>>     admm.grad_mask()
            >>> optimizer.step()
        """
        for i in range(len(self.W)):
            self.W[i].grad.data[self.mask[i]] = 0




class ADMM_quantization(ADMM):
    def __init__ (self, model, update_interval=3, bits=[4 for i in range(1000)], rho=1e-3, mu=10, tau_incr=1.5, tau_decr=1.5):
        """Implements ADMM quantization method.
        Args:
            model (nn.Module): model needs to be optimize
            bits (list): quantized bits of each layer. Defaults to 4 for every layer.
            update_interval (int, optional): ADMM update interval. Defaults to 10 epoch.
        """

        super(ADMM_quantization, self).__init__(model, update_interval, rho, mu, tau_incr, tau_decr)
        self.b = bits
        self.a = torch.zeros((len(self.W)),dtype=torch.float) # a is the scale factor of quatized data

        if len(self.b) < len(self.W):
            raise ValueError("Each layer should have a specify quantization bit count.")

        # init Z
        #for i in range(len(self.W)):
        #    self.Z[i], self.a[i] = self.quantization_Z_update(i)

        # init a
        if bits is not None:
            for i in range(len(self.W)):
                self.a[i] = 0.5 / (pow(2,bits[i])-1)


    def update(self, epoch):
        """ ADMM update function.
            Example:
            >>> for epoch in range(1, args.epochs + 1):
            >>>     train(args)
            >>>     test(args)
            >>>     admm.update(epoch)
        """
        if epoch % self.update_interval != 0:
            return
        for i in range(len(self.W)):
            # update Z
            self.Z[i], self.a[i] = self.quantization_Z_update(i)

            # update U
            self.U[i] += self.W[i] - self.Z[i]

            # admm-update end
            
            # update rho
            self.r[i] = self.W[i].data - self.Z[i]
            self.s[i] = self.rho[i] * (self.W[i].data - self.preW[i].data)
            # F-norm
            flatten_r = self.r[i].view(1,-1)
            flatten_s = self.s[i].view(1,-1)
            norm_r = torch.sqrt(torch.sum(torch.mul(flatten_r, flatten_r))) 
            norm_s = torch.sqrt(torch.sum(torch.mul(flatten_s, flatten_s))) 
            if norm_r > self.mu * norm_s:
                # set upper bound of rho
                if self.rho[i] < 0.4:
                    self.rho[i] = self.rho[i] * self.tau_incr
                    # the scaled dual variable u = (1/ρ)y must also be rescaled
                    #after updating ρ
                    self.U[i] = self.U[i] / self.tau_incr
            elif norm_s > self.mu * norm_r:
                self.rho[i] = self.rho[i] / self.tau_decr
                # the scaled dual variable u = (1/ρ)y must also be rescaled
                #after updating ρ
                self.U[i] = self.U[i] * self.tau_decr
            
            self.preW[i] = self.W[i].clone()
        self.print_info(epoch)
            

    def quantization_Z_update(self, i):
        """The opertaion that projects Z to a quantization set.

        Args:
            i (int): layer(tensor) index

        Returns:
            Z: Z after projection
            a: scaling factor of quantization
        """
        #Vi = (W + U)
        V = self.W[i] + self.U[i]
        Z = self.Z[i]
        a = self.a[i]
        Q = (Z / a) 

        # update a
        a = torch.sum(torch.mul(V,Q)) / torch.sum(torch.mul(Q,Q))

        # update Z
        Q = V / a
        Q = torch.round((Q-1)/2)*2+1
        Q = torch.clamp(Q,-(pow(2,self.b[i])-1),pow(2,self.b[i])-1)

        Z = Q * a
        return Z, a
    