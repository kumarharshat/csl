import torch
import csl
import numpy as np
torch.manual_seed(1234)

####################################
# SIMULATED DATA                   #
####################################
class linearData:
    def __init__(self, dim, n):
        self.wo = torch.ones(dim,1)
        self.x = torch.randn(n,dim)
        self.y = torch.mm(self.x, self.wo) + np.sqrt(1e-3)*torch.randn(n,1)

    def __getitem__(self, idx):
        return self.x[idx,:], self.y[idx]

    def __len__(self):
        return self.x.shape[0]

####################################
# LINEAR MODEL                     #
####################################
class Linear:
    def __init__(self, n_features):
        self.parameters = [torch.zeros([n_features,1], dtype = torch.float, requires_grad = True)]

    def __call__(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(1)

        yhat = torch.mm(x, self.parameters[0])

        return yhat.squeeze()

    def predict(self, x):
        return self(x)

####################################
# CSL PROBLEM                      #
####################################
class QCQP(csl.ConstrainedLearningProblem):
    def __init__(self):
        self.model = Linear(10)
        self.data = linearData(10,100)

        self.obj_function = self.loss
        self.constraints = [lambda batch, primal: torch.mean(self.model.parameters[0]**2)]
        self.rhs = [0.5]
        self.pointwise = [self.pointwise_loss]
        self.pointwise_rhs = [5*torch.ones(len(self.data), requires_grad = False)]

        super().__init__()

    def loss(self, batch_idx):
        # Evaluate objective
        x, y = self.data[batch_idx]
        yhat = self.model(x)

        return torch.mean((yhat - y.squeeze())**2)
        # return torch.ones(1, requires_grad=True)

    def pointwise_loss(self, batch_idx, primal):
        # Evaluate objective
        x, y = self.data[batch_idx]
        yhat = self.model(x)

        return (yhat - y.squeeze())**2

problem = QCQP()

####################################
# CSL SOLVER                       #
####################################
solver_settings = {'iterations': 2000,
                   'batch_size': 10,
                   'primal_solver': lambda p: torch.optim.Adam(p, lr=0.01),
                   'dual_solver': lambda p: torch.optim.Adam(p, lr=0.01),
                   }

solver = csl.PrimalDual(solver_settings)

####################################
# TRAINING                         #
####################################
solver.solve(problem)
solver.plot()