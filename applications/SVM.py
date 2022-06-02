"""SVM application

Simple SVM with blobs application to test solver

"""
import torch
import csl
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC
torch.manual_seed(1234)


####################################
# SIMULATED DATA                   #
####################################
class linearData:
    def __init__(self, dim, n):
        centers = [[0.0 for i in range(dim)], [3.5 for i in range(dim)]]
        X, y = make_blobs(n_samples=[int(n/2), int(n/2)], centers=centers, random_state=5)
        y[y == 0] = -1
        self.x = torch.from_numpy(np.hstack((X,np.ones((len(y),1)))), ).float()
        self.y = torch.from_numpy(y).float()

    def __getitem__(self, idx):
        return self.x[idx,:], self.y[idx]

    def __len__(self):
        return self.x.shape[0]

####################################
# LINEAR MODEL                     #
####################################
class Linear:
    def __init__(self, n_features):
        self.parameters = [torch.rand([n_features +1,1], dtype = torch.float, requires_grad = True)]

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
class SVM(csl.ConstrainedLearningProblem):
    def __init__(self, C = 1):
        self.C = C
        self.model = Linear(2)
        self.data = linearData(2,100)

        self.obj_function = self.loss
        self.constraints = []
        self.rhs = []
        self.pointwise = [self.pointwise_loss]
        self.pointwise_rhs = [-0.00001*torch.ones(len(self.data), requires_grad = False)]

        super().__init__()

    def loss(self, batch_idx):
        # Evaluate objective
        return 0*torch.norm(self.model.parameters[0]) + self.C*torch.dot(torch.ones(len(self.data)), self.pointwise_rhs[0])
        # return torch.ones(1, requires_grad=True)

    def pointwise_loss(self, batch_idx, primal):
        # Evaluate objective
        x, y = self.data[batch_idx]
        yhat = self.model(x)


        return torch.ones_like(yhat) - torch.mul(yhat, y)


problem = SVM()

####################################
# CSL SOLVER                       #
####################################
solver_settings = {'iterations': 10000,
                   'batch_size': 100,
                   'primal_solver': lambda p: torch.optim.Adam(p, lr=0.01),
                   'dual_solver': lambda p: torch.optim.Adam(p, lr=0.01),
                   }

solver = csl.PrimalDual(solver_settings)

####################################
# TRAINING                         #
####################################
solver.solve(problem)
solver.plot()

#####################################
# Compare against SCIKIT            #
#####################################
print('SCIKIT l1')
LD = linearData(2,100)
# svclassifier = SVC(kernel='linear', max_iter=n_iter)
svclassifier = LinearSVC(max_iter=100, penalty='l1', dual=False, C = 1)
svclassifier.fit(LD.x, LD.y)
y_pred = svclassifier.predict(LD.x)
print(len(np.where(torch.from_numpy(y_pred) == LD.y)[0])/len(LD.y))