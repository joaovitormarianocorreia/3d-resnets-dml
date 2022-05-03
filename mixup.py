import torch
import numpy as np

def find_neighbours(num_neighbours, centres, queries=None):

	if queries is None:
		dist_matrix = torch.cdist(centres, centres)
		dist_matrix[torch.diag(torch.ones(dist_matrix.size(0))).type(torch.bool)] = float('inf')
	else:
		dist_matrix = torch.cdist(queries, centres)

	return torch.argsort(dist_matrix)[:,0:num_neighbours]


def update_centres(model, device, batch_size, centres, data_loader):
	model.eval()

	with torch.no_grad():
		for i, data in enumerate(data_loader, 0):
			inputs, labels = data
			inputs = inputs.to(device)
			extracted_features = model(inputs)
			idx = i * batch_size
			centres[idx:idx + extracted_features.shape[0], :] = extracted_features

	model.eval()
	return centres

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



class GaussianKernels(torch.nn.Module):
    def __init__(self, num_classes, num_neighbours, num_centres, sigma):
        torch.nn.Module.__init__(self)

        self.num_classes       = num_classes
        self.num_neighbours    = num_neighbours
        self.gaussian_constant = 1.0/(2.0 * (sigma**2))

        # trainable per-kernel weights - parameterised in log-space (exp(0) = 1)
        self.weight = torch.nn.Parameter(torch.zeros(num_centres, requires_grad=True))
        
    def forward(self, features, centres, centre_labels, neighbours=None):

        batch_size_it  = features.size(0)
        num_dims       = features.size(1)
        p      = []
        p2 = []
        
        # tensor with [0,1,...,num_classes]
        classes = torch.linspace(0,self.num_classes-1,self.num_classes).type(torch.LongTensor).view(self.num_classes,1).to(features.device)

        # per-kernel weights (self.weight in log space)
        kernel_weights = torch.exp(self.weight)

        # if in test phase, find neighbours
        if neighbours is None:
            neighbours = find_neighbours(self.num_neighbours, centres, features)
        
        for ii in range(batch_size_it):
            
            # neighbour indices for current example
            neighbours_ii = neighbours[ii,:]
            
            # squared Euclidean distance to neighbours
            d = torch.pow( features[ii,:] - centres[neighbours_ii], 2 ).sum(1)
            
            # weighted Gaussian distance
            d = torch.exp( -d * self.gaussian_constant ) * kernel_weights[neighbours_ii]

            # labels of neighbouring centres
            neighbour_labels = centre_labels[neighbours_ii].view(1,neighbours_ii.size(0))

            # sum per-class influence of neighbouring centres (avoiding loops - need 2D tensors)
            p_arr      = torch.zeros(self.num_classes,d.size(0)).type(torch.FloatTensor).to(features.device)
            idx        = classes==neighbour_labels
            p_arr[idx] = d.expand(self.num_classes,-1)[idx]
            p_ii       = p_arr.sum(1) # unnormalsied class probability distribution

            # avoid divide by zero and log(0)
            p_ii[p_ii==0] = 1e-10

            # normalise
            p_ii = p_ii / p_ii.sum()

            # convert to log-prob
            p.append(torch.log(p_ii).view(1,self.num_classes))
            p2.append(p_ii.view(1,self.num_classes))
        return torch.cat(p), torch.cat(p2)
