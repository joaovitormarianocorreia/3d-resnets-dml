import torch
import numpy as np

def update_centres(centres, model, update_loader, batch_size, device):

	# disable dropout, use global stats for batchnorm
	model.eval()

	# disable learning
	with torch.no_grad():

		# update stored centres
		for i, data in enumerate(update_loader, 0):

			# get the inputs; data is a list of [inputs, labels]. Send to GPU
			inputs, labels, index = data
			inputs = inputs.to(device)
    
			# extract features for batch
			extracted_features = model(inputs)

			# save to centres tensor
			idx = i*batch_size
			centres[idx:idx + extracted_features.shape[0], :] = extracted_features

	model.train()

	return centres


def mixup_data(x, y, device, alpha=1.0):
    """
    Return mixed inputs, pair of targets and lambda
    """
    if (alpha > 0):
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]

    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1-lam)*x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1-lam) * criterion(pred, y_b)