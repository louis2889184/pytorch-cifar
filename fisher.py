import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def calculate_the_importance_label(model, data_loader, num_samples, cuda_device, grad_type):
    """
    Args:
        grad_type: (square or absolute) 
    """
    gradients_dict = {}

    for name, param in model.named_parameters():
        gradients_dict[name] = torch.zeros_like(param).to(cuda_device)

    if grad_type == "absolute":
        grad_method = torch.abs
    elif grad_type == "square":
        grad_method = torch.square

    criterion = nn.CrossEntropyLoss()

    for idx, (inputs, labels) in enumerate(data_loader):
        if idx >= num_samples:
            break

        # print(idx)

        inputs = inputs.to(cuda_device)
        labels = labels.to(cuda_device)

        logits = model(inputs)

        loss = criterion(logits, labels)

        loss.backward()

        for name, param in model.named_parameters():
            gradients_dict[name] += grad_method(param.grad).data

        model.zero_grad()

    return gradients_dict


def calculate_the_importance_expect(model, data_loader, num_samples, cuda_device, grad_type):
    """
    Args:
        grad_type: (square or absolute) 
    """
    gradients_dict = {}

    for name, param in model.named_parameters():
        gradients_dict[name] = torch.zeros_like(param).to(cuda_device)

    if grad_type == "absolute":
        grad_method = torch.abs
    elif grad_type == "square":
        grad_method = torch.square

    for idx, (inputs, labels) in enumerate(data_loader):
        if idx >= num_samples:
            break

        inputs = inputs.to(cuda_device)
        labels = labels.to(cuda_device)

        logits = model(inputs)

        log_probs = torch.nn.functional.log_softmax(logits, -1)
        probs = torch.nn.functional.softmax(logits, -1)

        for b in range(logits.shape[0]):
            for i in range(logits.shape[1]):
                loss = - log_probs[b, i]
                loss.backward(retain_graph=True)

                prob = probs[b, i]

                for name, param in model.named_parameters():
                    gradients_dict[name] += (prob * grad_method(param.grad)).data

                model.zero_grad()

    return gradients_dict


def create_mask_gradient_key(model, train_dataset, num_samples, keep_ratio, sample_type, grad_type, key):
    original_device = list(model.parameters())[0].device
    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(cuda_device)

    data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=1)

    if sample_type == "label":
        importance_method = calculate_the_importance_label
    elif sample_type == "expect":
        importance_method = calculate_the_importance_expect
    else:
        raise NotImplementedError

    gradients = importance_method(model, data_loader, num_samples, cuda_device, grad_type)

    # add sizes and aggregate tensors
    sizes = {}
    tensors = []

    classifier_size = 0
    all_params_size = 0

    classifier_mask_dict = {}

    for k, v in gradients.items():
        # don't count classifier layer, they should be all trainable
        if key in k:
            classifier_size += torch.prod(torch.tensor(v.shape)).item()
            classifier_mask_dict[k] = torch.ones_like(v).to(original_device)
        else:
            sizes[k] = v.shape
            tensors.append(v.view(-1))

        all_params_size += torch.prod(torch.tensor(v.shape)).item()

    tensors = torch.cat(tensors, 0)

    keep_num = int(all_params_size * keep_ratio) - classifier_size

    assert keep_num > 0

    top_pos = torch.topk(tensors, keep_num)[1]

    masks = torch.zeros_like(tensors, device=cuda_device)

    masks[top_pos] = 1

    assert masks.long().sum() == len(top_pos)

    mask_dict = {}

    now_idx = 0
    for k, v in sizes.items():
        end_idx = now_idx + torch.prod(torch.tensor(v))
        mask_dict[k] = masks[now_idx: end_idx].reshape(v).to(original_device)
        now_idx = end_idx

    assert now_idx == len(masks)

    # Add the classifier's mask to mask_dict
    mask_dict.update(classifier_mask_dict)

    model.to(original_device)

    # Print the parameters for checking
    classifier_size = 0
    all_params_size = 0
    pretrain_weight_size = 0
    
    for k, v in mask_dict.items():
        if key in k:
            classifier_size += (v == 1).sum().item()
        else:
            pretrain_weight_size += (v == 1).sum().item()

        all_params_size += torch.prod(torch.tensor(v.shape)).item()
    
    print(pretrain_weight_size, classifier_size, all_params_size)
    print(f"trainable parameters: {(pretrain_weight_size + classifier_size) / all_params_size * 100} %")

    return mask_dict



def create_mask_gradient(model, train_dataset, num_samples, keep_ratio, sample_type, grad_type):
    original_device = list(model.parameters())[0].device
    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(cuda_device)

    data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=1)

    if sample_type == "label":
        importance_method = calculate_the_importance_label
    elif sample_type == "expect":
        importance_method = calculate_the_importance_expect
    else:
        raise NotImplementedError

    gradients = importance_method(model, data_loader, num_samples, cuda_device, grad_type)

    # add sizes and aggregate tensors
    sizes = {}
    tensors = []

    classifier_size = 0
    all_params_size = 0

    classifier_mask_dict = {}

    for k, v in gradients.items():
        # don't count classifier layer, they should be all trainable
        if "classifier" in k:
            classifier_size += torch.prod(torch.tensor(v.shape)).item()
            classifier_mask_dict[k] = torch.ones_like(v).to(original_device)
        else:
            sizes[k] = v.shape
            tensors.append(v.view(-1))

        all_params_size += torch.prod(torch.tensor(v.shape)).item()

    tensors = torch.cat(tensors, 0)

    keep_num = int(all_params_size * keep_ratio) - classifier_size

    assert keep_num > 0

    top_pos = torch.topk(tensors, keep_num)[1]

    masks = torch.zeros_like(tensors, device=cuda_device)

    masks[top_pos] = 1

    assert masks.long().sum() == len(top_pos)

    mask_dict = {}

    now_idx = 0
    for k, v in sizes.items():
        end_idx = now_idx + torch.prod(torch.tensor(v))
        mask_dict[k] = masks[now_idx: end_idx].reshape(v).to(original_device)
        now_idx = end_idx

    assert now_idx == len(masks)

    # Add the classifier's mask to mask_dict
    mask_dict.update(classifier_mask_dict)

    model.to(original_device)

    # Print the parameters for checking
    classifier_size = 0
    all_params_size = 0
    pretrain_weight_size = 0
    
    for k, v in mask_dict.items():
        if "classifier" in k:
            classifier_size += (v == 1).sum().item()
        else:
            pretrain_weight_size += (v == 1).sum().item()

        all_params_size += torch.prod(torch.tensor(v.shape)).item()
    
    print(pretrain_weight_size, classifier_size, all_params_size)
    print(f"trainable parameters: {(pretrain_weight_size + classifier_size) / all_params_size * 100} %")

    return mask_dict


def create_all_ones_mask(model, *args, split=2, **kwargs):
    mask_list = []
    for _ in range(split):
        mask = {}
        
        for n, p in model.named_parameters():
            mask[n] = torch.ones_like(p)

        mask_list.append(mask)

    return mask_list


def create_mask_gradient_list(model, train_dataset, num_samples, keep_ratio, sample_type, grad_type, split=2):
    original_device = list(model.parameters())[0].device
    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(cuda_device)

    data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=1)

    if sample_type == "label":
        importance_method = calculate_the_importance_label
    elif sample_type == "expect":
        importance_method = calculate_the_importance_expect
    else:
        raise NotImplementedError

    gradients = importance_method(model, data_loader, num_samples, cuda_device, grad_type)

    # add sizes and aggregate tensors
    sizes = {}
    tensors = []

    classifier_size = 0
    all_params_size = 0

    classifier_mask_dict = {}

    for k, v in gradients.items():
        # don't count classifier layer, they should be all trainable
        if "classifier" in k:
            classifier_size += torch.prod(torch.tensor(v.shape)).item()
            classifier_mask_dict[k] = torch.ones_like(v).to(original_device)
        else:
            sizes[k] = v.shape
            tensors.append(v.view(-1))

        all_params_size += torch.prod(torch.tensor(v.shape)).item()

    tensors = torch.cat(tensors, 0)

    keep_num = int(all_params_size * keep_ratio) - classifier_size

    keep_num = split * keep_num

    assert keep_num > 0

    top_pos = torch.topk(tensors, keep_num)[1]

    # random_indices = torch.randperm(len(top_pos), device=cuda_device)
    
    # Actually is interleave, just not change the naming used before
    random_indices = [torch.arange(i, len(top_pos), split) for i in range(split)]
    random_indices = torch.hstack(random_indices)

    random_indices = random_indices.reshape(split, -1)

    mask_list = []

    for i in range(random_indices.shape[0]):
        # get the indices of the the split
        split_indices = random_indices[i]
        split_top_pos = top_pos[split_indices]
        
        masks = torch.zeros_like(tensors, device=cuda_device)

        masks[split_top_pos] = 1

        assert masks.sum() == len(split_top_pos)

        mask_dict = {}

        now_idx = 0
        for k, v in sizes.items():
            end_idx = now_idx + torch.prod(torch.tensor(v))
            mask_dict[k] = masks[now_idx: end_idx].reshape(v).to(original_device)
            now_idx = end_idx

        assert now_idx == len(masks)

        # Add the classifier's mask to mask_dict
        mask_dict.update(classifier_mask_dict)

        model.to(original_device)

        # Print the parameters for checking
        classifier_size = 0
        all_params_size = 0
        pretrain_weight_size = 0
        
        for k, v in mask_dict.items():
            if "classifier" in k:
                classifier_size += (v == 1).sum().item()
            else:
                pretrain_weight_size += (v == 1).sum().item()

            all_params_size += torch.prod(torch.tensor(v.shape)).item()
        
        print(pretrain_weight_size, classifier_size, all_params_size)
        print(f"trainable parameters: {(pretrain_weight_size + classifier_size) / all_params_size * 100} %")

        mask_list.append(mask_dict)

    return mask_list
