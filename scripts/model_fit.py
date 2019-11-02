from .imports import *

def fit_model(epochs, model, dataloader, phase = 'training', volatile = False):
    
    pprint("Epoch: {}".format(epochs))

    if phase == 'training':
        model.train()
        
    if phase == 'validataion':
        model.eval()
        volatile = True
        
    running_loss = []
    running_acc = []
    b = 0
    for i, data in enumerate(dataloader):
        

        inputs, target = data['image'].cuda(), data['label'].float().cuda()
        
        inputs, target = Variable(inputs), Variable(target)
        
        if phase == 'training':
            optimizer.zero_grad()
            
        ops = model(inputs)
        
        # print("DDD")
        acc_ = []
        for i, d in enumerate(ops, 0):
            # print("OPS: {}".format(torch.round(d)))
            # print("Target: {}".format(target[i]))
            # cd = d.eq()
            acc = pred_acc(torch.Tensor.cpu(target[i]), torch.Tensor.cpu(d))
            # pprint("AACC: {}".format(acc))
            acc_.append(acc)
            # print("DONE")
        # print("DDDD")
        loss = criterion(ops, target)
                
        running_loss.append(loss.item())
        running_acc.append(np.asarray(acc_).mean())
        b += 1
        # pprint("Batch: {} Accuracy: {}".format(b, np.asarray(acc_).mean()))
        # pprint("Batch: {} Loss: {}".format(b, loss.item()))
  
        if phase == 'training':
            
            loss.backward()
        
            optimizer.step()
            
    total_batch_loss = np.asarray(running_loss).mean()
    total_batch_acc = np.asarray(running_acc).mean()
    

    pprint("{} loss is {} ".format(phase,total_batch_loss))
    pprint("{} accuracy is {} ".format(phase, total_batch_acc))
    
    return total_batch_loss, total_batch_acc