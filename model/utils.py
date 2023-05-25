import os
def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * ((1 - 0.015) ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(dista, distb):
    margin = 0
    pred = (dista - distb - margin).cpu().data
    return float((pred > 0).sum())/(dista.size()[0]+1e-12)

def accuracy_id(dista, distb, c, c_id):
    margin = 0
    pred = (dista - distb - margin).cpu().data
    a = float(((pred > 0)*(c.cpu().data == c_id)).sum())
    b = float((c.cpu().data == c_id).sum())
    if b==0:
        c = 0
    else:
        c = a/b
    return b,c


# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#     """Saves checkpoint to disk"""
#     directory = "runs/%s/"%(args.name)
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     filename = directory + filename
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')

# def save_checkpoint(is_best, state, save_path, filename='checkpoint.pth.tar'):      
#     torch.save(state, os.path.join(save_path, filename))
#     print('i have saved................................................')
#     if is_best:
#         shutil.copyfile(os.path.join(args.save_path, filename), os.path.join(args.save_path, 'model_best.pth.tar'))
