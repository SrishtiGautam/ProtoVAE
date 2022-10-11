import os
import torch

def save_model_w_condition(model, model_dir, model_name, accu, target_accu, log=print):
    '''
    model: this is not the multigpu model
    '''
    if accu > target_accu:
        log('\tabove {0:.2f}%'.format(target_accu * 100))
        print(os.path.join(model_dir, (model_name + '____{0:.4f}.pth').format(accu)))
        with open(os.path.join(model_dir,'saved_model.txt'), 'w') as f:
            f.write('epoch:'+model_name+", acc:{0:.4f}".format(accu))
        # torch.save(obj=model, f=os.path.join(model_dir, ('model.pth').format(accu)))
        torch.save(obj=model.state_dict(), f=os.path.join(model_dir, ('model.pth').format(accu)))
