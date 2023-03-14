#coding:utf8
import ipdb;
import models
import time

from models.rec_models.ACNN.config import opt
from models.rec_models.ACNN import models as mds
from models.rec_models.ACNN.data.dataset import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchnet import meter
from models.rec_models.ACNN.utils.visualize import Visualizer
from skimage.metrics import structural_similarity as ssim

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

@t.no_grad() # pytorch>=0.5

def test(**kwargs):
    opt._parse(kwargs)

    vis = Visualizer(opt.env + '_test',port = opt.vis_port)

    
    model = getattr(mds, opt.model)(8 * 2 * opt.slice_num, 8 * 2 * opt.slice_num, 64, 5, 0, opt.slice_num, False)
    # if opt.test_model_path:
    #     model.load('ACNN/'+opt.test_model_path)
    model.to(opt.device)

    # data
    mask_smp_path = opt.undersample_mask #"./data/mask_smp.mat"
    test_data = Mridata(opt.val_data_root, mask_smp_path, slice_num=opt.slice_num, test=True)
    test_dataloader = DataLoader(test_data,opt.val_batch_size,
                        shuffle=False,num_workers=opt.num_workers)
    model.eval()
    rec_total_nmse  = 0
    rec_total_psnr  = 0
    rec_total_ssim  = 0
    data_total_nmse = 0
    data_total_psnr = 0
    data_total_ssim = 0
    for ii, (val_input, label, sen, dc_fft) in tqdm(enumerate(test_dataloader)):
        print(ii)
        val_input = val_input.to(opt.device)
        
        kl = fft2(label).data.numpy()   
        label = label.data.numpy() # b*c*W*h*2 real
        
        s_t = time.clock()
        val_pred, ks, ke, c_weight, c_weight2 = model(val_input)
        e_t = time.clock()
        print(e_t - s_t)
        ipdb.set_trace()

        for i in range(11):
            cv2.imwrite(str(i)+'.png', cv2.resize(c_weight[i][0,0,:,:].cpu().numpy()*255, dsize=(320, 320), interpolation=cv2.INTER_CUBIC))
        ipdb.set_trace()
        data_input = val_input.data.cpu().numpy()       
        val_pred = val_pred.data.cpu().numpy()
        sen = sen.data.numpy()       
        ks = ks.data.cpu().numpy() 
        ke = ke.data.cpu().numpy() 

        label = label[:,8:16,:,:,:]
        data_input = data_input[:,8:16,:,:,:]
        val_pred = val_pred[:,:,:,:,:]
        sen = sen[:,8:16,:,:,:]

        kl = kl[:,8:16,:,:,:]
        ks = ks[:,8:16,:,:,:]
        ke = ke[:,:,:,:,:]

        sen = sen[...,0] - 1j * sen[...,1]       #bcwh imag

        label = label[...,0] + 1j * label[...,1] #bcwh imag
        sen_label = np.sum(label * sen, 1)

        val_pred = val_pred[...,0] + 1j * val_pred[...,1] #bcwh imag
        sen_val_pred = np.sum(val_pred * sen, 1)
 
        data_input = data_input[...,0] + 1j * data_input[...,1]
        sen_data_input = np.sum(data_input * sen, 1)

        ks = ks[...,0] + 1j * ks[...,1]
        ks = np.sum(ks, 1)

        ke = ke[...,0] + 1j * ke[...,1]
        ke = np.sum(ke, 1)

        kl = kl[...,0] + 1j * kl[...,1]
        kl = np.sum(kl, 1)

        norval = np.absolute(sen_label.max())
        sen_label = np.absolute(sen_label) / norval
        sen_val_pred = np.absolute(sen_val_pred) / norval
        sen_data_input = np.absolute(sen_data_input) / norval
        ks = np.absolute(ks) / norval         
        ke = np.absolute(ke) / norval         
        kl = np.absolute(kl) / norval         

        vis.images(sen_label.reshape(1,1,320,256), opts=dict(title='Label', caption='Label'), win=1)
        vis.images(sen_data_input.reshape(1,1,320,256), opts=dict(title='Input', caption='Input'), win=2)
        vis.images(sen_val_pred.reshape(1,1,320,256), opts=dict(title='Output', caption='Output'), win=3)
        vis.images(ks.reshape(1,1,320,256), opts=dict(title='K-input', caption='K-input'), win=4)
        vis.images(ke.reshape(1,1,320,256), opts=dict(title='K-Output', caption='K-Output'), win=5)
        vis.images(kl.reshape(1,1,320,256), opts=dict(title='K-label', caption='K-label'), win=6)

           # print("REC  nmse: {} psnr: {} ssim: {}".format(b_nmse(sen_label, sen_val_pred), b_psnr(sen_label, sen_val_pred), b_ssim(sen_label, sen_val_pred) ))
        
        rec_total_nmse += b_nmse(sen_label, sen_val_pred)
        rec_total_psnr += b_psnr(sen_label, sen_val_pred)
        rec_total_ssim += b_ssim(sen_label, sen_val_pred)

        data_total_nmse += b_nmse(sen_label, sen_data_input)
        data_total_psnr += b_psnr(sen_label, sen_data_input)
        data_total_ssim += b_ssim(sen_label, sen_data_input)

    rec_total_nmse /= opt.test_num 
    rec_total_psnr /= opt.test_num
    rec_total_ssim /= opt.test_num
    data_total_nmse /=opt.test_num
    data_total_psnr /=opt.test_num
    data_total_ssim /=opt.test_num

    print("REC  nmse: {} psnr: {} ssim: {}".format(rec_total_nmse, rec_total_psnr, rec_total_ssim))
    print("DATA nmse: {} psnr: {} ssim: {}".format(data_total_nmse, data_total_psnr, data_total_ssim))

def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerows(results)

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp(-grad_clip, grad_clip)

def weight_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform(m.weight)


def train(**kwargs):
    opt._parse(kwargs)
    vis = Visualizer(opt.env,port = opt.vis_port)

    # step1: configure model
    model = getattr(models, opt.model)(8 * 2 * opt.slice_num, 8 * 2 * opt.slice_num, 64, 5, 0, opt.slice_num, False)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    else:
        print('Initialize the model!')
        model.apply(weight_init)
    model.to(opt.device)

    # step2: data
    mask_smp_path = opt.undersample_mask #"./data/mask_smp.mat"
    
    train_data = Mridata(opt.train_data_root, mask_smp_path, slice_num=opt.slice_num, test=False)
    val_data = Mridata(opt.val_data_root, mask_smp_path, slice_num=1, test=True)
    train_dataloader = DataLoader(train_data,opt.batch_size,
                        shuffle=True,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data,opt.val_batch_size,
                        shuffle=False,num_workers=opt.num_workers)
    
    # step3: criterion and optimizer
    criterion = t.nn.MSELoss(reduction='sum')
    lr = opt.lr
    optimizer = model.get_optimizer(lr, opt.weight_decay)
        
    # step4: meters
    loss_meter = meter.AverageValueMeter()
    #previous_loss = 1e10
    acc_n = 0
    acc_ave = 0
    
    # train
    for epoch in range(opt.max_epoch):
        for ii,(data,label,sen,dc_fft) in tqdm(enumerate(train_dataloader)):
            # train model 
            input = data.to(opt.device)
            target = label.to(opt.device)
            optimizer.zero_grad()
            score, ks, ke, c_weight, c_weight2 = model(input)
            #kl = fft2(label).cuda() * mask_weight.reshape(320, 256, 1)
            #ke_ = ke * mask_weight.reshape(320, 256, 1)
            kl = fft2(label).cuda()
            ke_ = ke
            
            loss = criterion(score,target[:,8:16,:,:,:]) # + 10 * criterion(kl, ke_)  3 slice
            loss.backward()
            
            # meters update and visualize
            loss_meter.add(loss.item())
            acc_m = acc_n + 320 * 256 * opt.batch_size
            acc_ave = (acc_n * acc_ave + loss) / acc_m     
            acc_n = acc_m

            print('Epoch: {} Iter: {} Loss: {} Objective: {} Env: {}'.format(epoch, ii, loss, acc_ave, opt.env))
            if (ii + 1)%opt.print_freq == 0:
                vis.plot('loss', loss_meter.value()[0])
                vis.plot('objective', acc_ave.data.cpu().numpy())
                
                # enter debug mode
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()
            if (ii + 0)% (opt.print_freq * 10) == 0:
                KL = np.absolute(fft2(label).data.cpu().numpy()[0][...,0] + 1j*fft2(label).data.cpu().numpy()[0][...,1])[4,np.newaxis,:,:]
                KS = np.absolute(ks.data.cpu().numpy()[0][...,0] + 1j * ks.data.cpu().numpy()[0][...,1])[4,np.newaxis,:,:]
                KE = np.absolute(ke.data.cpu().numpy()[0][...,0] + 1j * ke.data.cpu().numpy()[0][...,1])[4,np.newaxis,:,:]
                vis.images(KL / 1, opts=dict(title='Label_kspace', caption='Kspace'), win=6)
                vis.images(KS / 1, opts=dict(title='Input_kspace', caption='Kspace'), win=4)
                vis.images(KE / 1, opts=dict(title='Output_kspace', caption='Kspace'), win=5)
                vis.images((KS / 1) - (KE / 1), opts=dict(title='Diff_kspace', caption='Kspace'), win=7)
                vis.images(np.absolute(label.data.cpu().numpy()[0][...,0] + 1j*label.data.cpu().numpy()[0][...,1])[4,np.newaxis,:,:], opts=dict(title='Label', caption='Label'), win=1)
                vis.images(np.absolute(data.data.cpu().numpy()[0][...,0] + 1j*data.data.cpu().numpy()[0][...,1])[4,np.newaxis,:,:], opts=dict(title='Input', caption='Input'), win=2)
                vis.images(np.absolute(score.data.cpu().numpy()[0][...,0] + 1j*score.data.cpu().numpy()[0][...,1])[4,np.newaxis,:,:], opts=dict(title='Output', caption='Output'), win=3)
                '''
                label = label.detach()
                data = data.detach()
                score = score.detach()
                
                kl = fft2(label.sum(1))
                ks = fft2(data.sum(1))
                ke = fft2(score.sum(1))

                K_label = np.log(np.absolute(kl.data.cpu().numpy()[0][...,0] + 1j*kl.data.cpu().numpy()[0][...,1])[np.newaxis,np.newaxis,:,:])  
                KS = np.log(np.absolute(ks.data.cpu().numpy()[0][...,0] + 1j * ks.data.cpu().numpy()[0][...,1])[np.newaxis,np.newaxis,:,:])
                KE = np.log(np.absolute(ke.data.cpu().numpy()[0][...,0] + 1j * ke.data.cpu().numpy()[0][...,1])[np.newaxis,np.newaxis,:,:])
                vis.images(KS, opts=dict(title='Input_kspace', caption='Kspace'), win=4)
                vis.images(KE, opts=dict(title='Output_kspace', caption='Kspace'), win=5)
                vis.images(K_label, opts=dict(title='Label_kspace', caption='Kspace'), win=6)
                vis.images(KS - KE, opts=dict(title='Diff_kspace', caption='Kspace'), win=7)

                D_label = (np.sum(np.absolute(((label.data.cpu().numpy()[0][...,0] + 1j*label.data.cpu().numpy()[0][...,1]) / 8) ** 2), 0) ** (0.5))[np.newaxis,np.newaxis,:,:]
                D_input = (np.sum(np.absolute(((data.data.cpu().numpy()[0][...,0] + 1j*data.data.cpu().numpy()[0][...,1]) / 8) ** 2), 0) ** (0.5))[np.newaxis,np.newaxis,:,:]
                D_output = (np.sum(np.absolute(((score.data.cpu().numpy()[0][...,0] + 1j*score.data.cpu().numpy()[0][...,1]) / 8) ** 2), 0) ** (0.5))[np.newaxis,np.newaxis,:,:]
                vis.images(D_label, opts=dict(title='Label', caption='Label'), win=1)
                vis.images(D_input, opts=dict(title='Input', caption='Input'), win=2)
                vis.images(D_output, opts=dict(title='Output', caption='Output'), win=3)
                '''
        if (epoch) % 20 == 0:
            model.save('./checkpoints/' + opt.env + '_' + str(epoch) + '.pth')
        # validate and visualize
        #total_nmse, total_psnr, total_ssim = val(model,val_dataloader, mask_weight)
        #print("nmse: {}  psnr: {}  ssim: {}".format(total_nmse, total_psnr, total_ssim))
        #vis.plot('nmse', total_nmse)
        #is.plot('psnr', total_psnr)
        #vis.plot('ssim', total_ssim)
        
        # update learning rate
        #if loss_meter.value()[0] > previous_loss:          
        #    lr = lr * opt.lr_decay
        for param_group in optimizer.param_groups:
            if epoch < 300:
                param_group['lr'] = opt.lr_list[epoch]
            else:
                param_group['lr'] = opt.lr_list[299]
        previous_loss = loss_meter.value()[0]

@t.no_grad()
def mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)

def b_nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    batch_size = gt.shape[0]
    batch_nmse = 0
    for i in range(batch_size):
        batch_nmse += np.linalg.norm(gt[i] - pred[i]) ** 2 / np.linalg.norm(gt[i]) ** 2
    return batch_nmse

def b_psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    batch_size = gt.shape[0]
    batch_psnr = 0
    for i in range(batch_size):
        mse = ((gt[i] - pred[i]) ** 2).mean()
        max_i = gt[i].max()
        s_psnr = 10 * np.log10((max_i ** 2) / mse)
        batch_psnr += s_psnr
    return batch_psnr

def b_ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    batch_size = gt.shape[0]
    batch_ssim = 0
    for i in range(batch_size):
        s_ssim = ssim(gt[i], pred[i])
        batch_ssim += s_ssim
    return batch_ssim


def val(model,dataloader,mask_weight):
    """
    Calculate accuracy in val_dataset
    """
    model.eval()
    total_nmse = 0
    total_psnr = 0
    total_ssim = 0
    for ii, (val_input, label, sen, dc_fft) in tqdm(enumerate(dataloader)):
        val_input = val_input.to(opt.device)
        
        label = label.data.numpy() # b*c*W*h*2 real
        val_pred, ks, ke = model(val_input, mask_weight, dc_fft)
        val_pred = val_pred.data.cpu().numpy()
        sen = sen.data.numpy()       
 
        sen = sen[...,0] + 1j * sen[...,1]       #bcwh imag

        label = label[...,0] + 1j * label[...,1] #bcwh imag
        sen_label = np.absolute(np.sum(label * sen, 1))

        val_pred = val_pred[...,0] + 1j * val_pred[...,1] #bcwh imag
        sen_val_pred = np.absolute(np.sum(val_pred * sen, 1))

        total_nmse += b_nmse(sen_label, sen_val_pred)
        total_psnr += b_psnr(sen_label, sen_val_pred)
        total_ssim += b_ssim(sen_label, sen_val_pred)

    total_nmse /= 512
    total_psnr /= 512
    total_ssim /= 512

    model.train()
    return total_nmse, total_psnr, total_ssim

def help():
    """
    python file.py help
    """
    
    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

if __name__=='__main__':
    train()
    test()
