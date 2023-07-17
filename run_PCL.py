from utils.utils import *
import argparse
from methods.PCL_matchingnet_xnoise import PCL_Matching_xnoise


def _train():
    print('Start training!')
    model = PCL_Matching_xnoise(model_dict[model_name], n_way=train_n_way, n_support=n_shot, image_size=image_size,
                                noise_type=noise_type, noise_rate=noise_rate, tao=tao, device=device)
    optimizer = get_optimizer(model, model_name)
    max_acc = 0
    for epoch in range(start_epoch, stop_epoch):
        model.train()
        model.train_loop(epoch, base_loader, optimizer)
        if epoch == stop_epoch - 1:
            outfile = os.path.join(checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
        model.eval()
        acc = model.test_loop(val_loader)
        if acc > max_acc:  # for baseline and baseline++, we don't use validation here so we let acc = -1
            print("--> Best model! save...", acc)
            max_acc = acc
            outfile = os.path.join(checkpoint_dir, 'best_model.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)


def _test():
    print('Start testing!')
    model = PCL_Matching_xnoise(model_dict[model_name], n_way=train_n_way, n_support=n_shot, image_size=image_size,
                                noise_type=noise_type, noise_rate=noise_rate, tao=tao, device=device)
    modelfile = get_best_file(checkpoint_dir)
    assert modelfile is not None
    tmp = torch.load(modelfile)
    model.load_state_dict(tmp['state'])
    loadfile = get_novel_file(dataset=dataset, split='novel')
    datamgr = SetDataManager(image_size, n_eposide=test_iter_num, n_query=15, n_way=test_n_way,
                             n_support=n_shot, num_workers=num_workers)
    novel_loader = datamgr.get_data_loader(loadfile, aug=False)
    model.eval()
    acc_mean, acc_std = model.test_loop(novel_loader, return_std=True)
    print('%d Test Acc = %4.2f±%4.2f%%' % (test_iter_num, acc_mean, 1.96 * acc_std / np.sqrt(test_iter_num)))


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar')
parser.add_argument('--algorithm', type=str, default='PCL_matchingnet')
parser.add_argument('--tao', type=float, default=2.0)
parser.add_argument('--noise_type', type=str, default='feature')
parser.add_argument('--noise_rate', type=float, default=0.2)
parser.add_argument('--train_n_way', type=int, default=5)
parser.add_argument('--test_n_way', type=int, default=5)
parser.add_argument('--n_shot', type=int, default=5)
parser.add_argument('--model_name', type=str, default='Conv4')
parser.add_argument('--stop_epoch', type=int, default=-1)
parser.add_argument('--device', type=str, default='cuda:0')

if __name__ == '__main__':
    args = parser.parse_args()
    dataset = args.dataset
    algorithm = args.algorithm
    tao = args.tao
    noise_type = args.noise_type
    noise_rate = args.noise_rate
    train_n_way = args.train_n_way
    test_n_way = args.test_n_way
    n_shot = args.n_shot
    model_name = args.model_name
    stop_epoch = args.stop_epoch
    device = args.device

    image_size = get_image_size(model_name=model_name, dataset=dataset)
    model_name = get_model_name(model_name=model_name, dataset=dataset)
    if stop_epoch == -1:
        stop_epoch = get_stop_epoch(algorithm=algorithm, dataset=dataset, n_shot=n_shot)

    checkpoint_dir = get_checkpoint_dir(algorithm=algorithm, model_name=model_name, dataset=dataset,
                                        train_n_way=train_n_way, n_shot=n_shot,
                                        addition=f'{noise_type}_{noise_rate}_{tao}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    base_file, val_file = get_train_files(dataset=dataset)
    base_loader, val_loader = get_train_loader(algorithm=algorithm, image_size=image_size, base_file=base_file,
                                               val_file=val_file, train_n_way=train_n_way,
                                               test_n_way=test_n_way,
                                               n_shot=n_shot, num_workers=num_workers)
    if not os.path.exists(os.path.join(checkpoint_dir, '{:d}.tar'.format(stop_epoch - 1))):
        _train()
    _test()
