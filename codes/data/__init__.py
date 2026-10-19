import os
from torch.utils.data import DataLoader

# Default dataset roots, used when --dir_data is not given
DEFAULT_ROOTS = {
    'UCMerced': './datasets/UCMerced-dataset',
}


def create_dataloaders(args):
    """create dataloader"""
    root = args.dir_data if args.dir_data != '.' else DEFAULT_ROOTS.get(args.dataset)
    if args.dataset == 'UCMerced':
        from data.ucmerced import UCMercedDataset
        training_set = UCMercedDataset(args, root_dir=os.path.join(root, 'train'), train=True)
        # --test_only evaluates on the full test split instead of the validation split
        val_split = 'test' if args.test_only else 'val'
        val_set = UCMercedDataset(args, root_dir=os.path.join(root, val_split), train=False)
        test_set = UCMercedDataset(args, root_dir=os.path.join(root, 'test'), train=False)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s ' % args.dataset)

    dataloaders = {'train': DataLoader(training_set, batch_size=args.batch_size,
                                       shuffle=True, num_workers=args.n_threads, drop_last=True),
                   'val': DataLoader(val_set, batch_size=1,
                                     shuffle=False, num_workers=args.n_threads),
                   'test': DataLoader(test_set, batch_size=1,
                                      shuffle=False, num_workers=args.n_threads),
                   }

    return dataloaders
