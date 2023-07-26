def print_tensor(x, desc=''):
    if desc:
        print(f'{desc} : {x.shape} {x.dtype} {x.max()} {x.min()}')
    else:
        print(f'{x.shape} {x.dtype} {x.max()} {x.min()}')