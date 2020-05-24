if __name__ == '__main__':
    train_path = '../data/wmt16/train_enro.txt'

    with open('train_en.txt', 'w') as f0:
        with open('train_ro.txt', 'w') as f1:
            with open(train_path, 'r') as f:
                for line in f:
                    f0.write('{}\n'.format(line.split('\t')[0].strip()))
                    f1.write('{}\n'.format(line.split('\t')[1].strip()))
