import os
import shutil
from bs4 import BeautifulSoup

def generate_dataset_test_val(l0_path, l1_path, out_path):
    dataset = {}
    l0 = BeautifulSoup(open(l0_path, 'r'), 'html.parser')
    l1 = BeautifulSoup(open(l1_path, 'r'), 'html.parser')

    for doc in l0.find_all('doc'):
        doc_dict = {}
        assert len(doc.find_all('p')) == 1, 'Multiple paragraphs in document {}!'.format(doc.attrs['docid'])

        for p in doc.find_all('p'):
            for seg in p.find_all('seg'):
                doc_dict[seg.attrs['id']] = seg.text
            dataset[doc.attrs['docid']] = doc_dict

    for doc in l1.find_all('doc'):
        assert doc.attrs['docid'] in dataset, 'Document {} not in dict!'.format(doc.attrs['docid'])
        assert len(doc.find_all('p')) == 1, 'Multiple paragraphs in document {}!'.format(doc.attrs['docid'])

        for p in doc.find_all('p'):
            for seg in p.find_all('seg'):
                assert seg.attrs['id'] in dataset[doc.attrs['docid']], 'Sentence not in dict!'

                l0_text = dataset[doc.attrs['docid']][seg.attrs['id']]
                dataset[doc.attrs['docid']][seg.attrs['id']] = (l0_text, seg.text)

    enro_path = out_path + '_enro.txt'
    roen_path = out_path + '_roen.txt'

    f_enro = open(enro_path, 'w')
    f_roen = open(roen_path, 'w')

    sz = 0
    for sentences in dataset.values():
        for sent in sentences.values():
            sz += 1
            f_enro.write('{}\t{}\n'.format(sent[0], sent[1]))
            f_roen.write('{}\t{}\n'.format(sent[1], sent[0]))

    f_enro.close()
    f_roen.close()

    print('Len: {}'.format(sz))

def check_train_data(text):
    if len(text) < 3 \
            or text.startswith('(') \
            or text.startswith('-') \
            or text[0].isdigit():
        return 0

    return 1

def generate_dataset_train(l0, l1, out_path):
    en = []
    ro = []
    f_en = open(l0, 'r')
    f_ro = open(l1, 'r')
    for line0, line1 in zip(f_en, f_ro):
        line0 = line0.strip()
        line0 = ' '.join(line0.split('\t'))
        line1 = line1.strip()
        line1 = ' '.join(line1.split('\t'))

        if not check_train_data(line0) or not check_train_data(line1):
            continue

        en.append(line0)
        ro.append(line1)
    f_en.close()
    f_ro.close()

    assert len(en) == len(ro), 'len(en) != len(ro)'

    enro_path = out_path + '_enro.txt'
    roen_path = out_path + '_roen.txt'

    f_enro = open(enro_path, 'w')
    f_roen = open(roen_path, 'w')

    for i in range(len(ro)):
        f_enro.write('{}\t{}\n'.format(en[i], ro[i]))
        f_roen.write('{}\t{}\n'.format(ro[i], en[i]))

    f_enro.close()
    f_roen.close()

    print('Len: {}'.format(len(ro)))

if __name__ == '__main__':
    if os.path.isdir('./data/wmt16'):
        shutil.rmtree('./data/wmt16')
    os.makedirs('./data/wmt16')

    l0 = './data/wmt16_raw/val_newsdev2016.src.en.sgm'
    l1 = './data/wmt16_raw/val_newsdev2016.ref.ro.sgm'
    out_path = './data/wmt16/val'
    generate_dataset_test_val(l0, l1, out_path)

    l0 = './data/wmt16_raw/test_newstest2016.src.en.sgm'
    l1 = './data/wmt16_raw/test_newstest2016.ref.ro.sgm'
    out_path = './data/wmt16/test'
    generate_dataset_test_val(l0, l1, out_path)

    l0 = './data/wmt16_raw/train_corpus.en'
    l1 = './data/wmt16_raw/train_corpus.ro'
    out_path = './data/wmt16/train'
    generate_dataset_train(l0, l1, out_path)

    print('Done!')
