import os


def run_func(description, ppi_path, pseq_path, vec_path, pre_emb_path,
             index_path, gnn_model, test_all):
    os.system("python ppi_test.py \
            --description={} \
            --ppi_path={} \
            --pseq_path={} \
            --vec_path={} \
            --pre_emb_path={} \
            --index_path={} \
            --gnn_model={} \
            --test_all={} \
            ".format(description, ppi_path, pseq_path, vec_path, pre_emb_path,
                     index_path, gnn_model, test_all))


if __name__ == "__main__":
    finetune_type = 'MASSA'  # random_initial, only_sequence, sequence_structure, structure_go, sequence_go, our_model
    dataset_type = 'shs27k'  # shs27k, shs148k, string
    mode = 'random'  # dfs, bfs

    description = "test"


    # SHS27k
    ppi_path = "./data/protein.actions.SHS27k.STRING.txt"
    pseq_path = "./data/protein.SHS27k.sequences.dictionary.tsv"
    vec_path = "./data/vec5_CTC.txt"
    pre_emb_path = './pre_train_data/shs_%s.pickle' % finetune_type

    # SHS148k
    # ppi_path = "./data/protein.actions.SHS148k.STRING.txt"
    # pseq_path = "./data/protein.SHS148k.sequences.dictionary.tsv"
    # vec_path = "./data/vec5_CTC.txt"
    # pre_emb_path = './pre_train_data/shs_%s.pickle' % finetune_type


    index_path = "./train_valid_index_json/%s.%s.fold1.json" % (dataset_type, mode)

    gnn_model='./save_model/%s_%s/gnn_test_shs27k_random/gnn_model_valid_best.ckpt'%(mode,dataset_type)

    test_all = "True"

    # test test

    run_func(description, ppi_path, pseq_path, vec_path, pre_emb_path, index_path, gnn_model, test_all)
