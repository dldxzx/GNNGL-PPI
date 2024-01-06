import os


def run_func(description, ppi_path, pseq_path, vec_path, pre_emb_path,
             split_new, split_mode, train_valid_index_path,
             use_lr_scheduler, save_path, graph_only_train,
             batch_size, epochs):
    os.system("python -u gnn_train.py \
            --description={} \
            --ppi_path={} \
            --pseq_path={} \
            --vec_path={} \
            --pre_emb_path={}\
            --split_new={} \
            --split_mode={} \
            --train_valid_index_path={} \
            --use_lr_scheduler={} \
            --save_path={} \
            --graph_only_train={} \
            --batch_size={} \
            --epochs={} \
            ".format(description, ppi_path, pseq_path, vec_path, pre_emb_path,
                     split_new, split_mode, train_valid_index_path,
                     use_lr_scheduler, save_path, graph_only_train,
                     batch_size, epochs))


if __name__ == "__main__":
    finetune_type = 'MASSA'  # random_initial, only_sequence, sequence_structure, structure_go, sequence_go, our_model
    dataset_type = 'shs27k'  # shs27k, shs148k, string
    mode = 'random'  # dfs, bfs
    #     description = "test_string_bfs"
    description = "test_%s_%s" % (dataset_type, mode)

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

    split_new = "True"
    split_mode = "%s" % mode
    index_path = "./train_valid_index_json_%s/"
    if not os.path.isdir(index_path):
        os.mkdir(index_path)
    train_valid_index_path = os.path.join(index_path, "%s.%s.fold1.json" % (dataset_type, mode))

    use_lr_scheduler = "True"
    save_path = "./save_model/%s_%s/" % (mode, dataset_type)
    graph_only_train = "False"

    batch_size = 1024
    epochs = 400

    run_func(description, ppi_path, pseq_path, vec_path, pre_emb_path,
             split_new, split_mode, train_valid_index_path,
             use_lr_scheduler, save_path, graph_only_train,
             batch_size, epochs)
