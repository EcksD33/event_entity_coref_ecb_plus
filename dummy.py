import torch
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())

# python src/data/make_dataset.py --ecb_path data/raw/ECB+_LREC2014/ECB+/ --output_dir out_dataset/ --data_setup 2 --selected_sentences_file data/raw/ECB+_LREC2014/ECBplus_coreference_sentences.csv
# python src/features/build_features.py --config_path build_features_config_experimental.json --output_path out_featurex/
