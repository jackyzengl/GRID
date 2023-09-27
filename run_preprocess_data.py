from dataloader.data_preprocessor import data_preprocessor
from arguments import create_parser


args_, config_ = create_parser('preprocessor')
processor = data_preprocessor(arg=args_, config=config_)
processor.preprocess_from_file(data_path=args_.data_path)
