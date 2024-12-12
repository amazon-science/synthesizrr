# from simpletransformers.classification import MultiLabelClassificationModel
#
#
# model = MultiLabelClassificationModel(
#     'roberta',
#     'roberta-base',
#     num_labels=6,
#     args={
#         'train_batch_size':2,
#         'gradient_accumulation_steps':16,
#         'learning_rate': 3e-5,
#         'num_train_epochs': 3,
#         'max_seq_length': 512,
#     })
#
# from transformers import BertForSequenceClassification
#
#
#
# with optional_dependency('transformers', 'torch'):
#     from transformers import AutoTokenizer, AutoModelForSequenceClassification
#     from torch.nn import CrossEntropyLoss
#     import torch
#
#     BERTClassifier(hyperparams=dict(
#         optimizer=dict(
#             name='Adam',
#             weight_decay=0.99,
#         )
#     ))
#
#     class HuggingFaceSequenceClassification(Classifier):
#         label_encoding_range = EncodingRange.ZERO_TO_N_MINUS_ONE
#
#         model: Optional[AutoModelForSequenceClassification] = None
#         tokenizer: Optional[AutoTokenizer] = None
#         loss: Optional[Any] = None
#         optimizer: Optional[torch.optim.Optimizer] = None
#
#         class Hyperparameters(Classifier.Hyperparameters):
#             model_name: str = 'bert-base-uncased'
#             adam_params: Dict = {
#                 'weight_decay': 0.95
#             }
#
#         def initialize(self, model_dir: Optional[FileMetadata] = None):
#             if model_dir is None:
#                 self.model: AutoModelForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(
#                     self.hyperparams.model_name,
#                     num_labels=self.num_labels,
#                 )
#                 self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(self.hyperparams.model_name)
#             else:
#                 assert model_dir.storage is Storage.LOCAL_FILE_SYSTEM, 'Can only load models from disk.'
#                 self.model: SGDClassifier = joblib.load(os.path.join(model_dir.path, 'model.pkl'))
#             self.optimizer = torch.optim.Adam(
#                 params=self.model.get_params(),
#                 weight_decay=self.hyperparams.adam_params['weight_decay']
#             )
#             self.loss = CrossEntropyLoss()
#
#         def train_step(self, batch: ClassificationData, **kwargs):
#             self.model.train()
#             ## Convert from our internal format to Torch:
#             features: torch.Tensor = batch.features(MLType.TEXT).torch()
#             labels: torch.Tensor = batch.ground_truths().torch()  ## 0, ..., N-1
#             tokens = self.tokenizer(features.to(device))
#             outputs = self.model(tokens)
#             loss = self.loss(outputs, labels)
#             loss.backward()
#             self.optimizer.step()
#
#         def predict_step(self, batch: ClassificationData, **kwargs) -> Dict:
#             self.model.eval()
#             ## Convert from our internal format to Pandas DataFrame:
#             features: pd.DataFrame = batch.features({MLType.FLOAT, MLType.INT}).pandas()
#             scores: np.ndarray = self.model.predict_proba(features)
#             return {'scores': scores, 'labels': self.model.classes_}
#
#         def save(self, model_dir: FileMetadata):
#             joblib.dump(self.model, os.path.join(model_dir.path, 'model.pkl'))
