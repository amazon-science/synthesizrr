from typing import *
from synthesizrr.base.util import *
from synthesizrr.base.data import *
from synthesizrr.base.constants import *
from synthesizrr.base.framework import *
from synthesizrr.base.data.reader import Reader
from synthesizrr.base.data.writer import Writer
from synthesizrr.base.framework.dl.torch import *
from synthesizrr.base.framework.task_data import DataSplit, Datasets, Dataset
from datasets import load_dataset as hf_load_dataset, load_from_disk
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR: str = ''  # TODO: fill this out!


class SynthesizRRDataset(Registry):
    name: ClassVar[str]
    task: ClassVar[Task]
    data_schema: ClassVar[Dict[str, MLType]]

    @classmethod
    def _registry_keys(cls) -> Optional[Union[List[Any], Any]]:
        return cls.name

    @classmethod
    def get(cls, name: str) -> Type['SynthesizRRDataset']:
        return cls.get_subclass(name)

    @classproperty
    def dataset_dir(cls) -> FileMetadata:
        dataset_dir: FileMetadata = FileMetadata.of(f'{DATA_DIR}/{cls.name}/')
        dataset_dir.mkdir()
        return dataset_dir

    @classproperty
    def train_path(cls) -> str:
        return cls.dataset_dir.file_in_dir(f'{cls.name}_train.parquet')

    @classproperty
    def validation_path(cls) -> str:
        return cls.dataset_dir.file_in_dir(f'{cls.name}_validation.parquet')

    @classproperty
    def test_path(cls) -> str:
        return cls.dataset_dir.file_in_dir(f'{cls.name}_test.parquet')

    @classproperty
    def unsupervised_path(cls) -> str:
        return cls.dataset_dir.file_in_dir(f'{cls.name}_unsupervised.parquet')

    @classproperty
    def schema_path(cls) -> str:
        return cls.dataset_dir.file_in_dir(f'{cls.name}_schema.json')

    @classproperty
    def unsupervised_schema_path(cls) -> str:
        return cls.dataset_dir.file_in_dir(f'{cls.name}_unsupervised_schema.json')

    @classproperty
    def supervised_data_schema(cls) -> MLTypeSchema:
        return copy.deepcopy(MLType.convert_values(cls.data_schema))

    @classproperty
    def unsupervised_data_schema(cls) -> MLTypeSchema:
        data_schema: Dict[str, MLType] = copy.deepcopy(MLType.convert_values(cls.data_schema))
        data_schema: MLTypeSchema = {
            col: mltype
            for col, mltype in data_schema.items()
            if mltype not in GROUND_TRUTH_ML_TYPES
        }
        return data_schema

    @classmethod
    def raw_train(cls) -> Optional[pd.DataFrame]:
        return None

    @classmethod
    def has_train(cls) -> bool:
        return 'raw_train' in cls.__dict__

    @classmethod
    def raw_validation(cls) -> Optional[pd.DataFrame]:
        return None

    @classmethod
    def has_validation(cls) -> bool:
        return 'raw_validation' in cls.__dict__

    @classmethod
    def raw_test(cls) -> Optional[pd.DataFrame]:
        return None

    @classmethod
    def has_test(cls) -> bool:
        return 'raw_test' in cls.__dict__

    @classmethod
    def raw_unsupervised(cls) -> Optional[pd.DataFrame]:
        return None

    @classmethod
    def has_unsupervised(cls) -> bool:
        return 'raw_unsupervised' in cls.__dict__

    @classmethod
    def decode_labels(cls, row) -> pd.Series:
        raise NotImplementedError()

    @classmethod
    def has_decode_labels(cls) -> bool:
        return 'decode_labels' in cls.__dict__

    @classmethod
    def setup(cls):
        if cls.has_train():
            Writer.of('json').write(
                data=cls.supervised_data_schema,
                destination=cls.schema_path,
                overwrite=True,
            )
            idx_col: str = only_item([
                col
                for col, mltype in cls.supervised_data_schema.items()
                if mltype is MLType.INDEX
            ])
            train: pd.DataFrame = cls.raw_train()
            if cls.has_decode_labels():
                train: pd.DataFrame = train.apply(cls.decode_labels, axis=1)
            if len(train) != train[idx_col].nunique():
                raise ValueError(f'Expected unique index column "{idx_col}" for split="train".')
            train.reset_index(drop=True).to_parquet(cls.train_path)
        if cls.has_validation():
            idx_col: str = only_item([
                col
                for col, mltype in cls.supervised_data_schema.items()
                if mltype is MLType.INDEX
            ])
            validation: pd.DataFrame = cls.raw_validation()
            if cls.has_decode_labels():
                validation: pd.DataFrame = validation.apply(cls.decode_labels, axis=1)
            if len(validation) != validation[idx_col].nunique():
                raise ValueError(f'Expected unique index column "{idx_col}" for split="validation".')
            validation.reset_index(drop=True).to_parquet(cls.validation_path)
        if cls.has_test():
            idx_col: str = only_item([
                col
                for col, mltype in cls.supervised_data_schema.items()
                if mltype is MLType.INDEX
            ])
            test: pd.DataFrame = cls.raw_test()
            if cls.has_decode_labels():
                test: pd.DataFrame = test.apply(cls.decode_labels, axis=1)
            if len(test) != test[idx_col].nunique():
                raise ValueError(f'Expected unique index column "{idx_col}" for split="test".')
            test.reset_index(drop=True).to_parquet(cls.test_path)
        if cls.has_unsupervised():
            idx_col: str = only_item([
                col
                for col, mltype in cls.unsupervised_data_schema.items()
                if mltype is MLType.INDEX
            ])
            Writer.of('json').write(
                data=cls.unsupervised_data_schema,
                destination=cls.unsupervised_schema_path,
                overwrite=True,
            )
            unsupervised: pd.DataFrame = cls.raw_unsupervised()
            if len(unsupervised) != unsupervised[idx_col].nunique():
                raise ValueError(f'Expected unique index column "{idx_col}" for split="unsupervised".')
            unsupervised.reset_index(drop=True).to_parquet(cls.unsupervised_path)

    @classproperty
    def train(cls) -> Optional[Dataset]:
        return cls.datasets.train

    @classmethod
    @safe_validate_arguments
    def create_seed_set(
            cls,
            seed_size: int,
            *,
            data_split: DataSplit = DataSplit.TRAIN,
            random_state: int = 42,
            stratify_on_ground_truth: bool = False,
    ) -> Dataset:
        dataset: Dataset = cls.datasets[data_split].read(read_as=DataLayout.PANDAS)
        dataset_df: pd.DataFrame = dataset.data.pandas()
        gt_col: str = only_key(dataset.data_schema.ground_truths_schema)
        if stratify_on_ground_truth:
            _, seed_dataset_df = train_test_split(
                dataset_df,
                test_size=seed_size,
                random_state=random_state,
                stratify=dataset_df[gt_col],
            )
        else:
            _, seed_dataset_df = train_test_split(
                dataset_df,
                test_size=seed_size,
                random_state=random_state,
            )
        return dataset.update_params(data=seed_dataset_df)

    @classproperty
    def validation(cls) -> Optional[Dataset]:
        return cls.datasets.validation

    @classproperty
    def test(cls) -> Optional[Dataset]:
        return cls.datasets.test

    @classproperty
    def unsupervised(cls) -> Optional[Dataset]:
        return cls.datasets.unsupervised

    @classproperty
    def datasets(cls) -> Datasets:
        datasets: Dict[str, Dataset] = {}
        if cls.has_train():
            datasets[DataSplit.TRAIN] = Dataset.of(
                data_split=DataSplit.TRAIN,
                task=cls.task,
                data=FileMetadata.of(cls.train_path),
                data_schema=cls.supervised_data_schema,
            )
        if cls.has_validation():
            datasets[DataSplit.VALIDATION] = Dataset.of(
                data_split=DataSplit.VALIDATION,
                task=cls.task,
                data=FileMetadata.of(cls.validation_path),
                data_schema=cls.supervised_data_schema,
            )
        if cls.has_test():
            datasets[DataSplit.TEST] = Dataset.of(
                data_split=DataSplit.TEST,
                task=cls.task,
                data=FileMetadata.of(cls.test_path),
                data_schema=cls.supervised_data_schema,
            )
        if cls.has_unsupervised():
            datasets[DataSplit.UNSUPERVISED] = Dataset.of(
                data_split=DataSplit.UNSUPERVISED,
                task=cls.task,
                data=FileMetadata.of(cls.unsupervised_path),
                data_schema=cls.unsupervised_data_schema,
            )
        return Datasets.of(**datasets)

    @classmethod
    def setup_datasets(cls):
        for SynthesizRRDatasetSubclass in cls.subclasses():
            assert issubclass(SynthesizRRDatasetSubclass, SynthesizRRDataset)
            SynthesizRRDatasetSubclass.setup()

    @classproperty
    def label_verbalizer(cls) -> Optional[Dict[str, str]]:
        return None


class HyperpartisanNewsDataset(SynthesizRRDataset):
    name = 'hyperpartisan_news'
    task = Task.BINARY_CLASSIFICATION
    data_schema = dict(
        id=MLType.INDEX,
        text=MLType.TEXT,
        label_text=MLType.GROUND_TRUTH,
    )

    @classmethod
    def raw_train(cls) -> Optional[pd.DataFrame]:
        return hf_load_dataset("zapsdcn/hyperpartisan_news", split='train').to_pandas().rename(
            columns={'label': 'label_text'}
        )

    @classmethod
    def raw_validation(cls) -> Optional[pd.DataFrame]:
        return hf_load_dataset("zapsdcn/hyperpartisan_news", split='validation').to_pandas().rename(
            columns={'label': 'label_text'}
        )

    @classmethod
    def raw_test(cls) -> Optional[pd.DataFrame]:
        return hf_load_dataset("zapsdcn/hyperpartisan_news", split='test').to_pandas().rename(
            columns={'label': 'label_text'}
        )

    @classproperty
    def label_verbalizer(cls) -> Optional[Dict[str, str]]:
        return {
            'true': 'using harsh political language, using a mocking tone and toxic commentary',
            'false': 'using neutral language, using a reasonable tone and politically correct commentary',
        }


class AGNewsDataset(SynthesizRRDataset):
    name = 'ag_news'
    task = Task.MULTI_CLASS_CLASSIFICATION
    data_schema = dict(
        id=MLType.INDEX,
        text=MLType.TEXT,
        # headline=MLType.TEXT,
        label_text=MLType.GROUND_TRUTH,
    )

    @classproperty
    def label_verbalizer(cls) -> Optional[Dict[str, str]]:
        return {
            'Business': 'about companies, industries, markets, trade, investments, entrepreneurship, economic policies, and other business-related developments',
            'World': 'about international news, such as politics, diplomacy, conflicts, global events, international relations, human rights issues, and significant global trends',
            'Sci/Tech': 'about scientific discoveries, technological advancements, innovations, research breakthroughs',
            'Sports': 'related to coverage of professional sports leagues, major tournaments, athletes, teams, match results, player transfers, coaching changes, sports-related controversies',
        }

    @classmethod
    def raw_train(cls) -> Optional[pd.DataFrame]:
        return hf_load_dataset("zapsdcn/ag", split='train', features=ds.Features({
            'label': ds.Value('int64'),
            'text': ds.Value('string'),
            'headline': ds.Value('string'),
            'id': ds.Value('string'),
        })).to_pandas()

    @classmethod
    def raw_validation(cls) -> Optional[pd.DataFrame]:
        return hf_load_dataset("zapsdcn/ag", split='validation', features=ds.Features({
            'label': ds.Value('int64'),
            'text': ds.Value('string'),
            'headline': ds.Value('string'),
            'id': ds.Value('string'),
        })).to_pandas()

    @classmethod
    def raw_test(cls) -> Optional[pd.DataFrame]:
        test_df: pd.DataFrame = hf_load_dataset("zapsdcn/ag", split='test', features=ds.Features({
            'label': ds.Value('int64'),
            'text': ds.Value('string'),
            'headline': ds.Value('string'),
            'id': ds.Value('string'),
        })).to_pandas()
        test_df = test_df.drop(['id'], axis=1).reset_index(drop=True).reset_index().rename(columns={'index': 'id'})
        test_df['id'] = 'idtest' + test_df['id'].astype(str)
        return test_df

    @classmethod
    def decode_labels(cls, row) -> pd.Series:
        ## Ref: https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset
        lb_decoder = {
            1: 'World',
            2: 'Sports',
            3: 'Business',
            4: 'Sci/Tech',
        }
        row['label_text'] = lb_decoder[row['label']]
        return row


class AmazonReviewsPolarity(SynthesizRRDataset):
    name = 'amazon-polarity'
    task = Task.BINARY_CLASSIFICATION
    data_schema = dict(
        idx=MLType.INDEX,
        # title=MLType.TEXT,
        text=MLType.TEXT,
        label_text=MLType.GROUND_TRUTH,
    )

    @classproperty
    def label_verbalizer(cls) -> Optional[Dict[str, str]]:
        return {
            'positive': "what the reviewer liked about the product, how the reviewer found it easy to use the product, or the reviewer's positive experience with the product",
            'negative': "what the reviewer disliked about the product, how the reviewer found it challenging to use the product, or the reviewer's negative experience with the product",
        }

    @classmethod
    def raw_train(cls) -> Optional[pd.DataFrame]:
        return pd.read_parquet(
            f'{DATA_DIR}/data/amazon-polarity-mini/amazon-polarity-mini_train.parquet'
        ).rename(columns={'index': 'idx', 'content': 'text'})

    @classmethod
    def raw_validation(cls) -> Optional[pd.DataFrame]:
        return pd.read_parquet(
            f'{DATA_DIR}/data/amazon-polarity-mini/amazon-polarity-mini_validation.parquet'
        ).rename(columns={'index': 'idx', 'content': 'text'})

    @classmethod
    def raw_test(cls) -> Optional[pd.DataFrame]:
        return pd.read_parquet(
            f'{DATA_DIR}/data/amazon-polarity-mini/amazon-polarity-mini_test.parquet'
        ).rename(columns={'index': 'idx', 'content': 'text'})

    @staticmethod
    def create_dataset():
        from datasets import load_dataset as hf_load_dataset, load_from_disk
        amazon_polarity_train = hf_load_dataset("amazon_polarity", split='train') \
            .to_pandas().reset_index(drop=True).reset_index().rename(
            columns={'index': 'idx'}
        )
        amazon_polarity_test = hf_load_dataset("amazon_polarity", split='test') \
            .to_pandas().reset_index(drop=True).reset_index().rename(
            columns={'index': 'idx'}
        )
        amazon_polarity_train['label_text'] = amazon_polarity_train['label'].map({
            1: 'positive',
            0: 'negative',
        })
        amazon_polarity_test['label_text'] = amazon_polarity_test['label'].map({
            1: 'positive',
            0: 'negative',
        })
        amazon_polarity_mini_train = pd.concat([
            amazon_polarity_train.query('label_text == "positive"').sample(
                n=72000 // 2,
                random_state=42,
            ).reset_index(drop=True),
            amazon_polarity_train.query('label_text == "negative"').sample(
                n=72000 // 2,
                random_state=42,
            ).reset_index(drop=True),
        ]).reset_index(drop=True)
        amazon_polarity_mini_test = pd.concat([
            amazon_polarity_test.query('label_text == "positive"').sample(
                n=40_000 // 2,
                random_state=42,
            ).reset_index(drop=True),
            amazon_polarity_test.query('label_text == "negative"').sample(
                n=40_000 // 2,
                random_state=42,
            ).reset_index(drop=True),
        ]).reset_index(drop=True)
        amazon_polarity_train_non_mini = amazon_polarity_train.query(
            f'idx not in {amazon_polarity_mini_train["idx"].tolist()}')

        amazon_polarity_mini_validation = pd.concat([
            amazon_polarity_train_non_mini.query('label_text == "positive"').sample(
                n=3600 // 2,
                random_state=42,
            ).reset_index(drop=True),
            amazon_polarity_train_non_mini.query('label_text == "negative"').sample(
                n=3600 // 2,
                random_state=42,
            ).reset_index(drop=True),
        ]).reset_index(drop=True)
        FileMetadata.of(f'{DATA_DIR}/data/amazon-polarity-mini/').mkdir()
        amazon_polarity_mini_train.to_parquet(
            f'{DATA_DIR}/data/amazon-polarity-mini/amazon-polarity-mini_train.parquet'
        )
        amazon_polarity_mini_validation.to_parquet(
            f'{DATA_DIR}/data/amazon-polarity-mini/amazon-polarity-mini_validation.parquet'
        )
        amazon_polarity_mini_test.to_parquet(
            f'{DATA_DIR}/data/amazon-polarity-mini/amazon-polarity-mini_test.parquet'
        )


class AmazonReviewsProductCategory(SynthesizRRDataset):
    name = 'amazon-reviews-category'
    task = Task.MULTI_CLASS_CLASSIFICATION
    data_schema = dict(
        idx=MLType.INDEX,
        # asin=MLType.CATEGORICAL,
        # product_name=MLType.TEXT,
        # product_type=MLType.CATEGORICAL,
        # helpful=MLType.CATEGORICAL,
        # rating=MLType.CATEGORICAL,
        # title=MLType.TEXT,
        # date=MLType.CATEGORICAL,
        # reviewer=MLType.TEXT,
        # reviewer_location=MLType.CATEGORICAL,
        text=MLType.TEXT,
        label_text=MLType.GROUND_TRUTH,
        # sentiment=MLType.CATEGORICAL,
    )

    @classproperty
    def label_verbalizer(cls) -> Optional[Dict[str, str]]:
        return {
            "magazines": "magazines or periodicals covering various topics",
            "camera_photo": "photography gear including cameras, lenses, accessories, or photo editing tools",
            "office_products": "office supplies or equipment for professional and home office setups",
            "kitchen": "kitchenware, appliances, or culinary tools for cooking and dining",
            "cell_phones_service": "cell phone service accessories or service plans for communication and connectivity",
            "computer_video_games": "computers, gaming consoles, video games, or related accessories",
            "grocery_and_gourmet_food": "groceries, fruits and vegetables, gourmet treats, or specialty food items",
            "tools_hardware": "tools, hardware, or equipment for DIY projects and home repairs",
            "automotive": "auto parts, accessories, or tools for vehicle maintenance and enhancements",
            "music_album": "music albums spanning various genres and artists",
            "health_and_personal_care": "healthcare products, personal care items, or wellness essentials",
            "electronics": "electronic devices, gadgets, personal tech, or home electronics",
            "outdoor_living": "products for outdoor activities, gardening, or patio living",
            "video": "movies, TV shows, and documentaries spanning various genres and artists",
            "apparel": "clothing including casual wear, formal attire, seasonal outfits, activewear, or fashion accessories for men, women, and children",
            "toys_games": "fun or educational toys and games for kids of all ages",
            "sports_outdoors": "products for various sports and outdoor activities",
            "books": "books in various genres and formats",
            "software": "computer software for productivity or gaming covering either personal or professional needs",
            "baby": "baby essentials, gear, or toys for infants and toddlers",
            "musical_and_instruments": "musical instruments, accessories, or music production equipment",
            "beauty": "beauty products, cosmetics, or skincare essentials, makeup, hair care, fragrances, or grooming essentials",
            "jewelry_and_watches": "watches or jewelry pieces such as necklaces, bracelets, earrings, or rings, crafted in precious metals or adorned with gemstones for special occasions",
        }

    @classmethod
    def raw_train(cls) -> Optional[pd.DataFrame]:
        return pd.read_parquet(
            f'{DATA_DIR}/data/amazon-reviews-category/amazon-reviews-category-train.parquet'
        ).rename(columns={'unique_id': 'idx', 'review_text': 'text', 'product_category': 'label_text'})

    @classmethod
    def raw_validation(cls) -> Optional[pd.DataFrame]:
        return pd.read_parquet(
            f'{DATA_DIR}/data/amazon-reviews-category/amazon-reviews-category-validation.parquet'
        ).rename(columns={'unique_id': 'idx', 'review_text': 'text', 'product_category': 'label_text'})

    @classmethod
    def raw_test(cls) -> Optional[pd.DataFrame]:
        return pd.read_parquet(
            f'{DATA_DIR}/data/amazon-reviews-category/amazon-reviews-category-test.parquet'
        ).rename(columns={'unique_id': 'idx', 'review_text': 'text', 'product_category': 'label_text'})

    @staticmethod
    def create_dataset():
        from bs4 import BeautifulSoup
        dataset_dir: FileMetadata = FileMetadata.of(
            f'{DATA_DIR}/raw-data/amazon-reviews-category/sorted_data/'
        ).mkdir(return_metadata=True)
        if len(dataset_dir.list()) == 0:
            raise SystemError(
                f'Expected Amazon Reviews Category data to be in folder "{dataset_dir.path}". '
                f'Please download and unzip the data from '
                f'https://www.cs.jhu.edu/~mdredze/datasets/sentiment/unprocessed.tar.gz'
            )

        def parse_review(text) -> pd.DataFrame:
            df = []
            for review_BS in BeautifulSoup(text).find_all('review'):
                d = {}
                for child in review_BS.children:
                    k = child.name
                    if k is not None:
                        v = child.text.strip()
                        if k in ['product_type', 'unique_id']:  ## Allow list
                            d.setdefault(k, [])
                        if isinstance(d.get(k), list):
                            d[k].append(v)
                        else:
                            if k in d:
                                raise ValueError(f'"{k}" key already exists')
                            d[k] = v
                if 'unique_id' in d:
                    d['unique_id'] = '-'.join(d['unique_id'])
                if 'product_type' in d:
                    d['product_type'] = ','.join(set(d['product_type']))
                if len(d) > 0:
                    df.append(d)
            return pd.DataFrame(df)

        dfs = []
        from pathlib import Path
        for fpath in dataset_dir.list(only_subdirs=True):
            category = Path(fpath).stem
            neg = FileSystemUtil.get_file_str(
                str(Path(fpath) / f'negative.review'),
                encoding='cp1252',
                raise_error=True,
            )
            neg_df = parse_review(neg)
            neg_df['product_category'] = category
            neg_df['sentiment'] = 'negative'
            dfs.append(neg_df)

            pos = FileSystemUtil.get_file_str(
                str(Path(fpath) / f'positive.review'),
                encoding='cp1252',
                raise_error=True,
            )
            pos_df = parse_review(pos)
            pos_df['product_category'] = category
            pos_df['sentiment'] = 'positive'

            dfs.append(pos_df)
        reviews = pd.concat(dfs).reset_index(drop=True)
        reviews['product_category'] = reviews['product_category'].replace({
            'grocery': 'grocery_and_gourmet_food',
            'gourmet_food': 'grocery_and_gourmet_food',
        })
        attrprompt_cats = {
            "magazines",
            "camera_photo",
            "office_products",
            "kitchen",
            "cell_phones_service",
            "computer_video_games",
            "grocery_and_gourmet_food",
            "tools_hardware",
            "automotive",
            "music_album",
            "health_and_personal_care",
            "electronics",
            "outdoor_living",
            "video",
            "apparel",
            "toys_games",
            "sports_outdoors",
            "books",
            "software",
            "baby",
            "musical_and_instruments",
            "beauty",
            "jewelry_and_watches",
        }
        reviews = reviews.query(f'product_category in {list(attrprompt_cats)}').reset_index(drop=True)
        assert set(reviews['product_category']) == attrprompt_cats
        assert reviews['unique_id'].nunique() == len(reviews)
        # reviews['product_category'].value_counts()
        reviews_train = reviews.sample(n=30_000, random_state=42).reset_index(drop=True)
        print(reviews_train.shape)

        reviews_test = reviews.query(
            f"unique_id not in {reviews_train['unique_id'].tolist()}"
        ).sample(n=2_400, random_state=42).reset_index(drop=True)
        print(reviews_test.shape)

        reviews_validation = reviews.query(
            f"unique_id not in {reviews_train['unique_id'].tolist() + reviews_test['unique_id'].tolist()}"
        ).reset_index(drop=True)
        print(reviews_validation.shape)

        FileMetadata.of(f'{DATA_DIR}/data/amazon-reviews-category/').mkdir()
        reviews_train.to_parquet(
            f'{DATA_DIR}/data/amazon-reviews-category/amazon-reviews-category-train.parquet'
        )
        reviews_validation.to_parquet(
            f'{DATA_DIR}/data/amazon-reviews-category/amazon-reviews-category-validation.parquet'
        )
        reviews_test.to_parquet(
            f'{DATA_DIR}/data/amazon-reviews-category/amazon-reviews-category-test.parquet'
        )


class AmazonHumorousProductQuestions(SynthesizRRDataset):
    name = 'amazon-humor'
    task = Task.BINARY_CLASSIFICATION
    data_schema = dict(
        idx=MLType.INDEX,
        text=MLType.TEXT,
        # product_description=MLType.TEXT,
        # image_url=MLType.URL,
        label_text=MLType.GROUND_TRUTH,
    )

    @classproperty
    def label_verbalizer(cls) -> Optional[Dict[str, str]]:
        return {
            'non_humorous': "solemn",
            'humorous': "humorous",
        }

    @classmethod
    def raw_train(cls) -> Optional[pd.DataFrame]:
        return pd.read_parquet(f'{DATA_DIR}/data/amazon-humor/amazon-humor_train.parquet').rename(
            columns={'question': 'text'}
        )

    @classmethod
    def raw_validation(cls) -> Optional[pd.DataFrame]:
        return pd.read_parquet(f'{DATA_DIR}/data/amazon-humor/amazon-humor_validation.parquet').rename(
            columns={'question': 'text'}
        )

    @classmethod
    def raw_test(cls) -> Optional[pd.DataFrame]:
        return pd.read_parquet(f'{DATA_DIR}/data/amazon-humor/amazon-humor_test.parquet').rename(
            columns={'question': 'text'}
        )

    @staticmethod
    def create_dataset():
        dataset_dir: FileMetadata = FileMetadata.of(f'{DATA_DIR}/raw-data/amazon-humor/').mkdir(return_metadata=True)
        if len(dataset_dir.list()) == 0:
            raise SystemError(
                f'Expected Amazon Humor data to be in folder "{dataset_dir.path}". '
                f'Please download the data files from https://registry.opendata.aws/humor-detection/'
            )

        humor_pos = pd.read_csv(
            dataset_dir.file_in_dir('Humorous.csv')
        ).reset_index(drop=True).reset_index().rename(columns={'index': 'idx'})
        humor_pos['label_text'] = humor_pos['label'].map({
            1: 'humorous',
            0: 'non_humorous',
        })

        humor_neg = pd.read_csv(
            dataset_dir.file_in_dir('Non-humorous-biased.csv')
        ).reset_index(drop=True).reset_index().rename(columns={'index': 'idx'})
        humor_neg['label_text'] = humor_neg['label'].map({
            1: 'humorous',
            0: 'non_humorous',
        })

        humor_pos_train = humor_pos.sample(n=15_000 // 2, random_state=42).reset_index(drop=True)
        # print(f'humor_pos_train: {len(humor_pos_train)}')

        humor_pos_validation = humor_pos.query(f'idx not in {humor_pos_train["idx"].tolist()}').sample(
            n=1142 // 2,
            random_state=42
        ).reset_index(drop=True)
        # print(f'humor_pos_validation: {len(humor_pos_validation)}')

        humor_pos_test = humor_pos.query(
            f'idx not in {humor_pos_train["idx"].tolist() + humor_pos_validation["idx"].tolist()}'
        ).reset_index(drop=True)
        # print(f'humor_pos_test: {len(humor_pos_test)}')

        humor_neg_train = humor_neg.sample(n=15_000 // 2, random_state=42).reset_index(drop=True)
        # print(f'humor_neg_train: {len(humor_neg_train)}')

        humor_neg_validation = humor_neg.query(f'idx not in {humor_neg_train["idx"].tolist()}').sample(
            n=1142 // 2,
            random_state=42
        ).reset_index(drop=True)
        # print(f'humor_neg_validation: {len(humor_neg_validation)}')

        humor_neg_test = humor_neg.query(
            f'idx not in {humor_neg_train["idx"].tolist() + humor_neg_validation["idx"].tolist()}').reset_index(
            drop=True)
        # print(f'humor_neg_test: {len(humor_neg_test)}')

        humor_train = pd.concat([humor_pos_train, humor_neg_train]).sample(frac=1, random_state=42).reset_index(
            drop=True)
        # print(f'humor_train: {len(humor_train)}')
        # display(humor_train.head(3))

        humor_validation = pd.concat([humor_pos_validation, humor_neg_validation]).sample(
            frac=1,
            random_state=42
        ).reset_index(drop=True)
        # print(f'humor_validation: {len(humor_validation)}')
        # display(humor_validation.head(3))

        humor_test = pd.concat([humor_pos_test, humor_neg_test]).sample(frac=1, random_state=42).reset_index(drop=True)
        # print(f'humor_test: {len(humor_test)}')
        # display(humor_test.head(3))
        humor_train['idx'] = humor_train['idx'].astype(str) + '-' + humor_train['label_text'].astype(str)
        humor_validation['idx'] = humor_validation['idx'].astype(str) + '-' + humor_validation['label_text'].astype(str)
        humor_test['idx'] = humor_test['idx'].astype(str) + '-' + humor_test['label_text'].astype(str)

        FileMetadata.of(f'{DATA_DIR}/data/amazon-humor/').mkdir()
        humor_train.to_parquet(
            f'{DATA_DIR}/data/amazon-humor/amazon-humor_train.parquet'
        )
        humor_validation.to_parquet(
            f'{DATA_DIR}/data/amazon-humor/amazon-humor_validation.parquet'
        )
        humor_test.to_parquet(
            f'{DATA_DIR}/data/amazon-humor/amazon-humor_test.parquet'
        )


class ToiHeadlinesDataset(SynthesizRRDataset):
    name = 'toi_headlines'
    task = Task.MULTI_CLASS_CLASSIFICATION
    data_schema = dict(
        idx=MLType.INDEX,
        text=MLType.TEXT,
        # publish_date=MLType.TEXT,
        # headline_category=MLType.CATEGORICAL,
        # headline_text_len=MLType.INT,
        label_text=MLType.GROUND_TRUTH,
    )

    @classmethod
    def raw_train(cls) -> Optional[pd.DataFrame]:
        return pd.read_parquet(f'{DATA_DIR}/data/toi_headlines/toi_headlines_train.parquet').rename(
            columns={'headline_text': 'text', 'headline_root': 'label_text'}
        )

    @classmethod
    def raw_validation(cls) -> Optional[pd.DataFrame]:
        return pd.read_parquet(f'{DATA_DIR}/data/toi_headlines/toi_headlines_validation.parquet').rename(
            columns={'headline_text': 'text', 'headline_root': 'label_text'}
        )

    @classmethod
    def raw_test(cls) -> Optional[pd.DataFrame]:
        return pd.read_parquet(f'{DATA_DIR}/data/toi_headlines/toi_headlines_test.parquet').rename(
            columns={'headline_text': 'text', 'headline_root': 'label_text'}
        )

    @classproperty
    def label_verbalizer(cls) -> Optional[Dict[str, str]]:
        raise NotImplementedError()

    @staticmethod
    def create_dataset():
        label_space: List[str] = [
            'sports',
            'life-style',
            'education',
            'entertainment',
            'business',
            'city',
            'environment',
            'tech',
            'elections',
            'world',
        ]
        dataset_dir: FileMetadata = FileMetadata.of(f'{DATA_DIR}/raw-data/toi_headlines/').mkdir(return_metadata=True)
        if len(dataset_dir.list()) == 0:
            raise SystemError(
                f'Expected ToI Headlines data to be in folder "{dataset_dir.path}". '
                f'Please download the CSV file from '
                f'https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DPQMQH'
            )
        df = pd.read_csv(
            dataset_dir.file_in_dir('india-news-headlines.csv')
        )
        df['headline_root'] = df['headline_category'].apply(lambda x: x.strip().removeprefix('home.').split('.')[0])
        df['headline_text_len'] = df['headline_text'].apply(len)
        df = df.query('headline_text_len >= 40').reset_index(drop=True).reset_index().rename(columns=dict(index='idx'))
        full_idxs: np.ndarray = sample_idxs_match_distribution(
            source=df['headline_root'],
            target=pd.Series([lb for lb in label_space]),  ## Balanced
            n=None,
            seed=42,
        )
        full = df.loc[full_idxs].sample(frac=1, random_state=42).reset_index(drop=True)
        train = full.loc[sample_idxs_match_distribution(
            full['headline_root'],
            target=pd.Series([lb for lb in label_space]),  ## Balanced
            n=52_000,
            seed=42,
        )].reset_index(drop=True)
        remaining_wo_train = full.query(f'idx not in {train["idx"].tolist()}').reset_index(drop=True)
        test = remaining_wo_train.loc[sample_idxs_match_distribution(
            remaining_wo_train['headline_root'],
            target=pd.Series([lb for lb in label_space]),  ## Balanced
            n=10_000,
            seed=42,
        )].reset_index(drop=True)
        validation = full.query(f'idx not in {train["idx"].tolist() + test["idx"].tolist()}').reset_index(drop=True)
        train['idx'] = train['idx'].apply(lambda x: f'train-{x}')
        test['idx'] = test['idx'].apply(lambda x: f'test-{x}')
        validation['idx'] = validation['idx'].apply(lambda x: f'validation-{x}')
        print(train['headline_root'].value_counts())
        print(validation['headline_root'].value_counts())
        print(test['headline_root'].value_counts())
        FileMetadata.of(f'{DATA_DIR}/data/toi_headlines/').mkdir()
        train.to_parquet(f'{DATA_DIR}/data/toi_headlines/toi_headlines_train.parquet')
        validation.to_parquet(f'{DATA_DIR}/data/toi_headlines/toi_headlines_validation.parquet')
        test.to_parquet(f'{DATA_DIR}/data/toi_headlines/toi_headlines_test.parquet')


if __name__ == '__main__':
    ## Convert from raw data file:
    ToiHeadlinesDataset.create_dataset()
    AmazonReviewsProductCategory.create_dataset()
    AmazonReviewsPolarity.create_dataset()
    AmazonHumorousProductQuestions.create_dataset()
    ## Copy schema and files to be accessible to all workers:
    HyperpartisanNewsDataset.setup()
    AGNewsDataset.setup()
    ToiHeadlinesDataset.setup()
    AmazonReviewsProductCategory.setup()
    AmazonReviewsPolarity.setup()
    AmazonHumorousProductQuestions.setup()
