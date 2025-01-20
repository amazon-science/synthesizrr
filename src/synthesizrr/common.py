import math
import re
from abc import ABC, abstractmethod

import pandas as pd
import ray
from fmcore.constants import DataLayout, DataSplit, FileFormat
from fmcore.data import FileMetadata, Reader, Writer
from fmcore.framework import ActorComposite
from fmcore.framework.chain.Chain import Step
from fmcore.framework.metric import Metric, Metrics
from fmcore.framework.task.classification import ClassificationData
from fmcore.framework.task.text_generation import (
    GENERATED_TEXTS_COL,
    TextGenerationsPredictionsBase,
)
from fmcore.framework.task_data import Dataset
from fmcore.util import (
    AutoEnum,
    FileSystemUtil,
    Parameters,
    ProgressBar,
    StringUtil,
    alias,
    all_are_none,
    as_list,
    auto,
    binary_search,
    get_default,
    irange,
    multiple_are_not_none,
    not_impl,
    only_item,
    parameterized_flatten,
    punct_normalize,
    remove_nulls,
    safe_validate_arguments,
    shuffle_items,
    str_normalize,
    type_str,
)
from fmcore.util.concurrency import accumulate
from nltk import word_tokenize
from pydantic import confloat, conint, root_validator
from pydantic.typing import Literal
from sklearn.model_selection import train_test_split

from synthesizrr.corpus import CORPUS_DIR
from synthesizrr.data import SynthesizRRDataset

RESULTS_DIR = FileMetadata.of(
    ""  # TODO: fill this out!
)

IDX_COL: str = "idx"
LABEL_TEXT_COL: str = "label_text"
LABEL_VERBALIZATION_COL: str = "label_verbalization"
LABEL_VERBALIZATION_TEMPL: str = "{label_verbalization}"
LABEL_VERBALIZATION_META_TEMPL: str = "{{label_verbalization}}"
ICL_EXAMPLES_TEMPL: str = "{icl_examples}"
ICL_EXAMPLES_META_TEMPL: str = "{{icl_examples}}"
EXAMPLE_TEXT_COL: str = "example_text"
QUERY_TEXT_COL: str = "query_text"
RETRIEVED_TOP_K_COL: str = "retrieved_top_k"
RETRIEVED_CONTEXT_COL: str = "retrieved_context"
ICL_RETRIEVED_CONTEXT_TEMPL: str = "{{icl[retrieved_context]}}"
ICL_EXAMPLE_TEXT_TEMPL: str = "{{icl[example_text]}}"
RETRIEVED_CONTEXT_TEMPL: str = "{{retrieved_context}}"
DISTANCE_COL: str = "distance"
DISTANCE_METRIC_COL: str = "distance_metric"
EFS_HUGGINGFACE_CACHE_DIR: FileMetadata = FileMetadata.of(
    "/efs/.cache/huggingface/hub/"
)
DEFAULT_SEED_SET_DATA_SPLIT: DataSplit = DataSplit.TRAIN
DEFAULT_SEED_SET_STRATIFY_ON_GROUND_TRUTH: bool = True
DEFAULT_SEED: int = 42
LABEL_OVERALL: str = "Overall"
DEFAULT_TOP_P: confloat(ge=0.0, le=1.0) = 0.9
DEFAULT_TEMPERATURE: confloat(ge=0.0, le=1.0) = 0.95


def TEXT_GEN_REFERENCES_COL(data_split: DataSplit) -> str:
    return f"{str(data_split).lower()}_text_references"


class Experiment(AutoEnum):
    Gold = alias("gold-dataset")
    SynthesizRR = alias("retrieval-augmented-dataset-generation", "synthesizrr")
    FewGen = alias("few-shot-generation")

    def canonical(self) -> str:
        return {
            Experiment.Gold: "gold-dataset",
            Experiment.SynthesizRR: "retrieval-augmented-dataset-generation",
            Experiment.FewGen: "few-shot-generation",
        }[self]


class CachedResultsStep(Step, ABC):
    @abstractmethod
    def save_to(
        self,
        *,
        results_dir: FileMetadata,
        **kwargs,
    ) -> FileMetadata:
        pass


def expand_num_samples_per_label(
    num_samples_per_label: Optional[Union[Dict[str, conint(ge=1)], conint(ge=1)]],
    label_verbalizer: Dict[str, str],
) -> Optional[Dict[str, int]]:
    if num_samples_per_label is None:
        return None
    if isinstance(num_samples_per_label, dict):
        return num_samples_per_label
    assert isinstance(num_samples_per_label, int)
    num_samples_per_label: Dict[str, conint(ge=1)] = {
        label_text: num_samples_per_label for label_text, _ in label_verbalizer.items()
    }
    assert isinstance(num_samples_per_label, dict)
    return num_samples_per_label


class Student(AutoEnum):
    TinyBert = alias("huawei-noah/TinyBERT_General_4L_312D")
    MiniLM = alias("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2")
    DistilBERT = alias("distilbert-base-uncased")
    DistilBERT_AttrPromptTable13 = alias("distilbert-base-uncased-attrprompt-table-13")
    BERT = alias("bert-base-uncased")
    DeBERTaV3Base = alias("microsoft/deberta-v3-base")
    DeBERTaV3Large = alias("microsoft/deberta-v3-large")

    def canonical(self) -> str:
        return {
            Student.TinyBert: "tinybert",
            Student.MiniLM: "all_minilm_l6_v2",
            Student.DistilBERT: "distilbert",
            Student.DistilBERT_AttrPromptTable13: "distilbert_attrprompt_t13",
            Student.BERT: "bert_base_uncased",
            Student.DeBERTaV3Base: "deberta_v3_base",
            Student.DeBERTaV3Large: "deberta_v3_large",
        }[self]

    def hf_model_name(self) -> str:
        return {
            Student.TinyBert: "huawei-noah/TinyBERT_General_4L_312D",
            Student.MiniLM: "sentence-transformers/all-MiniLM-L6-v2",
            Student.DistilBERT: "distilbert-base-uncased",
            Student.DistilBERT_AttrPromptTable13: "distilbert-base-uncased",
            Student.BERT: "bert-base-uncased",
            Student.DeBERTaV3Base: "microsoft/deberta-v3-base",
            Student.DeBERTaV3Large: "microsoft/deberta-v3-large",
        }[self]

    def algorithm(self) -> str:
        return {
            Student.TinyBert: "huggingface-SequenceClassification",
            Student.MiniLM: "pytorch",
            Student.DistilBERT: "huggingface-SequenceClassification",
            Student.DistilBERT_AttrPromptTable13: "huggingface-SequenceClassification",
            Student.BERT: "huggingface-SequenceClassification",
            Student.DeBERTaV3Base: "huggingface-SequenceClassification",
            Student.DeBERTaV3Large: "huggingface-SequenceClassification",
        }[self]

    def hyperparams(self, training_steps: int, student_hpo: bool) -> Dict:
        if student_hpo:
            batch_size: int = self.hpo_batch_size()
            gradient_accumulation_steps: int = self.hpo_gradient_accumulation_steps()
        else:
            batch_size: int = self.batch_size()
            gradient_accumulation_steps: int = self.gradient_accumulation_steps()

        if self in {Student.MiniLM}:
            hyperparams: Dict = dict(
                base_model=dict(
                    name="SentenceTransformers",
                    hyperparams=dict(
                        model_name=self.hf_model_name(),
                        max_sequence_length=self.max_sequence_length(),
                    ),
                ),
            )
        else:
            hyperparams: Dict = dict(
                model_name=self.hf_model_name(),
                tokenizer_encode=dict(max_length=self.max_sequence_length()),
            )
        hyperparams: Dict = {
            **hyperparams,
            **dict(
                batch_size=batch_size,
                dropout=self.dropout(),
                optimizer=dict(
                    name="AdamW",
                    lr=self.lr(),
                    weight_decay=self.adam_weight_decay(),
                    eps=self.adam_epsilon(),
                ),
                lr_scheduler=dict(
                    name="linear_schedule_with_warmup",
                    num_warmup_steps=max(
                        1,
                        int(
                            training_steps
                            * self.warmup_frac()
                            / gradient_accumulation_steps
                        ),
                    ),
                ),
                gradient_accumulation_steps=gradient_accumulation_steps,
                steps=training_steps,
            ),
        }
        return hyperparams

    def search_algorithm(self) -> str:
        return "grid"

    def search_space(self, student_hpo: bool) -> Dict:
        if student_hpo:
            batch_size: int = self.hpo_batch_size()
            gradient_accumulation_steps: int = self.hpo_gradient_accumulation_steps()
        else:
            batch_size: int = self.batch_size()
            gradient_accumulation_steps: int = self.gradient_accumulation_steps()

        search_space: Dict = dict(
            optimizer=dict(
                name="grid_search",
                values=[
                    dict(
                        name="AdamW",
                        lr=lr,
                        weight_decay=self.adam_weight_decay(),
                        eps=self.adam_epsilon(),
                    )
                    for (
                        lr,
                    ) in parameterized_flatten(  ## Note: parameterized_flatten returns a tuple
                        [2e-5, 5e-5, 1e-4],  ## lr
                    )
                ],
            ),
        )
        if gradient_accumulation_steps >= 4:
            ## For large models like DeBERTa, vary the gradient accumulation steps.
            search_space["gradient_accumulation_steps"] = dict(
                name="grid_search",
                values=[
                    gradient_accumulation_steps // 4,
                    gradient_accumulation_steps,
                    gradient_accumulation_steps * 4,
                ],
            )
        else:
            assert batch_size >= 4
            ## For smaller models like TinyBERT, vary the batch size steps.
            search_space["batch_size"] = dict(
                name="grid_search",
                values=[
                    batch_size // 4,
                    batch_size,
                    batch_size * 4,
                ],
            )
        return search_space

    def lr(self) -> float:
        ## From AttrPrompt:
        return {
            Student.TinyBert: 1e-4,
            Student.MiniLM: 5e-5,
            Student.DistilBERT: 5e-5,
            Student.DistilBERT_AttrPromptTable13: 2e-5,
            Student.BERT: 5e-5,
            Student.DeBERTaV3Base: 5e-5,
            Student.DeBERTaV3Large: 2e-5,
        }[self]

    def gradient_accumulation_steps(self) -> int:
        return {
            Student.TinyBert: 1,
            Student.MiniLM: 1,
            Student.DistilBERT: 1,
            Student.DistilBERT_AttrPromptTable13: 1,
            Student.BERT: 4,
            Student.DeBERTaV3Base: 8,
            Student.DeBERTaV3Large: 8,
        }[self]

    def batch_size(self) -> int:
        ## as per AttrPrompt paper. We need batch_size * gradient_accumulation_steps = 32 (effective batch size)
        return {
            Student.TinyBert: 32,
            Student.MiniLM: 32,
            Student.DistilBERT: 32,
            Student.DistilBERT_AttrPromptTable13: 32,
            Student.BERT: 8,
            Student.DeBERTaV3Base: 4,
            Student.DeBERTaV3Large: 4,
        }[self]

    def hpo_gradient_accumulation_steps(self) -> int:
        return {
            Student.TinyBert: 1,
            Student.MiniLM: 1,
            Student.DistilBERT: 4,
            Student.DistilBERT_AttrPromptTable13: 4,
            Student.BERT: 4,
            Student.DeBERTaV3Base: 8,
            Student.DeBERTaV3Large: 8,
        }[self]

    def hpo_batch_size(self) -> int:
        ## as per AttrPrompt paper. We need batch_size * gradient_accumulation_steps = 32 (effective batch size)
        return {
            Student.TinyBert: 32,
            Student.MiniLM: 32,
            Student.DistilBERT: 8,
            Student.DistilBERT_AttrPromptTable13: 8,
            Student.BERT: 8,
            Student.DeBERTaV3Base: 4,
            Student.DeBERTaV3Large: 4,
        }[self]

    def adam_weight_decay(self) -> float:
        return 1e-4  ## as per AttrPrompt paper

    def warmup_frac(self) -> float:
        return 0.06  ## 6% as per AttrPrompt paper

    def max_sequence_length(self) -> int:
        return 512  ## Increased from 128 in AttrPrompt to accommodate datasets like hyperpartisan etc.

    def dropout(self) -> float:
        return 0.0  ## Don't use dropout

    def adam_epsilon(self) -> float:
        return 1e-6  ## as per AttrPrompt codebase: https://github.com/yueyu1030/AttrPrompt/blob/3a05ffdcced6cacfd338e89ea8d0cd64ff449fa7/train_classifier/plm_model/main.py#L82

    def num_epochs(self) -> int:
        if self == Student.DistilBERT_AttrPromptTable13:
            return 5
        return 6  ## as per AttrPrompt paper

    def resources_per_model(self) -> Dict[str, int]:
        return {
            Student.TinyBert: dict(cpu=2, gpu=0.5),
            Student.MiniLM: dict(cpu=2, gpu=0.5),
            Student.DistilBERT: dict(cpu=2, gpu=1),
            Student.DistilBERT_AttrPromptTable13: dict(cpu=2, gpu=1),
            Student.BERT: dict(cpu=2, gpu=1),
            Student.DeBERTaV3Base: dict(cpu=2, gpu=1),
            Student.DeBERTaV3Large: dict(cpu=2, gpu=1),
        }[self]


class DatasetName(AutoEnum):
    AgNews = alias("ag")
    HyperpartisanNews = alias("hyperpartisan")
    ToiHeadlines = alias("toi-head")
    AmazonReviewsPolarity = alias("amazon-polarity")
    AmazonReviewsProductCategory = alias("amazon-reviews-category")
    AmazonHumorousProductQuestions = alias("amazon-humor")
    IMDb = alias("imdb-movie-reviews")
    SST2 = alias("stanford-sentiment-treebank")

    def canonical(self) -> str:
        return {
            DatasetName.AgNews: "ag_news",
            DatasetName.HyperpartisanNews: "hyperpartisan_news",
            DatasetName.ToiHeadlines: "toi_headlines",
            DatasetName.AmazonReviewsPolarity: "amazon_polarity",
            DatasetName.AmazonReviewsProductCategory: "amazon_reviews_category",
            DatasetName.AmazonHumorousProductQuestions: "amazon_humor",
            DatasetName.IMDb: "imdb",
            DatasetName.SST2: "sst2",
        }[self]

    def query_col(self) -> str:
        return self.text_col()

    def text_col(self) -> str:
        return {
            DatasetName.AgNews: "text",
            DatasetName.HyperpartisanNews: "text",
            DatasetName.ToiHeadlines: "text",
            DatasetName.AmazonReviewsPolarity: "text",
            DatasetName.AmazonReviewsProductCategory: "text",
            DatasetName.AmazonHumorousProductQuestions: "text",
            DatasetName.IMDb: "text",
            DatasetName.SST2: "text",
        }[self]

    def label_col(self) -> str:
        return {
            DatasetName.AgNews: "label_text",
            DatasetName.HyperpartisanNews: "label_text",
            DatasetName.ToiHeadlines: "label_text",
            DatasetName.AmazonReviewsPolarity: "label_text",
            DatasetName.AmazonReviewsProductCategory: "label_text",
            DatasetName.AmazonHumorousProductQuestions: "label_text",
            DatasetName.IMDb: "label_text",
            DatasetName.SST2: "label_text",
        }[self]

    def create_seed_set(
        self,
        seed_size: int,
        *,
        data_split: DataSplit,
        seed: int,
        stratify_on_ground_truth: bool,
        label_col: str,
        label_verbalizer: Dict[str, str],
    ) -> Dataset:
        dataset: Dataset = (
            SynthesizRRDataset.get(self.canonical())
            .datasets[data_split]
            .read(read_as=DataLayout.PANDAS)
        )
        dataset_df: pd.DataFrame = dataset.data.pandas()
        if stratify_on_ground_truth:
            ## Similar to dataset distribution:
            _, seed_dataset_df = train_test_split(
                dataset_df,
                test_size=seed_size,
                random_state=seed,
                stratify=dataset_df[label_col],
            )
        else:
            ## Balanced sample:
            num_labels: int = len(label_verbalizer)
            lb_sample_size: int = int(seed_size / num_labels)
            if lb_sample_size != (
                seed_size / num_labels
            ):  ## Check it is perfectly divisible.
                raise ValueError(
                    f"Expected seed size {seed_size} to be perfectly divisible "
                    f"by number of labels {len(label_verbalizer)}"
                )
            seed_dataset_df: pd.DataFrame = pd.concat(
                [
                    label_df.sample(n=lb_sample_size, random_state=seed, replace=True)
                    for label_text, label_df in dataset_df.groupby(label_col)
                    if label_text in label_verbalizer
                ]
            )
        seed_dataset_df = seed_dataset_df.reset_index(drop=True)
        if len(seed_dataset_df) != seed_size:
            raise ValueError(
                f"Expected seed dataset to be of size {seed_size}, but found {len(seed_dataset_df)} rows with"
                f"label distribution: {seed_dataset_df[label_col].to_dict()}"
            )
        return dataset.update_params(data=seed_dataset_df)

    def text_gens_parser(self) -> Callable:
        return GEN_PARSERS[self]

    def text_gens_parser_rejection(self, expt: Experiment) -> Callable:
        return TEXT_GENS_PARSERS_REJECTION[(self, expt)]

    def label_preservation_model_dir(
        self,
        *,
        results_dir: FileMetadata,  ## E.g. RESULTS_DIR/,
        student: Student,
    ) -> FileMetadata:
        return (
            results_dir.subdir_in_dir("label-preservation", return_metadata=True)
            .subdir_in_dir(self.canonical(), return_metadata=True)
            .subdir_in_dir(student.canonical(), return_metadata=True)
        )

    def label_preservation_best_trial(
        self,
        *,
        results_dir: FileMetadata,  ## E.g. RESULTS_DIR/,
        student: Student,
    ) -> Dict[str, Union[str, Metrics, FileMetadata]]:
        label_preservation_model_dir: FileMetadata = self.label_preservation_model_dir(
            results_dir=results_dir,
            student=student,
        )
        trialwise_final_model_metrics: Dict[str, Metrics] = Reader.of(
            FileFormat.PICKLE
        ).read(
            self.trialwise_final_model_metrics_file(label_preservation_model_dir),
        )
        best_trial_id, best_accuracy = sorted(
            [
                (
                    trial_id,
                    trial_metrics.find("Accuracy", data_split=DataSplit.TEST).value,
                )
                for trial_id, trial_metrics in trialwise_final_model_metrics.items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )[0]
        best_trial_metrics: Metrics = trialwise_final_model_metrics[best_trial_id]
        best_trial_label_preservation_model_dir: FileMetadata = (
            label_preservation_model_dir.subdir_in_dir(
                best_trial_id,
                return_metadata=True,
            )
        )
        return dict(
            best_trial_id=best_trial_id,
            best_trial_metrics=best_trial_metrics,
            best_trial_label_preservation_model_dir=best_trial_label_preservation_model_dir,
        )

    def trialwise_final_model_metrics_file(
        self,
        label_preservation_model_dir: FileMetadata,
    ) -> FileMetadata:
        return label_preservation_model_dir.file_in_dir(
            "trialwise_final_model_metrics.pkl",
            return_metadata=True,
        )

    def tune_metrics_file(
        self,
        label_preservation_model_dir: FileMetadata,
    ) -> FileMetadata:
        return label_preservation_model_dir.file_in_dir(
            "tune_metrics.pkl",
            return_metadata=True,
        )

    def max_num_tokens(self) -> int:
        return {
            DatasetName.AgNews: 200,
            DatasetName.HyperpartisanNews: 600,
            DatasetName.ToiHeadlines: 200,
            DatasetName.AmazonReviewsPolarity: 200,
            DatasetName.AmazonReviewsProductCategory: 400,
            DatasetName.AmazonHumorousProductQuestions: 50,
            DatasetName.IMDb: 400,
            DatasetName.SST2: 50,
        }[self]

    def seed_size(self) -> int:
        ## Pick 50 examples per class for multiclass & 100 examples per class for binary:
        return {
            DatasetName.AgNews: 4 * 50,
            DatasetName.HyperpartisanNews: 2 * 100,
            DatasetName.ToiHeadlines: 10 * 50,
            DatasetName.AmazonReviewsPolarity: 2 * 100,
            DatasetName.AmazonReviewsProductCategory: 23 * 50,
            DatasetName.AmazonHumorousProductQuestions: 2 * 100,
            DatasetName.IMDb: 2 * 100,
            DatasetName.SST2: 2 * 100,
        }[self]

    def label_verbalizer(self) -> Dict[str, str]:
        return {
            DatasetName.HyperpartisanNews: {
                "true": "using harsh political language, using a mocking tone and toxic commentary",
                "false": "using neutral language, using a reasonable tone and politically correct commentary",
            },
            DatasetName.AgNews: {
                "Business": "about companies, industries, markets, trade, investments, entrepreneurship, economic policies, and other business-related developments",
                "World": "about international news, such as politics, diplomacy, conflicts, global events, international relations, human rights issues, and significant global trends",
                "Sci/Tech": "about scientific discoveries, technological advancements, innovations, research breakthroughs",
                "Sports": "related to coverage of professional sports leagues, major tournaments, athletes, teams, match results, player transfers, coaching changes, sports-related controversies",
            },
            DatasetName.ToiHeadlines: {
                "sports": "sports in India",
                "life-style": "health and lifestyle trends in India",
                "education": "Indian examinations and education",
                "entertainment": "the Indian entertainment industry",
                "business": "business-related developments in India",
                "city": "ongoing matters in any Indian city",
                "environment": "environment-related events in Indian cities",
                "tech": "technology news and the tech industry in India",
                "elections": "elections and politics in India",
                "world": "international news and events outside of India",
            },
            DatasetName.AmazonReviewsPolarity: {
                "positive": "what the reviewer liked about the product, how the reviewer found it easy to use the product, or the reviewer's positive experience with the product",
                "negative": "what the reviewer disliked about the product, how the reviewer found it challenging to use the product, or the reviewer's negative experience with the product",
            },
            DatasetName.AmazonReviewsProductCategory: {
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
            },
            DatasetName.AmazonHumorousProductQuestions: {
                "non_humorous": "solemn",
                "humorous": "humorous",
            },
            DatasetName.IMDb: {
                "positive": "what the reviewer liked about the movie",
                "negative": "what the reviewer disliked about the movie",
            },
            DatasetName.SST2: {
                "positive": "what the reviewer liked about the movie",
                "negative": "what the reviewer disliked about the movie",
            },
        }[self]

    def icl_and_prompt_template(
        self, expt: Experiment, model_name: str
    ) -> Dict[str, str]:
        if expt is Experiment.Gold:
            return {}
        model_name: ModelName = ModelName(model_name)

        icl_and_prompt_template_dict: Dict[str, Union[Dict, List[Tuple], str]] = (
            ICL_AND_PROMPT_TEMPLATE_DICT[(self, expt)]
        )
        assert "icl_template" in icl_and_prompt_template_dict
        assert "prompt_template" in icl_and_prompt_template_dict
        assert "claude_replacements" in icl_and_prompt_template_dict
        icl_template: str = icl_and_prompt_template_dict["icl_template"]
        prompt_template: str = icl_and_prompt_template_dict["prompt_template"]

        chat_templates: Dict[str, str] = icl_and_prompt_template_dict["chat_templates"]
        chat_templates_chat_prompt_template: str = chat_templates[
            "chat_prompt_template"
        ]
        chat_templates_system_template: str = chat_templates["system_template"]
        chat_templates_icl_user_template: str = chat_templates["icl_user_template"]
        chat_templates_icl_assistant_template: str = chat_templates[
            "icl_assistant_template"
        ]
        chat_templates_user_template: str = chat_templates["user_template"]

        claude_replacements: List[Tuple[str, str]] = icl_and_prompt_template_dict[
            "claude_replacements"
        ]

        if model_name.is_claude():
            for repl in as_list(claude_replacements):
                assert icl_template.find(repl[0]) >= 0
                icl_template: str = icl_template.replace(
                    repl[0],
                    repl[1],
                )
                assert prompt_template.find(repl[0]) >= 0
                prompt_template: str = prompt_template.replace(
                    repl[0],
                    repl[1],
                )
        elif model_name.is_hf() and model_name.use_chat_template():
            tokenizer = model_name.tokenizer()
            apply_chat_template_params: Dict = dict(
                tokenize=False,
                add_generation_prompt=True,
            )

            ## Create icl_template:
            icl_template_chat_messages: List[Dict] = []
            if chat_templates_system_template.strip() != "" and None is not None:
                icl_template_chat_messages.append(
                    {
                        "role": None,
                        "content": chat_templates_system_template,
                    }
                )
            icl_template_chat_messages.append(
                {
                    "role": "user",
                    "content": chat_templates_icl_user_template,
                }
            )
            icl_template_chat_messages.append(
                {
                    "role": "assistant",
                    "content": chat_templates_icl_assistant_template,
                }
            )
            icl_template: str = tokenizer.apply_chat_template(
                icl_template_chat_messages,
                **apply_chat_template_params,
            )

            ## Create prompt_template:
            user_prompt_template_chat_messages: List[Dict] = []
            if chat_templates_system_template.strip() != "" and None is not None:
                user_prompt_template_chat_messages.append(
                    {
                        "role": None,
                        "content": chat_templates_system_template,
                    }
                )
            user_prompt_template_chat_messages.append(
                {
                    "role": "user",
                    "content": chat_templates_user_template,
                }
            )
            user_prompt_template: str = tokenizer.apply_chat_template(
                user_prompt_template_chat_messages,
                **apply_chat_template_params,
            )

            prompt_template: str = chat_templates_chat_prompt_template.format(
                **{
                    "user_prompt_template": user_prompt_template,
                }
            )

        assert ICL_EXAMPLE_TEXT_TEMPL in icl_template
        assert ICL_EXAMPLES_META_TEMPL in prompt_template
        if expt is Experiment.FewGen:
            assert ICL_RETRIEVED_CONTEXT_TEMPL not in icl_template
            assert RETRIEVED_CONTEXT_TEMPL not in prompt_template
        elif expt is Experiment.SynthesizRR:
            assert ICL_RETRIEVED_CONTEXT_TEMPL in icl_template
            assert RETRIEVED_CONTEXT_TEMPL in prompt_template
        return dict(
            icl_template=icl_template,
            prompt_template=prompt_template,
        )


ICL_AND_PROMPT_TEMPLATE_DICT: Dict[Tuple[DatasetName, Experiment], Dict] = {
    ##  _  _                                         _    _
    ## | || | _  _  _ __  ___  _ _  _ __  __ _  _ _ | |_ (_) ___ __ _  _ _
    ## | __ || || || '_ \/ -_)| '_|| '_ \/ _` || '_||  _|| |(_-</ _` || ' \
    ## |_||_| \_, || .__/\___||_|  | .__/\__,_||_|   \__||_|/__/\__,_||_||_|
    ##        |__/ |_|             |_|
    (DatasetName.HyperpartisanNews, Experiment.FewGen): dict(
        icl_template="""
Write a single news article {label_verbalization}. The written article should be 2 to 3 paragraphs long.
News Article:
{{icl[example_text]}}""".strip()
        + " ",
        prompt_template="""
{{icl_examples}}

Write a single news article {label_verbalization}. The written article should be 2 to 3 paragraphs long.
News Article: """.strip()
        + "\n",
        claude_replacements=[
            ("Write a single news article", "Human: Write a single news article"),
            ("News Article:", "News Article by Assistant:"),
        ],
        chat_templates=dict(
            chat_prompt_template="""
{{{{icl_examples}}}}

{user_prompt_template}
            """.strip(),
            system_template="",
            icl_user_template="""
Write a single news article {label_verbalization}. The written article should be 2 to 3 paragraphs long.
            """.strip(),
            icl_assistant_template="""
News Article:
{{icl[example_text]}}
            """.strip(),
            user_template="""
Write a single news article {label_verbalization}. The written article should be 2 to 3 paragraphs long.
            """.strip(),
        ),
    ),
    (DatasetName.HyperpartisanNews, Experiment.SynthesizRR): dict(
        icl_template="""
News Article:
{{icl[retrieved_context]}}

Rewrite the above news article {label_verbalization}. The rewritten article should be 2 to 3 paragraphs long.
Rewritten Article: 
{{icl[example_text]}}""",
        prompt_template="""
{{icl_examples}}

News Article:
{{retrieved_context}}

Rewrite the above news article {label_verbalization}. The rewritten article should be 2 to 3 paragraphs long.
Rewritten Article: """.strip()
        + "\n",
        claude_replacements=[
            ("News Article:", "Human: News Article:"),
            ("Rewritten Article:", "Rewritten Article by Assistant:"),
        ],
        chat_templates=dict(
            chat_prompt_template="""
{{{{icl_examples}}}}

{user_prompt_template}
            """.strip(),
            system_template="",
            icl_user_template="""
News Article:
{{icl[retrieved_context]}}

Rewrite the above news article {label_verbalization}. The rewritten article should be 2 to 3 paragraphs long.
            """.strip(),
            icl_assistant_template="""
Rewritten Article: 
{{icl[example_text]}}
            """.strip(),
            user_template="""
News Article:
{{retrieved_context}}

Rewrite the above news article {label_verbalization}. The rewritten article should be 2 to 3 paragraphs long.
            """.strip(),
        ),
    ),
    ##    _    ___   _  _
    ##   /_\  / __| | \| | ___ __ __ __ ___
    ##  / _ \| (_ | | .` |/ -_)\ V  V /(_-<
    ## /_/ \_\\___| |_|\_|\___| \_/\_/ /__/
    (DatasetName.AgNews, Experiment.FewGen): dict(
        icl_template="""
Write a summary for a news article {label_verbalization}. The summary should be one or two short sentences.
Summary: {{icl[example_text]}}
""",
        prompt_template="""
{{icl_examples}}

Write a summary for a news article {label_verbalization}. The summary should be one or two short sentences.
Summary: """.strip()
        + " ",
        claude_replacements=[
            (
                "Write a summary for a news article",
                "Human: Write a summary for a news article",
            ),
            ("Summary:", "Summary by Assistant:"),
        ],
        chat_templates=dict(
            chat_prompt_template="""
{{{{icl_examples}}}}

{user_prompt_template}
            """.strip(),
            system_template="",
            icl_user_template="""
Write a summary for a news article {label_verbalization}. The summary should be one or two short sentences.
            """.strip(),
            icl_assistant_template="""
Summary: {{icl[example_text]}}
            """.strip(),
            user_template="""
Write a summary for a news article {label_verbalization}. The summary should be one or two short sentences.
            """.strip(),
        ),
    ),
    (DatasetName.AgNews, Experiment.SynthesizRR): dict(
        icl_template="""
News Article:
{{icl[retrieved_context]}}

Write a summary for the above news article {label_verbalization}. The summary should be one or two short sentences.
Summary: {{icl[example_text]}}""",
        prompt_template="""
{{icl_examples}}

News Article:
{{retrieved_context}}

Write a summary for the above news article {label_verbalization}. The summary should be one or two short sentences.
Summary: """.strip()
        + " ",
        claude_replacements=[
            ("News Article:", "Human: News Article:"),
            ("Summary:", "Summary by Assistant:"),
        ],
        chat_templates=dict(
            chat_prompt_template="""
{{{{icl_examples}}}}

{user_prompt_template}
            """.strip(),
            system_template="",
            icl_user_template="""
News Article:
{{icl[retrieved_context]}}

Write a summary for the above news article {label_verbalization}. The summary should be one or two short sentences.
            """.strip(),
            icl_assistant_template="""
Summary: {{icl[example_text]}}
            """.strip(),
            user_template="""
News Article:
{{retrieved_context}}

Write a summary for the above news article {label_verbalization}. The summary should be one or two short sentences.
            """.strip(),
        ),
    ),
    ##  _____      ___   _  _                _  _  _
    ## |_   _|___ |_ _| | || | ___  __ _  __| || |(_) _ _   ___  ___
    ##   | | / _ \ | |  | __ |/ -_)/ _` |/ _` || || || ' \ / -_)(_-<
    ##   |_| \___/|___| |_||_|\___|\__,_|\__,_||_||_||_||_|\___|/__/
    (DatasetName.ToiHeadlines, Experiment.FewGen): dict(
        icl_template="""
Write a headline for a news article about {label_verbalization}. The headline should be a single sentence.
Headline: {{icl[example_text]}}
""",
        prompt_template="""
{{icl_examples}}

Write a headline for a news article about {label_verbalization}. The headline should be a single sentence.
Headline: """.strip()
        + " ",
        claude_replacements=[
            (
                "Write a headline for a news article",
                "Human: Write a headline for a news article",
            ),
            ("Headline:", "Headline by Assistant:"),
        ],
        chat_templates=dict(
            chat_prompt_template="""
{{{{icl_examples}}}}

{user_prompt_template}
            """.strip(),
            system_template="",
            icl_user_template="""
Write a headline for a news article about {label_verbalization}. The headline should be a single sentence.
            """.strip(),
            icl_assistant_template="""
Headline: {{icl[example_text]}}
            """.strip(),
            user_template="""
Write a headline for a news article about {label_verbalization}. The headline should be a single sentence.
            """.strip(),
        ),
    ),
    (DatasetName.ToiHeadlines, Experiment.SynthesizRR): dict(
        icl_template="""
News Article:
{{icl[retrieved_context]}}

Write a headline for the above news article about {label_verbalization}. The headline should be a single sentence.
Headline: {{icl[example_text]}}""",
        prompt_template="""
{{icl_examples}}

News Article:
{{retrieved_context}}

Write a headline for the above news article about {label_verbalization}. The headline should be a single sentence.
Headline: """.strip()
        + " ",
        claude_replacements=[
            ("News Article:", "Human: News Article:"),
            ("Headline:", "Headline by Assistant:"),
        ],
        chat_templates=dict(
            chat_prompt_template="""
{{{{icl_examples}}}}

{user_prompt_template}
            """.strip(),
            system_template="",
            icl_user_template="""
News Article:
{{icl[retrieved_context]}}

Write a headline for the above news article about {label_verbalization}. The headline should be a single sentence.
            """.strip(),
            icl_assistant_template="""
Headline: {{icl[example_text]}}
            """.strip(),
            user_template="""
News Article:
{{retrieved_context}}

Write a headline for the above news article about {label_verbalization}. The headline should be a single sentence.
            """.strip(),
        ),
    ),
    ##  ___       _             _  _
    ## | _ \ ___ | | __ _  _ _ (_)| |_  _  _
    ## |  _// _ \| |/ _` || '_|| ||  _|| || |
    ## |_|  \___/|_|\__,_||_|  |_| \__| \_, |
    ##                                  |__/
    (DatasetName.AmazonReviewsPolarity, Experiment.FewGen): dict(
        icl_template="""
Write a review about a product on Amazon which discusses {label_verbalization}. Include relevant product details. The review should only be a single short sentence, or a single paragraph of 3 to 4 sentences. Add very minor typos.
Review: {{icl[example_text]}}""".strip()
        + " ",
        prompt_template="""
{{icl_examples}}

Write a review about a product on Amazon which discusses {label_verbalization}. Include relevant product details. The review should only be a single short sentence, or a single paragraph of 3 to 4 sentences. Add very minor typos.
Review: """.strip()
        + " ",
        claude_replacements=[
            (
                "Write a review about a product on Amazon which discusses",
                "Human: Write a review about a product on Amazon which discusses",
            ),
            ("Review:", "Review by Assistant:"),
        ],
        chat_templates=dict(
            chat_prompt_template="""
{{{{icl_examples}}}}

{user_prompt_template}
            """.strip(),
            system_template="",
            icl_user_template="""
Write a review about a product on Amazon which discusses {label_verbalization}. Include relevant product details. The review should only be a single short sentence, or a single paragraph of 3 to 4 sentences. Add very minor typos.
            """.strip(),
            icl_assistant_template="""
Review: {{icl[example_text]}}
            """.strip(),
            user_template="""
Write a review about a product on Amazon which discusses {label_verbalization}. Include relevant product details. The review should only be a single short sentence, or a single paragraph of 3 to 4 sentences. Add very minor typos.
            """.strip(),
        ),
    ),
    (DatasetName.AmazonReviewsPolarity, Experiment.SynthesizRR): dict(
        icl_template="""
Product details:
{{icl[retrieved_context]}}

Write a review about the above product on Amazon which discusses {label_verbalization}. Include relevant product details which are mentioned above. The review should only be a single short sentence, or a single paragraph of 3 to 4 sentences. Add very minor typos.
Review: {{icl[example_text]}}""".strip()
        + " ",
        prompt_template="""
{{icl_examples}}

Product details:
{{retrieved_context}}

Write a review about the above product on Amazon which discusses {label_verbalization}. Include relevant product details which are mentioned above. The review should only be a single short sentence, or a single paragraph of 3 to 4 sentences. Add very minor typos.
Review: """.strip()
        + " ",
        claude_replacements=[
            ("Product details:", "Human: Product details:"),
            ("Review:", "Review by Assistant:"),
        ],
        chat_templates=dict(
            chat_prompt_template="""
{{{{icl_examples}}}}

{user_prompt_template}
            """.strip(),
            system_template="",
            icl_user_template="""
Product details:
{{icl[retrieved_context]}}

Write a review about the above product on Amazon which discusses {label_verbalization}. Include relevant product details which are mentioned above. The review should only be a single short sentence, or a single paragraph of 3 to 4 sentences. Add very minor typos.
            """.strip(),
            icl_assistant_template="""
Review: {{icl[example_text]}}
            """.strip(),
            user_template="""
Product details:
{{retrieved_context}}

Write a review about the above product on Amazon which discusses {label_verbalization}. Include relevant product details which are mentioned above. The review should only be a single short sentence, or a single paragraph of 3 to 4 sentences. Add very minor typos.
            """.strip(),
        ),
    ),
    ##   ___        _
    ##  / __| __ _ | |_  ___  __ _  ___  _ _  _  _
    ## | (__ / _` ||  _|/ -_)/ _` |/ _ \| '_|| || |
    ##  \___|\__,_| \__|\___|\__, |\___/|_|   \_, |
    ##                       |___/            |__/
    (DatasetName.AmazonReviewsProductCategory, Experiment.FewGen): dict(
        icl_template="""
Write a product review about a product which is in the category of {label_verbalization}. Include relevant product details. The review should only be a single short sentence, or a single paragraph of 3 to 4 sentences. Add very minor typos.
Review: {{icl[example_text]}}""".strip()
        + " ",
        prompt_template="""
{{icl_examples}}

Write a product review about a product which is in the category of {label_verbalization}. Include relevant product details. The review should only be a single short sentence, or a single paragraph of 3 to 4 sentences. Add very minor typos.
Review: """.strip()
        + " ",
        claude_replacements=[
            (
                "Write a product review about a product which is in the category of",
                "Human: Write a product review about a product which is in the category of",
            ),
            ("Review:", "Review by Assistant:"),
        ],
        chat_templates=dict(
            chat_prompt_template="""
{{{{icl_examples}}}}

{user_prompt_template}
            """.strip(),
            system_template="",
            icl_user_template="""
Write a product review about a product which is in the category of {label_verbalization}. Include relevant product details. The review should only be a single short sentence, or a single paragraph of 3 to 4 sentences. Add very minor typos.
            """.strip(),
            icl_assistant_template="""
Review: {{icl[example_text]}}
            """.strip(),
            user_template="""
Write a product review about a product which is in the category of {label_verbalization}. Include relevant product details. The review should only be a single short sentence, or a single paragraph of 3 to 4 sentences. Add very minor typos.
            """.strip(),
        ),
    ),
    (DatasetName.AmazonReviewsProductCategory, Experiment.SynthesizRR): dict(
        icl_template="""
Product details:
{{icl[retrieved_context]}}

Write a product review about the above product which is in the category of {label_verbalization}. Include relevant product details which are mentioned above. The review should only be a single short sentence, or a single paragraph of 3 to 4 sentences. Add very minor typos.
Review: {{icl[example_text]}}""".strip()
        + " ",
        prompt_template="""
{{icl_examples}}

Product details:
{{retrieved_context}}

Write a product review about the above product which is in the category of {label_verbalization}. Include relevant product details which are mentioned above. The review should only be a single short sentence, or a single paragraph of 3 to 4 sentences. Add very minor typos.
Review: """.strip()
        + " ",
        claude_replacements=[
            ("Product details:", "Human: Product details:"),
            ("Review:", "Review by Assistant:"),
        ],
        chat_templates=dict(
            chat_prompt_template="""
{{{{icl_examples}}}}

{user_prompt_template}
            """.strip(),
            system_template="",
            icl_user_template="""
Product details:
{{icl[retrieved_context]}}

Write a product review about the above product which is in the category of {label_verbalization}. Include relevant product details which are mentioned above. The review should only be a single short sentence, or a single paragraph of 3 to 4 sentences. Add very minor typos.
            """.strip(),
            icl_assistant_template="""
Review: {{icl[example_text]}}
            """.strip(),
            user_template="""
Product details:
{{retrieved_context}}

Write a product review about the above product which is in the category of {label_verbalization}. Include relevant product details which are mentioned above. The review should only be a single short sentence, or a single paragraph of 3 to 4 sentences. Add very minor typos.
            """.strip(),
        ),
    ),
    ##  _  _
    ## | || | _  _  _ __   ___  _ _
    ## | __ || || || '  \ / _ \| '_|
    ## |_||_| \_,_||_|_|_|\___/|_|
    (DatasetName.AmazonHumorousProductQuestions, Experiment.FewGen): dict(
        icl_template="""
Write a short {label_verbalization} question about a product on Amazon. Only include the question.
Product Question: {{icl[example_text]}}""".strip()
        + " ",
        prompt_template="""
{{icl_examples}}

Write a short {label_verbalization} question about a product on Amazon. Only include the question.
Product Question: """.strip()
        + " ",
        claude_replacements=[
            ("Write a short", "Human: Write a short"),
            ("Product Question:", "Product Question by Assistant:"),
        ],
        chat_templates=dict(
            chat_prompt_template="""
{{{{icl_examples}}}}

{user_prompt_template}
            """.strip(),
            system_template="",
            icl_user_template="""
Write a short {label_verbalization} question about a product on Amazon. Only include the question.
            """.strip(),
            icl_assistant_template="""
Product Question: {{icl[example_text]}}
            """.strip(),
            user_template="""
Write a short {label_verbalization} question about a product on Amazon. Only include the question.
            """.strip(),
        ),
    ),
    (DatasetName.AmazonHumorousProductQuestions, Experiment.SynthesizRR): dict(
        icl_template="""
Product details:
{{icl[retrieved_context]}}

Write a short {label_verbalization} question about the above product on Amazon. Only include the question.
Product Question: {{icl[example_text]}}""".strip()
        + " ",
        prompt_template="""
{{icl_examples}}

Product details:
{{retrieved_context}}

Write a short {label_verbalization} question about the above product on Amazon. Only include the question.
Product Question: """.strip()
        + " ",
        claude_replacements=[
            ("Product details:", "Human: Product details:"),
            ("Product Question:", "Product Question by Assistant:"),
        ],
        chat_templates=dict(
            chat_prompt_template="""
{{{{icl_examples}}}}

{user_prompt_template}
            """.strip(),
            system_template="",
            icl_user_template="""
Product details:
{{icl[retrieved_context]}}

Write a short {label_verbalization} question about the above product on Amazon. Only include the question.
            """.strip(),
            icl_assistant_template="""
Product Question: {{icl[example_text]}}
            """.strip(),
            user_template="""
Product details:
{{retrieved_context}}

Write a short {label_verbalization} question about the above product on Amazon. Only include the question.
            """.strip(),
        ),
    ),
    ##  ___  __  __  ___   _
    ## |_ _||  \/  ||   \ | |__
    ##  | | | |\/| || |) || '_ \
    ## |___||_|  |_||___/ |_.__/
    (DatasetName.IMDb, Experiment.FewGen): dict(
        icl_template="""
Write a review which discusses {label_verbalization}. Include relevant details about the movie. The review should only be a single short sentence, or a single paragraph of 3 to 4 sentences. Add very minor typos.
Review: {{icl[example_text]}}""".strip()
        + " ",
        prompt_template="""
{{icl_examples}}

Write a review which discusses {label_verbalization}. Include relevant details about the movie. The review should only be a single short sentence, or a single paragraph of 3 to 4 sentences. Add very minor typos.
Review: """.strip()
        + " ",
        claude_replacements=[
            ("Write a review which discusses", "Human: Write a review which discusses"),
            ("Review:", "Review by Assistant:"),
        ],
        chat_templates=dict(
            chat_prompt_template="""
{{{{icl_examples}}}}

{user_prompt_template}
            """.strip(),
            system_template="",
            icl_user_template="""
Write a review which discusses {label_verbalization}. Include relevant details about the movie. The review should only be a single short sentence, or a single paragraph of 3 to 4 sentences. Add very minor typos.
            """.strip(),
            icl_assistant_template="""
Review: {{icl[example_text]}}
            """.strip(),
            user_template="""
Write a review which discusses {label_verbalization}. Include relevant details about the movie. The review should only be a single short sentence, or a single paragraph of 3 to 4 sentences. Add very minor typos.
            """.strip(),
        ),
    ),
    (DatasetName.IMDb, Experiment.SynthesizRR): dict(
        icl_template="""
Movie details:
{{icl[retrieved_context]}}

Write a review which discusses {label_verbalization}. Include relevant details about the movie which are mentioned above. The review should only be a single short sentence, or a single paragraph of 3 to 4 sentences. Add very minor typos.
Review: {{icl[example_text]}}""".strip()
        + " ",
        prompt_template="""
{{icl_examples}}

Movie details:
{{retrieved_context}}

Write a review which discusses {label_verbalization}. Include relevant details about the movie which are mentioned above. The review should only be a single short sentence, or a single paragraph of 3 to 4 sentences. Add very minor typos.
Review: """.strip()
        + " ",
        claude_replacements=[
            ("Movie details:", "Human: Movie details:"),
            ("Review:", "Review by Assistant:"),
        ],
        chat_templates=dict(
            chat_prompt_template="""
{{{{icl_examples}}}}

{user_prompt_template}
            """.strip(),
            system_template="",
            icl_user_template="""
Movie details:
{{icl[retrieved_context]}}

Write a review which discusses {label_verbalization}. Include relevant details about the movie which are mentioned above. The review should only be a single short sentence, or a single paragraph of 3 to 4 sentences. Add very minor typos.
            """.strip(),
            icl_assistant_template="""
Review: {{icl[example_text]}}
            """.strip(),
            user_template="""
Movie details:
{{retrieved_context}}

Write a review which discusses {label_verbalization}. Include relevant details about the movie which are mentioned above. The review should only be a single short sentence, or a single paragraph of 3 to 4 sentences. Add very minor typos.
            """.strip(),
        ),
    ),
    ##  ___  ___  _____     ___
    ## / __|/ __||_   _|___|_  )
    ## \__ \\__ \  | | |___|/ /
    ## |___/|___/  |_|     /___|
    (DatasetName.SST2, Experiment.FewGen): dict(
        icl_template="""
Write a single sentence from a review which discusses {label_verbalization}. Include relevant details about the movie. The review should only be a single short sentence. Add very minor typos.
Review: {{icl[example_text]}}""".strip()
        + " ",
        prompt_template="""
{{icl_examples}}

Write a single sentence from a review which discusses {label_verbalization}. Include relevant details about the movie. The review should only be a single short sentence. Add very minor typos.
Review: """.strip()
        + " ",
        claude_replacements=[
            (
                "Write a single sentence from a review which discusses",
                "Human: Write a single sentence from a review which discusses",
            ),
            ("Review:", "Review by Assistant:"),
        ],
        chat_templates=dict(
            chat_prompt_template="""
{{{{icl_examples}}}}

{user_prompt_template}
            """.strip(),
            system_template="",
            icl_user_template="""
Write a single sentence from a review which discusses {label_verbalization}. Include relevant details about the movie. The review should only be a single short sentence. Add very minor typos.
            """.strip(),
            icl_assistant_template="""
Review: {{icl[example_text]}}
            """.strip(),
            user_template="""
Write a single sentence from a review which discusses {label_verbalization}. Include relevant details about the movie. The review should only be a single short sentence. Add very minor typos.
            """.strip(),
        ),
    ),
    (DatasetName.SST2, Experiment.SynthesizRR): dict(
        icl_template="""
Movie details:
{{icl[retrieved_context]}}

Write a single sentence from a review which discusses {label_verbalization}. Include relevant details about the movie which are mentioned above. The review should only be a single short sentence. Add very minor typos.
Review: {{icl[example_text]}}""".strip()
        + " ",
        prompt_template="""
{{icl_examples}}

Movie details:
{{retrieved_context}}

Write a single sentence from a review which discusses {label_verbalization}. Include relevant details about the movie which are mentioned above. The review should only be a single short sentence. Add very minor typos.
Review: """.strip()
        + " ",
        claude_replacements=[
            ("Movie details:", "Human: Movie details:"),
            ("Review:", "Review by Assistant:"),
        ],
        chat_templates=dict(
            chat_prompt_template="""
{{{{icl_examples}}}}

{user_prompt_template}
            """.strip(),
            system_template="",
            icl_user_template="""
Movie details:
{{icl[retrieved_context]}}

Write a single sentence from a review which discusses {label_verbalization}. Include relevant details about the movie which are mentioned above. The review should only be a single short sentence. Add very minor typos.
            """.strip(),
            icl_assistant_template="""
Review: {{icl[example_text]}}
            """.strip(),
            user_template="""
Movie details:
{{retrieved_context}}

Write a single sentence from a review which discusses {label_verbalization}. Include relevant details about the movie which are mentioned above. The review should only be a single short sentence. Add very minor typos.
            """.strip(),
        ),
    ),
}


class Corpus(AutoEnum):
    RealNews = alias("realnews-full")
    RealNews_India = alias("realnews-india-only")
    RealNews_Regional = alias("realnews-regional", "realnews-regional-only")
    RealNews_Dominant = auto()
    AmazonProducts = alias("products")
    CMU_Movies = alias("cmu-movie-summary")

    def canonical(self) -> str:
        return {
            Corpus.RealNews: "realnews",
            Corpus.RealNews_India: "realnews_india",
            Corpus.RealNews_Regional: "realnews_regional",
            Corpus.RealNews_Dominant: "realnews_dominant",
            Corpus.AmazonProducts: "amazon_products",
            Corpus.CMU_Movies: "cmu_movie_summary",
        }[self]

    def context_col(self) -> str:
        return {
            Corpus.RealNews: "text",
            Corpus.RealNews_India: "text",
            Corpus.RealNews_Regional: "text",
            Corpus.RealNews_Dominant: "text",
            Corpus.AmazonProducts: "product_text",
            Corpus.CMU_Movies: "text",
        }[self]

    def context_token_range(self) -> Tuple[int, int]:
        ## (min, max) tokens of retrieved context
        return {
            Corpus.RealNews_India: (20, 400),
            Corpus.RealNews_Regional: (20, 400),
            Corpus.RealNews_Dominant: (20, 400),
            Corpus.AmazonProducts: (20, 400),
            Corpus.CMU_Movies: (20, 400),
        }[self]

    def num_shards(self) -> int:
        return {
            Corpus.RealNews_India: 16,
            Corpus.RealNews_Regional: 16,
            Corpus.RealNews_Dominant: 384,
            Corpus.AmazonProducts: 202,
            Corpus.CMU_Movies: 8,
        }[self]

    def shard_num_cpus(self) -> int:
        return {
            Corpus.RealNews_India: 6,
            Corpus.RealNews_Regional: 6,
            Corpus.RealNews_Dominant: 2,
            Corpus.AmazonProducts: 2,
            Corpus.CMU_Movies: 6,
        }[self]

    def synthesizrr_top_k_range(self) -> range:
        ## (min, max) tokens of retrieved context
        return {
            Corpus.RealNews_India: irange(1, 50, 1),
            Corpus.RealNews_Regional: irange(1, 50, 1),
            Corpus.RealNews_Dominant: irange(1, 50, 1),
            Corpus.AmazonProducts: irange(1, 50, 1),
            Corpus.CMU_Movies: irange(1, 50, 1),
        }[self]

    def _raw_text_path(self) -> str:
        return {
            # Corpus.RealNews: '',
            Corpus.RealNews_India: f"{CORPUS_DIR}/data/realnews/realnews-india/",
            Corpus.RealNews_Regional: f"{CORPUS_DIR}/data/realnews/realnews-regional/",
            Corpus.RealNews_Dominant: f"{CORPUS_DIR}/data/realnews/realnews-dominant/",
            Corpus.AmazonProducts: f"{CORPUS_DIR}/data/amazon-reviews/2018/meta/raw-text/",
            Corpus.CMU_Movies: f"{CORPUS_DIR}/data/cmu-movie-summary/raw-text/",
        }[self]

    def raw_text_dir(self) -> FileMetadata:
        return {
            Corpus.RealNews_India: FileMetadata.of(
                self._raw_text_path(),
                data_schema={
                    "idx": "index",
                    "title": "object",
                    "text": "text",
                    "summary": "object",
                    "authors": "categorical",
                    "publish_date": "object",
                    "status": "categorical",
                    "url": "categorical",
                    "domain": "categorical",
                    "warc_date": "object",
                    "split": "categorical",
                },
                file_format="parquet",
                file_glob="*.parquet",
            ),
            Corpus.RealNews_Regional: FileMetadata.of(
                self._raw_text_path(),
                data_schema={
                    "idx": "index",
                    "title": "object",
                    "text": "text",
                    "summary": "object",
                    "authors": "categorical",
                    "publish_date": "object",
                    "status": "categorical",
                    "url": "categorical",
                    "domain": "categorical",
                    "warc_date": "object",
                    "split": "categorical",
                },
                file_format="parquet",
                file_glob="*.parquet",
            ),
            Corpus.RealNews_Dominant: FileMetadata.of(
                self._raw_text_path(),
                data_schema={
                    "idx": "index",
                    "title": "object",
                    "text": "text",
                    "summary": "object",
                    "authors": "categorical",
                    "publish_date": "object",
                    "status": "categorical",
                    "url": "categorical",
                    "domain": "categorical",
                    "warc_date": "object",
                    "split": "categorical",
                },
                file_format="parquet",
                file_glob="*.parquet",
            ),
            Corpus.AmazonProducts: FileMetadata.of(
                self._raw_text_path(),
                data_schema={
                    "asin": "index",
                    # "also_buy": 'object',
                    # "also_view": 'object',
                    "title": "object",
                    "description": "object",
                    "brand": "object",
                    "category": "object",
                    "date": "object",
                    # "details": 'object',
                    "feature": "object",
                    "fit": "object",
                    # "image": 'object',
                    "main_cat": "object",
                    "price": "object",
                    # "rank": 'object',
                    # "similar_item": 'object',
                    # "tech1": 'object',
                    # "tech2": 'object',
                    "product_text": "text",
                },
                file_format="parquet",
                file_glob="*.parquet",
            ),
            Corpus.CMU_Movies: FileMetadata.of(
                self._raw_text_path(),
                data_schema={
                    "idx": "index",
                    "freebase_movie_id": "object",
                    "title": "object",
                    "text": "text",
                    "release_date": "object",
                    "box_office_revenue": "object",
                    "runtime": "object",
                    "languages": "object",
                    "countries": "object",
                    "genres": "object",
                },
                file_format="parquet",
                file_glob="*.parquet",
            ),
        }[self]


class ModelName(AutoEnum):
    LLaMa_2_7B = auto()
    LLaMa_2_7B_Chat = auto()
    LLaMa_2_13B = auto()
    LLaMa_2_13B_Chat = auto()
    ChatGPT = alias("gpt-3.5-turbo")
    GPT_4 = auto()
    Claude_Instant_v1 = auto()
    Claude_v2 = auto()
    Phi_2 = auto()
    Phi_3_mini = auto()
    Qwen_1_8B_Chat = auto()
    Mistral_7B_Instruct = auto()
    Mixtral_8x7B_Instruct = auto()

    def canonical(self) -> str:
        return {
            ModelName.LLaMa_2_7B: "Llama-2-7B",
            ModelName.LLaMa_2_7B_Chat: "Llama-2-7B-Chat",
            ModelName.LLaMa_2_13B: "Llama-2-13B",
            ModelName.LLaMa_2_13B_Chat: "Llama-2-13B-Chat",
            ModelName.ChatGPT: "gpt3.5turbo",
            ModelName.GPT_4: "gpt-4",
            ModelName.Claude_Instant_v1: "claude-instant-v1",
            ModelName.Claude_v2: "claude-v2",
            ModelName.Qwen_1_8B_Chat: "Qwen-1.8B-Chat",
            ModelName.Phi_2: "phi-2",
            ModelName.Phi_3_mini: "phi-3-mini-4k-instruct",
            ModelName.Mistral_7B_Instruct: "mistral-7b-instruct-v0.2",
            ModelName.Mixtral_8x7B_Instruct: "mixtral-8x7b-instruct-v0.1",
        }[self]

    def is_hf(self) -> bool:
        return self in {
            ModelName.LLaMa_2_7B,
            ModelName.LLaMa_2_7B_Chat,
            ModelName.LLaMa_2_13B,
            ModelName.LLaMa_2_13B_Chat,
            ModelName.Phi_2,
            ModelName.Phi_3_mini,
            ModelName.Qwen_1_8B_Chat,
            ModelName.Mistral_7B_Instruct,
            ModelName.Mixtral_8x7B_Instruct,
        }

    def is_claude(self) -> bool:
        return self in {
            ModelName.Claude_Instant_v1,
            ModelName.Claude_v2,
        }

    def is_openai(self) -> bool:
        return self in {
            ModelName.ChatGPT,
            ModelName.GPT_4,
        }

    def model_name(self) -> str:
        return {
            ModelName.LLaMa_2_7B: "TheBloke/Llama-2-7B-fp16",
            ModelName.LLaMa_2_7B_Chat: "TheBloke/Llama-2-7B-Chat-fp16",
            ModelName.LLaMa_2_13B: "TheBloke/Llama-2-13B-fp16",
            ModelName.LLaMa_2_13B_Chat: "TheBloke/Llama-2-13B-Chat-fp16",
            ModelName.ChatGPT: "gpt-3.5-turbo-1106",
            ModelName.GPT_4: "gpt-4-0613",
            ModelName.Qwen_1_8B_Chat: "Qwen/Qwen-1_8B-Chat",
            ModelName.Claude_Instant_v1: "anthropic.claude-instant-v1",
            ModelName.Claude_v2: "anthropic.claude-v2",
            ModelName.Phi_2: "microsoft/phi-2",
            ModelName.Phi_3_mini: "microsoft/Phi-3-mini-4k-instruct",
            ModelName.Mistral_7B_Instruct: "mistralai/Mistral-7B-Instruct-v0.2",
            ModelName.Mixtral_8x7B_Instruct: "mistralai/Mixtral-8x7B-Instruct-v0.1",
        }[self]

    def algorithm_name(self) -> str:
        return {
            ModelName.LLaMa_2_7B: "huggingface-CausalLM",
            ModelName.LLaMa_2_7B_Chat: "huggingface-CausalLM",
            ModelName.LLaMa_2_13B: "huggingface-CausalLM",
            ModelName.LLaMa_2_13B_Chat: "huggingface-CausalLM",
            ModelName.Phi_2: "huggingface-CausalLM",
            ModelName.Phi_3_mini: "huggingface-CausalLM",
            ModelName.Mistral_7B_Instruct: "huggingface-CausalLM",
            ModelName.Mixtral_8x7B_Instruct: "huggingface-CausalLM",
            ModelName.Qwen_1_8B_Chat: "huggingface-CausalLM",
            ModelName.ChatGPT: "langchain",
            ModelName.GPT_4: "langchain",
            ModelName.Claude_Instant_v1: "bedrock",
            ModelName.Claude_v2: "bedrock",
        }[self]

    def nested_evaluator(self) -> Optional[str]:
        if not self.is_hf():
            return None

        return {
            ModelName.LLaMa_2_7B: "accelerate",
            ModelName.LLaMa_2_7B_Chat: "accelerate",
            ModelName.LLaMa_2_13B: "accelerate",
            ModelName.LLaMa_2_13B_Chat: "local",
            ModelName.Mistral_7B_Instruct: "local",
            ModelName.Mixtral_8x7B_Instruct: "local",
            ModelName.Phi_2: "local",
            ModelName.Phi_3_mini: "local",
            ModelName.Qwen_1_8B_Chat: "local",
        }[self]

    def nested_evaluator_use_hf_from_pretrained(self) -> bool:
        if not self.is_hf():
            return False
        return {
            ModelName.LLaMa_2_7B: False,
            ModelName.LLaMa_2_7B_Chat: False,
            ModelName.LLaMa_2_13B: False,
            ModelName.LLaMa_2_13B_Chat: False,
            ModelName.Mistral_7B_Instruct: False,
            ModelName.Mixtral_8x7B_Instruct: False,
            ModelName.Phi_2: False,
            ModelName.Phi_3_mini: False,
            ModelName.Qwen_1_8B_Chat: False,
        }[self]

    def model_weights_dtype(self) -> Optional[str]:
        return {
            ModelName.LLaMa_2_7B: "float16",
            ModelName.LLaMa_2_7B_Chat: "float16",
            ModelName.LLaMa_2_13B: "float16",
            ModelName.LLaMa_2_13B_Chat: "float16",
            ModelName.Phi_2: "float16",
            ModelName.Phi_3_mini: "float16",
            ModelName.Qwen_1_8B_Chat: "float16",
            ModelName.Mistral_7B_Instruct: "bfloat16",
            ModelName.Mixtral_8x7B_Instruct: "float16",
        }[self]

    def max_input_length(self) -> int:
        return {
            ModelName.LLaMa_2_7B: 4000,
            ModelName.LLaMa_2_7B_Chat: 4000,
            ModelName.LLaMa_2_13B: 4000,
            ModelName.LLaMa_2_13B_Chat: 4000,
            ModelName.Qwen_1_8B_Chat: 8000,
            ModelName.Phi_2: 2040,
            ModelName.Phi_3_mini: 4000,
            ModelName.Mistral_7B_Instruct: 32_000,
            ModelName.Mixtral_8x7B_Instruct: 32_000,
        }[self]

    def llm_resources_per_model(self) -> Dict[str, int]:
        return {
            ModelName.LLaMa_2_7B: dict(cpu=2, gpu=2),
            ModelName.LLaMa_2_7B_Chat: dict(cpu=2, gpu=2),
            ModelName.LLaMa_2_13B: dict(cpu=2, gpu=2),
            ModelName.LLaMa_2_13B_Chat: dict(cpu=2, gpu=2),
            ModelName.Phi_2: dict(cpu=2, gpu=1),
            ModelName.Phi_3_mini: dict(cpu=2, gpu=1),
            ModelName.Qwen_1_8B_Chat: dict(cpu=2, gpu=1),
            ModelName.Claude_Instant_v1: dict(cpu=1),
            ModelName.Claude_v2: dict(cpu=1),
            ModelName.ChatGPT: dict(cpu=1),
            ModelName.GPT_4: dict(cpu=1),
            ModelName.Mistral_7B_Instruct: dict(cpu=2, gpu=1),
            ModelName.Mixtral_8x7B_Instruct: dict(cpu=4, gpu=4),
        }[self]

    def llm_batch_size(self, *, dataset_name: DatasetName, expt: Experiment) -> int:
        llm_batch_sizes_dict: Dict = {
            ModelName.Claude_Instant_v1: 3,
            ModelName.Claude_v2: 3,
            ModelName.ChatGPT: 3,
            ModelName.GPT_4: 3,
            ModelName.Qwen_1_8B_Chat: 3,
            ModelName.Phi_2: 3,
            ModelName.Phi_3_mini: 3,
            ModelName.LLaMa_2_7B: 1,
            ModelName.LLaMa_2_7B_Chat: 1,
            ModelName.LLaMa_2_13B: 1,
            (ModelName.LLaMa_2_13B, DatasetName.AmazonHumorousProductQuestions): 3,
            (ModelName.LLaMa_2_13B, DatasetName.AmazonReviewsPolarity): 3,
            (ModelName.LLaMa_2_13B, DatasetName.IMDb): 2,
            (ModelName.LLaMa_2_13B, DatasetName.AgNews): 2,
            (ModelName.LLaMa_2_13B, DatasetName.ToiHeadlines): 2,
            ModelName.LLaMa_2_13B_Chat: 1,
            (ModelName.LLaMa_2_13B_Chat, DatasetName.AmazonHumorousProductQuestions): 3,
            (ModelName.LLaMa_2_13B_Chat, DatasetName.AmazonReviewsPolarity): 3,
            (ModelName.LLaMa_2_13B_Chat, DatasetName.IMDb): 2,
            (ModelName.LLaMa_2_13B_Chat, DatasetName.AgNews): 2,
            (ModelName.LLaMa_2_13B, DatasetName.ToiHeadlines): 2,
            ModelName.Mistral_7B_Instruct: 1,
            ModelName.Mixtral_8x7B_Instruct: 1,
        }
        return get_default(
            llm_batch_sizes_dict.get((self, dataset_name, expt)),
            llm_batch_sizes_dict.get((self, dataset_name)),
            llm_batch_sizes_dict[self],  ## Default for the model_name
        )

    def pad_token(self) -> str:
        return {
            ModelName.LLaMa_2_7B: "[PAD]",
            ModelName.LLaMa_2_7B_Chat: "[PAD]",
            ModelName.LLaMa_2_13B: "[PAD]",
            ModelName.LLaMa_2_13B_Chat: "[PAD]",
            ModelName.Qwen_1_8B_Chat: "[PAD]",
            ModelName.Phi_2: None,
            ModelName.Phi_3_mini: "<|endoftext|>",
            ModelName.Mistral_7B_Instruct: "</s>",
            ModelName.Mixtral_8x7B_Instruct: "</s>",
        }[self]

    def api_key(self) -> Optional[str]:
        return {
            ModelName.ChatGPT: "sk-examplekeyabcd",
            ModelName.GPT_4: "sk-examplekeyabcd",
            ModelName.Mistral_7B_Instruct: "hf_examplekeyabcd",
            ModelName.Mixtral_8x7B_Instruct: "hf_examplekeyabcd",
        }.get(self)

    def input_price_per_1M_tokens(self) -> float:
        ## Price in USD:
        return {
            ## https://aws.amazon.com/bedrock/pricing/
            self.LLaMa_2_13B_Chat: 0.00075 * 1e3,
            self.Claude_Instant_v1: 0.00080 * 1e3,
            self.Claude_v2: 0.00800 * 1e3,
            ## https://openai.com/pricing
            self.ChatGPT: 0.50,
            self.GPT_4: 30.00,
            self.Mistral_7B_Instruct: 0.00015 * 1e3,
            self.Mixtral_8x7B_Instruct: 0.00045 * 1e3,
        }.get(self, 0.0)

    def output_price_per_1M_tokens(self) -> float:
        ## Price in USD:
        return {
            ## https://aws.amazon.com/bedrock/pricing/
            self.LLaMa_2_13B_Chat: 0.00100 * 1e3,
            self.Claude_Instant_v1: 0.00240 * 1e3,
            self.Claude_v2: 0.02400 * 1e3,
            ## https://openai.com/pricing
            self.ChatGPT: 1.50,
            self.GPT_4: 60.00,
            self.Mistral_7B_Instruct: 0.00020 * 1e3,
            self.Mixtral_8x7B_Instruct: 0.0007 * 1e3,
        }.get(self, 0.0)

    def use_chat_template(self) -> bool:
        if not self.is_hf():
            return False
        return {
            ModelName.LLaMa_2_7B: False,
            ModelName.LLaMa_2_7B_Chat: False,
            ModelName.LLaMa_2_13B: False,
            ModelName.LLaMa_2_13B_Chat: False,
            ModelName.Qwen_1_8B_Chat: False,
            ModelName.Phi_2: False,
            ModelName.Phi_3_mini: True,
            ModelName.Mistral_7B_Instruct: True,
            ModelName.Mixtral_8x7B_Instruct: True,
        }[self]

    def system_role(self) -> Optional[str]:
        if not self.use_chat_template():
            return None
        return {
            ModelName.Phi_3_mini: None,
            ModelName.Mistral_7B_Instruct: None,
            ModelName.Mixtral_8x7B_Instruct: None,
        }[self]

    def user_role(self) -> Optional[str]:
        if not self.use_chat_template():
            return None
        return {
            ModelName.Phi_3_mini: "user",
            ModelName.Mistral_7B_Instruct: "user",
            ModelName.Mixtral_8x7B_Instruct: "user",
        }[self]

    def assistant_role(self) -> Optional[str]:
        if not self.use_chat_template():
            return None
        return {
            ModelName.Phi_3_mini: "assistant",
            ModelName.Mistral_7B_Instruct: "assistant",
            ModelName.Mixtral_8x7B_Instruct: "assistant",
        }[self]

    def tokenizer(self) -> Any:
        if not hasattr(self, "_tokenizer"):
            if self.is_hf():
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name(),
                    pad_token=self.pad_token(),
                    padding_side="left",
                    truncation_side="left",
                    token=self.api_key(),
                )
            elif self is ModelName.ChatGPT:
                import tiktoken

                tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
            elif self is ModelName.GPT_4:
                import tiktoken

                tokenizer = tiktoken.encoding_for_model("gpt-4")
            elif self.is_claude():
                from transformers import PreTrainedTokenizerFast

                tokenizer = PreTrainedTokenizerFast(
                    tokenizer_file=FileSystemUtil.expand_dir(
                        "~/claude/claude-v1-tokenization.json"
                    )
                )
            else:
                raise NotImplementedError(f"Unsupported: {self}")
            setattr(self, "_tokenizer", tokenizer)
        return getattr(self, "_tokenizer")


class Retriever(AutoEnum):
    Contriever = alias("facebook/contriever")
    MiniLM = alias("MiniLM", "all-MiniLM-L6-v2")
    BM25Okapi = alias("BM25")
    Random = alias("random-retriever")

    def canonical(self) -> str:
        return {
            Retriever.Contriever: "contriever",
            Retriever.MiniLM: "all_minilm_l6_v2",
            Retriever.BM25Okapi: "bm25_okapi",
            Retriever.Random: "random",
        }[self]

    def batch_size(self) -> int:
        return {
            Retriever.Contriever: 16,
            Retriever.MiniLM: 16,
            Retriever.BM25Okapi: 16,
            Retriever.Random: 16,
        }[self]

    def is_dense(self):
        return self in {Retriever.Contriever, Retriever.MiniLM}

    def is_sparse(self):
        return self in {Retriever.BM25Okapi}


def count_num_tokens(text: str, *, tokenizer: Any):
    import tiktoken
    import transformers

    if isinstance(tokenizer, tiktoken.core.Encoding):
        return len(tokenizer.encode(text))
    elif isinstance(
        tokenizer, transformers.tokenization_utils_base.PreTrainedTokenizerBase
    ):
        return len(tokenizer.encode(text, add_special_tokens=False))
    else:
        raise NotImplementedError(
            f"Unrecognized type for tokenizer: {type_str(tokenizer)}"
        )


def shorten(
    text: str,
    *,
    tokenizer: Any,
    min_tokens: Optional[int] = None,
    max_tokens: int,
    sentence_end: str = ".",
) -> Optional[str]:
    import tiktoken
    import transformers

    if isinstance(tokenizer, tiktoken.core.Encoding):
        sentence_end_token_id: int = tokenizer.encode_single_token(sentence_end)
        token_ids: List[int] = tokenizer.encode(text)
        if min_tokens is not None and len(token_ids) < min_tokens:
            return None
        token_ids: List[int] = token_ids[:max_tokens]
        max_tok_idx: int = -1
        for tok_idx, tok_id in enumerate(token_ids):
            if tok_id == sentence_end_token_id:
                max_tok_idx: int = max(max_tok_idx, tok_idx)
        if max_tok_idx == -1:
            return text  ## Not a single period in the sentence, so probably a short one like a chat.
        else:
            return tokenizer.decode(token_ids[: max_tok_idx + 1])
    elif isinstance(
        tokenizer, transformers.tokenization_utils_base.PreTrainedTokenizerBase
    ):
        sentence_end_token_id: int = tokenizer.get_vocab()[sentence_end]
        assert sentence_end == tokenizer.decode(sentence_end_token_id)
        token_ids: List[int] = tokenizer.encode(text, add_special_tokens=False)
        if min_tokens is not None and len(token_ids) < min_tokens:
            return None
        token_ids: List[int] = token_ids[:max_tokens]
        max_tok_idx: int = -1
        for tok_idx, tok_id in enumerate(token_ids):
            if tok_id == sentence_end_token_id:
                max_tok_idx: int = max(max_tok_idx, tok_idx)
        if max_tok_idx == -1:
            return text  ## Not a single period in the sentence, so probably a short one like a chat.
        else:
            return tokenizer.decode(token_ids[: max_tok_idx + 1])
    else:
        raise NotImplementedError(
            f"Unrecognized type for tokenizer: {type_str(tokenizer)}"
        )


@ray.remote
class ShortenActor:
    def __init__(self, model_name: ModelName):
        self.tokenizer = model_name.tokenizer()

    def shorten(self, text: str, **kwargs) -> str:
        return shorten(text=text, tokenizer=self.tokenizer, **kwargs)


def shorten_batch(
    texts: List[str],
    *,
    model_name: ModelName,
    verbosity: int,
    **kwargs,
) -> List[str]:
    def actor_factory(*, actor_id: str, **kwargs):
        return ShortenActor.options(
            num_cpus=1,
        ).remote(
            model_name=model_name,
        )

    texts: List[str] = as_list(texts)
    actors: List[ActorComposite] = ActorComposite.create_actors(
        actor_factory,
        num_actors=max(1, len(texts) // 30),
    )
    try:
        shortened_texts: List[str] = accumulate(
            [
                actors[text_i % len(actors)].actor.shorten.remote(text, **kwargs)
                for text_i, text in enumerate(texts)
            ],
            progress_bar={"desc": ""} if verbosity >= 2 else False,
        )
        return shortened_texts
    finally:
        for actor in actors:
            actor.kill()
            del actor
        del actors


def tokenize_batch(texts: List[str], *, model_name) -> List[List[int]]:
    tokenizer = model_name.tokenizer()
    import tiktoken
    import transformers

    if isinstance(tokenizer, tiktoken.core.Encoding):
        return [list(tokenizer.encode(text)) for text in texts]
    elif isinstance(
        tokenizer, transformers.tokenization_utils_base.PreTrainedTokenizerBase
    ):
        return [
            list(tokenizer.encode(text, add_special_tokens=False)) for text in texts
        ]
    else:
        raise NotImplementedError(
            f"Unrecognized type for tokenizer: {type_str(tokenizer)}"
        )


class MetricName(AutoEnum):
    NoMetric = auto()
    RowCount = auto()
    TextLength = auto()
    EntityCount = auto()
    SelfBLEU = auto()
    LabelwiseCosineSimilarity = auto()
    PairwiseCosineSimilarity = auto()
    Mauve = auto()
    LabelPreservation = auto()
    StudentDatasetCartography = auto()
    StudentTinyBert = alias("student-huawei-noah/TinyBERT_General_4L_312D")
    StudentMiniLM = alias(
        "student-all-MiniLM-L6-v2", "student-sentence-transformers/all-MiniLM-L6-v2"
    )
    StudentDistilBERT = alias("student-distilbert-base-uncased")
    StudentDistilBERT_AttrPromptTable13 = alias(
        "student-distilbert-base-uncased-attrprompt-table13"
    )
    StudentBERT = alias("student-bert-base-uncased")
    StudentDeBERTaV3Base = alias("student-microsoft/deberta-v3-base")
    StudentDeBERTaV3Large = alias("student-microsoft/deberta-v3-large")

    StudentHPOTinyBert = alias("hpo-student-huawei-noah/TinyBERT_General_4L_312D")
    StudentHPOMiniLM = alias(
        "hpo-student-all-MiniLM-L6-v2",
        "hpo-student-sentence-transformers/all-MiniLM-L6-v2",
    )
    StudentHPODistilBERT = alias("hpo-student-distilbert-base-uncased")
    StudentHPODistilBERT_AttrPromptTable13 = alias(
        "hpo-student-distilbert-base-uncased-attrprompt-table13"
    )
    StudentHPOBERT = alias("hpo-student-bert-base-uncased")
    StudentHPODeBERTaV3Base = alias("hpo-student-microsoft/deberta-v3-base")
    StudentHPODeBERTaV3Large = alias("hpo-student-microsoft/deberta-v3-large")

    RagasFaithfulness = alias("faithfulness")
    RagasContextRelevance = alias("context_relevance")
    RagasAnswerRelevance = alias("answer_relevance")

    SaveFilteredDataset = alias("save")

    def canonical(self) -> str:
        if self.is_student():
            return f"student_{self.to_student().canonical()}"
        if self.is_student_hpo():
            return f"student_hpo_{self.to_student_hpo().canonical()}"
        return {
            MetricName.NoMetric: "no_metric",
            MetricName.RowCount: "row_count",
            MetricName.TextLength: "text_length",
            MetricName.EntityCount: "entity_count",
            MetricName.SelfBLEU: "self_bleu",
            MetricName.LabelwiseCosineSimilarity: "label_cosine_sim",
            MetricName.PairwiseCosineSimilarity: "pairwise_cosine_sim",
            MetricName.Mauve: "mauve",
            MetricName.LabelPreservation: "label_preservation",
            MetricName.StudentDatasetCartography: "cartography",
            MetricName.RagasFaithfulness: "ragas_faithfulness",
            MetricName.RagasContextRelevance: "ragas_context_relevance",
            MetricName.RagasAnswerRelevance: "ragas_answer_relevance",
            MetricName.SaveFilteredDataset: "filtered_dataset",
        }[self]

    @classmethod
    def from_metric(cls, metric: Metric):
        from fmcore.framework.trainer import RowCount, SaveDatasetOrPredictions
        from fmcore.metric.text_generation_metrics import (
            EntityCount,
            LabelPreservation,
            LabelwiseCosineSimilarity,
            Mauve,
            PairwiseCosineSimilarity,
            SelfBLEU,
            TextGenerationStudent,
            TextLength,
        )

        if isinstance(metric, TextGenerationStudent):
            return cls.from_student_metric(metric)
        elif isinstance(metric, RowCount):
            return cls.RowCount
        elif isinstance(metric, TextLength):
            return cls.TextLength
        elif isinstance(metric, EntityCount):
            return cls.EntityCount
        elif isinstance(metric, SelfBLEU):
            return cls.SelfBLEU
        elif isinstance(metric, LabelwiseCosineSimilarity):
            return cls.LabelwiseCosineSimilarity
        elif isinstance(metric, PairwiseCosineSimilarity):
            return cls.PairwiseCosineSimilarity
        elif isinstance(metric, Mauve):
            return cls.Mauve
        elif isinstance(metric, LabelPreservation):
            return cls.LabelPreservation
        elif isinstance(metric, SaveDatasetOrPredictions):
            return cls.SaveFilteredDataset
        raise not_impl("metric", metric)

    @classmethod
    def from_student_metric(cls, metric: Metric):
        from fmcore.metric.classification_metrics import DatasetCartography
        from fmcore.metric.text_generation_metrics import TextGenerationStudent

        assert isinstance(metric, TextGenerationStudent)
        if str_normalize(metric.params.algorithm) == str_normalize("pytorch"):
            student: Student = Student(
                metric.params.hyperparams["base_model"]["hyperparams"]["model_name"]
            )
        elif str_normalize(metric.params.algorithm) == str_normalize(
            "huggingface-SequenceClassification"
        ):
            student: Student = Student(metric.params.hyperparams["model_name"])
            if student == Student.DistilBERT:
                if metric.params.hyperparams["optimizer"]["lr"] == 2e-5:
                    student: Student = Student.DistilBERT_AttrPromptTable13
        else:
            raise not_impl("metric.params.algorithm", metric.params.algorithm)
        if metric.params.hpo:
            return cls.from_student_hpo(student)
        else:
            student_metrics: Metrics = metric.params.metrics
            assert isinstance(student_metrics, Metrics)
            for _train_student_metric in student_metrics[DataSplit.TRAIN]:
                if isinstance(_train_student_metric, DatasetCartography):
                    return cls.StudentDatasetCartography
            return cls.from_student(student)

    # @safe_validate_arguments
    def get_metric(
        self,
        *,
        results_dir: FileMetadata,
        expt: Experiment,
        text_gens: TextGenerationsPredictionsBase,
        dataset_name: DatasetName,
        model_name: ModelName,
        references_col: str,
        label_col: str,
        verbosity: int,
        label_preservation_student: Student,
        dataset_cartography_student: Student,
        dataset_cartography_text_col: str,
        val_set: Optional[ClassificationData],
        student_text_col: str,
        student_hpo_validation_set: Literal["seed", "train_set", "val_set"],
        entity_count_num_cpus: int = 16,
        self_bleu_num_cpus: int = 36,
        mauve_num_cpus: int = 12,
        label_cosine_sim_num_cpus: int = 8,
    ) -> Metric:
        if self.is_student():
            return self.get_student_metric(
                text_gens=text_gens,
                dataset_name=dataset_name,
                student=self.to_student(),
                train_text_col=student_text_col,
                verbosity=1 if verbosity >= 3 else 0,
            )

        if self is MetricName.StudentDatasetCartography:
            return self.get_student_metric(
                text_gens=text_gens,
                dataset_name=dataset_name,
                student=dataset_cartography_student,
                train_text_col=dataset_cartography_text_col,
                verbosity=1 if verbosity >= 3 else 0,
                student_num_models=3,
            )

        if self.is_student_hpo():
            return self.get_student_hpo_metric(
                text_gens=text_gens,
                val_set=val_set,
                student_hpo_validation_set=student_hpo_validation_set,
                dataset_name=dataset_name,
                student=self.to_student_hpo(),
                verbosity=1 if verbosity >= 3 else 0,
            )

        if self is MetricName.LabelPreservation:
            best_trial_label_preservation_model_dir: FileMetadata = (
                dataset_name.label_preservation_best_trial(
                    results_dir=results_dir,
                    student=label_preservation_student,
                )["best_trial_label_preservation_model_dir"]
            )
            if len(best_trial_label_preservation_model_dir.list()) == 0:
                raise ValueError(
                    f"Could not find label preservation best trial model artifacts for "
                    f'dataset={dataset_name.canonical()} at "{best_trial_label_preservation_model_dir.path}".'
                )
            return Metric.of(
                "LabelPreservation",
                params=dict(
                    batch_size=16,
                    metrics=["Accuracy", "MacroF1", "ConfusionMatrix"],
                    label_col=label_col,
                    text_col=GENERATED_TEXTS_COL,
                    evaluator_params=dict(
                        evaluator="ray",
                        task=SynthesizRRDataset.get(dataset_name.canonical()).task,
                        model_dir=best_trial_label_preservation_model_dir,
                        cache_dir=EFS_HUGGINGFACE_CACHE_DIR,
                        resources_per_model=dict(cpu=1, gpu=1),
                        num_models={
                            50_000: 1,  ## 1 model upto 50k
                            250_000: 2,  ## 2 models between 50k - 250k
                            1_000_000: 3,  ## 3 models between 250k - 1MM
                        }[
                            binary_search(
                                [50_000, 250_000, 1_000_000],
                                target=len(text_gens),
                                return_tuple=True,
                            )[1]
                        ],
                    ),
                    max_retries=1,
                    verbosity=max(0, verbosity - 2),
                ),
            )

        return {
            MetricName.RowCount: Metric.of("RowCount"),
            MetricName.TextLength: Metric.of(
                "TextLength", params=dict(tokenizer=word_tokenize)
            ),
            MetricName.EntityCount: Metric.of(
                "EntityCount",
                params=dict(
                    batch_size=50,
                    num_cpus=entity_count_num_cpus,
                    spacy_ner_model="en_core_web_lg",
                    max_retries=1,
                ),
            ),
            MetricName.SelfBLEU: Metric.of(
                "Self-BLEU",
                params=dict(
                    batch_size=50,
                    num_cpus=self_bleu_num_cpus,
                    spacy_ner_model="en_core_web_lg",
                    max_retries=1,
                ),
            ),
            MetricName.LabelwiseCosineSimilarity: Metric.of(
                "LabelwiseCosineSimilarity",
                params=dict(
                    label_col=label_col,
                    generations_col=GENERATED_TEXTS_COL,
                    num_cpus=label_cosine_sim_num_cpus,
                    hf_embedding_model_name="all-mpnet-base-v2",
                    max_retries=2,
                ),
            ),
            MetricName.PairwiseCosineSimilarity: Metric.of(
                "PairwiseCosineSimilarity",
                params=dict(
                    generations_col=GENERATED_TEXTS_COL,
                    num_cpus=label_cosine_sim_num_cpus,
                    hf_embedding_model_name="all-mpnet-base-v2",
                    max_retries=2,
                ),
            ),
            MetricName.Mauve: Metric.of(
                "Mauve",
                params=dict(
                    references_col=references_col,
                    num_cpus=mauve_num_cpus,
                    max_retries=2,
                ),
            ),
            MetricName.SaveFilteredDataset: Metric.of(
                "SaveDatasetOrPredictions", params=dict(destination=None)
            ),
        }[self]

    def get_student_metric(
        self,
        *,
        text_gens: TextGenerationsPredictionsBase,
        dataset_name: DatasetName,
        train_text_col: str,
        student: Student,
        verbosity: int,
        num_epochs: Optional[int] = None,
        student_num_models: int = 5,
        dataset_metrics: Tuple[str, ...] = (
            "RowCount",
            "Accuracy",
            "MacroF1",
            "ConfusionMatrix",
        ),
    ) -> Metric:
        dataset_metrics: List[str] = as_list(dataset_metrics)
        if self is MetricName.StudentDatasetCartography:
            dataset_metrics.append("DatasetCartography")
        student_metrics = Metrics.of(
            train=dataset_metrics,
            validation=dataset_metrics,
            test=dataset_metrics,
        )
        test_dataset: ClassificationData = (
            SynthesizRRDataset.get(dataset_name.canonical())
            .datasets[DataSplit.TEST]
            .read()
        )
        num_epochs: int = get_default(num_epochs, student.num_epochs())
        student_training_steps: int = math.ceil(
            num_epochs * len(text_gens) / student.batch_size()
        )

        return Metric.of(
            "TextGenerationStudent",
            params=dict(
                test_dataset=test_dataset,
                train_text_col=train_text_col,
                hpo=False,
                algorithm=student.algorithm(),
                hyperparams=student.hyperparams(
                    training_steps=student_training_steps, student_hpo=False
                ),
                eval_steps=math.ceil(
                    len(text_gens) / student.batch_size()
                ),  ## Log once per epoch
                resources_per_model=student.resources_per_model(),
                test_num_models=student_num_models,
                metrics=student_metrics,
                max_retries=1,
                verbosity=verbosity,
            ),
        )

    def get_student_hpo_metric(
        self,
        *,
        text_gens: TextGenerationsPredictionsBase,
        val_set: Optional[ClassificationData],
        student_hpo_validation_set: Literal["seed", "train_set", "val_set"],
        dataset_name: DatasetName,
        student: Student,
        verbosity: int,
        student_num_models: int = 5,
        num_epochs: Optional[int] = None,
        val_frac: float = 0.2,
    ) -> Metric:
        student_metrics = Metrics.of(
            train=["RowCount", "Accuracy", "MacroF1", "ConfusionMatrix"],
            validation=["RowCount", "Accuracy", "MacroF1", "ConfusionMatrix"],
            test=["RowCount", "Accuracy", "MacroF1", "ConfusionMatrix"],
        )
        test_dataset: ClassificationData = (
            SynthesizRRDataset.get(dataset_name.canonical())
            .datasets[DataSplit.TEST]
            .read()
        )
        num_epochs: int = get_default(num_epochs, student.num_epochs())
        student_training_steps: int = math.ceil(
            num_epochs * len(text_gens) / student.hpo_batch_size()
        )

        if student_hpo_validation_set == "seed":
            assert val_set is not None
            validation_dataset: Dataset = val_set
            val_frac: Optional[float] = None
        elif student_hpo_validation_set == "val_set":
            validation_dataset: Optional[Dataset] = (
                SynthesizRRDataset.get(dataset_name.canonical())
                .datasets[DataSplit.VALIDATION]
                .read()
            )
            val_frac: Optional[float] = None
        elif student_hpo_validation_set == "train_set":
            validation_dataset: Optional[Dataset] = None
            assert val_frac is not None
        else:
            raise not_impl("student_hpo_validation_set", student_hpo_validation_set)

        return Metric.of(
            "TextGenerationStudent",
            params=dict(
                test_dataset=test_dataset,
                hpo=True,
                algorithm=student.algorithm(),
                hyperparams=student.hyperparams(
                    training_steps=student_training_steps, student_hpo=True
                ),
                eval_steps=math.ceil(
                    len(text_gens) / student.hpo_batch_size()
                ),  ## Log once per epoch
                resources_per_model=student.resources_per_model(),
                search_algorithm=student.search_algorithm(),
                search_space=student.search_space(student_hpo=True),
                tune_num_models=1 if student.search_algorithm() == "grid" else 12,
                test_num_models=student_num_models,
                validation_dataset=validation_dataset,
                val_frac=val_frac,
                metrics=student_metrics,
                objective_metric="Accuracy",
                objective_type="maximize",
                max_retries=1,
                verbosity=verbosity,
            ),
        )

    def get_label_preservation_student_metric(
        self,
        *,
        dataset_name: DatasetName,
        verbosity: int,
        save_to: Optional[FileMetadata] = None,
        num_final_models: int = 3,
        val_frac: float = 0.2,
        student_training_steps: Optional[int] = None,
        student_eval_steps: Optional[int] = None,
    ) -> Metric:
        student: Student = self.to_student()
        student_metrics = Metrics.of(
            train=["RowCount", "Accuracy", "MacroF1"],
            validation=["RowCount", "Accuracy", "MacroF1"],
            test=["RowCount", "Accuracy", "MacroF1"],
        )

        train_dataset: ClassificationData = (
            SynthesizRRDataset.get(dataset_name.canonical())
            .datasets[DataSplit.TRAIN]
            .read()
        )
        test_dataset: ClassificationData = (
            SynthesizRRDataset.get(dataset_name.canonical())
            .datasets[DataSplit.TEST]
            .read()
        )
        if student_training_steps is None:
            student_training_steps: int = math.ceil(
                student.num_epochs() * len(train_dataset) / student.batch_size()
            )
        if student_eval_steps is None:
            student_eval_steps: int = math.ceil(
                len(train_dataset) / student.batch_size()
            )  ## Log once per epoch

        return Metric.of(
            "TextGenerationStudent",
            params=dict(
                test_dataset=test_dataset,
                hpo=True,
                algorithm=student.algorithm(),
                hyperparams=student.hyperparams(
                    training_steps=student_training_steps, student_hpo=False
                ),
                eval_steps=student_eval_steps,
                resources_per_model=student.resources_per_model(),
                search_algorithm=student.search_algorithm(),
                search_space=student.search_space(student_hpo=False),
                tune_num_models=1 if student.search_algorithm() == "grid" else 12,
                test_num_models=num_final_models,
                val_frac=val_frac,
                metrics=student_metrics,
                objective_metric="Accuracy",
                objective_type="maximize",
                save_to=save_to,
                max_retries=1,
                verbosity=verbosity,
            ),
        )

    def is_student(self) -> bool:
        return self in self.student_metrics()

    @classmethod
    def student_metrics(cls) -> Set:
        return {
            MetricName.StudentTinyBert,
            MetricName.StudentMiniLM,
            MetricName.StudentDistilBERT,
            MetricName.StudentDistilBERT_AttrPromptTable13,
            MetricName.StudentBERT,
            MetricName.StudentDeBERTaV3Base,
            MetricName.StudentDeBERTaV3Large,
        }

    def is_student_hpo(self) -> bool:
        return self in self.student_hpo_metrics()

    @classmethod
    def student_hpo_metrics(cls) -> Set:
        return {
            MetricName.StudentHPOTinyBert,
            MetricName.StudentHPOMiniLM,
            MetricName.StudentHPODistilBERT,
            MetricName.StudentHPODistilBERT_AttrPromptTable13,
            MetricName.StudentHPOBERT,
            MetricName.StudentHPODeBERTaV3Base,
            MetricName.StudentHPODeBERTaV3Large,
        }

    def is_rag(self) -> bool:
        return self in self.rag_metrics()

    @classmethod
    def rag_metrics(cls) -> Set:
        return {
            MetricName.RagasFaithfulness,
            MetricName.RagasContextRelevance,
            MetricName.RagasAnswerRelevance,
        }

    @classmethod
    def non_student_metrics(cls) -> Set:
        return set(list(cls)) - cls.student_metrics()

    def to_student(self) -> Student:
        return METRIC_NAME_TO_STUDENT_MAP[self]

    def to_student_hpo(self) -> Student:
        return HPO_METRIC_NAME_TO_STUDENT_MAP[self]

    @classmethod
    def from_student(cls, student: Student):
        return STUDENT_TO_METRIC_NAME_MAP[student]

    @classmethod
    def from_student_hpo(cls, student: Student):
        return STUDENT_HPO_TO_METRIC_NAME_MAP[student]


STUDENT_TO_METRIC_NAME_MAP: Dict[Student, MetricName] = {
    Student.TinyBert: MetricName.StudentTinyBert,
    Student.MiniLM: MetricName.StudentMiniLM,
    Student.DistilBERT: MetricName.StudentDistilBERT,
    Student.DistilBERT_AttrPromptTable13: MetricName.StudentDistilBERT_AttrPromptTable13,
    Student.BERT: MetricName.StudentBERT,
    Student.DeBERTaV3Base: MetricName.StudentDeBERTaV3Base,
    Student.DeBERTaV3Large: MetricName.StudentDeBERTaV3Large,
}
METRIC_NAME_TO_STUDENT_MAP: Dict[MetricName, Student] = {
    v: k for k, v in STUDENT_TO_METRIC_NAME_MAP.items()
}

STUDENT_HPO_TO_METRIC_NAME_MAP: Dict[Student, MetricName] = {
    Student.TinyBert: MetricName.StudentHPOTinyBert,
    Student.MiniLM: MetricName.StudentHPOMiniLM,
    Student.DistilBERT: MetricName.StudentHPODistilBERT,
    Student.DistilBERT_AttrPromptTable13: MetricName.StudentHPODistilBERT_AttrPromptTable13,
    Student.BERT: MetricName.StudentHPOBERT,
    Student.DeBERTaV3Base: MetricName.StudentHPODeBERTaV3Base,
    Student.DeBERTaV3Large: MetricName.StudentHPODeBERTaV3Large,
}
HPO_METRIC_NAME_TO_STUDENT_MAP: Dict[MetricName, Student] = {
    v: k for k, v in STUDENT_HPO_TO_METRIC_NAME_MAP.items()
}


def parse_text_gen_clean(x: Optional[str]):
    if x is None:
        x = ""
    x = x.strip()
    if x.startswith('"') and x.endswith('"'):
        x = x.removeprefix('"').removesuffix('"')
    if x.startswith("'") and x.endswith("'"):
        x = x.removeprefix("'").removesuffix("'")
    return x.strip()


def parse_text_gens_rejection(
    gens_text: str,
    *,
    dataset_name: DatasetName,
    expt: Experiment,
    model_name: ModelName,
    split: Optional[str] = None,
    rejection_command: Optional[str] = None,
) -> Generator[str, None, None]:
    split: str = get_default(
        split,
        get_split_prefix(dataset_name=dataset_name, expt=expt, model_name=model_name),
    )
    if rejection_command is not None and str_normalize(
        rejection_command
    ) in str_normalize(gens_text):
        return []
    for gen_text in gens_text.split(split):
        gen_text = parse_text_gen_clean(gen_text)
        if len(gen_text) > 0:
            yield gen_text


def get_split_prefix(
    *, dataset_name: DatasetName, expt: Experiment, model_name: ModelName
) -> str:
    replacement: Tuple[str, str] = ICL_AND_PROMPT_TEMPLATE_DICT[(dataset_name, expt)][
        "claude_replacements"
    ][1]
    assert replacement[0] in {
        "News Article:",
        "Rewritten Article:",
        "Summary:",
        "Review:",
        "Product Question:",
        "Headline:",
    }
    if model_name.is_claude():
        return replacement[1]
    return replacement[0]


def rag_hyperpartisan_news_parse_text_gens_rejection(
    *args, **kwargs
) -> Generator[str, None, None]:
    kwargs["dataset_name"] = DatasetName.HyperpartisanNews
    kwargs["expt"] = Experiment.SynthesizRR
    return parse_text_gens_rejection(
        *args,
        rejection_command="UNABLE TO REWRITE",
        **kwargs,
    )


def fewgen_hyperpartisan_news_parse_text_gens_rejection(
    *args, **kwargs
) -> Generator[str, None, None]:
    kwargs["dataset_name"] = DatasetName.HyperpartisanNews
    kwargs["expt"] = Experiment.FewGen
    return parse_text_gens_rejection(
        *args,
        **kwargs,
    )


def rag_toi_headlines_parse_text_gens_rejection(
    *args, **kwargs
) -> Generator[str, None, None]:
    kwargs["dataset_name"] = DatasetName.ToiHeadlines
    kwargs["expt"] = Experiment.SynthesizRR
    return parse_text_gens_rejection(
        *args,
        rejection_command="UNABLE TO WRITE HEADLINE",
        **kwargs,
    )


def fewgen_toi_headlines_parse_text_gens_rejection(
    *args, **kwargs
) -> Generator[str, None, None]:
    kwargs["dataset_name"] = DatasetName.ToiHeadlines
    kwargs["expt"] = Experiment.FewGen
    return parse_text_gens_rejection(
        *args,
        **kwargs,
    )


def rag_ag_news_parse_text_gens_rejection(
    *args, **kwargs
) -> Generator[str, None, None]:
    kwargs["dataset_name"] = DatasetName.AgNews
    kwargs["expt"] = Experiment.SynthesizRR
    return parse_text_gens_rejection(
        *args,
        rejection_command="UNABLE TO SUMMARIZE",
        **kwargs,
    )


def fewgen_ag_news_parse_text_gens_rejection(
    *args, **kwargs
) -> Generator[str, None, None]:
    kwargs["dataset_name"] = DatasetName.AgNews
    kwargs["expt"] = Experiment.FewGen
    return parse_text_gens_rejection(
        *args,
        **kwargs,
    )


def rag_amazon_reviews_category_parse_text_gens_rejection(
    *args, **kwargs
) -> Generator[str, None, None]:
    kwargs["dataset_name"] = DatasetName.AmazonReviewsProductCategory
    kwargs["expt"] = Experiment.SynthesizRR
    return parse_text_gens_rejection(
        *args,
        rejection_command="UNABLE TO REVIEW",
        **kwargs,
    )


def fewgen_amazon_reviews_category_parse_text_gens_rejection(
    *args, **kwargs
) -> Generator[str, None, None]:
    kwargs["dataset_name"] = DatasetName.AmazonReviewsProductCategory
    kwargs["expt"] = Experiment.FewGen
    return parse_text_gens_rejection(
        *args,
        **kwargs,
    )


def rag_amazon_humor_parse_text_gens_rejection(
    *args, **kwargs
) -> Generator[str, None, None]:
    kwargs["dataset_name"] = DatasetName.AmazonHumorousProductQuestions
    kwargs["expt"] = Experiment.SynthesizRR
    return parse_text_gens_rejection(
        *args,
        rejection_command="UNABLE TO ASK QUESTION",
        **kwargs,
    )


def fewgen_amazon_humor_parse_text_gens_rejection(
    *args, **kwargs
) -> Generator[str, None, None]:
    kwargs["dataset_name"] = DatasetName.AmazonHumorousProductQuestions
    kwargs["expt"] = Experiment.FewGen
    return parse_text_gens_rejection(
        *args,
        **kwargs,
    )


def rag_amazon_polarity_parse_text_gens_rejection(
    *args, **kwargs
) -> Generator[str, None, None]:
    kwargs["dataset_name"] = DatasetName.AmazonReviewsPolarity
    kwargs["expt"] = Experiment.SynthesizRR
    return parse_text_gens_rejection(
        *args,
        rejection_command="UNABLE TO REVIEW",
        **kwargs,
    )


def fewgen_amazon_polarity_parse_text_gens_rejection(
    *args, **kwargs
) -> Generator[str, None, None]:
    kwargs["dataset_name"] = DatasetName.AmazonReviewsPolarity
    kwargs["expt"] = Experiment.FewGen
    return parse_text_gens_rejection(
        *args,
        **kwargs,
    )


TEXT_GENS_PARSERS_REJECTION: Dict[Tuple[DatasetName, Experiment], Callable] = {
    (
        DatasetName.HyperpartisanNews,
        Experiment.SynthesizRR,
    ): rag_hyperpartisan_news_parse_text_gens_rejection,
    (
        DatasetName.HyperpartisanNews,
        Experiment.FewGen,
    ): fewgen_hyperpartisan_news_parse_text_gens_rejection,
    (
        DatasetName.ToiHeadlines,
        Experiment.SynthesizRR,
    ): rag_toi_headlines_parse_text_gens_rejection,
    (
        DatasetName.ToiHeadlines,
        Experiment.FewGen,
    ): fewgen_toi_headlines_parse_text_gens_rejection,
    (DatasetName.AgNews, Experiment.SynthesizRR): rag_ag_news_parse_text_gens_rejection,
    (DatasetName.AgNews, Experiment.FewGen): fewgen_ag_news_parse_text_gens_rejection,
    (
        DatasetName.AmazonReviewsProductCategory,
        Experiment.SynthesizRR,
    ): rag_amazon_reviews_category_parse_text_gens_rejection,
    (
        DatasetName.AmazonReviewsProductCategory,
        Experiment.FewGen,
    ): fewgen_amazon_reviews_category_parse_text_gens_rejection,
    (
        DatasetName.AmazonHumorousProductQuestions,
        Experiment.SynthesizRR,
    ): rag_amazon_humor_parse_text_gens_rejection,
    (
        DatasetName.AmazonHumorousProductQuestions,
        Experiment.FewGen,
    ): fewgen_amazon_humor_parse_text_gens_rejection,
    (
        DatasetName.AmazonReviewsPolarity,
        Experiment.SynthesizRR,
    ): rag_amazon_polarity_parse_text_gens_rejection,
    (
        DatasetName.AmazonReviewsPolarity,
        Experiment.FewGen,
    ): fewgen_amazon_polarity_parse_text_gens_rejection,
}


def parse_amazon_polarity_gen_text(
    gen_text: str,
    *,
    model_name: ModelName,
) -> Generator[str, None, None]:
    gen_text = parse_text_gen_clean(gen_text)
    if len(gen_text) > 0:
        if model_name.is_claude():
            for claude_gen_example in gen_text.split("\n"):
                claude_gen_example = claude_gen_example.strip()
                if (
                    claude_gen_example.lower().startswith("here is a")
                    or claude_gen_example.lower().startswith("i apologize")
                    or claude_gen_example.lower().startswith("thank you")
                ):
                    continue
                claude_gen_example = parse_text_gen_clean(claude_gen_example)
                if len(claude_gen_example) > 0:
                    yield claude_gen_example
        else:
            gen_text = gen_text.split("Product details:")[0].strip()
            gen_text = parse_text_gen_clean(gen_text)
            if len(gen_text) > 0:
                yield gen_text


def parse_amazon_humor_gen_text(
    gen_text: str,
    *,
    model_name: ModelName,
) -> Generator[str, None, None]:
    gen_text = parse_text_gen_clean(gen_text)
    if len(gen_text) > 0:
        yield gen_text


def parse_amazon_reviews_category_gen_text(
    gen_text: str,
    *,
    model_name: ModelName,
) -> Generator[str, None, None]:
    gen_text = parse_text_gen_clean(gen_text)
    if len(gen_text) > 0:
        if model_name.is_claude():
            for claude_gen_example in gen_text.split("\n"):
                claude_gen_example = claude_gen_example.strip()
                if (
                    claude_gen_example.lower().startswith("here is a")
                    or claude_gen_example.lower().startswith("i apologize")
                    or claude_gen_example.lower().startswith("thank you")
                ):
                    continue
                claude_gen_example = parse_text_gen_clean(claude_gen_example)
                if len(claude_gen_example) > 0:
                    yield claude_gen_example
        else:
            gen_text = gen_text.split("Product details:")[0].strip()
            gen_text = parse_text_gen_clean(gen_text)
            if len(gen_text) > 0:
                yield gen_text


def parse_ag_news_gen_text(
    gen_text: str,
    *,
    model_name: ModelName,
    invalid: Tuple[str, ...] = (
        ## Weird crap generated by LLaMa during 0-shot:
        "Sentence 1:",
        "Sentence 1:",
        "Sentence 2:",
        "Sentence 3:",
    ),
) -> List[str]:
    if "\nSentences:" in gen_text:
        gen_text: str = gen_text.split("\nSentences:")[0].strip()
    elif "\nSentence 1:" in gen_text:
        gen_text: str = gen_text.split("\nSentence 1:")[0].strip()
    ag_news_regex = '(Article:)?( )*"(.+)"($|\n)'
    gen_examples = [x[2] for x in re.findall(ag_news_regex, gen_text)]
    if len(gen_examples) > 0:
        for gen_example in gen_examples:
            if gen_example in invalid:
                continue
            yield gen_example
    else:
        num_examples_this_generated_text: int = 0
        for gen_example in gen_text.strip().split("\n"):
            gen_example: str = gen_example.strip()
            gen_example: str = (
                gen_example.removeprefix("Article:").removeprefix("article:").strip()
            )
            gen_example: str = gen_example.removeprefix('"').removeprefix("'").strip()
            if gen_example.endswith('"') or gen_example.endswith("'"):
                gen_example: str = gen_example.removesuffix('"').removesuffix("'")
            elif num_examples_this_generated_text > 0:
                ## If it does not end with " and we already have some examples, it is a cutoff.
                gen_example: Optional[str] = None
            if gen_example is not None and gen_example.strip() != "":
                yield gen_example


def parse_hyperpartisan_news_gen_text(
    gen_text: str,
    *,
    model_name: ModelName,
    invalid: Tuple[str, ...] = (
        ## Weird crap generated by LLaMa during 0-shot:
        "Sentence 1:",
        "Sentence 1:",
        "Sentence 2:",
        "Sentence 3:",
    ),
) -> List[str]:
    gen_text: str = gen_text.split("\nArticle:")[0].strip()
    if len(gen_text) > 0:
        yield gen_text


def parse_toi_headlines_gen_text(
    gen_text: str,
    *,
    model_name: ModelName,
    invalid: Tuple[str, ...] = (
        ## Weird crap generated by LLaMa during 0-shot:
        "Sentence 1:",
        "Sentence 1:",
        "Sentence 2:",
        "Sentence 3:",
    ),
) -> List[str]:
    invalid: Set[str] = set(invalid)
    if model_name.is_claude():
        toi_claude_gen: str = gen_text
        has_ended_in_period = None
        added_examples = False
        for toi_claude_gen_example in toi_claude_gen.split("\n"):
            toi_claude_gen_example = toi_claude_gen_example.strip()
            if (
                toi_claude_gen_example.lower().startswith("here are")
                or toi_claude_gen_example.lower().startswith("i apologize")
                or toi_claude_gen_example.lower().startswith("thank you")
            ):
                continue
            if toi_claude_gen_example.startswith(
                '"'
            ) and not toi_claude_gen_example.endswith('"'):
                continue
            if toi_claude_gen_example.startswith(
                "'"
            ) and not toi_claude_gen_example.endswith("'"):
                continue
            for prefix in (
                [f"{x}." for x in [1, 2, 3, 4, 5]]
                + [f"{x})" for x in [1, 2, 3, 4, 5]]
                + [f"({x})" for x in [1, 2, 3, 4, 5]]
                + ["-", '"', "'"]
            ):
                toi_claude_gen_example = toi_claude_gen_example.removeprefix(
                    prefix
                ).strip()
            for suffix in ['"', "'"]:
                toi_claude_gen_example = toi_claude_gen_example.removesuffix(
                    suffix
                ).strip()
            toi_claude_gen_example = toi_claude_gen_example.strip()
            if len(toi_claude_gen_example) == 0:
                continue
            if has_ended_in_period is True and not toi_claude_gen_example.endswith("."):
                continue

            if toi_claude_gen_example.endswith("."):
                has_ended_in_period = True
            added_examples = True
            yield toi_claude_gen_example
        # if added_examples is False:
        #     print(f'\n\nNo examples for text:\n{toi_claude_gen}')
    else:
        if "\nSentences:" in gen_text:
            gen_text: str = gen_text.split("\nSentences:")[0].strip()
        elif "\nSentence 1:" in gen_text:
            gen_text: str = gen_text.split("\nSentence 1:")[0].strip()
        if isinstance(gen_text, str) and len(gen_text.strip()) == 0:
            return []
        toi_headlines_regex = '(Article:)?(Headline:)?( )*"(.+)"($|\n)'
        gen_examples = [x[3] for x in re.findall(toi_headlines_regex, gen_text)]
        if len(gen_examples) > 0:
            for gen_example in gen_examples:
                if gen_example in invalid:
                    continue
                yield gen_example
        else:
            num_examples_this_generated_text: int = 0
            for gen_example in gen_text.strip().split("\n"):
                gen_example: str = gen_example.strip()
                gen_example: str = (
                    gen_example.removeprefix("Article:")
                    .removeprefix("article:")
                    .strip()
                )
                gen_example: str = (
                    gen_example.removeprefix("Headline:")
                    .removeprefix("headline:")
                    .strip()
                )
                if len(gen_example) == 0:
                    continue
                if gen_example.startswith('"') and not gen_example.endswith('"'):
                    continue
                if gen_example.startswith("'") and not gen_example.endswith("'"):
                    continue
                gen_example: str = (
                    gen_example.removeprefix('"').removeprefix("'").strip()
                )
                if gen_example.endswith('"') or gen_example.endswith("'"):
                    gen_example: str = gen_example.removesuffix('"').removesuffix("'")
                elif num_examples_this_generated_text > 0:
                    ## If it does not end with " and we already have some examples, it is a cutoff.
                    gen_example: Optional[str] = None
                if gen_example is not None and gen_example.strip() != "":
                    yield gen_example


GEN_PARSERS: Dict[str, Callable] = {
    DatasetName.AgNews: parse_ag_news_gen_text,
    DatasetName.HyperpartisanNews: parse_hyperpartisan_news_gen_text,
    DatasetName.ToiHeadlines: parse_toi_headlines_gen_text,
    DatasetName.AmazonReviewsPolarity: parse_amazon_polarity_gen_text,
    DatasetName.AmazonHumorousProductQuestions: parse_amazon_humor_gen_text,
    DatasetName.AmazonReviewsProductCategory: parse_amazon_reviews_category_gen_text,
}


def calc_label_dist(data_df: pd.DataFrame, *, label_col: str) -> pd.DataFrame:
    label_dist: pd.Series = data_df[label_col].value_counts()
    label_dist.name = "counts"
    label_dist.index.name = label_col
    label_dist: pd.DataFrame = label_dist.reset_index(drop=True).reset_index()
    label_dist["pct"] = (label_dist["counts"] / label_dist["counts"].sum()).apply(
        lambda x: f"{100 * x:.2f}%"
    )
    return label_dist


@safe_validate_arguments
def check_gen_parser(
    text_gens: TextGenerationsPredictionsBase,
    *,
    text_gens_parser: Callable,
    model_name: ModelName,
    n: int = 10,
    prompts: bool = True,
):
    from IPython.display import display
    from termcolor import cprint

    gens_parsed = []
    for gen, prompt, label_text in ProgressBar.iter(
        zip(
            text_gens.data["generations"],
            text_gens.data["prompts"],
            text_gens.data["label_text"],
        ),
        miniters=1000,
    ):
        for gen_parsed in text_gens_parser(gen, model_name=model_name):
            gens_parsed.append(
                dict(
                    gen=gen,
                    gen_parsed=gen_parsed,
                    prompts=prompt,
                    label_text=label_text,
                )
            )
    gens_parsed = pd.DataFrame(gens_parsed)
    print("Label distribution (after expansion):")
    display(calc_label_dist(gens_parsed, label_col="label_text"))

    i = 0
    for gen, gen_df in shuffle_items(list(gens_parsed.groupby("gen"))):
        print("▔" * 80)
        if prompts:
            prompt: str = only_item(gen_df["prompts"].unique())
            cprint(prompt, color="light_grey")
            print()

        cprint(
            "[" + only_item(gen_df["label_text"].unique()) + "]", color="red", end=""
        )
        cprint(gen, color="black")
        print()
        colors = ["blue", "green", "red", "magenta", "cyan"]
        colors_iter = iter(colors)
        for gen_parsed in gen_df["gen_parsed"]:
            cprint(gen_parsed, color=next(colors_iter))
        print("▔" * 80)

        i += 1
        if i > n:
            break


class LabelPreservationModel(CachedResultsStep):
    @safe_validate_arguments
    def run(
        self,
        *,
        results_dir: FileMetadata,
        dataset_name: DatasetName,
        label_preservation_student: Student,
        student_training_steps: Optional[conint(ge=1)] = None,
        student_eval_steps: Optional[conint(ge=1)] = None,
        **kwargs,
    ) -> Dict:
        label_preservation_model_dir: FileMetadata = self.save_to(
            results_dir=results_dir,
            dataset_name=dataset_name,
            student=label_preservation_student,
        )
        if len(label_preservation_model_dir.list()) == 0:
            self.info(
                f'No models found at: "{label_preservation_model_dir.path}", training...'
            )
            label_preservation_student_metric_name: MetricName = (
                MetricName.from_student(label_preservation_student)
            )
            label_preservation_student_metric: Metric = label_preservation_student_metric_name.get_label_preservation_student_metric(
                dataset_name=dataset_name,
                save_to=label_preservation_model_dir,
                student_training_steps=student_training_steps,
                student_eval_steps=student_eval_steps,
                verbosity=1 if self.verbosity >= 3 else 0,
            )
            train_dataset: ClassificationData = (
                SynthesizRRDataset.get(dataset_name.canonical())
                .datasets[DataSplit.TRAIN]
                .read()
            )

            label_preservation_student_metric: Metric = (
                label_preservation_student_metric.evaluate(
                    train_dataset,
                    rolling=False,
                    inplace=False,
                )
            )
            (
                trialwise_final_model_metrics,
                detailed_final_model_metrics,
                tune_metrics,
            ) = label_preservation_student_metric.value
            self._write_tuning_metrics(
                dataset_name=dataset_name,
                label_preservation_model_dir=label_preservation_model_dir,
                trialwise_final_model_metrics=trialwise_final_model_metrics,
                tune_metrics=tune_metrics,
            )
        return dict(
            label_preservation_model_dir=label_preservation_model_dir,
            **self._load_tuning_metrics(
                dataset_name=dataset_name,
                label_preservation_model_dir=label_preservation_model_dir,
            ),
        )

    def save_to(
        self,
        *,
        results_dir: FileMetadata,
        dataset_name: DatasetName,
        student: Student,
        **kwargs,
    ) -> FileMetadata:
        return dataset_name.label_preservation_model_dir(
            results_dir=results_dir,
            student=student,
        )

    def _write_tuning_metrics(
        self,
        *,
        dataset_name: DatasetName,
        label_preservation_model_dir: FileMetadata,
        trialwise_final_model_metrics: Dict[str, Metrics],
        tune_metrics: pd.DataFrame,
    ):
        self.info(
            f"Writing tuning metrics for dataset_name={dataset_name.canonical()}..."
        )

        trialwise_final_model_metrics_file: FileMetadata = (
            dataset_name.trialwise_final_model_metrics_file(
                label_preservation_model_dir
            )
        )
        self.info(
            f'>> Writing trialwise_final_model_metrics to "{trialwise_final_model_metrics_file.path}"...'
        )
        Writer.of(FileFormat.PICKLE).write(
            trialwise_final_model_metrics_file,
            data=trialwise_final_model_metrics,
            overwrite=True,
        )
        self.info(
            f'...wrote trialwise_final_model_metrics to "{trialwise_final_model_metrics_file.path}".'
        )

        tune_metrics_file: FileMetadata = dataset_name.tune_metrics_file(
            label_preservation_model_dir
        )
        self.info(f'>> Writing tune_metrics to "{tune_metrics_file.path}"...')
        Writer.of(FileFormat.PICKLE).write(
            tune_metrics_file, data=tune_metrics, overwrite=True
        )
        self.info(f'...wrote tune_metrics to "{tune_metrics_file.path}".')

        self.info(
            f"...done writing tuning metrics for dataset_name={dataset_name.canonical()}."
        )

    def _load_tuning_metrics(
        self,
        *,
        dataset_name: DatasetName,
        label_preservation_model_dir: FileMetadata,
    ) -> Dict:
        self.info(
            f"Loading tuning metrics for dataset_name={dataset_name.canonical()}..."
        )

        trialwise_final_model_metrics_file: FileMetadata = (
            dataset_name.trialwise_final_model_metrics_file(
                label_preservation_model_dir
            )
        )
        trialwise_final_model_metrics: Dict[str, Metrics] = Reader.of(
            FileFormat.PICKLE
        ).read(trialwise_final_model_metrics_file)
        self.info(
            f'>> Loaded trialwise_final_model_metrics from "{trialwise_final_model_metrics_file.path}".'
        )

        tune_metrics_file: FileMetadata = dataset_name.tune_metrics_file(
            label_preservation_model_dir
        )
        tune_metrics: pd.DataFrame = Reader.of(FileFormat.PICKLE).read(
            tune_metrics_file
        )
        self.info(f'>> Loaded tune_metrics from "{tune_metrics_file.path}".')

        self.info(
            f"...done loading tuning metrics for dataset_name={dataset_name.canonical()}."
        )
        return dict(
            trialwise_final_model_metrics=trialwise_final_model_metrics,
            tune_metrics=tune_metrics,
        )


def get_templates_and_hashes(
    *,
    expt: Experiment,
    dataset_name: DatasetName,
    model_name: ModelName,
    icl_template: Optional[str],
    prompt_template: Optional[str],
) -> Tuple[str, Optional[str], str, Optional[str]]:
    icl_and_prompt_template: Dict[str, str] = dataset_name.icl_and_prompt_template(
        expt=expt,
        model_name=model_name,
    )
    if icl_template is not None:
        icl_template_hash: str = StringUtil.hash(
            punct_normalize(icl_template), max_len=6
        )
    else:
        icl_template: str = icl_and_prompt_template["icl_template"]
        icl_template_hash: Optional[str] = (
            None  ## Picking the default, do not add the hash.
        )
    if prompt_template is not None:
        prompt_template_hash: str = StringUtil.hash(
            punct_normalize(prompt_template), max_len=6
        )
    else:
        prompt_template: str = icl_and_prompt_template["prompt_template"]
        prompt_template_hash: Optional[str] = (
            None  ## Picking the default, do not add the hash.
        )
    return icl_template, icl_template_hash, prompt_template, prompt_template_hash


class DatasetFilterParams(Parameters):
    filter_type: Literal["none", "cartography"]
    cartography_apply: Literal["overall", "label"] = "label"
    cartography_confidence_range: Optional[
        Tuple[confloat(ge=0.0, le=1.0), confloat(ge=0.0, le=1.0)]
    ] = None
    cartography_confidence_frac: Optional[
        Tuple[Literal["top", "bottom"], confloat(ge=0.0, le=1.0)]
    ] = None

    cartography_variability_range: Optional[
        Tuple[confloat(ge=0.0, le=1.0), confloat(ge=0.0, le=1.0)]
    ] = None
    cartography_variability_frac: Optional[
        Tuple[Literal["top", "bottom"], confloat(ge=0.0, le=1.0)]
    ] = None

    cartography_correctness_range: Optional[
        Tuple[confloat(ge=0.0, le=1.0), confloat(ge=0.0, le=1.0)]
    ] = None
    cartography_correctness_frac: Optional[
        Tuple[Literal["top", "bottom"], confloat(ge=0.0, le=1.0)]
    ] = None

    @classmethod
    @root_validator(pre=False)
    def check_params(cls, params: Dict) -> Dict:
        if params["filter_type"] == "cartography":
            if all_are_none(
                params["cartography_confidence_range"],
                params["cartography_confidence_frac"],
                params["cartography_variability_range"],
                params["cartography_variability_frac"],
                params["cartography_correctness_range"],
                params["cartography_correctness_frac"],
            ):
                raise ValueError(
                    "At least one of the dataset cartography filters must be specified."
                )
            if multiple_are_not_none(
                params["cartography_confidence_range"],
                params["cartography_confidence_frac"],
            ):
                raise ValueError(
                    "At most one of the confidence filters must be specified."
                )
            if multiple_are_not_none(
                params["cartography_variability_range"],
                params["cartography_variability_frac"],
            ):
                raise ValueError(
                    "At most one of the variability filters must be specified."
                )
            if multiple_are_not_none(
                params["cartography_correctness_range"],
                params["cartography_correctness_frac"],
            ):
                raise ValueError(
                    "At most one of the correctness filters must be specified."
                )

    cartography_confidence_frac: Optional[
        Tuple[Literal["top", "bottom"], confloat(ge=0.0, le=1.0)]
    ] = None

    cartography_variability_range: Optional[
        Tuple[confloat(ge=0.0, le=1.0), confloat(ge=0.0, le=1.0)]
    ] = None
    cartography_variability_frac: Optional[
        Tuple[Literal["top", "bottom"], confloat(ge=0.0, le=1.0)]
    ] = None

    cartography_correctness_range: Optional[
        Tuple[confloat(ge=0.0, le=1.0), confloat(ge=0.0, le=1.0)]
    ] = None
    cartography_correctness_frac: Optional[
        Tuple[Literal["top", "bottom"], confloat(ge=0.0, le=1.0)]
    ] = None

    @classmethod
    @root_validator(pre=False)
    def check_params(cls, params: Dict) -> Dict:
        if params["filter_type"] == "cartography":
            if all_are_none(
                params["cartography_confidence_range"],
                params["cartography_confidence_frac"],
                params["cartography_variability_range"],
                params["cartography_variability_frac"],
                params["cartography_correctness_range"],
                params["cartography_correctness_frac"],
            ):
                raise ValueError(
                    "At least one of the dataset cartography filters must be specified."
                )
            if multiple_are_not_none(
                params["cartography_confidence_range"],
                params["cartography_confidence_frac"],
            ):
                raise ValueError(
                    "At most one of the confidence filters must be specified."
                )
            if multiple_are_not_none(
                params["cartography_variability_range"],
                params["cartography_variability_frac"],
            ):
                raise ValueError(
                    "At most one of the variability filters must be specified."
                )
            if multiple_are_not_none(
                params["cartography_correctness_range"],
                params["cartography_correctness_frac"],
            ):
                raise ValueError(
                    "At most one of the correctness filters must be specified."
                )
        return params

    def save_key(
        self,
        *,
        dataset_cartography_student: Student,
        dataset_cartography_text_col: str,
    ) -> str:
        filter_params: Dict = remove_nulls(self.dict())
        filter_params["dataset_cartography_student"] = (
            dataset_cartography_student.canonical()
        )
        filter_params["dataset_cartography_text_col"] = dataset_cartography_text_col
        return StringUtil.stringify(filter_params)

    def save_key_and_hash(self, **kwargs) -> Tuple[str, save_key]:
        save_key: str = self.save_key(**kwargs)
        return save_key, StringUtil.hash(save_key, max_len=6)
