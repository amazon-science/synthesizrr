import gc
import json
from abc import ABC
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from bears.util import (
    StringUtil,
    all_are_false,
    get_default,
    get_fn_spec,
    not_impl,
    remove_keys,
    safe_validate_arguments,
    sample_idxs_match_distribution,
)
from bears.util.concurrency import (
    ThreadPoolExecutor,
    accumulate,
    get_result,
    retry,
    run_concurrent,
    stop_executor,
)
from fmcore.constants import (
    DataLayout,
    DataSplit,
    FileFormat,
    MLType,
    Parallelize,
)
from fmcore.data import (
    FileMetadata,
    Reader,
    Writer,
)
from fmcore.framework._metric import Metric, Metrics
from fmcore.framework._predictions import load_predictions, save_predictions
from fmcore.framework._task.classification import ClassificationData
from fmcore.framework._task.text_generation import (
    GENERATED_TEXTS_COL,
    TextGenerationsPredictionsBase,
    TextInputs,
)
from fmcore.framework._dataset import Dataset
from fmcore.framework._trainer.RayTuneTrainer import (
    _RAY_TRAINING_ITERATION,
    _RAY_TRIAL_ID,
)
from fmcore.metric.classification_metrics import DatasetCartography
from fmcore.metric.text_generation_metrics import TextGenerationStudent, _clf_dataset_to_text_gens
from pydantic import ValidationError, confloat, conint, constr

from synthesizrr.common import (
    DEFAULT_SEED,
    DEFAULT_SEED_SET_DATA_SPLIT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    IDX_COL,
    LABEL_OVERALL,
    RETRIEVED_CONTEXT_COL,
    TEXT_GEN_REFERENCES_COL,
    CachedResultsStep,
    Corpus,
    DatasetFilterParams,
    DatasetName,
    Experiment,
    MetricName,
    ModelName,
    Retriever,
    Student,
    calc_label_dist,
    expand_num_samples_per_label,
    get_templates_and_hashes,
)
from synthesizrr.data import SynthesizRRDataset


class BaseCalculateTextGenMetrics(CachedResultsStep, ABC):
    def should_run_metric(
        self,
        *,
        metric_name: MetricName,
        metric_file: FileMetadata,
    ) -> bool:
        if not metric_file.exists():
            return True
        if metric_name is MetricName.StudentDatasetCartography:
            cart_metric: Metric = retry(
                Reader.of(FileFormat.PICKLE).read,
                metric_file,
                retries=10,
                wait=10,
            )
            if not isinstance(cart_metric, TextGenerationStudent):
                return True
            try:
                _, detailed_final_model_metrics = cart_metric.value
                DatasetCartography.calc_data_map(detailed_final_model_metrics, index_col=IDX_COL)
                return False
            except Exception:  ## Failed to calculate data map, rerun
                return True
        return False

    def expand_text_gens(
        self,
        text_gens: TextGenerationsPredictionsBase,
        *,
        num_shots: int,
        text_gens_parser: Callable,
        model_name: ModelName,
        dataset_name: DatasetName,
        label_verbalizer: Dict[str, str],
        label_col: str,
        references_col: str,
        references_data_split: DataSplit,
        seed: int,
    ) -> TextGenerationsPredictionsBase:
        assert text_gens_parser is not None
        text_gens_data: List[Dict] = []
        index_col: str = text_gens.data_schema.index_col
        num_ex: int = 0
        for d in text_gens.data.to_list_of_dict():
            for text_gen in text_gens_parser(d[GENERATED_TEXTS_COL], model_name=model_name):
                if isinstance(text_gen, str) and len(text_gen.strip()) > 0:
                    text_gens_data.append(
                        {
                            **remove_keys(d, [index_col, GENERATED_TEXTS_COL]),
                            index_col: f"{d[index_col]}###{num_ex}",
                            GENERATED_TEXTS_COL: text_gen,
                        }
                    )
                    num_ex += 1
        text_gens_expanded: TextGenerationsPredictionsBase = text_gens.update_params(data=text_gens_data)

        ## Add references column:
        text_gens_expanded_df: pd.DataFrame = text_gens_expanded.data.pandas()
        ref_dataset: ClassificationData = (
            SynthesizRRDataset.get(dataset_name.canonical()).datasets[references_data_split].read()
        )
        ref_dataset_df: pd.DataFrame = ref_dataset.data.pandas()
        ref_dataset_text_col: str = dataset_name.text_col()

        text_gens_expanded_with_refs_df: List[pd.DataFrame] = []
        for label_text in label_verbalizer.keys():
            labelwise_ref_dataset_df: pd.DataFrame = ref_dataset_df.query(
                f'{label_col} == "{label_text}"'
            ).reset_index(drop=True)
            labelwise_text_gens_expanded_df: pd.DataFrame = text_gens_expanded_df.query(
                f'{label_col} == "{label_text}"'
            ).reset_index(drop=True)
            labelwise_ref_text: List[str] = list(
                np.random.RandomState(seed).permutation(
                    labelwise_ref_dataset_df[ref_dataset_text_col].tolist()
                )
            )
            labelwise_text_gens_expanded_refs: List[str] = []
            while len(labelwise_text_gens_expanded_refs) < len(labelwise_text_gens_expanded_df):
                labelwise_text_gens_expanded_refs.extend(labelwise_ref_text)
            labelwise_text_gens_expanded_df[references_col]: pd.Series = pd.Series(
                labelwise_text_gens_expanded_refs[: len(labelwise_text_gens_expanded_df)]
            )
            text_gens_expanded_with_refs_df.append(labelwise_text_gens_expanded_df)
        text_gens_expanded_with_refs_df: pd.DataFrame = pd.concat(
            text_gens_expanded_with_refs_df,
            axis=0,
        ).reset_index(drop=True)
        text_gens_expanded_with_refs: TextGenerationsPredictionsBase = text_gens_expanded.update_params(
            data=text_gens_expanded_with_refs_df
        )
        text_gens_expanded_with_refs.data_schema.features_schema[references_col] = MLType.TEXT
        assert isinstance(text_gens_expanded_with_refs, TextGenerationsPredictionsBase)

        self.info(
            f"Expanded text generations for "
            f"{self._combo_str(dataset_name, model_name, num_shots, label_text=None)}: "
            f"[{len(text_gens)} => {len(text_gens_expanded_with_refs)}] "
            f"{len(text_gens_expanded_with_refs) / len(text_gens):.2f} examples/gen."
        )
        self.info(
            f"Expanded text generations has following number of examples per class:"
            f"\n{json.dumps(text_gens_expanded_with_refs.data.pandas()[label_col].value_counts().to_dict())}"
        )
        return text_gens_expanded_with_refs

    # @safe_validate_arguments
    def filter_text_gens(
        self,
        text_gens: TextGenerationsPredictionsBase,
        *,
        filter_text_gens_results_dir: FileMetadata,  ## Dir for storing and caching values.
        filter_params: DatasetFilterParams,
        dataset_cartography_student: Student,
        dataset_cartography_text_col: str,
        num_shots: int,
        model_name: ModelName,
        dataset_name: DatasetName,
        label_verbalizer: Dict[str, str],
        label_col: str,
        seed: int,
        verbosity: int,
    ) -> TextGenerationsPredictionsBase:
        if filter_params.filter_type == "none":
            self.info("No filter applied to text generations.")
            return text_gens
        filter_str, filter_hash = filter_params.save_key_and_hash(
            dataset_cartography_student=dataset_cartography_student,
            dataset_cartography_text_col=dataset_cartography_text_col,
        )
        self.info(f"Filtering {len(text_gens)} text generations according to params: {filter_str}")
        filtered_text_gens_file: FileMetadata = filter_text_gens_results_dir.file_in_dir(
            f"text_gens-f={filter_hash}.parquet",
            return_metadata=True,
        ).update_params(file_format=FileFormat.PARQUET)
        if not filtered_text_gens_file.exists():
            if filter_params.filter_type == "cartography":
                cartography_metric = MetricName.StudentDatasetCartography.get_student_metric(
                    text_gens=text_gens,
                    dataset_name=dataset_name,
                    student=dataset_cartography_student,
                    train_text_col=dataset_cartography_text_col,
                    verbosity=1 if verbosity >= 3 else 0,
                    student_num_models=1,
                )
                self.info(f">> Running dataset cartography on {len(text_gens)} generations.")
                cartography_metric: Metric = cartography_metric.evaluate(text_gens, inplace=False)
                cartography_metric_file: FileMetadata = filter_text_gens_results_dir.file_in_dir(
                    f"cartography_metric-f={filter_hash}.pkl",
                    return_metadata=True,
                ).update_params(file_format=FileFormat.PICKLE)
                Writer.of(FileFormat.PICKLE).write(
                    cartography_metric_file,
                    data=cartography_metric,
                    overwrite=True,
                )
                self.info(f'>> Wrote dataset cartography metric to: "{cartography_metric_file.path}"')
                _, detailed_final_model_metrics = cartography_metric.value
                trial_cart_metrics: List[Dict] = []
                trial_id: str = ""
                for trial_id, trial_df in detailed_final_model_metrics.groupby(_RAY_TRIAL_ID):
                    if trial_df[_RAY_TRAINING_ITERATION].nunique() != 6:
                        continue
                    trial_df: pd.DataFrame = trial_df.sort_values(
                        _RAY_TRAINING_ITERATION, ascending=True
                    ).reset_index(drop=True)
                    trial_cart_df: List[pd.DataFrame] = []
                    for training_iteration, iter_data_map_df in (
                        trial_df.set_index(_RAY_TRAINING_ITERATION)["Train/DatasetCartography"]
                        .to_dict()
                        .items()
                    ):
                        iter_data_map_df[_RAY_TRAINING_ITERATION] = training_iteration
                        trial_cart_df.append(iter_data_map_df)
                    trial_cart_df: pd.DataFrame = (
                        pd.concat(trial_cart_df)
                        .sort_values(
                            [IDX_COL, _RAY_TRAINING_ITERATION],
                            ascending=True,
                        )
                        .reset_index(drop=True)
                    )
                    for idx, idx_df in trial_cart_df.groupby(IDX_COL):
                        trial_cart_metrics.append(
                            {
                                IDX_COL: idx,
                                "confidence": float(idx_df["gold_label_prob"].mean()),
                                "variability": float(
                                    idx_df["gold_label_prob"].std(ddof=0)
                                ),  ## Biased std for some reason?
                                "correctness": float(idx_df["predicted_label_matches_gold"].mean()),
                            }
                        )
                    break  ## Only take first trial
                trial_cart_metrics: pd.DataFrame = pd.DataFrame(trial_cart_metrics)
                if len(trial_cart_metrics) != len(text_gens):
                    raise ValueError(
                        f"Expected {len(text_gens)} examples (unique indexes) in dataset cartography, "
                        f"but found {len(trial_cart_metrics)}"
                    )
                index_col: str = text_gens.data_schema.index_col
                trial_cart_metrics["correctness_bucket"] = trial_cart_metrics["correctness"].apply(
                    lambda x: round(x, 1)
                )
                trial_cart_metrics[label_col] = trial_cart_metrics[IDX_COL].map(
                    text_gens.data.pandas().set_index(index_col)[label_col].to_dict()
                )

                def get_range_filter(col_range: Tuple[float, float], col: str) -> str:
                    low, high = col_range
                    return f"""{low} <= {col} <= {high}"""

                def get_frac_filter(
                    cart_df: pd.DataFrame,
                    col_frac: Tuple[Literal["top", "bottom"], float],
                    col: str,
                ) -> str:
                    crit, frac = col_frac
                    if crit == "top":
                        quant_val: float = cart_df[
                            col
                        ].quantile(
                            1
                            - frac  ## 0.8 becomes 0.2, so we filter by those above 20th quantile value, i.e top-80%
                        )
                        return f"""{quant_val} <= {col}"""
                    elif crit == "bottom":
                        quant_val: float = cart_df[col].quantile(
                            frac  ## 0.8, so we filter by those below 80th quantile value, i.e bottom-80%
                        )
                        return f"""{col} <= {quant_val}"""
                    else:
                        raise not_impl("crit", crit)

                def filter_cart_df(cart_df: pd.DataFrame) -> pd.DataFrame:
                    df_filters: List[str] = []
                    ## Filter by confidence:
                    if filter_params.cartography_confidence_range is not None:
                        df_filters.append(
                            get_range_filter(filter_params.cartography_confidence_range, "confidence")
                        )
                    elif filter_params.cartography_confidence_frac is not None:
                        df_filters.append(
                            get_frac_filter(
                                cart_df,
                                filter_params.cartography_confidence_frac,
                                "confidence",
                            )
                        )
                    ## Filter by variability:
                    if filter_params.cartography_variability_range is not None:
                        df_filters.append(
                            get_range_filter(
                                filter_params.cartography_variability_range,
                                "variability",
                            )
                        )
                    elif filter_params.cartography_variability_frac is not None:
                        df_filters.append(
                            get_frac_filter(
                                cart_df,
                                filter_params.cartography_variability_frac,
                                "variability",
                            )
                        )
                    ## Filter by correctness:
                    if filter_params.cartography_correctness_range is not None:
                        df_filters.append(
                            get_range_filter(
                                filter_params.cartography_correctness_range,
                                "correctness",
                            )
                        )
                    elif filter_params.cartography_correctness_frac is not None:
                        df_filters.append(
                            get_frac_filter(
                                cart_df,
                                filter_params.cartography_correctness_frac,
                                "correctness",
                            )
                        )
                    ## Apply filters:
                    assert len(df_filters) > 0
                    return cart_df.query(" and ".join(df_filters)).reset_index(drop=True)

                if filter_params.cartography_apply == "overall":
                    trial_cart_metrics_filtered: pd.DataFrame = filter_cart_df(trial_cart_metrics)
                elif filter_params.cartography_apply == "label":
                    trial_cart_metrics_filtered: pd.DataFrame = pd.concat(
                        [
                            filter_cart_df(labelwise_trial_cart_metrics)
                            for label_text, labelwise_trial_cart_metrics in trial_cart_metrics.groupby(
                                label_col
                            )
                        ]
                    ).reset_index(drop=True)
                else:
                    raise not_impl(
                        "filter_params.cartography_apply",
                        filter_params.cartography_apply,
                    )
                trial_cartography_file: FileMetadata = filter_text_gens_results_dir.file_in_dir(
                    f"dataset_cartography-trial_id={trial_id}-f={filter_hash}.parquet",
                    return_metadata=True,
                ).update_params(file_format=FileFormat.PARQUET)
                Writer.of(FileFormat.PARQUET).write(
                    trial_cartography_file,
                    data=trial_cart_metrics_filtered,
                    overwrite=True,
                )
                self.info(f'>> Wrote trial-run cartography to: "{trial_cartography_file.path}"')
                ## Filter by cartography
                filtered_idxs: Set[str] = set(trial_cart_metrics_filtered[IDX_COL].unique())
                filtered_text_gens: TextGenerationsPredictionsBase = text_gens.filter(
                    lambda row: row[index_col] in filtered_idxs
                )
            else:
                raise not_impl("dataset_filter_params.filter_type", filter_params.filter_type)
            self.info(
                f">> Filtered from {len(text_gens)} => {len(filtered_text_gens)} generations for ("
                f"dataset={dataset_name.canonical()}, "
                f"model_name={model_name.canonical()}, "
                f"num_shots={num_shots}, "
                f"filter={filter_str}"
                f")."
            )

            self.info(
                f">> Saving {len(filtered_text_gens)} filtered generations for ("
                f"dataset={dataset_name.canonical()}, "
                f"model_name={model_name.canonical()}, "
                f"num_shots={num_shots}, "
                f"filter={filter_str}"
                f') to "{filtered_text_gens_file.path}".'
            )
            save_predictions(
                predictions=filtered_text_gens,
                predictions_destination=filtered_text_gens_file,
                overwrite=True,
            )
        self.info(f">> Loading filtered text generations for num_shots={num_shots}...")
        filtered_text_gens: TextGenerationsPredictionsBase = load_predictions(
            filtered_text_gens_file, retry=10, retry_wait=10
        )
        self.info(
            f">> Loaded {len(filtered_text_gens)} filtered generations ("
            f"dataset={dataset_name.canonical()}, "
            f"model_name={model_name.canonical()}, "
            f"num_shots={num_shots}, "
            f"filter={filter_str}"
            f') from "{filtered_text_gens_file.path}".'
            f"\nLabel-distribution:"
        )
        self.info(calc_label_dist(filtered_text_gens.data.pandas(), label_col=label_col))
        return filtered_text_gens

    @classmethod
    @safe_validate_arguments
    def sample_dataset(
        cls,
        dataset_name: DatasetName,
        *,
        data_split: DataSplit,
        n_sample: Optional[int] = None,
        seed: int = 42,
        stratify_on_ground_truth: bool = True,
        filter_label: Optional[str] = None,
    ) -> ClassificationData:
        from sklearn.model_selection import train_test_split

        dataset: ClassificationData = (
            SynthesizRRDataset.get(dataset_name.canonical())
            .datasets[data_split]
            .read(read_as=DataLayout.PANDAS)
        )
        label_col: str = dataset.ground_truth_label_col_name
        ## Filter before sampling:
        if filter_label not in {LABEL_OVERALL, None}:
            dataset: ClassificationData = dataset.filter(lambda row: row[label_col] == filter_label)
            if len(dataset) == 0:
                raise ValueError(
                    f'Empty dataset after filtering "{dataset_name}" by {label_col}="{filter_label}"'
                )
        dataset_df: pd.DataFrame = dataset.data.pandas()
        if n_sample is not None and len(dataset_df) > n_sample:
            if stratify_on_ground_truth:
                _, dataset_df = train_test_split(
                    dataset_df,
                    test_size=n_sample,
                    random_state=seed,
                    stratify=dataset_df[label_col],
                )
            else:
                _, dataset_df = train_test_split(
                    dataset_df,
                    test_size=n_sample,
                    random_state=seed,
                )
        return dataset.update_params(data=dataset_df)

    # @safe_validate_arguments
    def match_dataset_distribution(
        self,
        *,
        text_gens: TextGenerationsPredictionsBase,
        label_col: str,
        dataset_name: DatasetName,
        model_name: Optional[ModelName],
        num_shots: Optional[int],
        metrics_num_samples_per_label: Optional[int],
        metrics_label_distribution: Literal["balanced", "train_set"],
        metrics_override_row_count: bool,
        label_verbalizer: Dict[str, str],
        label_text: Optional[str],
        seed: int,
        tol: float = 20 / 1000,  ## X% above or below is fine
    ) -> TextGenerationsPredictionsBase:
        ## If label_text is *not* Overall, just sample.
        ## Otherwise, we must match the label-distributions.
        ## Step 1: pick the *Train* dataset (we don't know the test dataset).
        ## Step 2: match the distribution of labels in the generated texts to the Train dataset distribution.
        ## Step 3: select indexes from generated texts.

        if label_text == LABEL_OVERALL:
            ## If label is Overall, sample:
            if metrics_label_distribution == "train_set":
                dataset_data_split: DataSplit = DataSplit.TRAIN  ## Match the train dataset distribution.
                dataset: ClassificationData = (
                    SynthesizRRDataset.get(dataset_name.canonical())
                    .datasets[dataset_data_split]
                    .read(retry=10, retry_wait=10)
                    .to_layout(DataLayout.PANDAS)
                )
                dataset_label_col: str = dataset.ground_truth_label_col_name
                text_gens: TextGenerationsPredictionsBase = text_gens.to_layout(DataLayout.PANDAS)
                text_gens_idxs: np.ndarray = sample_idxs_match_distribution(
                    source=text_gens.data.pandas()[label_col],
                    target=dataset.data.pandas()[dataset_label_col],  ## Train set distribution
                    n=metrics_num_samples_per_label,
                    seed=seed,
                )
                text_gens: TextGenerationsPredictionsBase = text_gens.update_params(
                    data=text_gens.data.iloc[text_gens_idxs]
                )
            elif metrics_label_distribution == "balanced":
                text_gens_idxs: np.ndarray = sample_idxs_match_distribution(
                    source=text_gens.data.pandas()[label_col],
                    target=pd.Series([lb for lb in label_verbalizer.keys()]),  ## Balanced
                    n=metrics_num_samples_per_label,
                    seed=seed,
                )
                text_gens: TextGenerationsPredictionsBase = text_gens.update_params(
                    data=text_gens.data.iloc[text_gens_idxs]
                )
            else:
                raise not_impl("metrics_label_distribution", metrics_label_distribution)
            filtered_sampled_text_gens: TextGenerationsPredictionsBase = text_gens
        else:
            ## If label is not Overall, only sample:
            text_gens: TextGenerationsPredictionsBase = text_gens.filter(
                lambda row: row[label_col] == label_text
            )
            if len(text_gens) == 0:
                raise ValueError(f'Empty Text Generations after filtering by {label_col}="{label_text}"')
            if (
                isinstance(metrics_num_samples_per_label, int)
                and len(text_gens) > metrics_num_samples_per_label
            ):
                text_gens: TextGenerationsPredictionsBase = text_gens.sample(
                    n=metrics_num_samples_per_label,
                    seed=seed,
                )
            filtered_sampled_text_gens: TextGenerationsPredictionsBase = text_gens

        self.info(
            f"After matching {len(text_gens)} generations with ("
            f"metrics_num_samples_per_label={metrics_num_samples_per_label}, "
            f"metrics_label_distribution={metrics_label_distribution}"
            f") label distribution for {self._combo_str(dataset_name, model_name, num_shots, label_text)} is:"
        )
        self.info(calc_label_dist(filtered_sampled_text_gens.data.pandas(), label_col=label_col))
        if metrics_num_samples_per_label is not None:
            tol_num_rows: int = max(int(metrics_num_samples_per_label * tol), 1)
            rows_diff: int = abs(len(filtered_sampled_text_gens) - metrics_num_samples_per_label)
            if rows_diff > tol_num_rows and not metrics_override_row_count:
                raise ValueError(
                    f"Expected text gens for label {label_text} to be between ["
                    f"{metrics_num_samples_per_label - tol_num_rows}, "
                    f"{metrics_num_samples_per_label + tol_num_rows}"
                    f"] rows after sampling, but found {len(filtered_sampled_text_gens)} rows."
                )
        filtered_sampled_text_gens: TextGenerationsPredictionsBase = filtered_sampled_text_gens.update_params(
            data_split=DataSplit.TRAIN
        )
        return filtered_sampled_text_gens

    @classmethod
    def dataset_labels(
        cls,
        dataset_name: DatasetName,
        *,
        seed_set_data_split: DataSplit,
        label_col: str,
    ) -> List[str]:
        dataset: Dataset = (
            SynthesizRRDataset.get(dataset_name.canonical())
            .datasets[seed_set_data_split]
            .read(read_as=DataLayout.PANDAS)
        )
        return sorted(dataset.data[label_col].unique())

    def _labels_to_evaluate(
        self,
        *,
        metrics_labels_to_evaluate: Optional[List[str]],
        metrics_calc_overall: bool,
        metrics_calc_labels: bool,
        label_verbalizer: Dict[str, str],
    ) -> List[str]:
        if all_are_false(metrics_calc_overall, metrics_calc_labels):
            raise ValueError(
                "Should specify either `metrics_calc_overall=True` or `metrics_calc_labels=True` or both."
            )
        if metrics_labels_to_evaluate is None:
            metrics_labels_to_evaluate: List[str] = []
            if metrics_calc_overall:
                metrics_labels_to_evaluate.append(LABEL_OVERALL)
            if metrics_calc_labels:
                metrics_labels_to_evaluate.extend(list(label_verbalizer.keys()))
        assert len(metrics_labels_to_evaluate) > 0
        self.info(f"Evaluating labels: {metrics_labels_to_evaluate}")
        return metrics_labels_to_evaluate

    # @safe_validate_arguments
    def _run_metrics_calculation(
        self,
        *,
        results_dir: FileMetadata,
        expt: Experiment,
        metrics_files: Dict[Optional[int], Dict[str, Dict[MetricName, FileMetadata]]],
        text_gens: Dict[Optional[int], TextGenerationsPredictionsBase],
        model_name: Optional[ModelName],
        dataset_name: DatasetName,
        references_col: str,
        label_col: str,
        label_verbalizer: Dict[str, str],
        metrics_num_samples_per_label: Optional[Dict[str, int]],
        metrics_label_distribution: Literal["balanced", "train_set"],
        metrics_override_row_count: bool,
        val_set: Optional[ClassificationData],
        student_text_col: str,
        student_hpo_validation_set: Literal["seed", "train_set", "val_set"],
        label_preservation_student: Student,
        dataset_cartography_student: Student,
        dataset_cartography_text_col: str,
        metrics_parallelize: Parallelize,
        metrics_max_parallel: Optional[conint(ge=1)],
        seed: int,
    ) -> Dict[Optional[int], Dict[str, Dict[MetricName, Metric]]]:
        num_combos: int = sum(
            [
                1  ## Create a thread for every combination of num_shots x label
                for num_shots in metrics_files.keys()
                for label_text in metrics_files[num_shots].keys()
            ]
        )
        metrics_max_parallel: int = min(get_default(metrics_max_parallel, num_combos), num_combos)
        executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=metrics_max_parallel)
        try:
            self.info(
                f"Evaluating metrics for {num_combos} combinations of "
                f"{self._combo_str(dataset_name, model_name, num_shots=None, label_text=None)}..."
            )
            futs: Dict[Optional[int], Dict[str, Any]] = {}
            for num_shots in metrics_files.keys():
                futs.setdefault(num_shots, {})
                for label_text in metrics_files[num_shots].keys():
                    futs[num_shots][label_text] = run_concurrent(
                        self._run_num_shots_labelwise_metrics,
                        results_dir=results_dir,
                        expt=expt,
                        text_gens=text_gens[num_shots],
                        num_shots_labelwise_metric_files=metrics_files[num_shots][label_text],
                        model_name=model_name,
                        dataset_name=dataset_name,
                        num_shots=num_shots,
                        label_text=label_text,
                        references_col=references_col,
                        label_col=label_col,
                        label_preservation_student=label_preservation_student,
                        dataset_cartography_student=dataset_cartography_student,
                        dataset_cartography_text_col=dataset_cartography_text_col,
                        metrics_num_samples_per_label=get_default(metrics_num_samples_per_label, {}).get(
                            label_text
                        ),
                        metrics_label_distribution=metrics_label_distribution,
                        metrics_override_row_count=metrics_override_row_count,
                        val_set=val_set,
                        student_text_col=student_text_col,
                        student_hpo_validation_set=student_hpo_validation_set,
                        metrics_parallelize=metrics_parallelize,
                        label_verbalizer=label_verbalizer,
                        seed=seed,
                        executor=executor,
                    )
            evaluated_metrics: Dict[Optional[int], Dict[str, Dict[MetricName, Metric]]] = {}
            for num_shots in futs.keys():
                evaluated_metrics.setdefault(num_shots, {})
                for label_text in futs[num_shots].keys():
                    evaluated_metrics[num_shots][label_text] = get_result(futs[num_shots][label_text])
            self.info(f"...done evaluating metrics for {num_combos} combinations.")
        finally:
            stop_executor(executor)
            gc.collect()
        return evaluated_metrics

    @safe_validate_arguments
    def _run_num_shots_labelwise_metrics(
        self,
        *,
        results_dir: FileMetadata,
        expt: Experiment,
        text_gens: TextGenerationsPredictionsBase,
        num_shots_labelwise_metric_files: Dict[MetricName, FileMetadata],
        dataset_name: DatasetName,
        model_name: Optional[ModelName],
        num_shots: Optional[int],
        label_text: str,
        metrics_num_samples_per_label: Optional[int],
        metrics_label_distribution: Literal["balanced", "train_set"],
        metrics_override_row_count: bool,
        val_set: Optional[ClassificationData],
        student_text_col: str,
        student_hpo_validation_set: Literal["seed", "train_set", "val_set"],
        label_verbalizer: Dict[str, str],
        references_col: str,
        label_col: str,
        label_preservation_student: Student,
        dataset_cartography_student: Student,
        dataset_cartography_text_col: str,
        metrics_parallelize: Parallelize,
        seed: int,
    ) -> Dict[MetricName, Metric]:
        filtered_sampled_text_gens: TextGenerationsPredictionsBase = self.match_dataset_distribution(
            text_gens=text_gens,
            label_col=label_col,
            dataset_name=dataset_name,
            model_name=model_name,
            num_shots=num_shots,
            metrics_num_samples_per_label=metrics_num_samples_per_label,
            metrics_label_distribution=metrics_label_distribution,
            metrics_override_row_count=metrics_override_row_count,
            label_text=label_text,
            label_verbalizer=label_verbalizer,
            seed=seed,
        )
        num_shots_labelwise_metrics: Metrics = Metrics.of(
            **{
                str(filtered_sampled_text_gens.data_split).lower(): [
                    metric_name.get_metric(
                        results_dir=results_dir,
                        expt=expt,
                        text_gens=filtered_sampled_text_gens,
                        dataset_name=dataset_name,
                        model_name=model_name,
                        references_col=references_col,
                        label_col=label_col,
                        label_preservation_student=label_preservation_student,
                        dataset_cartography_student=dataset_cartography_student,
                        dataset_cartography_text_col=dataset_cartography_text_col,
                        val_set=val_set,
                        student_text_col=student_text_col,
                        student_hpo_validation_set=student_hpo_validation_set,
                        verbosity=self.verbosity,
                    )
                    for metric_name in num_shots_labelwise_metric_files.keys()
                ]
            }
        )
        num_shots_labelwise_evaluated_metrics_list: Generator = num_shots_labelwise_metrics.evaluate(
            filtered_sampled_text_gens,
            data_split=filtered_sampled_text_gens.data_split,
            inplace=False,
            allow_partial_metrics=True,
            parallelize=metrics_parallelize,
            progress_bar=dict(
                desc=f"Metrics for {self._combo_str(dataset_name, model_name, num_shots, label_text)}",
            )
            if self.verbosity >= 2
            else False,
            iter=True,
        )
        evaluated_metrics: Dict[MetricName, Metric] = {}
        for (
            evaluated_metric_i,
            evaluated_metric,
        ) in num_shots_labelwise_evaluated_metrics_list:
            evaluated_metric_name: MetricName = MetricName.from_metric(evaluated_metric)
            metric_file: FileMetadata = num_shots_labelwise_metric_files[evaluated_metric_name]
            self.info(
                f'Writing metric "{evaluated_metric_name.canonical()}" for '
                f"{self._combo_str(dataset_name, model_name, num_shots, label_text)} "
                f'to "{metric_file.path}"...'
            )
            Writer.of(FileFormat.PICKLE).write(metric_file, data=evaluated_metric, overwrite=True)
            self.info(
                f'...wrote metric "{evaluated_metric_name.canonical()}" for '
                f"{self._combo_str(dataset_name, model_name, num_shots, label_text)}"
            )
            evaluated_metrics[evaluated_metric_name] = evaluated_metric
        return evaluated_metrics

    def _load_metrics(
        self,
        *,
        results_dir: FileMetadata,
        expt: Experiment,
        metrics_files: Dict[Optional[int], Dict[str, Dict[MetricName, FileMetadata]]],
        dataset_name: DatasetName,
        model_name: Optional[ModelName],
        label_preservation_student: Student,
    ) -> Dict[Optional[int], Dict[str, Dict[MetricName, Metric]]]:
        text_gens_metrics: Dict[Optional[int], Dict[str, Dict[MetricName, Metric]]] = {}
        for num_shots in metrics_files.keys():
            text_gens_metrics.setdefault(num_shots, {})
            self.info(f"\nLoading metrics for text generation for num_shots={num_shots}...")
            for label_text in metrics_files[num_shots].keys():
                text_gens_metrics[num_shots].setdefault(label_text, {})
                for metric_name, metric_file in metrics_files[num_shots][label_text].items():
                    if expt is Experiment.Gold and metric_name is MetricName.LabelPreservation:
                        best_trial_metrics: Metrics = dataset_name.label_preservation_best_trial(
                            results_dir=results_dir,
                            student=label_preservation_student,
                        )["best_trial_metrics"]
                        text_gens_metrics[num_shots][label_text][
                            metric_name
                        ]: Metric = best_trial_metrics.find(
                            "Accuracy",
                            data_split=DataSplit.VALIDATION,
                        )
                    else:
                        self.info(
                            f'>> Loading metric "{metric_name.canonical()}" for '
                            f"{self._combo_str(dataset_name, model_name, num_shots, label_text)} "
                            f'from "{metric_file.path}"...'
                        )
                        text_gens_metrics[num_shots][label_text][metric_name]: Metric = retry(
                            Reader.of(FileFormat.PICKLE).read,
                            metric_file,
                            retries=10,
                            wait=10,
                        )
            self.info(f"\n...loading metrics for text generation for num_shots={num_shots}.")
        return text_gens_metrics

    @safe_validate_arguments
    def _combo_str(
        self,
        dataset_name: DatasetName,
        model_name: Optional[ModelName],
        num_shots: Optional[conint(ge=0)],
        label_text: Optional[constr(min_length=1)],
    ) -> str:
        out = "("
        out += f"dataset={dataset_name.canonical()}, "
        if model_name is not None:
            out += f"model_name={model_name.canonical()}, "
        if num_shots is not None:
            out += f"num_shots={num_shots}, "
        if label_text is not None:
            out += f"label={label_text}"
        out: str = out.strip().removesuffix(",")
        out += ")"
        return out


class GoldDatasetMetrics(BaseCalculateTextGenMetrics):
    @safe_validate_arguments
    def run(
        self,
        *,
        results_dir: FileMetadata,
        dataset_name: DatasetName,
        label_verbalizer: Dict[str, str],
        metrics_to_evaluate: List[MetricName],
        metrics_label_distribution: Literal["balanced", "train_set"],
        metrics_num_samples_per_label: Union[Dict[str, conint(ge=1)], conint(ge=1)],
        metrics_calc_overall: bool = True,
        metrics_calc_labels: bool = False,
        metrics_labels_to_evaluate: Optional[List[str]] = None,
        metrics_parallelize: Parallelize = Parallelize.ray,
        metrics_max_parallel: Optional[conint(ge=1)] = None,
        metrics_override_row_count: bool = False,
        student_hpo_validation_set: Literal["train_set", "val_set"] = "val_set",
        label_preservation_student: Student = Student.DeBERTaV3Large,
        dataset_cartography_student: Student = Student.DeBERTaV3Large,
        seed: int = DEFAULT_SEED,
        text_col: Optional[str] = None,
        label_col: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        text_col: str = get_default(text_col, dataset_name.text_col())
        label_col: str = get_default(label_col, dataset_name.label_col())
        references_data_split: DataSplit = DataSplit.TEST
        references_col: str = TEXT_GEN_REFERENCES_COL(data_split=references_data_split)
        disallowed_metrics: List[MetricName] = [
            MetricName.Mauve,
        ]
        dataset_cartography_text_col: str = text_col
        for disallowed_metric in disallowed_metrics:
            if disallowed_metric in metrics_to_evaluate:
                self.warning(
                    f'Metric "{disallowed_metric}" is not allowed for {self.class_name}; it will be ignored.'
                )
        metrics_to_evaluate: List[MetricName] = [
            metric_to_evaluate
            for metric_to_evaluate in metrics_to_evaluate
            if metric_to_evaluate not in disallowed_metrics
        ]

        metrics_num_samples_per_label: Dict[str, int] = expand_num_samples_per_label(
            num_samples_per_label=metrics_num_samples_per_label,
            label_verbalizer={**label_verbalizer, LABEL_OVERALL: LABEL_OVERALL},
        )
        metrics_labels_to_evaluate: List[str] = self._labels_to_evaluate(
            metrics_labels_to_evaluate=metrics_labels_to_evaluate,
            metrics_calc_overall=metrics_calc_overall,
            metrics_calc_labels=metrics_calc_labels,
            label_verbalizer=label_verbalizer,
        )
        metrics_files: Dict[str, Dict[MetricName, FileMetadata]] = {}
        missing_metrics_files: Dict[str, Dict[MetricName, FileMetadata]] = {}
        for label_text in metrics_labels_to_evaluate:
            metrics_files.setdefault(label_text, {})
            for metric_name in metrics_to_evaluate:
                metrics_files[label_text][metric_name]: FileMetadata = self.save_to(
                    results_dir=results_dir,
                    dataset_name=dataset_name,
                    label_text=label_text,
                    metric_name=metric_name,
                    metrics_num_samples_per_label=metrics_num_samples_per_label[label_text],
                    metrics_label_distribution=metrics_label_distribution,
                    student_hpo_validation_set=student_hpo_validation_set,
                    dataset_cartography_student=dataset_cartography_student,
                    dataset_cartography_text_col=dataset_cartography_text_col,
                    seed=seed,
                )
                if self.should_run_metric(
                    metric_name=metric_name,
                    metric_file=metrics_files[label_text][metric_name],
                ):
                    missing_metrics_files.setdefault(label_text, {})
                    missing_metrics_files[label_text][metric_name]: FileMetadata = metrics_files[label_text][
                        metric_name
                    ]
                else:
                    self.info(
                        f'>> Metric "{metric_name.canonical()}" already exists for '
                        f"{self._combo_str(dataset_name, model_name=None, num_shots=None, label_text=label_text)}"
                        f'at "{metrics_files[label_text][metric_name].path}"...'
                    )
        gold_dataset: ClassificationData = SynthesizRRDataset.get(dataset_name.canonical()).train.read(
            retry=10, retry_wait=10
        )
        gold_dataset_text_gens: TextGenerationsPredictionsBase = _clf_dataset_to_text_gens(
            data=gold_dataset,
            data_split=DataSplit.TRAIN,
            text_col=text_col,
            label_col=label_col,
        )
        if len(missing_metrics_files) > 0:
            evaluated_metrics: Dict[Optional[int], Dict[str, Dict[MetricName, Metric]]] = (
                self._run_metrics_calculation(
                    results_dir=results_dir,
                    expt=Experiment.Gold,
                    metrics_files={None: missing_metrics_files},
                    text_gens={None: gold_dataset_text_gens},
                    model_name=None,
                    dataset_name=dataset_name,
                    label_col=label_col,
                    references_col=references_col,
                    label_verbalizer=label_verbalizer,
                    seed=seed,
                    metrics_num_samples_per_label=metrics_num_samples_per_label,
                    metrics_label_distribution=metrics_label_distribution,
                    metrics_override_row_count=metrics_override_row_count,
                    val_set=None,
                    student_text_col=text_col,
                    student_hpo_validation_set=student_hpo_validation_set,
                    label_preservation_student=label_preservation_student,
                    dataset_cartography_student=dataset_cartography_student,
                    dataset_cartography_text_col=dataset_cartography_text_col,
                    metrics_parallelize=metrics_parallelize,
                    metrics_max_parallel=metrics_max_parallel,
                )
            )
        gold_dataset_text_gens_metrics: Dict[Optional[int], Dict[str, Dict[MetricName, Metric]]] = (
            self._load_metrics(
                results_dir=results_dir,
                expt=Experiment.Gold,
                metrics_files={None: metrics_files},
                dataset_name=dataset_name,
                model_name=None,
                label_preservation_student=label_preservation_student,
            )
        )
        return dict(
            gold_dataset=gold_dataset,
            gold_dataset_text_gens=gold_dataset_text_gens,
            gold_metrics_files=metrics_files,
            gold_dataset_text_gens_metrics=gold_dataset_text_gens_metrics[None],
        )

    def save_to(
        self,
        results_dir: FileMetadata,  ## E.g. RESULTS_DIR/,
        dataset_name: DatasetName,  ## E.g. ag_news
        label_text: str,
        metric_name: MetricName,
        metrics_num_samples_per_label: Optional[conint(ge=1)],
        metrics_label_distribution: Literal["balanced", "train_set"],
        student_hpo_validation_set: Literal["train_set", "val_set"],
        dataset_cartography_student: Student,
        dataset_cartography_text_col: str,
        seed: int,
        **kwargs,
    ) -> FileMetadata:
        ## RESULTS_DIR/retrieval-augmented-dataset-generation/ag_news/retrieval-data/toi_without_datasets/ag_news_seed_retr_output_toi_without_datasets.jsonlines
        metrics_num_samples_per_label_str: str = get_default(metrics_num_samples_per_label, "all")

        student_hpo_validation_set_str: str = ""
        if metric_name.is_student_hpo():
            student_hpo_validation_set_str: str = f"-hpo_set={student_hpo_validation_set}"

        dataset_cartography_str: str = ""
        if metric_name is MetricName.StudentDatasetCartography:
            dataset_cartography_str: str = (
                f"-cart={dataset_cartography_student.canonical()}_col={dataset_cartography_text_col}"
            )

        return (
            results_dir.subdir_in_dir("gold-dataset", return_metadata=True)
            .subdir_in_dir(dataset_name.canonical(), return_metadata=True)
            .subdir_in_dir("gold-dataset-metrics", return_metadata=True)
            .subdir_in_dir(f"label={label_text}", return_metadata=True)
            .subdir_in_dir(
                f"metrics_label_distribution={metrics_label_distribution}",
                return_metadata=True,
            )
            .subdir_in_dir(
                f"metrics_num_samples_per_label={metrics_num_samples_per_label_str}",
                return_metadata=True,
            )
            .subdir_in_dir(f"metric_name={metric_name.canonical()}", return_metadata=True)
            .file_in_dir(
                f"metric"
                f"-metrics_label_distribution={metrics_label_distribution}"
                f"-metrics_num_samples_per_label={metrics_num_samples_per_label}"
                f"-metric_name={metric_name.canonical()}"
                f"-dataset={dataset_name.canonical()}"
                f"-label={label_text}"
                f"{student_hpo_validation_set_str}"
                f"{dataset_cartography_str}"
                f"-seed={seed}"
                f".pkl",
                return_metadata=True,
            )
            .update_params(file_format=FileFormat.PICKLE)
        )


class SynthesizRRTextGenMetrics(BaseCalculateTextGenMetrics):
    @safe_validate_arguments
    def run(
        self,
        *,
        results_dir: FileMetadata,
        text_gens: Dict[int, TextGenerationsPredictionsBase],
        dataset_name: DatasetName,
        icl_dataset: ClassificationData,
        text_input_dataset: TextInputs,
        corpus: Corpus,
        retriever: Retriever,
        model_name: ModelName,
        num_shots_list: List[conint(ge=0)],
        seed_set: ClassificationData,
        seed_type: Literal["generated", "train_set"],
        seed_generation_params_hash: Optional[str],
        icl_type: Literal["retrieved", "curated", "seed"],
        retr_icl_top_ks: List[conint(ge=1)],
        retr_icl_distance_range: Tuple[float, float],
        retr_icl_token_range: Tuple[conint(ge=1), conint(ge=1)],
        synthesizrr_top_k_range: range,
        synthesizrr_distance_range: Tuple[float, float],  ## (0.4, 0.9)
        synthesizrr_max_tokens: conint(ge=1),
        num_samples_per_label: Optional[Union[Dict[str, conint(ge=1)], conint(ge=1)]],
        label_verbalizer: Dict[str, str],
        icl_template: Optional[constr(min_length=1)] = None,
        prompt_template: Optional[constr(min_length=1)] = None,
        top_p: confloat(ge=0.0, le=1.0),
        temperature: confloat(ge=0.0, le=1e6),
        filter_params: DatasetFilterParams = DatasetFilterParams(filter_type="none"),
        metrics_to_evaluate: List[MetricName],
        metrics_label_distribution: Literal["balanced", "train_set"],
        metrics_num_samples_per_label: Optional[Union[Dict[str, conint(ge=1)], conint(ge=1)]],
        metrics_calc_overall: bool = True,
        metrics_calc_labels: bool = False,
        metrics_labels_to_evaluate: Optional[List[str]] = None,
        metrics_parallelize: Parallelize = Parallelize.ray,
        metrics_max_parallel: Optional[conint(ge=1)] = None,
        metrics_override_row_count: bool = False,
        student_hpo_validation_set: Literal["seed", "train_set", "val_set"] = "seed",
        text_gens_parser: Callable,
        label_preservation_student: Student = Student.DeBERTaV3Large,
        dataset_cartography_student: Student = Student.DeBERTaV3Large,
        dataset_cartography_text: Literal["context", "generations"] = "context",
        seed: int = DEFAULT_SEED,
        seed_set_data_split: DataSplit = DEFAULT_SEED_SET_DATA_SPLIT,
        query_col: Optional[str] = None,
        context_col: Optional[str] = None,
        label_col: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        if icl_type == "curated":
            raise not_impl("icl_type", icl_type)
        query_col: str = get_default(query_col, dataset_name.query_col())
        context_col: str = get_default(context_col, corpus.context_col())
        label_col: str = get_default(label_col, dataset_name.label_col())
        references_data_split: DataSplit = DataSplit.TEST
        references_col: str = TEXT_GEN_REFERENCES_COL(data_split=references_data_split)
        dataset_cartography_text_col: str = {
            "context": RETRIEVED_CONTEXT_COL,
            "generations": GENERATED_TEXTS_COL,
        }[dataset_cartography_text]

        num_samples_per_label: Optional[Dict[str, int]] = expand_num_samples_per_label(
            num_samples_per_label=num_samples_per_label,
            label_verbalizer={**label_verbalizer, LABEL_OVERALL: LABEL_OVERALL},
        )
        metrics_num_samples_per_label: Optional[Dict[str, int]] = expand_num_samples_per_label(
            num_samples_per_label=metrics_num_samples_per_label,
            label_verbalizer={**label_verbalizer, LABEL_OVERALL: LABEL_OVERALL},
        )
        metrics_labels_to_evaluate: List[str] = self._labels_to_evaluate(
            metrics_labels_to_evaluate=metrics_labels_to_evaluate,
            metrics_calc_overall=metrics_calc_overall,
            metrics_calc_labels=metrics_calc_labels,
            label_verbalizer=label_verbalizer,
        )

        icl_template, icl_template_hash, prompt_template, prompt_template_hash = get_templates_and_hashes(
            expt=Experiment.SynthesizRR,
            dataset_name=dataset_name,
            model_name=model_name,
            icl_template=icl_template,
            prompt_template=prompt_template,
        )

        text_gens_expanded: Dict[int, TextGenerationsPredictionsBase] = {
            num_shots: self.expand_text_gens(
                text_gens[num_shots],
                num_shots=num_shots,
                text_gens_parser=text_gens_parser,
                model_name=model_name,
                dataset_name=dataset_name,
                references_col=references_col,
                references_data_split=references_data_split,
                label_verbalizer=label_verbalizer,
                label_col=label_col,
                seed=seed,
            )
            for num_shots in text_gens
        }

        text_gens_expanded_filtered: Dict[int, TextGenerationsPredictionsBase] = accumulate(
            {
                num_shots: run_concurrent(
                    self.filter_text_gens,
                    text_gens_expanded[num_shots],
                    filter_params=filter_params,
                    dataset_cartography_student=dataset_cartography_student,
                    dataset_cartography_text_col=dataset_cartography_text_col,
                    num_shots=num_shots,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    label_verbalizer=label_verbalizer,
                    label_col=label_col,
                    seed=seed,
                    verbosity=self.verbosity,
                    filter_text_gens_results_dir=self.save_to(
                        save_type="expanded-filtered",
                        results_dir=results_dir,
                        dataset_name=dataset_name,
                        corpus=corpus,
                        retriever=retriever,
                        model_name=model_name,
                        num_shots=num_shots,
                        label_text=LABEL_OVERALL,
                        label_verbalizer=label_verbalizer,
                        seed_set_data_split=seed_set_data_split,
                        seed_type=seed_type,
                        seed_generation_params_hash=seed_generation_params_hash,
                        icl_type=icl_type,
                        retr_icl_top_ks=retr_icl_top_ks,
                        retr_icl_distance_range=retr_icl_distance_range,
                        retr_icl_token_range=retr_icl_token_range,
                        synthesizrr_top_k_range=synthesizrr_top_k_range,
                        synthesizrr_distance_range=synthesizrr_distance_range,
                        synthesizrr_max_tokens=synthesizrr_max_tokens,
                        top_p=top_p,
                        temperature=temperature,
                        icl_template_hash=icl_template_hash,
                        prompt_template_hash=prompt_template_hash,
                        num_samples_per_label=get_default(num_samples_per_label, {}).get(LABEL_OVERALL),
                        text_gens_parser=text_gens_parser,
                        filter_params=filter_params,
                        metric_name=MetricName.NoMetric,
                        metrics_num_samples_per_label=get_default(metrics_num_samples_per_label, {}).get(
                            LABEL_OVERALL
                        ),
                        metrics_label_distribution=metrics_label_distribution,
                        student_hpo_validation_set=student_hpo_validation_set,
                        dataset_cartography_student=dataset_cartography_student,
                        dataset_cartography_text_col=dataset_cartography_text_col,
                        seed=seed,
                    ),
                )
                for num_shots in text_gens_expanded
            }
        )

        metrics_files: Dict[int, Dict[str, Dict[MetricName, FileMetadata]]] = {}
        missing_metrics_files: Dict[int, Dict[str, Dict[MetricName, FileMetadata]]] = {}
        for num_shots, _ in text_gens.items():
            metrics_files.setdefault(num_shots, {})
            for label_text in metrics_labels_to_evaluate:
                metrics_files[num_shots].setdefault(label_text, {})
                for metric_name in metrics_to_evaluate:
                    metrics_files[num_shots][label_text][metric_name]: FileMetadata = self.save_to(
                        results_dir=results_dir,
                        dataset_name=dataset_name,
                        corpus=corpus,
                        retriever=retriever,
                        model_name=model_name,
                        num_shots=num_shots,
                        label_text=label_text,
                        label_verbalizer=label_verbalizer,
                        seed_set_data_split=seed_set_data_split,
                        seed_type=seed_type,
                        seed_generation_params_hash=seed_generation_params_hash,
                        icl_type=icl_type,
                        retr_icl_top_ks=retr_icl_top_ks,
                        retr_icl_distance_range=retr_icl_distance_range,
                        retr_icl_token_range=retr_icl_token_range,
                        synthesizrr_top_k_range=synthesizrr_top_k_range,
                        synthesizrr_distance_range=synthesizrr_distance_range,
                        synthesizrr_max_tokens=synthesizrr_max_tokens,
                        top_p=top_p,
                        temperature=temperature,
                        icl_template_hash=icl_template_hash,
                        prompt_template_hash=prompt_template_hash,
                        num_samples_per_label=get_default(num_samples_per_label, {}).get(label_text),
                        text_gens_parser=text_gens_parser,
                        filter_params=filter_params,
                        metric_name=metric_name,
                        metrics_num_samples_per_label=get_default(metrics_num_samples_per_label, {}).get(
                            label_text
                        ),
                        metrics_label_distribution=metrics_label_distribution,
                        student_hpo_validation_set=student_hpo_validation_set,
                        dataset_cartography_student=dataset_cartography_student,
                        dataset_cartography_text_col=dataset_cartography_text_col,
                        seed=seed,
                    )
                    if self.should_run_metric(
                        metric_name=metric_name,
                        metric_file=metrics_files[num_shots][label_text][metric_name],
                    ):
                        missing_metrics_files.setdefault(num_shots, {})
                        missing_metrics_files[num_shots].setdefault(label_text, {})
                        missing_metrics_files[num_shots][label_text][
                            metric_name
                        ]: FileMetadata = metrics_files[num_shots][label_text][metric_name]
                    else:
                        self.info(
                            f'>> Metric "{metric_name.canonical()}" already exists for '
                            f"{self._combo_str(dataset_name, model_name, num_shots, label_text)} "
                            f'at "{metrics_files[num_shots][label_text][metric_name].path}"...'
                        )
        if len(missing_metrics_files) > 0:
            evaluated_metrics: Dict[int, Dict[str, Dict[MetricName, Metric]]] = self._run_metrics_calculation(
                results_dir=results_dir,
                expt=Experiment.SynthesizRR,
                metrics_files=missing_metrics_files,
                text_gens=text_gens_expanded_filtered,
                model_name=model_name,
                dataset_name=dataset_name,
                label_col=label_col,
                references_col=references_col,
                label_verbalizer=label_verbalizer,
                seed=seed,
                metrics_num_samples_per_label=metrics_num_samples_per_label,
                metrics_label_distribution=metrics_label_distribution,
                metrics_override_row_count=metrics_override_row_count,
                val_set=seed_set,
                student_text_col=GENERATED_TEXTS_COL,
                student_hpo_validation_set=student_hpo_validation_set,
                label_preservation_student=label_preservation_student,
                dataset_cartography_student=dataset_cartography_student,
                dataset_cartography_text_col=dataset_cartography_text_col,
                metrics_parallelize=metrics_parallelize,
                metrics_max_parallel=metrics_max_parallel,
            )
        text_gens_expanded_filtered_metrics: Dict[int, Dict[str, Dict[MetricName, Metric]]] = (
            self._load_metrics(
                results_dir=results_dir,
                expt=Experiment.SynthesizRR,
                metrics_files=metrics_files,
                dataset_name=dataset_name,
                model_name=model_name,
                label_preservation_student=label_preservation_student,
            )
        )
        return dict(
            text_gens_expanded=text_gens_expanded,
            text_gens_expanded_filtered=text_gens_expanded_filtered,
            text_gens_expanded_metrics_files=metrics_files,
            text_gens_expanded_metrics=text_gens_expanded_filtered_metrics,
        )

    def save_to(
        self,
        save_type: Literal["metrics", "expanded", "filtered", "expanded-filtered"] = "metrics",
        *,
        results_dir: FileMetadata,  ## E.g. RESULTS_DIR/,
        dataset_name: DatasetName,  ## E.g. ag_news
        corpus: Corpus,
        retriever: Retriever,
        model_name: ModelName,
        num_shots: int,
        label_text: str,
        label_verbalizer: Dict[str, str],
        seed_set_data_split: DataSplit,
        seed_type: Literal["generated", "train_set"],
        seed_generation_params_hash: Optional[str],
        icl_type: Literal["retrieved", "curated", "seed"],
        retr_icl_top_ks: List[conint(ge=1)],
        retr_icl_distance_range: Tuple[float, float],
        retr_icl_token_range: Tuple[conint(ge=1), conint(ge=1)],
        synthesizrr_top_k_range: range,
        synthesizrr_distance_range: Tuple[float, float],  ## (0.4, 0.9)
        synthesizrr_max_tokens: conint(ge=1),
        top_p: confloat(ge=0.0, le=1.0),
        temperature: confloat(ge=0.0, le=1e6),
        icl_template_hash: Optional[constr(min_length=6)],
        prompt_template_hash: Optional[constr(min_length=6)],
        num_samples_per_label: Optional[conint(ge=1)],
        text_gens_parser: Callable,
        filter_params: DatasetFilterParams,
        metric_name: MetricName,
        metrics_num_samples_per_label: Optional[conint(ge=1)],
        metrics_label_distribution: Literal["balanced", "train_set"],
        student_hpo_validation_set: Literal["seed", "train_set", "val_set"],
        dataset_cartography_student: Student,
        dataset_cartography_text_col: str,
        seed: int,
        # **kwargs,
    ) -> FileMetadata:
        ## RESULTS_DIR/retrieval-augmented-dataset-generation/ag_news/retrieval-data/toi_without_datasets/ag_news_seed_retr_output_toi_without_datasets.jsonlines
        if seed_type == "generated":
            assert seed_generation_params_hash is not None
            seed_type_str: str = f"generated_seed_set={seed_generation_params_hash}"
        elif seed_type == "train_set":
            seed_type_str: str = f"{seed_set_data_split.lower()}_set_seed_set"
        else:
            raise not_impl("seed_type", seed_type)

        if icl_type == "retrieved":
            icl_type_str: str = "retrieved_icl_dataset"
        elif icl_type == "curated":
            icl_type_str: str = "curated_icl_dataset"
        elif icl_type == "seed":
            icl_type_str: str = f"{seed_type}_seed_icl_dataset"
        else:
            raise not_impl("icl_type", icl_type)

        num_samples_per_label_str: str = get_default(num_samples_per_label, "all")
        metrics_num_samples_per_label_str: str = get_default(metrics_num_samples_per_label, "all")

        if save_type != "metrics":
            if icl_template_hash is not None:
                icl_template_hash: str = icl_template_hash[:6]
            if prompt_template_hash is not None:
                prompt_template_hash: str = prompt_template_hash[:6]

        icl_template_hash_str: str = (
            "" if icl_template_hash is None else f"-icl_template_hash={icl_template_hash}"
        )
        prompt_template_hash_str: str = (
            "" if prompt_template_hash is None else f"-prompt_template_hash={prompt_template_hash}"
        )

        label_verbalizer_str: str = ""
        if label_text == LABEL_OVERALL:
            if label_verbalizer != dataset_name.label_verbalizer():
                label_verbalizer_str: str = f"-vbs={StringUtil.hash(label_verbalizer, max_len=4)}"
        else:
            assert label_text in label_verbalizer
            assert label_text in dataset_name.label_verbalizer()
            if label_verbalizer[label_text] != dataset_name.label_verbalizer()[label_text]:
                label_verbalizer_str: str = f"-vb={StringUtil.hash(label_verbalizer[label_text], max_len=4)}"

        text_gens_parser_str: str = ""
        text_gens_parser_body_hash: str = StringUtil.hash(get_fn_spec(text_gens_parser).source_body)
        if text_gens_parser_body_hash != StringUtil.hash(
            get_fn_spec(dataset_name.text_gens_parser()).source_body
        ):
            text_gens_parser_str: str = f"-pr={text_gens_parser_body_hash[:6]}"

        filter_params_str: str = ""
        if filter_params.filter_type != "none":
            filter_str, filter_hash = filter_params.save_key_and_hash(
                dataset_cartography_student=dataset_cartography_student,
                dataset_cartography_text_col=dataset_cartography_text_col,
            )
            filter_params_str: str = f"-f={filter_hash}"

        top_p_str: str = ""
        if top_p != DEFAULT_TOP_P:
            top_p_str = f"-top_p={top_p:.2f}"
        temperature_str: str = ""
        if temperature != DEFAULT_TEMPERATURE:
            temperature_str = f"-temp={temperature}"

        student_hpo_validation_set_str: str = ""
        if metric_name.is_student_hpo():
            student_hpo_validation_set_str: str = f"-hpo_set={student_hpo_validation_set}"

        dataset_cartography_str: str = ""
        if metric_name is MetricName.StudentDatasetCartography:
            dataset_cartography_str: str = (
                f"-cart={dataset_cartography_student.canonical()}_col={dataset_cartography_text_col}"
            )

        results_path = (
            results_dir.subdir_in_dir("retrieval-augmented-dataset-generation", return_metadata=True)
            .subdir_in_dir(dataset_name.canonical(), return_metadata=True)
            .subdir_in_dir(f"synthesizrr-generations-{save_type}", return_metadata=True)
            .subdir_in_dir(corpus.canonical(), return_metadata=True)
            .subdir_in_dir(retriever.canonical(), return_metadata=True)
            .subdir_in_dir(model_name.canonical(), return_metadata=True)
            .subdir_in_dir(
                f"num_samples_per_label={num_samples_per_label_str}",
                return_metadata=True,
            )
            .subdir_in_dir(f"num_shots={num_shots}", return_metadata=True)
        )
        _params = (
            f"-{icl_type_str}"
            f"-{seed_type_str}-retr_output"
            f"-dataset={dataset_name.canonical()}"
            f"-corpus={corpus.canonical()}"
            f"-retriever={retriever.canonical()}"
            f"-model_name={model_name.canonical()}"
            f"-num_samples_per_label={num_samples_per_label_str}"
            f"-num_shots={num_shots}"
            f"-seed={seed}"
            f"-retr_icl_top_ks={retr_icl_top_ks}"
            f"-retr_icl_distance_range={retr_icl_distance_range}"
            f"-retr_icl_token_range={retr_icl_token_range}"
            f"-synthesizrr_top_k_range=range({synthesizrr_top_k_range.start}, {synthesizrr_top_k_range.stop}, {synthesizrr_top_k_range.step})"
            f"-synthesizrr_distance_range={synthesizrr_distance_range}"
            f"-synthesizrr_max_tokens={synthesizrr_max_tokens}"
            f"{icl_template_hash_str}"
            f"{prompt_template_hash_str}"
            f"{label_verbalizer_str}"
            f"{student_hpo_validation_set_str}"
            f"{dataset_cartography_str}"
            f"{text_gens_parser_str}"
            f"{filter_params_str}"
            f"{top_p_str}"
            f"{temperature_str}"
        )
        if save_type == "metrics":
            try:
                return (
                    results_path.subdir_in_dir(f"label={label_text}", return_metadata=True)
                    .subdir_in_dir(
                        f"metrics_label_distribution={metrics_label_distribution}",
                        return_metadata=True,
                    )
                    .subdir_in_dir(
                        f"metrics_num_samples_per_label={metrics_num_samples_per_label_str}",
                        return_metadata=True,
                    )
                    .subdir_in_dir(f"metric_name={metric_name.canonical()}", return_metadata=True)
                    .file_in_dir(
                        f"metric"
                        f"-label={label_text}"
                        f"-metrics_label_distribution={metrics_label_distribution}"
                        f"-metrics_num_samples_per_label={metrics_num_samples_per_label}"
                        f"-metric_name={metric_name.canonical()}"
                        f"{_params}"
                        f".pkl",
                        return_metadata=True,
                    )
                    .update_params(file_format=FileFormat.PICKLE)
                )
            except ValidationError:
                return (
                    results_path.subdir_in_dir(f"label={label_text}", return_metadata=True)
                    .subdir_in_dir(
                        f"metrics_label_distribution={metrics_label_distribution}",
                        return_metadata=True,
                    )
                    .subdir_in_dir(
                        f"metrics_num_samples_per_label={metrics_num_samples_per_label_str}",
                        return_metadata=True,
                    )
                    .subdir_in_dir(f"metric_name={metric_name.canonical()}", return_metadata=True)
                    .file_in_dir(
                        f"metric"
                        f"-label={label_text}"
                        f"-metrics_label_distribution={metrics_label_distribution}"
                        f"-metrics_num_samples_per_label={metrics_num_samples_per_label}"
                        f"-metric_name={metric_name.canonical()}"
                        f"-params_hash={StringUtil.hash(_params, max_len=6)}"
                        f".pkl",
                        return_metadata=True,
                    )
                    .update_params(file_format=FileFormat.PICKLE)
                )

        else:
            ## Return a directory to save the text generations
            try:
                return results_path.subdir_in_dir(
                    f"synthesizrr-generations-{save_type}{_params}",
                    return_metadata=True,
                )
            except ValidationError:
                return results_path.subdir_in_dir(
                    f"synthesizrr-generations-{save_type}-params_hash={StringUtil.hash(_params, max_len=6)}",
                    return_metadata=True,
                )


class FewGenTextGenMetrics(BaseCalculateTextGenMetrics):
    @safe_validate_arguments
    def run(
        self,
        *,
        text_gens: Dict[int, TextGenerationsPredictionsBase],
        results_dir: FileMetadata,
        dataset_name: DatasetName,
        icl_dataset: ClassificationData,
        text_input_dataset: TextInputs,
        model_name: ModelName,
        num_shots_list: List[conint(ge=0)],
        seed_set: ClassificationData,
        seed_type: Literal["generated", "train_set"],
        seed_generation_params_hash: Optional[str],
        fewgen_max_tokens: conint(ge=1),
        num_samples_per_label: Union[Dict[str, conint(ge=1)], conint(ge=1)],
        label_verbalizer: Dict[str, str],
        icl_template: Optional[constr(min_length=1)] = None,
        prompt_template: Optional[constr(min_length=1)] = None,
        top_p: confloat(ge=0.0, le=1.0),
        temperature: confloat(ge=0.0, le=1e6),
        filter_params: DatasetFilterParams = DatasetFilterParams(filter_type="none"),
        metrics_to_evaluate: List[MetricName],
        metrics_label_distribution: Literal["balanced", "train_set"],
        metrics_num_samples_per_label: Union[Dict[str, conint(ge=1)], conint(ge=1)],
        metrics_calc_overall: bool = True,
        metrics_calc_labels: bool = False,
        metrics_labels_to_evaluate: Optional[List[str]] = None,
        metrics_parallelize: Parallelize = Parallelize.ray,
        metrics_max_parallel: Optional[conint(ge=1)] = None,
        metrics_override_row_count: bool = False,
        text_gens_parser: Callable,
        label_preservation_student: Student = Student.DeBERTaV3Large,
        dataset_cartography_student: Student = Student.DeBERTaV3Large,
        student_hpo_validation_set: Literal["seed", "train_set", "val_set"] = "seed",
        seed: int = DEFAULT_SEED,
        seed_set_data_split: DataSplit = DEFAULT_SEED_SET_DATA_SPLIT,
        text_col: Optional[str] = None,
        label_col: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        text_col: str = get_default(text_col, dataset_name.text_col())
        label_col: str = get_default(label_col, dataset_name.label_col())
        references_data_split: DataSplit = DataSplit.TEST
        references_col: str = TEXT_GEN_REFERENCES_COL(data_split=references_data_split)
        dataset_cartography_text_col: str = GENERATED_TEXTS_COL

        num_samples_per_label: Dict[str, int] = expand_num_samples_per_label(
            num_samples_per_label=num_samples_per_label,
            label_verbalizer={**label_verbalizer, LABEL_OVERALL: LABEL_OVERALL},
        )
        metrics_num_samples_per_label: Dict[str, int] = expand_num_samples_per_label(
            num_samples_per_label=metrics_num_samples_per_label,
            label_verbalizer={**label_verbalizer, LABEL_OVERALL: LABEL_OVERALL},
        )
        metrics_labels_to_evaluate: List[str] = self._labels_to_evaluate(
            metrics_labels_to_evaluate=metrics_labels_to_evaluate,
            metrics_calc_overall=metrics_calc_overall,
            metrics_calc_labels=metrics_calc_labels,
            label_verbalizer=label_verbalizer,
        )

        icl_template, icl_template_hash, prompt_template, prompt_template_hash = get_templates_and_hashes(
            expt=Experiment.FewGen,
            dataset_name=dataset_name,
            model_name=model_name,
            icl_template=icl_template,
            prompt_template=prompt_template,
        )

        text_gens_expanded: Dict[int, TextGenerationsPredictionsBase] = {
            num_shots: self.expand_text_gens(
                text_gens[num_shots],
                num_shots=num_shots,
                text_gens_parser=text_gens_parser,
                model_name=model_name,
                dataset_name=dataset_name,
                references_col=references_col,
                references_data_split=references_data_split,
                label_verbalizer=label_verbalizer,
                label_col=label_col,
                seed=seed,
            )
            for num_shots in text_gens
        }

        text_gens_expanded_filtered: Dict[int, TextGenerationsPredictionsBase] = {
            num_shots: self.filter_text_gens(
                text_gens_expanded[num_shots],
                filter_params=filter_params,
                dataset_cartography_student=dataset_cartography_student,
                dataset_cartography_text_col=dataset_cartography_text_col,
                num_shots=num_shots,
                model_name=model_name,
                dataset_name=dataset_name,
                label_verbalizer=label_verbalizer,
                label_col=label_col,
                seed=seed,
                verbosity=self.verbosity,
                filter_text_gens_results_dir=self.save_to(
                    save_type="expanded-filtered",
                    results_dir=results_dir,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    num_shots=num_shots,
                    label_text=LABEL_OVERALL,
                    label_verbalizer=label_verbalizer,
                    seed_set_data_split=seed_set_data_split,
                    seed_type=seed_type,
                    seed_generation_params_hash=seed_generation_params_hash,
                    fewgen_max_tokens=fewgen_max_tokens,
                    top_p=top_p,
                    temperature=temperature,
                    icl_template_hash=icl_template_hash,
                    prompt_template_hash=prompt_template_hash,
                    num_samples_per_label=num_samples_per_label[LABEL_OVERALL],
                    text_gens_parser=text_gens_parser,
                    filter_params=filter_params,
                    metric_name=MetricName.NoMetric,
                    metrics_num_samples_per_label=metrics_num_samples_per_label[LABEL_OVERALL],
                    metrics_label_distribution=metrics_label_distribution,
                    student_hpo_validation_set=student_hpo_validation_set,
                    dataset_cartography_student=dataset_cartography_student,
                    dataset_cartography_text_col=dataset_cartography_text_col,
                    seed=seed,
                ),
            )
            for num_shots in text_gens_expanded
        }

        metrics_files: Dict[int, Dict[str, Dict[MetricName, FileMetadata]]] = {}
        missing_metrics_files: Dict[int, Dict[str, Dict[MetricName, FileMetadata]]] = {}
        for num_shots in text_gens.keys():
            metrics_files.setdefault(num_shots, {})
            for label_text in metrics_labels_to_evaluate:
                metrics_files[num_shots].setdefault(label_text, {})
                for metric_name in metrics_to_evaluate:
                    metrics_files[num_shots][label_text][metric_name]: FileMetadata = self.save_to(
                        results_dir=results_dir,
                        dataset_name=dataset_name,
                        model_name=model_name,
                        num_shots=num_shots,
                        label_text=label_text,
                        label_verbalizer=label_verbalizer,
                        seed_set_data_split=seed_set_data_split,
                        seed_type=seed_type,
                        seed_generation_params_hash=seed_generation_params_hash,
                        fewgen_max_tokens=fewgen_max_tokens,
                        top_p=top_p,
                        temperature=temperature,
                        icl_template_hash=icl_template_hash,
                        prompt_template_hash=prompt_template_hash,
                        num_samples_per_label=num_samples_per_label[label_text],
                        text_gens_parser=text_gens_parser,
                        filter_params=filter_params,
                        metric_name=metric_name,
                        metrics_num_samples_per_label=metrics_num_samples_per_label[label_text],
                        metrics_label_distribution=metrics_label_distribution,
                        student_hpo_validation_set=student_hpo_validation_set,
                        dataset_cartography_student=dataset_cartography_student,
                        dataset_cartography_text_col=dataset_cartography_text_col,
                        seed=seed,
                    )
                    if self.should_run_metric(
                        metric_name=metric_name,
                        metric_file=metrics_files[num_shots][label_text][metric_name],
                    ):
                        missing_metrics_files.setdefault(num_shots, {})
                        missing_metrics_files[num_shots].setdefault(label_text, {})
                        missing_metrics_files[num_shots][label_text][
                            metric_name
                        ]: FileMetadata = metrics_files[num_shots][label_text][metric_name]
                    else:
                        self.info(
                            f'>> Metric "{metric_name.canonical()}" already exists for '
                            f"{self._combo_str(dataset_name, model_name, num_shots, label_text)} "
                            f'at "{metrics_files[num_shots][label_text][metric_name].path}"...'
                        )
        if len(missing_metrics_files) > 0:
            evaluated_metrics: Dict[int, Dict[str, Dict[MetricName, Metric]]] = self._run_metrics_calculation(
                results_dir=results_dir,
                expt=Experiment.FewGen,
                metrics_files=missing_metrics_files,
                text_gens=text_gens_expanded_filtered,
                model_name=model_name,
                dataset_name=dataset_name,
                label_col=label_col,
                references_col=references_col,
                label_verbalizer=label_verbalizer,
                seed=seed,
                metrics_num_samples_per_label=metrics_num_samples_per_label,
                metrics_label_distribution=metrics_label_distribution,
                metrics_override_row_count=metrics_override_row_count,
                val_set=seed_set,
                student_text_col=GENERATED_TEXTS_COL,
                student_hpo_validation_set=student_hpo_validation_set,
                label_preservation_student=label_preservation_student,
                dataset_cartography_student=dataset_cartography_student,
                dataset_cartography_text_col=dataset_cartography_text_col,
                metrics_parallelize=metrics_parallelize,
                metrics_max_parallel=metrics_max_parallel,
            )
        text_gens_expanded_filtered_metrics: Dict[int, Dict[str, Dict[MetricName, Metric]]] = (
            self._load_metrics(
                results_dir=results_dir,
                expt=Experiment.FewGen,
                metrics_files=metrics_files,
                dataset_name=dataset_name,
                model_name=model_name,
                label_preservation_student=label_preservation_student,
            )
        )
        return dict(
            text_gens_expanded=text_gens_expanded,
            text_gens_expanded_filtered=text_gens_expanded_filtered,
            text_gens_expanded_metrics_files=metrics_files,
            text_gens_expanded_metrics=text_gens_expanded_filtered_metrics,
        )

    def save_to(
        self,
        save_type: Literal["metrics", "expanded", "filtered", "expanded-filtered"] = "metrics",
        *,
        results_dir: FileMetadata,  ## E.g. RESULTS_DIR/,
        dataset_name: DatasetName,  ## E.g. ag_news
        model_name: ModelName,
        num_shots: int,
        label_text: str,
        label_verbalizer: Dict[str, str],
        seed_set_data_split: DataSplit,
        seed_type: Literal["generated", "train_set"],
        seed_generation_params_hash: Optional[str],
        fewgen_max_tokens: conint(ge=1),
        top_p: confloat(ge=0.0, le=1.0),
        temperature: confloat(ge=0.0, le=1e6),
        icl_template_hash: Optional[constr(min_length=6)],
        prompt_template_hash: Optional[constr(min_length=6)],
        num_samples_per_label: conint(ge=1),
        text_gens_parser: Callable,
        filter_params: DatasetFilterParams,
        metric_name: MetricName,
        metrics_num_samples_per_label: Optional[conint(ge=1)],
        metrics_label_distribution: Literal["balanced", "train_set"],
        student_hpo_validation_set: Literal["seed", "train_set", "val_set"],
        dataset_cartography_student: Student,
        dataset_cartography_text_col: str,
        seed: int,
        **kwargs,
    ) -> FileMetadata:
        ## RESULTS_DIR/retrieval-augmented-dataset-generation/ag_news/retrieval-data/toi_without_datasets/ag_news_seed_retr_output_toi_without_datasets.jsonlines
        if seed_type == "generated":
            assert seed_generation_params_hash is not None
            seed_type_str: str = f"generated_seed_set={seed_generation_params_hash}"
        elif seed_type == "train_set":
            seed_type_str: str = f"{seed_set_data_split.lower()}_set_seed_set"
        else:
            raise not_impl("seed_type", seed_type)

        num_samples_per_label_str: str = get_default(num_samples_per_label, "all")
        metrics_num_samples_per_label_str: str = get_default(metrics_num_samples_per_label, "all")

        if save_type != "metrics":
            if icl_template_hash is not None:
                icl_template_hash: str = icl_template_hash[:6]
            if prompt_template_hash is not None:
                prompt_template_hash: str = prompt_template_hash[:6]

        icl_template_hash_str: str = (
            "" if icl_template_hash is None else f"-icl_template_hash={icl_template_hash}"
        )
        prompt_template_hash_str: str = (
            "" if prompt_template_hash is None else f"-prompt_template_hash={prompt_template_hash}"
        )

        label_verbalizer_str: str = ""
        if label_text == LABEL_OVERALL:
            if label_verbalizer != dataset_name.label_verbalizer():
                label_verbalizer_str: str = f"-vbs={StringUtil.hash(label_verbalizer, max_len=4)}"
        else:
            assert label_text in label_verbalizer
            assert label_text in dataset_name.label_verbalizer()
            if label_verbalizer[label_text] != dataset_name.label_verbalizer()[label_text]:
                label_verbalizer_str: str = f"-vb={StringUtil.hash(label_verbalizer[label_text], max_len=4)}"

        text_gens_parser_str: str = ""
        text_gens_parser_body_hash: str = StringUtil.hash(get_fn_spec(text_gens_parser).source_body)
        if text_gens_parser_body_hash != StringUtil.hash(
            get_fn_spec(dataset_name.text_gens_parser()).source_body
        ):
            text_gens_parser_str: str = f"-pr={text_gens_parser_body_hash[:6]}"

        filter_params_str: str = ""
        if filter_params.filter_type != "none":
            filter_str, filter_hash = filter_params.save_key_and_hash(
                dataset_cartography_student=dataset_cartography_student,
                dataset_cartography_text_col=dataset_cartography_text_col,
            )
            filter_params_str: str = f"-f={filter_hash}"

        top_p_str: str = ""
        if top_p != DEFAULT_TOP_P:
            top_p_str = f"-top_p={top_p:.2f}"
        temperature_str: str = ""
        if temperature != DEFAULT_TEMPERATURE:
            temperature_str = f"-temp={temperature}"

        student_hpo_validation_set_str: str = ""
        if metric_name.is_student_hpo():
            student_hpo_validation_set_str: str = f"-hpo_set={student_hpo_validation_set}"

        dataset_cartography_str: str = ""
        if metric_name is MetricName.StudentDatasetCartography:
            dataset_cartography_str: str = (
                f"-cart={dataset_cartography_student.canonical()}_col={dataset_cartography_text_col}"
            )

        results_path = (
            results_dir.subdir_in_dir("few-shot-generation", return_metadata=True)
            .subdir_in_dir(dataset_name.canonical(), return_metadata=True)
            .subdir_in_dir(f"fewgen-generations-{save_type}", return_metadata=True)
            .subdir_in_dir(model_name.canonical(), return_metadata=True)
            .subdir_in_dir(
                f"num_samples_per_label={num_samples_per_label_str}",
                return_metadata=True,
            )
            .subdir_in_dir(f"num_shots={num_shots}", return_metadata=True)
        )

        if save_type == "metrics":
            return (
                results_path.subdir_in_dir(f"label={label_text}", return_metadata=True)
                .subdir_in_dir(
                    f"metrics_label_distribution={metrics_label_distribution}",
                    return_metadata=True,
                )
                .subdir_in_dir(
                    f"metrics_num_samples_per_label={metrics_num_samples_per_label_str}",
                    return_metadata=True,
                )
                .subdir_in_dir(f"metric_name={metric_name.canonical()}", return_metadata=True)
                .file_in_dir(
                    f"metric"
                    f"-metrics_label_distribution={metrics_label_distribution}"
                    f"-metrics_num_samples_per_label={metrics_num_samples_per_label}"
                    f"-metric_name={metric_name.canonical()}"
                    f"-seed_type={seed_type_str}"
                    f"-dataset={dataset_name.canonical()}"
                    f"-model_name={model_name.canonical()}"
                    f"-num_samples_per_label={num_samples_per_label_str}"
                    f"-num_shots={num_shots}"
                    f"-label={label_text}"
                    f"-seed={seed}"
                    f"-fewgen_max_tokens={fewgen_max_tokens}"
                    f"{icl_template_hash_str}"
                    f"{prompt_template_hash_str}"
                    f"{label_verbalizer_str}"
                    f"{student_hpo_validation_set_str}"
                    f"{dataset_cartography_str}"
                    f"{text_gens_parser_str}"
                    f"{filter_params_str}"
                    f"{top_p_str}"
                    f"{temperature_str}"
                    f".pkl",
                    return_metadata=True,
                )
                .update_params(file_format=FileFormat.PICKLE)
            )
        else:
            ## Return a directory to save the text generations
            return results_path.subdir_in_dir(f"fewgen-generations-{save_type}")
            if icl_template_hash is not None:
                icl_template_hash: str = icl_template_hash[:6]
            if prompt_template_hash is not None:
                prompt_template_hash: str = prompt_template_hash[:6]

        icl_template_hash_str: str = (
            "" if icl_template_hash is None else f"-icl_template_hash={icl_template_hash}"
        )
        prompt_template_hash_str: str = (
            "" if prompt_template_hash is None else f"-prompt_template_hash={prompt_template_hash}"
        )

        label_verbalizer_str: str = ""
        if label_text == LABEL_OVERALL:
            if label_verbalizer != dataset_name.label_verbalizer():
                label_verbalizer_str: str = f"-vbs={StringUtil.hash(label_verbalizer, max_len=4)}"
        else:
            assert label_text in label_verbalizer
            assert label_text in dataset_name.label_verbalizer()
            if label_verbalizer[label_text] != dataset_name.label_verbalizer()[label_text]:
                label_verbalizer_str: str = f"-vb={StringUtil.hash(label_verbalizer[label_text], max_len=4)}"

        text_gens_parser_str: str = ""
        text_gens_parser_body_hash: str = StringUtil.hash(get_fn_spec(text_gens_parser).source_body)
        if text_gens_parser_body_hash != StringUtil.hash(
            get_fn_spec(dataset_name.text_gens_parser()).source_body
        ):
            text_gens_parser_str: str = f"-pr={text_gens_parser_body_hash[:6]}"

        filter_params_str: str = ""
        if filter_params.filter_type != "none":
            filter_str, filter_hash = filter_params.save_key_and_hash(
                dataset_cartography_student=dataset_cartography_student,
                dataset_cartography_text_col=dataset_cartography_text_col,
            )
            filter_params_str: str = f"-f={filter_hash}"

        top_p_str: str = ""
        if top_p != DEFAULT_TOP_P:
            top_p_str = f"-top_p={top_p:.2f}"
        temperature_str: str = ""
        if temperature != DEFAULT_TEMPERATURE:
            temperature_str = f"-temp={temperature}"

        student_hpo_validation_set_str: str = ""
        if metric_name.is_student_hpo():
            student_hpo_validation_set_str: str = f"-hpo_set={student_hpo_validation_set}"

        dataset_cartography_str: str = ""
        if metric_name is MetricName.StudentDatasetCartography:
            dataset_cartography_str: str = (
                f"-cart={dataset_cartography_student.canonical()}_col={dataset_cartography_text_col}"
            )

        results_path = (
            results_dir.subdir_in_dir("few-shot-generation", return_metadata=True)
            .subdir_in_dir(dataset_name.canonical(), return_metadata=True)
            .subdir_in_dir(f"fewgen-generations-{save_type}", return_metadata=True)
            .subdir_in_dir(model_name.canonical(), return_metadata=True)
            .subdir_in_dir(f"num_samples_per_label={num_samples_per_label_str}", return_metadata=True)
            .subdir_in_dir(f"num_shots={num_shots}", return_metadata=True)
        )

        if save_type == "metrics":
            return (
                results_path.subdir_in_dir(f"label={label_text}", return_metadata=True)
                .subdir_in_dir(
                    f"metrics_label_distribution={metrics_label_distribution}", return_metadata=True
                )
                .subdir_in_dir(
                    f"metrics_num_samples_per_label={metrics_num_samples_per_label_str}", return_metadata=True
                )
                .subdir_in_dir(f"metric_name={metric_name.canonical()}", return_metadata=True)
                .file_in_dir(
                    f"metric"
                    f"-metrics_label_distribution={metrics_label_distribution}"
                    f"-metrics_num_samples_per_label={metrics_num_samples_per_label}"
                    f"-metric_name={metric_name.canonical()}"
                    f"-seed_type={seed_type_str}"
                    f"-dataset={dataset_name.canonical()}"
                    f"-model_name={model_name.canonical()}"
                    f"-num_samples_per_label={num_samples_per_label_str}"
                    f"-num_shots={num_shots}"
                    f"-label={label_text}"
                    f"-seed={seed}"
                    f"-fewgen_max_tokens={fewgen_max_tokens}"
                    f"{icl_template_hash_str}"
                    f"{prompt_template_hash_str}"
                    f"{label_verbalizer_str}"
                    f"{student_hpo_validation_set_str}"
                    f"{dataset_cartography_str}"
                    f"{text_gens_parser_str}"
                    f"{filter_params_str}"
                    f"{top_p_str}"
                    f"{temperature_str}"
                    f".pkl",
                    return_metadata=True,
                )
                .update_params(file_format=FileFormat.PICKLE)
            )
        else:
            ## Return a directory to save the text generations
            return results_path.subdir_in_dir(
                f"fewgen-generations-{save_type}"
                f"-seed_type={seed_type_str}"
                f"-dataset={dataset_name.canonical()}"
                f"-model_name={model_name.canonical()}"
                f"-num_samples_per_label={num_samples_per_label_str}"
                f"-num_shots={num_shots}"
                f"-seed={seed}"
                f"-fewgen_max_tokens={fewgen_max_tokens}"
                f"{icl_template_hash_str}"
                f"{prompt_template_hash_str}"
                f"{label_verbalizer_str}"
                f"{student_hpo_validation_set_str}"
                f"{dataset_cartography_str}"
                f"{text_gens_parser_str}"
                f"{filter_params_str}"
                f"{top_p_str}"
                f"{temperature_str}",
                return_metadata=True,
            )
