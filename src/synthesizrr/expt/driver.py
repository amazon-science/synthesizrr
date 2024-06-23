from typing import *
from synthesizrr.base.util import *
import ray

from synthesizrr.base.framework import *
from synthesizrr.base.framework.dl.torch import *
from synthesizrr.base.util import *
from synthesizrr.expt.common import MetricName, LABEL_OVERALL, Experiment, Student, DatasetFilterParams, \
    DEFAULT_TOP_P, DEFAULT_TEMPERATURE
from synthesizrr.expt.metrics import GoldDatasetMetrics, SynthesizRRTextGenMetrics, FewGenTextGenMetrics
from synthesizrr.expt.generation import DatasetName, Corpus, ModelName, Retriever
from synthesizrr.expt.generation import EmbedCorpus, CreateSeedSet, RetrieveFromSeedSet, CreateSynthesizRRDatasets, SynthesizRR, \
    CreateFewGenDatasets, FewGen


def get_wf(expt: Experiment, metrics: bool) -> Chain:
    return {
        ## Gold:
        (Experiment.Gold, True): Chain.of(
            GoldDatasetMetrics,
        ),
        ## Gold workflow without metrics does not exist.

        ## SynthesizRR:
        (Experiment.SynthesizRR, False): Chain.of(
            EmbedCorpus,
            CreateSeedSet,
            RetrieveFromSeedSet,
            CreateSynthesizRRDatasets,
            SynthesizRR,
            # SynthesizRRTextGenMetrics,
        ),
        (Experiment.SynthesizRR, True): Chain.of(
            EmbedCorpus,
            CreateSeedSet,
            RetrieveFromSeedSet,
            CreateSynthesizRRDatasets,
            SynthesizRR,
            SynthesizRRTextGenMetrics,
        ),

        ## FewGen:
        (Experiment.FewGen, False): Chain.of(
            CreateSeedSet,
            CreateFewGenDatasets,
            FewGen,
            # FewGenTextGenMetrics,
        ),
        (Experiment.FewGen, True): Chain.of(
            CreateSeedSet,
            CreateFewGenDatasets,
            FewGen,
            FewGenTextGenMetrics,
        ),
    }[(expt, metrics)]


@safe_validate_arguments
def run_chain(
        *,
        expt: Experiment,
        results_dir: FileMetadata,
        notifier: Optional[Notifier],
        tracker: Optional[Tracker],
        background: bool,
        step_wait: confloat(ge=0.0) = 30,  ## To avoid AWS creds error when running many in parallel
        pause: confloat(ge=0.0) = 3,

        dataset_name: DatasetName,
        model_name: Optional[ModelName] = None,

        num_samples_per_label: Optional[conint(ge=10)] = None,
        seed_type: Optional[Literal['generated', 'train_set']] = None,
        seed_size: Optional[conint(ge=1)] = None,
        seed_set_stratify_on_ground_truth: bool = True,
        seed_generation_params: Optional[Dict] = None,
        top_p: confloat(ge=0.0, le=1.0) = DEFAULT_TOP_P,
        temperature: confloat(ge=0.0, le=1e6) = DEFAULT_TEMPERATURE,
        icl_and_prompt_template: Optional[Dict[str, str]] = None,
        label_verbalizer: Optional[Dict[str, str]] = None,
        num_shots_list: Optional[List[conint(ge=0)]] = None,

        metrics_overall_num_samples_per_label: conint(ge=10),
        metrics_other_label_num_samples_per_label: Optional[conint(ge=10)] = None,

        corpus: Optional[Corpus] = None,
        retriever: Optional[Retriever] = None,
        icl_type: Literal['retrieved', 'curated', 'seed'] = 'retrieved',
        retrieval_top_k: conint(ge=1) = 500,
        retr_icl_top_ks: Tuple[conint(ge=1), ...] = (1, 2),
        retr_icl_distance_range: Tuple[float, float] = (0.5, 0.9),
        retr_icl_token_range: Optional[Tuple[conint(ge=1), conint(ge=1)]] = None,
        synthesizrr_top_k_range: Optional[range] = None,
        synthesizrr_distance_range: Tuple[float, float] = (0.4, 0.9),  ## (0.4, 0.9)

        llm_batch_size: Optional[conint(ge=1)] = None,
        llm_submission_batch_size: conint(ge=1) = 24,
        llm_tracking_batch_size: Optional[conint(ge=1)] = None,
        llm_num_concurrent_preds: conint(ge=1) = 6,
        llm_num_models: Optional[conint(ge=1)] = None,
        llm_resources_per_model: Optional[Dict[
            Literal['cpu', 'gpu'],
            Union[confloat(ge=0.0, lt=1.0), conint(ge=0)]
        ]] = None,
        llm_load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_USED,
        llm_evaluation_timeout: confloat(ge=0.0, allow_inf_nan=True) = math.inf,

        text_gens_parser: Optional[Callable] = None,
        text_gens_parser_type: Literal['default', 'rejection'] = 'rejection',
        filter_params: DatasetFilterParams = DatasetFilterParams(filter_type='none'),

        metrics_calc_overall: bool = True,
        metrics_calc_labels: bool = False,
        metrics_max_parallel: conint(ge=1) = 2,
        metrics_override_row_count: bool = False,

        dataset_cartography_student: Student = Student.DistilBERT,
        dataset_cartography_text: Literal['context', 'generations'] = 'generations',

        metrics_to_evaluate: Optional[Tuple[MetricName, ...]] = (
                MetricName.RowCount,
                MetricName.TextLength,
                MetricName.EntityCount,
                MetricName.SelfBLEU,
                MetricName.Mauve,

                MetricName.StudentDistilBERT_AttrPromptTable13,
                MetricName.StudentDeBERTaV3Large,

                # MetricName.StudentHPOTinyBert,
                # MetricName.StudentHPOMiniLM,
                # MetricName.StudentHPODistilBERT,
                # MetricName.StudentHPOBERT,
                # MetricName.StudentHPODeBERTaV3Base,
                # MetricName.StudentHPODeBERTaV3Large,

                MetricName.LabelPreservation,
                MetricName.StudentDatasetCartography,
                MetricName.SaveFilteredDataset,
        ),
        metrics_label_distribution: Literal['balanced', 'train_set'] = 'train_set',
        dry_run: bool = False,
        verbosity: conint(ge=0) = 2,
        cart_frac: Optional[confloat(gt=0.0)] = None,  # 0.83,
):
    if cart_frac is not None:
        filter_params = dict(
            filter_type='cartography',
            cartography_apply='label',
            cartography_confidence_frac=('top', cart_frac),
        )
        metrics_overall_num_samples_per_label: int = int(cart_frac * metrics_overall_num_samples_per_label)

    if expt is Experiment.SynthesizRR:
        assert corpus is not None
        assert retriever is not None

    if expt is not Experiment.Gold:
        assert model_name is not None
        assert num_samples_per_label is not None
        assert seed_type is not None

        if llm_num_models is None:
            ray_num_gpus: int = int(ray.cluster_resources()["GPU"])
            llm_num_gpus: int = model_name.llm_resources_per_model().get('gpu', 0)
            assert llm_num_gpus >= 0
            if llm_num_gpus > 0:
                ## LLaMa etc.
                llm_num_models: Optional[int] = ray_num_gpus // llm_num_gpus
            else:
                ## Claude, ChatGPT, etc.
                llm_num_models: Optional[int] = None
        llm_batch_size: int = get_default(
            llm_batch_size,
            model_name.llm_batch_size(dataset_name=dataset_name, expt=expt),
        )
        llm_resources_per_model: Dict[
            Literal['cpu', 'gpu'],
            Union[confloat(ge=0.0, lt=1.0), conint(ge=0)]
        ] = get_default(
            llm_resources_per_model,
            model_name.llm_resources_per_model(),
        )
    metrics_num_samples_per_label: Dict[str, int] = {
        LABEL_OVERALL: metrics_overall_num_samples_per_label,
    }

    label_verbalizer: Dict[str, str] = get_default(label_verbalizer, dataset_name.label_verbalizer())
    if metrics_other_label_num_samples_per_label is not None:
        for label_text in label_verbalizer.keys():
            metrics_num_samples_per_label[label_text] = metrics_other_label_num_samples_per_label

    if expt is Experiment.Gold:
        text_gens_parser: Optional[Callable] = None
    elif text_gens_parser is None:
        if text_gens_parser_type == 'default':
            text_gens_parser: Callable = dataset_name.text_gens_parser()
        elif text_gens_parser_type == 'rejection':
            text_gens_parser: Callable = dataset_name.text_gens_parser_rejection(expt=expt)
        else:
            raise not_impl('text_gens_parser_type', text_gens_parser_type)

    exn_input = dict(
        results_dir=results_dir,

        dataset_name=dataset_name,
        label_verbalizer=label_verbalizer,

        seed_type=seed_type,
        seed_size=get_default(seed_size, dataset_name.seed_size()),
        seed_set_stratify_on_ground_truth=seed_set_stratify_on_ground_truth,
        seed_generation_params=seed_generation_params,

        model_name=model_name,
        num_shots_list=get_default(num_shots_list, {
            Experiment.Gold: [None],
            Experiment.SynthesizRR: [0, 3],
            Experiment.FewGen: [0, 32],
        }[expt]),
        num_samples_per_label=num_samples_per_label,
        top_p=top_p,
        temperature=temperature,

        **get_default(icl_and_prompt_template, {}),
        **(
            {
                Experiment.Gold: lambda: dict(
                    dataset_cartography_student=dataset_cartography_student,
                ),
                Experiment.FewGen: lambda: dict(
                    fewgen_max_tokens=dataset_name.max_num_tokens(),  ## Max number of output  tokens
                    dataset_cartography_student=dataset_cartography_student,
                    text_gens_parser=text_gens_parser,
                    filter_params=filter_params,
                ),
                Experiment.SynthesizRR: lambda: dict(
                    synthesizrr_max_tokens=dataset_name.max_num_tokens(),  ## Max number of output  tokens
                    corpus=corpus,
                    corpus_raw_text_dir=corpus.raw_text_dir(),
                    retriever=retriever,
                    icl_type=icl_type,
                    retrieval_top_k=retrieval_top_k,
                    retr_icl_top_ks=retr_icl_top_ks,
                    retr_icl_distance_range=retr_icl_distance_range,
                    retr_icl_token_range=get_default(retr_icl_token_range, corpus.context_token_range()),
                    synthesizrr_top_k_range=get_default(synthesizrr_top_k_range, corpus.synthesizrr_top_k_range()),
                    synthesizrr_distance_range=synthesizrr_distance_range,
                    dataset_cartography_text=dataset_cartography_text,
                    dataset_cartography_student=dataset_cartography_student,
                    text_gens_parser=text_gens_parser,
                    filter_params=filter_params,
                ),
            }[expt]()
        ),

        llm_batch_size=llm_batch_size,
        llm_submission_batch_size=llm_submission_batch_size,
        llm_tracking_batch_size=llm_tracking_batch_size,
        llm_resources_per_model=llm_resources_per_model,
        llm_num_models=llm_num_models,
        llm_num_concurrent_preds=llm_num_concurrent_preds,
        llm_load_balancing_strategy=llm_load_balancing_strategy,
        llm_evaluation_timeout=llm_evaluation_timeout,

        metrics_to_evaluate=get_default(metrics_to_evaluate, []),
        metrics_num_samples_per_label=metrics_num_samples_per_label,
        metrics_label_distribution=metrics_label_distribution,
        metrics_calc_overall=metrics_calc_overall,
        metrics_calc_labels=metrics_calc_labels,
        metrics_max_parallel=metrics_max_parallel,
        metrics_override_row_count=metrics_override_row_count,
    )

    wf: Chain = get_wf(expt, metrics=metrics_to_evaluate is not None)
    print(set(wf.all_step_inputs(required_only=True)) - set(exn_input.keys()))
    if dry_run:
        return exn_input
    else:
        exn = wf.run(
            **exn_input,
            notifier=notifier,
            tracker=tracker,
            verbosity=verbosity,
            background=background,
            step_wait=step_wait,
        )
        time.sleep(pause)
        return exn
