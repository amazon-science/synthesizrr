from typing import *

from fmcore.framework import Tracker

from synthesizrr.common import (
    RESULTS_DIR,
    Corpus,
    DatasetName,
    Experiment,
    ModelName,
    Retriever,
)
from synthesizrr.driver import run_chain

TRACKER = Tracker.of('log', path='~/synthesizrr_run.log')  ## Execution outputs will get logged to this file.
TRACKER = Tracker.of(
    "log", path="~/synthesizrr_run.log"
)  ## Execution outputs will get logged to this file.
    "log", path="~/synthesizrr_run.log"
)  ## Execution outputs will get logged to this file.
BACKGROUND: bool = False
CART_FRAC: Optional[float] = 0.83  ## Make None to Cartography filtering.

if __name__ == "__main__":
    """
      _  _                                         _    _                 
     | || | _  _  _ __  ___  _ _  _ __  __ _  _ _ | |_ (_) ___ __ _  _ _  
     | __ || || || '_ \/ -_)| '_|| '_ \/ _` || '_||  _|| |(_-</ _` || ' \ 
     |_||_| \_, || .__/\___||_|  | .__/\__,_||_|   \__||_|/__/\__,_||_||_|
            |__/ |_|             |_|                                      
    """
    FEWGEN_NUM_SHOTS_LIST = [0, 32]
    if (
        "fewgen_hyperpartisan_news_llama_2_13b_chat_exn" not in globals()
        or fewgen_hyperpartisan_news_llama_2_13b_chat_exn.status is Status.FAILED
    ):
        fewgen_hyperpartisan_news_llama_2_13b_chat_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.FewGen,
            dataset_name=DatasetName.HyperpartisanNews,
            model_name=ModelName.LLaMa_2_13B_Chat,
            num_shots_list=FEWGEN_NUM_SHOTS_LIST,
            num_samples_per_label=3_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            llm_num_models=48,
            metrics_overall_num_samples_per_label=2_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=1,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    SYNTHESIZRR_NUM_SHOTS_LIST = [0, 3]
    if (
        "synthesizrr_retr_icl_hyperpartisan_news_llama_2_13b_chat_exn" not in globals()
        or synthesizrr_retr_icl_hyperpartisan_news_llama_2_13b_chat_exn.status
        is Status.FAILED
    ):
        synthesizrr_retr_icl_hyperpartisan_news_llama_2_13b_chat_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.SynthesizRR,
            dataset_name=DatasetName.HyperpartisanNews,
            model_name=ModelName.LLaMa_2_13B_Chat,
            num_shots_list=SYNTHESIZRR_NUM_SHOTS_LIST,
            corpus=Corpus.RealNews_Dominant,
            retriever=Retriever.Contriever,
            num_samples_per_label=3_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            llm_batch_size=1,
            llm_submission_batch_size=12,
            llm_num_models=48,
            llm_num_concurrent_preds=2,
            metrics_overall_num_samples_per_label=2_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=1,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    SYNTHESIZRR_NUM_SHOTS_LIST = [32]
    if (
        "synthesizrr_no_retr_icl_hyperpartisan_news_llama_2_13b_chat_exn"
        not in globals()
        or synthesizrr_no_retr_icl_hyperpartisan_news_llama_2_13b_chat_exn.status
        is Status.FAILED
    ):
        synthesizrr_no_retr_icl_hyperpartisan_news_llama_2_13b_chat_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.SynthesizRR,
            dataset_name=DatasetName.HyperpartisanNews,
            model_name=ModelName.LLaMa_2_13B_Chat,
            num_shots_list=SYNTHESIZRR_NUM_SHOTS_LIST,
            corpus=Corpus.RealNews_Dominant,
            retriever=Retriever.Contriever,
            num_samples_per_label=3_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            icl_type="seed",
            llm_batch_size=1,
            llm_submission_batch_size=12,
            llm_num_models=48,
            llm_num_concurrent_preds=2,
            metrics_overall_num_samples_per_label=2_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            icl_and_prompt_template=dict(
                icl_template="""
Rewritten Article: 
{{icl[example_text]}}""",
                prompt_template="""
{{icl_examples}}

News Article:
{{retrieved_context}}

Rewrite the above news article {label_verbalization}. The rewritten article should be 2 to 3 paragraphs long.
Rewritten Article: """.strip()
                + "\n",
            ),
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=1,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    FEWGEN_NUM_SHOTS_LIST = [0, 32]
    if (
        "fewgen_hyperpartisan_news_claude_instant_v1_exn" not in globals()
        or fewgen_hyperpartisan_news_claude_instant_v1_exn.status is Status.FAILED
    ):
        fewgen_hyperpartisan_news_claude_instant_v1_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.FewGen,
            dataset_name=DatasetName.HyperpartisanNews,
            model_name=ModelName.Claude_Instant_v1,
            num_shots_list=FEWGEN_NUM_SHOTS_LIST,
            num_samples_per_label=3_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            llm_batch_size=1,
            llm_submission_batch_size=12,
            llm_num_models=1,
            llm_num_concurrent_preds=6,
            metrics_overall_num_samples_per_label=2_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=1,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    SYNTHESIZRR_NUM_SHOTS_LIST = [0, 3]
    if (
        "synthesizrr_retr_icl_hyperpartisan_news_claude_instant_v1_exn" not in globals()
        or synthesizrr_retr_icl_hyperpartisan_news_claude_instant_v1_exn.status
        is Status.FAILED
    ):
        synthesizrr_retr_icl_hyperpartisan_news_claude_instant_v1_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.SynthesizRR,
            dataset_name=DatasetName.HyperpartisanNews,
            model_name=ModelName.Claude_Instant_v1,
            num_shots_list=SYNTHESIZRR_NUM_SHOTS_LIST,
            corpus=Corpus.RealNews_Dominant,
            retriever=Retriever.Contriever,
            num_samples_per_label=3_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            llm_batch_size=1,
            llm_submission_batch_size=12,
            llm_num_models=1,
            llm_num_concurrent_preds=6,
            metrics_overall_num_samples_per_label=2_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=1,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    SYNTHESIZRR_NUM_SHOTS_LIST = [32]
    if (
        "synthesizrr_no_retr_icl_hyperpartisan_news_claude_instant_v1_exn"
        not in globals()
        or synthesizrr_no_retr_icl_hyperpartisan_news_claude_instant_v1_exn.status
        is Status.FAILED
    ):
        synthesizrr_no_retr_icl_hyperpartisan_news_claude_instant_v1_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.SynthesizRR,
            dataset_name=DatasetName.HyperpartisanNews,
            model_name=ModelName.Claude_Instant_v1,
            num_shots_list=SYNTHESIZRR_NUM_SHOTS_LIST,
            corpus=Corpus.RealNews_Dominant,
            retriever=Retriever.Contriever,
            num_samples_per_label=3_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            icl_type="seed",
            llm_batch_size=1,
            llm_submission_batch_size=12,
            llm_num_models=1,
            llm_num_concurrent_preds=6,
            metrics_overall_num_samples_per_label=2_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            icl_and_prompt_template=dict(
                icl_template="""
Rewritten Article by Assistant:
{{icl[example_text]}}""".strip()
                + "\n",
                prompt_template="""
Human:
{{icl_examples}}

News Article:
{{retrieved_context}}

Rewrite the above news article {label_verbalization}. The rewritten article should be 2 to 3 paragraphs long.
Rewritten Article by Assistant: """.strip()
                + "\n",
            ),
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=1,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    """
        _    ___   _  _                  
       /_\  / __| | \| | ___ __ __ __ ___
      / _ \| (_ | | .` |/ -_)\ V  V /(_-<
     /_/ \_\\___| |_|\_|\___| \_/\_/ /__/
    """

    FEWGEN_NUM_SHOTS_LIST = [0, 32]
    if (
        "fewgen_ag_news_llama_2_13b_chat_exn" not in globals()
        or fewgen_ag_news_llama_2_13b_chat_exn.status is Status.FAILED
    ):
        fewgen_ag_news_llama_2_13b_chat_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.FewGen,
            dataset_name=DatasetName.AgNews,
            model_name=ModelName.LLaMa_2_13B_Chat,
            num_shots_list=FEWGEN_NUM_SHOTS_LIST,
            num_samples_per_label=3_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            llm_num_models=48,
            metrics_overall_num_samples_per_label=8_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=1,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    SYNTHESIZRR_NUM_SHOTS_LIST = [0, 3]
    if (
        "synthesizrr_retr_icl_ag_news_llama_2_13b_chat_exn" not in globals()
        or synthesizrr_retr_icl_ag_news_llama_2_13b_chat_exn.status is Status.FAILED
    ):
        synthesizrr_retr_icl_ag_news_llama_2_13b_chat_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.SynthesizRR,
            dataset_name=DatasetName.AgNews,
            model_name=ModelName.LLaMa_2_13B_Chat,
            num_shots_list=SYNTHESIZRR_NUM_SHOTS_LIST,
            corpus=Corpus.RealNews_Dominant,
            retriever=Retriever.Contriever,
            num_samples_per_label=3_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            llm_num_models=48,
            metrics_overall_num_samples_per_label=8_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=1,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    SYNTHESIZRR_NUM_SHOTS_LIST = [32]
    if (
        "synthesizrr_no_retr_icl_ag_news_llama_2_13b_chat_exn" not in globals()
        or synthesizrr_no_retr_icl_ag_news_llama_2_13b_chat_exn.status is Status.FAILED
    ):
        synthesizrr_no_retr_icl_ag_news_llama_2_13b_chat_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.SynthesizRR,
            dataset_name=DatasetName.AgNews,
            model_name=ModelName.LLaMa_2_13B_Chat,
            num_shots_list=SYNTHESIZRR_NUM_SHOTS_LIST,
            corpus=Corpus.RealNews_Dominant,
            retriever=Retriever.Contriever,
            num_samples_per_label=3_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            icl_type="seed",
            llm_batch_size=1,
            llm_submission_batch_size=24,
            llm_num_models=48,
            llm_num_concurrent_preds=4,
            metrics_overall_num_samples_per_label=8_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            icl_and_prompt_template=dict(
                icl_template="""
    Summary: {{icl[example_text]}}""",
                prompt_template="""
    {{icl_examples}}

    News Article:
    {{retrieved_context}}

    Write a summary for the above news article {label_verbalization}. The summary should be one or two short sentences.
    Summary: """.strip()
                + " ",
            ),
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=1,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    FEWGEN_NUM_SHOTS_LIST = [0, 32]
    if (
        "fewgen_ag_news_claude_instant_v1_exn" not in globals()
        or fewgen_ag_news_claude_instant_v1_exn.status is Status.FAILED
    ):
        fewgen_ag_news_claude_instant_v1_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.FewGen,
            dataset_name=DatasetName.AgNews,
            model_name=ModelName.Claude_Instant_v1,
            num_shots_list=FEWGEN_NUM_SHOTS_LIST,
            num_samples_per_label=3_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            llm_batch_size=1,
            llm_submission_batch_size=12,
            llm_num_models=1,
            llm_num_concurrent_preds=6,
            metrics_overall_num_samples_per_label=8_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=1,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    SYNTHESIZRR_NUM_SHOTS_LIST = [32]
    if (
        "synthesizrr_no_retr_icl_ag_news_claude_instant_v1_exn" not in globals()
        or synthesizrr_no_retr_icl_ag_news_claude_instant_v1_exn.status is Status.FAILED
    ):
        synthesizrr_no_retr_icl_ag_news_claude_instant_v1_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.SynthesizRR,
            dataset_name=DatasetName.AgNews,
            model_name=ModelName.Claude_Instant_v1,
            num_shots_list=SYNTHESIZRR_NUM_SHOTS_LIST,
            corpus=Corpus.RealNews_Dominant,
            retriever=Retriever.Contriever,
            num_samples_per_label=3_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            icl_type="seed",
            llm_batch_size=1,
            llm_submission_batch_size=12,
            llm_num_models=1,
            llm_num_concurrent_preds=6,
            metrics_overall_num_samples_per_label=8_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            icl_and_prompt_template=dict(
                icl_template="""
Summary by Assistant: {{icl[example_text]}}""".strip()
                + " ",
                prompt_template="""
Human:
{{icl_examples}}

News Article:
{{retrieved_context}}

Write a summary for the above news article {label_verbalization}. The summary should be one or two short sentences.
Summary by Assistant: """.strip()
                + " ",
            ),
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=1,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    SYNTHESIZRR_NUM_SHOTS_LIST = [0, 3]
    if (
        "synthesizrr_retr_icl_ag_news_claude_instant_v1_exn" not in globals()
        or synthesizrr_retr_icl_ag_news_claude_instant_v1_exn.status is Status.FAILED
    ):
        synthesizrr_retr_icl_ag_news_claude_instant_v1_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.SynthesizRR,
            dataset_name=DatasetName.AgNews,
            model_name=ModelName.Claude_Instant_v1,
            num_shots_list=SYNTHESIZRR_NUM_SHOTS_LIST,
            corpus=Corpus.RealNews_Dominant,
            retriever=Retriever.Contriever,
            num_samples_per_label=3_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            llm_batch_size=1,
            llm_submission_batch_size=12,
            llm_num_models=1,
            llm_num_concurrent_preds=8,
            metrics_overall_num_samples_per_label=8_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=1,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    """
      _____      ___   _  _                _  _  _                
     |_   _|___ |_ _| | || | ___  __ _  __| || |(_) _ _   ___  ___
       | | / _ \ | |  | __ |/ -_)/ _` |/ _` || || || ' \ / -_)(_-<
       |_| \___/|___| |_||_|\___|\__,_|\__,_||_||_||_||_|\___|/__/
    """

    FEWGEN_NUM_SHOTS_LIST = [0, 32]
    if (
        "fewgen_toi_headlines_llama_2_13b_chat_exn" not in globals()
        or fewgen_toi_headlines_llama_2_13b_chat_exn.status is Status.FAILED
    ):
        fewgen_toi_headlines_llama_2_13b_chat_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.FewGen,
            dataset_name=DatasetName.ToiHeadlines,
            model_name=ModelName.LLaMa_2_13B_Chat,
            num_shots_list=FEWGEN_NUM_SHOTS_LIST,
            num_samples_per_label=2_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            llm_batch_size=1,
            llm_submission_batch_size=12,
            llm_num_models=48,
            llm_num_concurrent_preds=20,
            metrics_overall_num_samples_per_label=8_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=2,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    SYNTHESIZRR_NUM_SHOTS_LIST = [0, 3]
    if (
        "synthesizrr_retr_icl_toi_headlines_llama_2_13b_chat_exn" not in globals()
        or synthesizrr_retr_icl_toi_headlines_llama_2_13b_chat_exn.status
        is Status.FAILED
    ):
        synthesizrr_retr_icl_toi_headlines_llama_2_13b_chat_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.SynthesizRR,
            dataset_name=DatasetName.ToiHeadlines,
            model_name=ModelName.LLaMa_2_13B_Chat,
            num_shots_list=SYNTHESIZRR_NUM_SHOTS_LIST,
            corpus=Corpus.RealNews_India,
            retriever=Retriever.Contriever,
            num_samples_per_label=2_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            llm_batch_size=1,
            llm_submission_batch_size=20,
            llm_tracking_batch_size=100,
            llm_num_models=48,
            llm_num_concurrent_preds=20,
            metrics_overall_num_samples_per_label=8_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=2,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    SYNTHESIZRR_NUM_SHOTS_LIST = [32]
    if (
        "synthesizrr_no_retr_icl_toi_headlines_llama_2_13b_chat_exn" not in globals()
        or synthesizrr_no_retr_icl_toi_headlines_llama_2_13b_chat_exn.status
        is Status.FAILED
    ):
        synthesizrr_no_retr_icl_toi_headlines_llama_2_13b_chat_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.SynthesizRR,
            dataset_name=DatasetName.ToiHeadlines,
            model_name=ModelName.LLaMa_2_13B_Chat,
            num_shots_list=SYNTHESIZRR_NUM_SHOTS_LIST,
            corpus=Corpus.RealNews_India,
            retriever=Retriever.Contriever,
            num_samples_per_label=2_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            icl_type="seed",
            llm_batch_size=1,
            llm_submission_batch_size=16,
            llm_num_models=48,
            llm_num_concurrent_preds=10,
            metrics_overall_num_samples_per_label=8_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            icl_and_prompt_template=dict(
                icl_template="""
Headline: {{icl[example_text]}}""",
                prompt_template="""
{{icl_examples}}

News Article:
{{retrieved_context}}

Write a headline for the above news article about {label_verbalization}. The headline should be a single sentence.
Headline: """.strip()
                + " ",
            ),
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=2,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    FEWGEN_NUM_SHOTS_LIST = [0, 32]
    if (
        "fewgen_toi_headlines_claude_instant_v1_exn" not in globals()
        or fewgen_toi_headlines_claude_instant_v1_exn.status is Status.FAILED
    ):
        fewgen_toi_headlines_claude_instant_v1_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.FewGen,
            dataset_name=DatasetName.ToiHeadlines,
            model_name=ModelName.Claude_Instant_v1,
            num_shots_list=FEWGEN_NUM_SHOTS_LIST,
            num_samples_per_label=2_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            llm_batch_size=1,
            llm_submission_batch_size=12,
            llm_num_models=1,
            llm_num_concurrent_preds=10,
            metrics_overall_num_samples_per_label=8_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=1,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    SYNTHESIZRR_NUM_SHOTS_LIST = [32]
    if (
        "synthesizrr_no_retr_icl_toi_headlines_claude_instant_v1_exn" not in globals()
        or synthesizrr_no_retr_icl_toi_headlines_claude_instant_v1_exn.status
        is Status.FAILED
    ):
        synthesizrr_no_retr_icl_toi_headlines_claude_instant_v1_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.SynthesizRR,
            dataset_name=DatasetName.ToiHeadlines,
            model_name=ModelName.Claude_Instant_v1,
            num_shots_list=SYNTHESIZRR_NUM_SHOTS_LIST,
            corpus=Corpus.RealNews_India,
            retriever=Retriever.Contriever,
            num_samples_per_label=2_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            icl_type="seed",
            llm_batch_size=1,
            llm_submission_batch_size=12,
            llm_num_models=1,
            llm_num_concurrent_preds=6,
            metrics_overall_num_samples_per_label=8_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            icl_and_prompt_template=dict(
                icl_template="""
Headline by Assistant: {{icl[example_text]}}""".strip()
                + " ",
                prompt_template="""
Human:
{{icl_examples}}

News Article:
{{retrieved_context}}

Write a headline for the above news article about {label_verbalization}. The headline should be a single sentence.
Headline by Assistant: """.strip()
                + " ",
            ),
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=2,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    SYNTHESIZRR_NUM_SHOTS_LIST = [0, 3]
    if (
        "synthesizrr_retr_icl_toi_headlines_claude_instant_v1_exn" not in globals()
        or synthesizrr_retr_icl_toi_headlines_claude_instant_v1_exn.status
        is Status.FAILED
    ):
        synthesizrr_retr_icl_toi_headlines_claude_instant_v1_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.SynthesizRR,
            dataset_name=DatasetName.ToiHeadlines,
            model_name=ModelName.Claude_Instant_v1,
            num_shots_list=SYNTHESIZRR_NUM_SHOTS_LIST,
            corpus=Corpus.RealNews_India,
            retriever=Retriever.Contriever,
            num_samples_per_label=2_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            llm_batch_size=1,
            llm_submission_batch_size=12,
            llm_num_models=1,
            llm_num_concurrent_preds=10,
            metrics_overall_num_samples_per_label=8_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=2,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    """
       ___        _                              
      / __| __ _ | |_  ___  __ _  ___  _ _  _  _ 
     | (__ / _` ||  _|/ -_)/ _` |/ _ \| '_|| || |
      \___|\__,_| \__|\___|\__, |\___/|_|   \_, |
                           |___/            |__/ 
    """
    FEWGEN_NUM_SHOTS_LIST = [0, 32]
    if (
        "fewgen_amazon_reviews_category_llama_2_13b_chat_exn" not in globals()
        or fewgen_amazon_reviews_category_llama_2_13b_chat_exn.status is Status.FAILED
    ):
        fewgen_amazon_reviews_category_llama_2_13b_chat_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.FewGen,
            dataset_name=DatasetName.AmazonReviewsProductCategory,
            model_name=ModelName.LLaMa_2_13B_Chat,
            num_shots_list=FEWGEN_NUM_SHOTS_LIST,
            num_samples_per_label=1_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            llm_num_models=48,
            metrics_overall_num_samples_per_label=8_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=1,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    SYNTHESIZRR_NUM_SHOTS_LIST = [32]
    if (
        "synthesizrr_no_retr_icl_amazon_reviews_category_llama_2_13b_chat_exn"
        not in globals()
        or synthesizrr_no_retr_icl_amazon_reviews_category_llama_2_13b_chat_exn.status
        is Status.FAILED
    ):
        synthesizrr_no_retr_icl_amazon_reviews_category_llama_2_13b_chat_exn = (
            run_chain(
                results_dir=RESULTS_DIR,
                expt=Experiment.SynthesizRR,
                dataset_name=DatasetName.AmazonReviewsProductCategory,
                model_name=ModelName.LLaMa_2_13B_Chat,
                num_shots_list=SYNTHESIZRR_NUM_SHOTS_LIST,
                corpus=Corpus.AmazonProducts,
                retriever=Retriever.Contriever,
                num_samples_per_label=1_000,
                seed_type="train_set",
                seed_set_stratify_on_ground_truth=False,
                icl_type="seed",
                llm_batch_size=1,
                llm_submission_batch_size=12,
                llm_num_models=48,
                llm_num_concurrent_preds=23,
                metrics_overall_num_samples_per_label=8_000,
                metrics_max_parallel=3,
                metrics_label_distribution="train_set",
                # metrics_to_evaluate=None,
                icl_and_prompt_template=dict(
                    icl_template="""
Review: {{icl[example_text]}}""".strip()
                    + " ",
                    prompt_template="""
{{icl_examples}}

Product details:
{{retrieved_context}}

Write a product review about the above product which is in the category of {label_verbalization}. Include relevant product details which are mentioned above. The review should only be a single short sentence, or a single paragraph of 3 to 4 sentences. Add very minor typos.
Review: """.strip()
                    + " ",
                ),
                tracker=TRACKER,
                background=BACKGROUND,
                verbosity=1,
                step_wait=5,
                cart_frac=CART_FRAC,
                # dry_run=True,
            )
        )

    SYNTHESIZRR_NUM_SHOTS_LIST = [0, 3]
    if (
        "synthesizrr_retr_icl_amazon_reviews_category_llama_2_13b_chat_exn"
        not in globals()
        or synthesizrr_retr_icl_amazon_reviews_category_llama_2_13b_chat_exn.status
        is Status.FAILED
    ):
        synthesizrr_retr_icl_amazon_reviews_category_llama_2_13b_chat_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.SynthesizRR,
            dataset_name=DatasetName.AmazonReviewsProductCategory,
            model_name=ModelName.LLaMa_2_13B_Chat,
            num_shots_list=SYNTHESIZRR_NUM_SHOTS_LIST,
            corpus=Corpus.AmazonProducts,
            retriever=Retriever.Contriever,
            num_samples_per_label=1_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            llm_num_models=48,
            metrics_overall_num_samples_per_label=8_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=1,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    FEWGEN_NUM_SHOTS_LIST = [0, 32]
    if (
        "fewgen_amazon_reviews_category_claude_instant_v1_exn" not in globals()
        or fewgen_amazon_reviews_category_claude_instant_v1_exn.status is Status.FAILED
    ):
        fewgen_amazon_reviews_category_claude_instant_v1_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.FewGen,
            dataset_name=DatasetName.AmazonReviewsProductCategory,
            model_name=ModelName.Claude_Instant_v1,
            num_shots_list=FEWGEN_NUM_SHOTS_LIST,
            num_samples_per_label=1_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            llm_batch_size=1,
            llm_submission_batch_size=12,
            llm_num_models=1,
            llm_num_concurrent_preds=6,
            metrics_overall_num_samples_per_label=8_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=1,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    SYNTHESIZRR_NUM_SHOTS_LIST = [32]
    if (
        "synthesizrr_no_retr_icl_amazon_reviews_category_claude_instant_v1_exn"
        not in globals()
        or synthesizrr_no_retr_icl_amazon_reviews_category_claude_instant_v1_exn.status
        is Status.FAILED
    ):
        synthesizrr_no_retr_icl_amazon_reviews_category_claude_instant_v1_exn = (
            run_chain(
                results_dir=RESULTS_DIR,
                expt=Experiment.SynthesizRR,
                dataset_name=DatasetName.AmazonReviewsProductCategory,
                model_name=ModelName.Claude_Instant_v1,
                num_shots_list=SYNTHESIZRR_NUM_SHOTS_LIST,
                corpus=Corpus.AmazonProducts,
                retriever=Retriever.Contriever,
                num_samples_per_label=1_000,
                seed_type="train_set",
                seed_set_stratify_on_ground_truth=False,
                icl_type="seed",
                llm_batch_size=1,
                llm_submission_batch_size=12,
                llm_num_models=1,
                llm_num_concurrent_preds=6,
                metrics_overall_num_samples_per_label=8_000,
                metrics_max_parallel=3,
                metrics_label_distribution="train_set",
                # metrics_to_evaluate=None,
                icl_and_prompt_template=dict(
                    icl_template="""
Review by Assistant: {{icl[example_text]}}""".strip()
                    + " ",
                    prompt_template="""
Human:
{{icl_examples}}

Product details:
{{retrieved_context}}

Write a product review about the above product which is in the category of {label_verbalization}. Include relevant product details which are mentioned above. The review should only be a single short sentence, or a single paragraph of 3 to 4 sentences. Add very minor typos.
Review by Assistant: """.strip()
                    + " ",
                ),
                tracker=TRACKER,
                background=BACKGROUND,
                verbosity=1,
                step_wait=5,
                cart_frac=CART_FRAC,
                # dry_run=True,
            )
        )

    SYNTHESIZRR_NUM_SHOTS_LIST = [0, 3]
    if (
        "synthesizrr_retr_icl_amazon_reviews_category_claude_instant_v1_exn"
        not in globals()
        or synthesizrr_retr_icl_amazon_reviews_category_claude_instant_v1_exn.status
        is Status.FAILED
    ):
        synthesizrr_retr_icl_amazon_reviews_category_claude_instant_v1_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.SynthesizRR,
            dataset_name=DatasetName.AmazonReviewsProductCategory,
            model_name=ModelName.Claude_Instant_v1,
            num_shots_list=SYNTHESIZRR_NUM_SHOTS_LIST,
            corpus=Corpus.AmazonProducts,
            retriever=Retriever.Contriever,
            num_samples_per_label=1_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            llm_batch_size=1,
            llm_submission_batch_size=12,
            llm_num_models=1,
            llm_num_concurrent_preds=6,
            metrics_overall_num_samples_per_label=8_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=1,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    """
      _  _                        
     | || | _  _  _ __   ___  _ _ 
     | __ || || || '  \ / _ \| '_|
     |_||_| \_,_||_|_|_|\___/|_|  
    """

    FEWGEN_NUM_SHOTS_LIST = [0, 32]
    if (
        "fewgen_amazon_humor_llama_2_13b_chat_exn" not in globals()
        or fewgen_amazon_humor_llama_2_13b_chat_exn.status is Status.FAILED
    ):
        fewgen_amazon_humor_llama_2_13b_chat_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.FewGen,
            dataset_name=DatasetName.AmazonHumorousProductQuestions,
            model_name=ModelName.LLaMa_2_13B_Chat,
            num_shots_list=FEWGEN_NUM_SHOTS_LIST,
            num_samples_per_label=5_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            llm_num_models=48,
            metrics_overall_num_samples_per_label=2_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=1,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    SYNTHESIZRR_NUM_SHOTS_LIST = [32]
    if (
        "synthesizrr_no_retr_icl_amazon_humor_llama_2_13b_chat_exn" not in globals()
        or synthesizrr_no_retr_icl_amazon_humor_llama_2_13b_chat_exn.status
        is Status.FAILED
    ):
        synthesizrr_no_retr_icl_amazon_humor_llama_2_13b_chat_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.SynthesizRR,
            dataset_name=DatasetName.AmazonHumorousProductQuestions,
            model_name=ModelName.LLaMa_2_13B_Chat,
            num_shots_list=SYNTHESIZRR_NUM_SHOTS_LIST,
            corpus=Corpus.AmazonProducts,
            retriever=Retriever.Contriever,
            num_samples_per_label=5_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            icl_type="seed",
            llm_batch_size=1,
            llm_submission_batch_size=12,
            llm_num_models=48,
            llm_num_concurrent_preds=2,
            metrics_overall_num_samples_per_label=2_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            icl_and_prompt_template=dict(
                icl_template="""
Product Question: {{icl[example_text]}}""".strip()
                + " ",
                prompt_template="""
{{icl_examples}}

Product details:
{{retrieved_context}}

Write a short {label_verbalization} question about the above product on Amazon. Only include the question.
Product Question: """.strip()
                + " ",
            ),
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=1,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    SYNTHESIZRR_NUM_SHOTS_LIST = [0, 3]
    if (
        "synthesizrr_retr_icl_amazon_humor_llama_2_13b_chat_exn" not in globals()
        or synthesizrr_retr_icl_amazon_humor_llama_2_13b_chat_exn.status
        is Status.FAILED
    ):
        synthesizrr_retr_icl_amazon_humor_llama_2_13b_chat_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.SynthesizRR,
            dataset_name=DatasetName.AmazonHumorousProductQuestions,
            model_name=ModelName.LLaMa_2_13B_Chat,
            num_shots_list=SYNTHESIZRR_NUM_SHOTS_LIST,
            corpus=Corpus.AmazonProducts,
            retriever=Retriever.Contriever,
            num_samples_per_label=5_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            llm_num_models=48,
            metrics_overall_num_samples_per_label=2_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=1,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    FEWGEN_NUM_SHOTS_LIST = [0, 32]
    if (
        "fewgen_amazon_humor_claude_instant_v1_exn" not in globals()
        or fewgen_amazon_humor_claude_instant_v1_exn.status is Status.FAILED
    ):
        fewgen_amazon_humor_claude_instant_v1_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.FewGen,
            dataset_name=DatasetName.AmazonHumorousProductQuestions,
            model_name=ModelName.Claude_Instant_v1,
            num_shots_list=FEWGEN_NUM_SHOTS_LIST,
            num_samples_per_label=5_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            llm_batch_size=1,
            llm_submission_batch_size=12,
            llm_num_models=1,
            llm_num_concurrent_preds=6,
            metrics_overall_num_samples_per_label=2_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=1,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    SYNTHESIZRR_NUM_SHOTS_LIST = [32]
    if (
        "synthesizrr_no_retr_icl_amazon_humor_claude_instant_v1_exn" not in globals()
        or synthesizrr_no_retr_icl_amazon_humor_claude_instant_v1_exn.status
        is Status.FAILED
    ):
        synthesizrr_no_retr_icl_amazon_humor_claude_instant_v1_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.SynthesizRR,
            dataset_name=DatasetName.AmazonHumorousProductQuestions,
            model_name=ModelName.Claude_Instant_v1,
            num_shots_list=SYNTHESIZRR_NUM_SHOTS_LIST,
            corpus=Corpus.AmazonProducts,
            retriever=Retriever.Contriever,
            num_samples_per_label=5_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            icl_type="seed",
            llm_batch_size=1,
            llm_submission_batch_size=12,
            llm_num_models=1,
            llm_num_concurrent_preds=6,
            metrics_overall_num_samples_per_label=2_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            icl_and_prompt_template=dict(
                icl_template="""
Product Question by Assistant: {{icl[example_text]}}""".strip()
                + " ",
                prompt_template="""
Human:
{{icl_examples}}

Product details:
{{retrieved_context}}

Write a short {label_verbalization} question about the above product on Amazon. Only include the question.
Product Question by Assistant: """.strip()
                + " ",
            ),
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=1,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    SYNTHESIZRR_NUM_SHOTS_LIST = [0, 3]
    if (
        "synthesizrr_retr_icl_amazon_humor_claude_instant_v1_exn" not in globals()
        or synthesizrr_retr_icl_amazon_humor_claude_instant_v1_exn.status
        is Status.FAILED
    ):
        synthesizrr_retr_icl_amazon_humor_claude_instant_v1_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.SynthesizRR,
            dataset_name=DatasetName.AmazonHumorousProductQuestions,
            model_name=ModelName.Claude_Instant_v1,
            num_shots_list=SYNTHESIZRR_NUM_SHOTS_LIST,
            corpus=Corpus.AmazonProducts,
            retriever=Retriever.Contriever,
            num_samples_per_label=5_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            llm_batch_size=1,
            llm_submission_batch_size=12,
            llm_num_models=1,
            llm_num_concurrent_preds=6,
            metrics_overall_num_samples_per_label=2_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=1,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    """
      ___       _             _  _         
     | _ \ ___ | | __ _  _ _ (_)| |_  _  _ 
     |  _// _ \| |/ _` || '_|| ||  _|| || |
     |_|  \___/|_|\__,_||_|  |_| \__| \_, |
                                      |__/ 
    """

    FEWGEN_NUM_SHOTS_LIST = [0, 32]
    if (
        "fewgen_amazon_polarity_llama_2_13b_chat_exn" not in globals()
        or fewgen_amazon_polarity_llama_2_13b_chat_exn.status is Status.FAILED
    ):
        fewgen_amazon_polarity_llama_2_13b_chat_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.FewGen,
            dataset_name=DatasetName.AmazonReviewsPolarity,
            model_name=ModelName.LLaMa_2_13B_Chat,
            num_shots_list=FEWGEN_NUM_SHOTS_LIST,
            num_samples_per_label=5_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            llm_num_models=48,
            metrics_overall_num_samples_per_label=4_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=1,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    SYNTHESIZRR_NUM_SHOTS_LIST = [32]
    if (
        "synthesizrr_no_retr_icl_amazon_polarity_llama_2_13b_chat_exn" not in globals()
        or synthesizrr_no_retr_icl_amazon_polarity_llama_2_13b_chat_exn.status
        is Status.FAILED
    ):
        synthesizrr_no_retr_icl_amazon_polarity_llama_2_13b_chat_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.SynthesizRR,
            dataset_name=DatasetName.AmazonReviewsPolarity,
            model_name=ModelName.LLaMa_2_13B_Chat,
            num_shots_list=SYNTHESIZRR_NUM_SHOTS_LIST,
            corpus=Corpus.AmazonProducts,
            retriever=Retriever.Contriever,
            num_samples_per_label=5_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            icl_type="seed",
            llm_batch_size=1,
            llm_submission_batch_size=12,
            llm_num_models=48,
            llm_num_concurrent_preds=2,
            metrics_overall_num_samples_per_label=4_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            icl_and_prompt_template=dict(
                icl_template="""
Review: {{icl[example_text]}}""".strip()
                + " ",
                prompt_template="""
{{icl_examples}}

Product details:
{{retrieved_context}}

Write a review about the above product on Amazon which discusses {label_verbalization}. Include relevant product details which are mentioned above. The review should only be a single short sentence, or a single paragraph of 3 to 4 sentences. Add very minor typos.
Review: """.strip()
                + " ",
            ),
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=1,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    SYNTHESIZRR_NUM_SHOTS_LIST = [0, 3]
    if (
        "synthesizrr_retr_icl_amazon_polarity_llama_2_13b_chat_exn" not in globals()
        or synthesizrr_retr_icl_amazon_polarity_llama_2_13b_chat_exn.status
        is Status.FAILED
    ):
        synthesizrr_retr_icl_amazon_polarity_llama_2_13b_chat_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.SynthesizRR,
            dataset_name=DatasetName.AmazonReviewsPolarity,
            model_name=ModelName.LLaMa_2_13B_Chat,
            num_shots_list=SYNTHESIZRR_NUM_SHOTS_LIST,
            corpus=Corpus.AmazonProducts,
            retriever=Retriever.Contriever,
            num_samples_per_label=5_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            llm_num_models=48,
            metrics_overall_num_samples_per_label=4_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=1,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    FEWGEN_NUM_SHOTS_LIST = [0, 32]
    if (
        "fewgen_amazon_polarity_claude_instant_v1_exn" not in globals()
        or fewgen_amazon_polarity_claude_instant_v1_exn.status is Status.FAILED
    ):
        fewgen_amazon_polarity_claude_instant_v1_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.FewGen,
            dataset_name=DatasetName.AmazonReviewsPolarity,
            model_name=ModelName.Claude_Instant_v1,
            num_shots_list=FEWGEN_NUM_SHOTS_LIST,
            num_samples_per_label=5_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            llm_batch_size=1,
            llm_submission_batch_size=12,
            llm_num_models=1,
            llm_num_concurrent_preds=6,
            metrics_overall_num_samples_per_label=4_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=1,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    SYNTHESIZRR_NUM_SHOTS_LIST = [32]
    if (
        "synthesizrr_no_retr_icl_amazon_polarity_claude_instant_v1_exn" not in globals()
        or synthesizrr_no_retr_icl_amazon_polarity_claude_instant_v1_exn.status
        is Status.FAILED
    ):
        synthesizrr_no_retr_icl_amazon_polarity_claude_instant_v1_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.SynthesizRR,
            dataset_name=DatasetName.AmazonReviewsPolarity,
            model_name=ModelName.Claude_Instant_v1,
            num_shots_list=SYNTHESIZRR_NUM_SHOTS_LIST,
            corpus=Corpus.AmazonProducts,
            retriever=Retriever.Contriever,
            num_samples_per_label=5_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            icl_type="seed",
            llm_batch_size=1,
            llm_submission_batch_size=12,
            llm_num_models=1,
            llm_num_concurrent_preds=6,
            metrics_overall_num_samples_per_label=4_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            icl_and_prompt_template=dict(
                icl_template="""
Review by Assistant: {{icl[example_text]}}""".strip()
                + " ",
                prompt_template="""
Human:
{{icl_examples}}

Product details:
{{retrieved_context}}

Write a review about the above product on Amazon which discusses {label_verbalization}. Include relevant product details which are mentioned above. The review should only be a single short sentence, or a single paragraph of 3 to 4 sentences. Add very minor typos.
Review by Assistant: """.strip()
                + " ",
            ),
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=1,
            step_wait=5,
            cart_frac=CART_FRAC,
            # dry_run=True,
        )

    SYNTHESIZRR_NUM_SHOTS_LIST = [0, 3]
    if (
        "synthesizrr_retr_icl_amazon_polarity_claude_instant_v1_exn" not in globals()
        or synthesizrr_retr_icl_amazon_polarity_claude_instant_v1_exn.status
        is Status.FAILED
    ):
        synthesizrr_retr_icl_amazon_polarity_claude_instant_v1_exn = run_chain(
            results_dir=RESULTS_DIR,
            expt=Experiment.SynthesizRR,
            dataset_name=DatasetName.AmazonReviewsPolarity,
            model_name=ModelName.Claude_Instant_v1,
            num_shots_list=SYNTHESIZRR_NUM_SHOTS_LIST,
            corpus=Corpus.AmazonProducts,
            retriever=Retriever.Contriever,
            num_samples_per_label=5_000,
            seed_type="train_set",
            seed_set_stratify_on_ground_truth=False,
            llm_batch_size=1,
            llm_submission_batch_size=12,
            llm_num_models=1,
            llm_num_concurrent_preds=6,
            metrics_overall_num_samples_per_label=4_000,
            metrics_max_parallel=3,
            metrics_label_distribution="train_set",
            # metrics_to_evaluate=None,
            tracker=TRACKER,
            background=BACKGROUND,
            verbosity=1,
            step_wait=5,
            cart_frac=CART_FRAC,
        )
