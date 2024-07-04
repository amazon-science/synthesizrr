from typing import *
import math, pandas as pd, io, orjson, gc
from synthesizrr.base.util import Timer, accumulate, as_list, get_default, whitespace_normalize, run_parallel_ray, \
    ProgressBar, run_concurrent, StringUtil, wait, is_null, as_tuple, shuffle_items
from synthesizrr.base.data import FileMetadata, Reader, to_sdf
from synthesizrr.base.constants import Status, DataLayout
from bs4 import BeautifulSoup as BS

CORPUS_DIR: str = ''  # TODO: fill this out! Recommended to use S3.


def create_amazon_products():
    corpus_dir: FileMetadata = FileMetadata.of(
        f'{CORPUS_DIR}/data/amazon-reviews/2018/meta/'
    )
    corpus_dir.mkdir()
    if len(corpus_dir.list()) == 0:
        raise SystemError(
            f'Expected Amazon Products metadata to be in folder "{corpus_dir.path}". '
            f'Please download the data file All_Amazon_Meta.json.gz and unzip to this directory '
            f'(to get this data you need to submit the form at https://nijianmo.github.io/amazon/index.html#complete-data)'
        )

    source: str = corpus_dir.file_in_dir('All_Amazon_Meta.json')
    all_dfs: List[pd.DataFrame] = []
    buf = []
    df_pbar = ProgressBar.of(unit='file')
    row_pbar = ProgressBar.of(unit='rows', miniters=10_000)
    with io.open(source, 'rb') as inp:
        for line in inp:
            buf.append(orjson.loads(line))
            row_pbar.update(1)
            if len(buf) == 100_000:
                all_dfs.append(pd.DataFrame(buf))
                df_pbar.update(1)
                buf = []
        row_pbar.success()
    all_dfs.append(pd.DataFrame(buf))
    df_pbar.update(1)
    # futs.append(run_concurrent(
    #     write_df,
    #     df=all_dfs[-1],
    #     n=df_pbar.pbar.n,
    # ))
    buf = []
    gc.collect()

    # fpaths = accumulate(futs, progress=dict(desc='Writing'))

    def _convert_row(row):
        if not is_null(row['category']):
            row['category'] = as_tuple(row['category'])
        if not is_null(row['description']):
            row['description'] = as_tuple(row['description'])
        if not is_null(row['also_buy']):
            row['also_buy'] = as_tuple(row['also_buy'])
        if not is_null(row['image']):
            row['image'] = as_tuple(row['image'])
        if not is_null(row['feature']):
            row['feature'] = as_tuple(row['feature'])
        if not is_null(row['also_view']):
            row['also_view'] = as_tuple(row['also_view'])
        if not is_null(row['rank']):
            row['rank'] = as_tuple(row['rank'])
        if is_null(row['details']):
            row['details'] = {}
        return row

    def _convert_df(df_part):
        return to_sdf(df_part).to_layout(DataLayout.LIST_OF_DICT).apply(_convert_row, axis=1).pandas()

    corpus_split_dir: FileMetadata = corpus_dir.subdir_in_dir('split', return_metadata=True)
    futs = []
    for df_part_i, df_part in enumerate(all_dfs):
        df_part = _convert_df(df_part)
        dest: str = corpus_split_dir.file_in_dir(f'amazon-reviews-2018-meta-part-{StringUtil.pad_zeros(df_part_i)}.parquet')
        futs.append(run_concurrent(
            df_part.to_parquet,
            dest,
        ))
        print(df_part_i)
    accumulate(futs, progress=dict(desc='Writing', unit='file'))

    with Timer():
        prods = Reader.of(
            'parquet',
            data_schema={
                "asin": 'object',
                # "also_buy": 'object',
                # "also_view": 'object',
                "title": 'object',
                "description": 'object',
                "brand": 'object',
                "category": 'object',
                "date": 'object',
                # "details": 'object',
                "feature": 'object',
                "fit": 'object',
                # "image": 'object',
                "main_cat": 'object',
                "price": 'object',
                # "rank": 'object',
                # "similar_item": 'object',
                # "tech1": 'object',
                # "tech2": 'object',
            }
        ).read(
            corpus_split_dir,
            read_as=DataLayout.PANDAS,
        )
        prods = prods.drop_duplicates('asin').persist(wait=True)

    corpus_raw_text_dir: FileMetadata = corpus_dir.subdir_in_dir('raw-text', return_metadata=True)
    with Timer():
        def create_product_text(row):
            product_text: str = ''
            for col in ['title', 'description']:
                for text in as_list(row[col]):
                    product_text += f"<item>{get_default(text, '')}</item><br>"
            product_text: str = BS(product_text).get_text(separator="\n")
            for i in range(1, 10):
                product_text: str = product_text.replace(f'{i}.', f'{i}. ')
            product_text: str = whitespace_normalize(product_text)
            return product_text

        def set_product_text(_prods_part):
            _prods_part['product_text'] = _prods_part.apply(create_product_text, axis=1)
            return _prods_part

        prods_list = []
        for prods_part_i, prods_part in ProgressBar.iter(
                enumerate(prods.stream(stream_as=DataLayout.PANDAS, batch_size=10_000)),
                total=math.ceil(len(prods) / 10_000)
        ):
            prods_list.append(
                run_parallel_ray(set_product_text, prods_part)
            )
        prods_list: List[pd.DataFrame] = accumulate(prods_list, progress=True)

        futs = []
        for prods_part_i, prods_part in ProgressBar.iter(
                enumerate(prods_list),
                total=len(prods_list),
        ):
            prods_part = prods_part.reset_index(drop=True)
            prods_part['asin'] = prods_part['asin'].astype(str)
            fpath: str = corpus_raw_text_dir.file_in_dir(
                f'amazon-products-2018-raw-text-part-{StringUtil.pad_zeros(prods_part_i)}.parquet'
            )
            futs.append(run_concurrent(
                prods_part.to_parquet,
                fpath
            ))
        wait(futs, progress=True)
        print(f'Done creating Amazon Products corpus, final data is at: "{corpus_raw_text_dir.path}"')

        def amazon_products_count_num_tokens(df_path):
            df_part = Reader.of(
                'parquet',
                data_schema={
                    'asin': 'index',
                    'title': 'text',
                    'description': 'text',
                }
            ).read(df_path, raw=True)
            ser_part = df_part['title'].fillna('').astype(str) + ' ' + df_part['description'].apply(
                lambda x: ' '.join(x)).astype(str)
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('TheBloke/Llama-2-13B-fp16')
            sum_is: int = sum(
                [len(input_ids) for input_ids in tokenizer(ser_part.tolist(), add_special_tokens=False)['input_ids']]
            )
            return sum_is

        counts: List[int] = accumulate([
            run_parallel_ray(
                amazon_products_count_num_tokens,
                df_path=df_path,
            )
            for df_path in FileMetadata.of(
                corpus_raw_text_dir.path,
                file_glob='*.parquet',
            ).list()
        ], progress=True)
        print(f'Amazon Products corpus has {round(sum(counts) / 1e9, 2)} billion tokens')


REALNEWS_REGIONAL_NEWS_DOMAINS: List[str] = [
    "mid-day.com", "financialexpress.com", "thenationonlineng.net", "livemint.com", "hindustantimes.com", "vanguardngr.com",
    "capitalfm.co.ke", "straitstimes.com", "indianexpress.com", "nation.com.pk", "jamaica-gleaner.com", "trend.az",
    "stabroeknews.com", "dawn.com", "emirates247.com", "mangalorean.com", "vccircle.com", "thisdaylive.com", "gulfnews.com",
    "tribune.com.pk", "arabnews.com", "pakobserver.net", "nation.co.ke", "eurasiareview.com", "thedailystar.net",
    "deccanchronicle.com", "jewishpress.com", "app.com.pk", "err.ee", "lankabusinessonline.com", "koreatimes.co.kr",
    "newera.com.na", "ticotimes.net", "codewit.com", "sunnewsonline.com", "afaqs.com", "ameinfo.com", "malaysiakini.com",
    "ynetnews.com", "palestinechronicle.com", "zmescience.com", "cyprus-mail.com", "colombiareports.com",
    "arabtimesonline.com", "bollywoodhungama.com", "pattayamail.com", "insightcrime.org", "medianewsline.com",
    "dailytimes.com.pk", "chinadigitaltimes.net", "saudigazette.com.sa", "newsday.co.zw", "sunstar.com.ph",
    "nehandaradio.com", "freemalaysiatoday.com", "onlanka.com", "thezimbabwemail.com", "theeastafrican.co.ke",
    "thecitizen.co.tz", "lusakatimes.com", "orissadiary.com", "aljazeera.com", "tehrantimes.com", "theborneopost.com",
    "morungexpress.com", "monitor.co.ug", "countercurrents.org", "businessworld.in", "governancenow.com", "itweb.co.za",
    "972mag.com", "memeburn.com", "themediaonline.co.za", "koimoi.com", "caribbean360.com", "yalibnan.com",
    "milligazette.com", "thefrontierpost.com", "kuwaittimes.net", "somalilandpress.com", "thestkittsnevisobserver.com",
    "news24.com", "livinginperu.com", "journal.com.ph", "bworldonline.com", "venezuelanalysis.com", "businessdayonline.com",
    "macaudailytimes.com.mo", "ghanabusinessnews.com", "trinidadexpress.com", "pmnewsnigeria.com", "lankanewspapers.com",
    "asiasentinel.com", "maravipost.com", "dayafterindia.com", "defense-update.com", "antiguaobserver.com", "newsbytes.ph",
    "truthdive.com", "thehimalayantimes.com", "standardmedia.co.ke", "groundviews.org", "japantoday.com", "kbc.co.ke",
    "mindanews.com", "thejakartaglobe.com", "actionforex.com", "modernghana.com", "newstodaynet.com",
    "centralchronicle.com", "dalje.com", "escambray.cu", "middle-east-online.com", "theminaretonline.com",
    "pakistankakhudahafiz.com", "meed.com", "tribwekchron.com", "thenews.com.pk", "iafrica.com", "philstar.com",
    "praguepost.com", "yonhapnews.co.kr", "china.org.cn", "rtn.asia", "nationalturk.com", "thebraziltimes.com",
    "businessdailyafrica.com", "hku.hk", "intifada-palestine.com", "realbollywood.com", "pak1stanfirst.com", "mutiny.in",
    "mareeg.com", "paltelegraph.com", "pakwatan.com", "mybroadband.co.za", "african-bulletin.com", "thedailynewsegypt.com",
    "7days.ae", "dailyforex.com", "melodika.net"
]

REALNEWS_INDIAN_NEWS_DOMAINS: List[str] = [
    "mid-day.com", "financialexpress.com", "livemint.com", "hindustantimes.com", "indianexpress.com", "mangalorean.com",
    "vccircle.com", "deccanchronicle.com", "afaqs.com", "bollywoodhungama.com", "medianewsline.com", "orissadiary.com",
    "morungexpress.com", "countercurrents.org", "businessworld.in", "governancenow.com", "koimoi.com", "milligazette.com",
    "dayafterindia.com", "truthdive.com", "newstodaynet.com", "centralchronicle.com", "dalje.com", "rtn.asia",
    "realbollywood.com", "mutiny.in"
]


def count_num_tokens(ser_part):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('TheBloke/Llama-2-13B-fp16')
    sum_is: int = sum([len(input_ids) for input_ids in tokenizer(ser_part.tolist(), add_special_tokens=False)['input_ids']])
    return sum_is


def write_realnews_partition(rn_data, *, corpus_dir: FileMetadata, rn_name: str):
    with Timer(f'Writing partition {rn_name}'):
        corpus_partition_dir: FileMetadata = corpus_dir.subdir_in_dir(rn_name, return_metadata=True)
        print(f'{rn_name} length: {len(rn_data)}')
        futs = []
        for rn_part_i, rn_part in ProgressBar.iter(
                enumerate(rn_data.stream(stream_as=DataLayout.PANDAS, batch_size=100_000)),
                total=math.ceil(len(rn_data) / 100_000)
        ):
            rn_part = rn_part.reset_index(drop=True)
            fpath: str = corpus_partition_dir.file_in_dir(
                f'realnews-{rn_name}-part-{StringUtil.pad_zeros(rn_part_i)}.parquet'
            )
            futs.append(run_concurrent(
                rn_part.to_parquet,
                fpath
            ))
        wait(futs, progress=True)
        print(f'Done creating {rn_name} partition, final data is at: "{corpus_partition_dir.path}"')

        counts: pd.Series = rn_data['text'].map_partitions(count_num_tokens).compute()
        print(f'{rn_name} corpus has {round(counts.sum() / 1e9, 2)} billion tokens')


def create_realnews():
    corpus_dir: FileMetadata = FileMetadata.of(
        f'{CORPUS_DIR}/data/realnews/'
    )
    corpus_dir.mkdir()
    if len(corpus_dir.list()) == 0:
        raise SystemError(
            f'Expected RealNews to be in folder "{corpus_dir.path}". '
            f'Please download the data file realnews.jsonl to this directory '
            f'(to get this data you need to submit the form at https://github.com/rowanz/grover/tree/master/realnews)'
        )

    with Timer('Reading and splitting realnews.jsonl'):
        source: str = corpus_dir.file_in_dir('realnews.jsonl')
        all_dfs: List[pd.DataFrame] = []
        buf = []
        df_pbar = ProgressBar.of(unit='file')
        row_pbar = ProgressBar.of(unit='rows', miniters=10_000)
        row_idx = 0
        with io.open(source, 'rb') as inp:
            for line in inp:
                buf.append(orjson.loads(line))
                buf[-1]['idx'] = row_idx
                row_idx += 1
                row_pbar.update(1)
                if len(buf) == 100_000:
                    all_dfs.append(pd.DataFrame(buf))
                    df_pbar.update(1)
                    buf = []
            row_pbar.success()
        all_dfs.append(pd.DataFrame(buf))
        df_pbar.update(1)
        buf = []
        gc.collect()

    corpus_split_dir: FileMetadata = corpus_dir.subdir_in_dir('split', return_metadata=True)
    futs = []
    for df_part_i, df_part in enumerate(all_dfs):
        dest: str = corpus_split_dir.file_in_dir(f'realnews-part-{StringUtil.pad_zeros(df_part_i)}.parquet')
        futs.append(run_concurrent(
            df_part.to_parquet,
            dest,
        ))
        print(df_part_i)
    accumulate(futs, progress=dict(desc='Writing', unit='file'))

    with Timer('Reading split files'):
        realnews_data_schema: Dict = {
            'idx': 'index',
            'title': 'object',
            'text': 'text',
            'summary': 'object',
            'authors': 'categorical',
            'publish_date': 'object',
            'status': 'categorical',
            'url': 'categorical',
            'domain': 'categorical',
            'warc_date': 'object',
            'split': 'categorical',
        }
        realnews = Reader.of(
            'parquet',
            data_schema=realnews_data_schema,
        ).read(
            FileMetadata.of(
                corpus_split_dir.path,
                file_format='parquet',
            ),
            read_as=DataLayout.DASK,
        )

    realnews_india = realnews.query(f'domain in {REALNEWS_INDIAN_NEWS_DOMAINS}').persist(wait=True)
    write_realnews_partition(realnews_india, corpus_dir=corpus_dir, rn_name='realnews-india')

    realnews_regional = realnews.query(f'domain in {REALNEWS_REGIONAL_NEWS_DOMAINS}').persist(wait=True)
    write_realnews_partition(realnews_regional, corpus_dir=corpus_dir, rn_name='realnews-regional')

    realnews_dominant = realnews.query(f'domain not in {REALNEWS_REGIONAL_NEWS_DOMAINS}').persist(wait=True)
    write_realnews_partition(realnews_dominant, corpus_dir=corpus_dir, rn_name='realnews-dominant')


def create_cmu_movies():
    corpus_dir: FileMetadata = FileMetadata.of(
        f'{CORPUS_DIR}/data/cmu_movies/'
    )
    corpus_dir.mkdir()
    if len(corpus_dir.list()) == 0:
        raise SystemError(
            f'Expected CMU Movies to be in folder "{corpus_dir.path}". '
            f'Please download the data from https://www.cs.cmu.edu/~ark/personas/ and extract it. '
            f'You should get the folder "MovieSummaries".'
        )
    with Timer('Reading and merging plot_summaries.txt and movie.metadata.tsv'):
        movie_plots: pd.DataFrame = pd.read_csv(
            corpus_dir.subdir_in_dir('MovieSummaries', return_metadata=True).file_in_dir('plot_summaries.txt'),
            sep='\t',
            header=None,
            names=[
                'wiki_movie_id',
                'plot_summary',
            ]
        )
        movie_meta: pd.DataFrame = pd.read_csv(
            corpus_dir.subdir_in_dir('MovieSummaries', return_metadata=True).file_in_dir('movie.metadata.tsv'),
            sep='\t',
            header=None,
            names=[
                'wiki_movie_id',
                'freebase_movie_id',
                'title',
                'release_date',
                'box_office_revenue',
                'runtime',
                'languages',
                'countries',
                'genres'
            ]
        )
        movies: pd.DataFrame = movie_meta.merge(
            movie_plots, on='wiki_movie_id'
        ).reset_index(drop=True).rename(
            columns=dict(plot_summary='text', wiki_movie_id='idx')
        )
        corpus_raw_text_dir: FileMetadata = corpus_dir.subdir_in_dir('raw-text', return_metadata=True)
        movies.to_parquet(corpus_raw_text_dir.file_in_dir(f'cmu-movie-summary.parquet'))
        print(f'Done creating CMU Moveis corpus, final data is at: "{corpus_raw_text_dir.path}"')

        def cmu_movies_count_num_tokens(df_path):
            df_part = Reader.of(
                'parquet',
                data_schema={
                    'wiki_movie_id': 'index',
                    'plot_summary': 'text',
                }
            ).read(df_path, raw=True)
            ser_part = df_part['plot_summary']
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('TheBloke/Llama-2-13B-fp16')
            sum_is: int = sum(
                [len(input_ids) for input_ids in tokenizer(ser_part.tolist(), add_special_tokens=False)['input_ids']]
            )
            return sum_is

        counts: List[int] = accumulate([
            run_parallel_ray(
                cmu_movies_count_num_tokens,
                df_path=df_path,
            )
            for df_path in FileMetadata.of(
                corpus_raw_text_dir.path,
                file_glob='*.parquet',
            ).list()
        ], progress=True)
        print(f'CMU Movies corpus has {round(sum(counts) / 1e6, 2)} million tokens')


if __name__ == '__main__':
    create_amazon_products()
    gc.collect()
    create_realnews()
    gc.collect()
    create_cmu_movies()
    gc.collect()
