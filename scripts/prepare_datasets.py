from plastinka_sales_predictor.data_preparation import (
    PlastinkaTrainingTSDataset,
    MultiColumnLabelBinarizer,
    GlobalLogMinMaxScaler,
    get_in_stock_conf,
    get_stock_history,
    get_starting_stocks,
    categorize_prices,
    get_monthly_sales_pivot,
    setup_dataset
)
from plastinka_sales_predictor import configure_logger
import pandas as pd
from typing_extensions import Optional, Iterable
import click
from datetime import timedelta
from warnings import filterwarnings
filterwarnings('ignore')


logger = configure_logger(child_logger_name='prepare_datasets')

DEFAULT_DATA_PATH = 'datasets/raw_data/sales'
DEFAULT_STOCKS_PATH = 'datasets/raw_data/stocks.xlsx'
DEFAULT_OUTPUT_DIR = 'datasets/'
DEFAULT_CUTOFF_DATE = '30-09-2022'
DEFAULT_STATIC_FEATURES = [
    'Конверт',
    'Тип',
    'Ценовая категория',
    'Стиль',
    'Год записи',
    'Год выпуска'
]
DEFAULT_PAST_COVARIATES_FNAMES = [
    'Тип',
    'Конверт',
    'Стиль',
    'Ценовая категория'
]


@click.command()
@click.option('--data_path', type=str, default=None)
@click.option('--stocks_path', type=str, default=None)
@click.option('--output_dir', type=str, default=None)
@click.option('--cutoff_date_lower', type=str, default=None)
@click.option('--cutoff_date_upper', type=str, default=None)
@click.option('--prices_bins', type=str, default=None)
def prepare_datasets(
    data_path: Optional[str] = None,
    stocks_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    cutoff_date_lower: Optional[str] = None,
    cutoff_date_upper: Optional[str] = None,
    prices_bins: Optional[Iterable[float]] = None,
) -> tuple[PlastinkaTrainingTSDataset, PlastinkaTrainingTSDataset]:
    if data_path is None:
        data_path = DEFAULT_DATA_PATH
    if stocks_path is None:
        stocks_path = DEFAULT_STOCKS_PATH
    if cutoff_date_lower is None:
        cutoff_date_lower = DEFAULT_CUTOFF_DATE
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    logger.info('Preprocessing data...')
    stocks = pd.read_excel(stocks_path, dtype='str')
    if prices_bins is None:
        _, prices_bins = categorize_prices(stocks, q=6)

    processed_stocks = get_starting_stocks(
        stocks,
        cutoff_date_lower,
        data_path,
        prices_bins
    )
    sales_history = get_stock_history(
        data_path,
        prices_bins,
        cutoff_date_lower
    )
    stocks_conf = get_in_stock_conf(processed_stocks, sales_history)

    if cutoff_date_upper is None:
        latest_date = sales_history.index.get_level_values('_date').max()
        if (latest_date + timedelta(days=1)).month == latest_date.month:
            cutoff_date_upper = latest_date.replace(day=1)
        else:
            cutoff_date_upper = latest_date + timedelta(days=1)
        
        cutoff_date_upper = cutoff_date_upper.strftime('%d-%m-%Y')

    rounded_sales = sales_history[
        sales_history.index.get_level_values('_date') <
        pd.to_datetime(cutoff_date_upper, dayfirst=True)
    ]
    rounded_stocks = stocks_conf[
        stocks_conf.index <
        pd.to_datetime(cutoff_date_upper, dayfirst=True)
    ]

    sales_pivot = get_monthly_sales_pivot(rounded_sales)

    static_transformer = MultiColumnLabelBinarizer()
    scaler = GlobalLogMinMaxScaler()
    input_chunk_length = sales_pivot.shape[0] - 1
    output_chunk_length = 1

    logger.info('Creating full dataset...')
    dataset = PlastinkaTrainingTSDataset(
        stocks=rounded_stocks,
        monthly_sales=sales_pivot,
        static_transformer=static_transformer,
        static_features=DEFAULT_STATIC_FEATURES,
        scaler=scaler,
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        save_dir=output_dir,
        dataset_name='full',
        past_covariates_span=14,
        past_covariates_fnames=DEFAULT_PAST_COVARIATES_FNAMES
    )

    logger.info('Creating train dataset...')
    L = dataset.L
    train_ds = setup_dataset(
        dataset,
        input_chunk_length=input_chunk_length - 1,
        output_chunk_length=output_chunk_length,
        window=(0, L - 1)
    )
    train_ds.dataset_name = 'train'
    train_ds.save(save_dir=output_dir)

if __name__ == '__main__':
    prepare_datasets()
