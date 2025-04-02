from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder, minmax_scale
from pathlib import Path
from typing_extensions import Callable, Optional, Union, Sequence, Self
from darts.timeseries import TimeSeries
from darts.utils.data.training_dataset import MixedCovariatesTrainingDataset
from abc import ABC, abstractmethod
import dill
from collections import defaultdict
from datetime import datetime


COLTYPES = {
    'Штрихкод': str,
    'Barcode': str,
    'Экземпляры': 'int64',
    'Ценовая категория': 'str',
    'Конверт': str,
    'Альбом': str,
    'Исполнитель': str,
    'Год записи': str,
    'Год выпуска': str,
    'Тип': str,
    'Стиль': str,
    'Дата создания': 'datetime64[ns]',
    'Дата продажи': 'datetime64[ns]',
    'Дата добавления': 'datetime64[ns]',
    'precise_record_year': 'int64'
}

GROUP_KEYS = (
    'Штрихкод',
    'Исполнитель',
    'Альбом',
    'Конверт',
    'Ценовая категория',
    'Тип',
    'Год записи',
    'Год выпуска',
    'Стиль',
    'precise_record_year'
)


class PlastinkaBaseTSDataset(ABC):
    def __init__(
            self,
            stocks: pd.DataFrame,
            monthly_sales: pd.DataFrame,
            static_transformer: BaseEstimator = None,
            static_features: Optional[Sequence[str]] = None,
            scaler: BaseEstimator = None,
            weight_coef: float = 0.,
            input_chunk_length: int = 12,
            output_chunk_length: int = 1,
            start: int = 0,
            end: Optional[int] = None,
            past_covariates_fnames: Sequence[str] = ('Тип', 'Конверт', 'Стиль', 'Ценовая категория'),
            past_covariates_span: int = 3,
            save_dir: Optional[str] = None, dataset_name: Optional[str] = None,
            dtype: Union[str, np.dtype] = np.float32
    ):
        self.stocks = stocks
        self._monthly_sales = monthly_sales
        self._n_time_steps = self._monthly_sales.shape[0]
        if end is None:
            end = self._n_time_steps
        self.end = end
        self.start = start
        self._past_covariates_fnames = past_covariates_fnames
        self.setup_dataset(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            span=past_covariates_span,
            weights_alpha=weight_coef,
            scaler=scaler,
            copy=False
        )

        self._index_names_mapping = {
            n: i for i, n in enumerate(self._monthly_sales.columns.names)
        }
        self.static_transformer = static_transformer
        self.static_features = static_features
        self.static_covariates_mapping = self.get_static_covariates()
        self.ds = self.prepare_dataset()
        self._idx_mapping = self._build_index_mapping()
        self.save_dir = save_dir
        self.dataset_name = dataset_name
        self.return_ts = None
        self.dtype = dtype

        if save_dir is not None:
            self.save(save_dir, dataset_name)

    @classmethod
    def from_dill(cls, dill_path: Union[str, Path]):
        with open(dill_path, 'rb') as f:
            return dill.load(f)

    def save(
            self, 
            save_dir: Union[str, Path], 
            dataset_name: Optional[str] = None
    ):
        if dataset_name is None:
            if not self.dataset_name:
                dataset_name = self.__class__.__name__
            else:
                dataset_name = self.dataset_name

        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        with open(save_dir / f'{dataset_name}.dill', 'wb') as f:
            dill.dump(self, f)

    def prepare_dataset(self):
        ds = defaultdict(list)

        for item_multiidx in self._monthly_sales:
            self._process_sample(ds, item_multiidx)

        return ds

    @abstractmethod
    def _process_sample(self, ds, item_multiidx):
        raise NotImplementedError

    def prepare_past_covariates(self, span=None):
        if span is not None:
            self._past_covariates_span = span

        self._past_covariates_cached = {
            feat: get_past_covariates_df(
                self._monthly_sales,
                (feat,),
                self._past_covariates_span
            ) for feat in self._past_covariates_fnames
        }

    def get_static_covariates(self):
        if self.static_transformer is None:
            return None

        unique_items = self._monthly_sales.columns.to_frame().drop_duplicates(
            ignore_index=True
        )
        full_index = unique_items.copy()
        if self.static_features is not None:
            unique_items = unique_items[self.static_features]

        if not self.static_transformer.is_fit():
            self.static_transformer = self.static_transformer.fit(unique_items)

        static_covariates = self.static_transformer.transform(unique_items)

        if not isinstance(static_covariates, pd.DataFrame):
            static_covariates = pd.DataFrame(
                static_covariates,
                columns=self.static_transformer.get_feature_names(),
                index=pd.MultiIndex.from_frame(
                    full_index
                )
            )

        else:
            static_covariates = static_covariates.set_index(
                pd.MultiIndex.from_frame(full_index)
            )

        static_covariates = static_covariates.astype('float32')

        return static_covariates

    def get_future_covariates(self, item_multiidx):  
        item_sales = self._monthly_sales[item_multiidx]
        is_hot = item_sales.index.map(lambda x: int(x.year)) == item_multiidx[-1]
        msin, mcos = transform_months(item_sales.index.map(lambda x: x.month))
        year = minmax_scale(item_sales.index.map(lambda x: x.year))

        return np.vstack([is_hot, msin, mcos, year]).T

    def get_past_covariates(self, item_multiidx):
        past_covariates = []
        release_year = item_multiidx[self._index_names_mapping['Год записи']]
        for feat_name in self._past_covariates_cached:
            feat_vals = item_multiidx[self._index_names_mapping[feat_name]]
            feat_df = self._past_covariates_cached[feat_name].loc[
                pd.IndexSlice[feat_vals, release_year, :]
            ]
            past_covariates.append(feat_df)

        past_covariates.append(
            self._monthly_sales[item_multiidx].ewm(
                span=self._past_covariates_span, adjust=False
            ).mean()
        )

        past_cov_arr = np.vstack(past_covariates)
        if self.scaler:
            past_cov_arr = self.scaler.transform(past_cov_arr)

        past_cov_arr = np.vstack(
            [past_cov_arr, self.stocks.loc[
                :, pd.IndexSlice[(slice(None),) + tuple(item_multiidx)]
            ].values.T]
        ).T

        return past_cov_arr

    def set_length(self, input_chunk_length, output_chunk_length):
        if all(
            [
                input_chunk_length > 1,
                output_chunk_length > 0,
                ((self.end - self.start) >= 
                 (input_chunk_length + output_chunk_length))
            ]
        ):
            self.input_chunk_length = input_chunk_length
            self.output_chunk_length = output_chunk_length

        else:
            raise ValueError(
                f'Invalid length: input_chunk_length={input_chunk_length}, '
                f'output_chunk_length={output_chunk_length}. '
            )

    def set_window(self, start, end):
        if end is None:
            end = self._n_time_steps
        length = self.input_chunk_length + self.output_chunk_length
        if all(
            [
                start >= 0,
                end <= self._n_time_steps,
                (end - start) >= length
            ]
        ):
            self.start = start
            self.end = end

        else:
            raise ValueError(
                f'Invalid window: start={start}, end={end}, length={length} '
                'Length must not be greater than the window size'
            )

    def set_scaler(self, scaler):
        if not scaler.is_fit():
            scaler = scaler.fit(self.monthly_sales)
        self.scaler = scaler

    def setup_dataset(
            self,
            window: tuple[int, int] | None = None,
            input_chunk_length: int | None = None,
            output_chunk_length: int = None,
            span: int | None = None,
            weights_alpha: float | None = None,
            scaler: BaseEstimator | None = None,
            copy: bool = True,
    ) -> Optional[Self]:
        if copy:
            ds = deepcopy(self)
        else:
            ds = self

        if input_chunk_length and output_chunk_length:
            ds.set_length(input_chunk_length, output_chunk_length)

        if window:
            ds.set_window(*window)

        if scaler:
            ds.set_scaler(scaler)

        if weights_alpha is not None:
            ds.set_reweight_fn(weights_alpha)

        if span:
            ds.prepare_past_covariates(span)

        if copy:
            return ds

    def _build_index_mapping(self):
        current_index = 0
        self._index_mapping = {}
        for i in range(len(self)):
            if self._validate_sample(i):
                self._index_mapping[current_index] = i
                current_index += 1

    def set_reweight_fn(self, alpha):
        def reweight_fn(array):
            min_weight = 0.1
            if alpha == 0.:
                min_weight = 1.
            weights = min_weight + alpha * np.log1p(array)

            return weights

        self.reweight_fn = reweight_fn

    def unravel_dataset(
            self,
            prefix=''
    ):
        dataset_dict = defaultdict(list)
        self.return_ts = True
        for i in range(len(self)):
            (
                past_target,
                past_covariates,
                historic_future_covariates,
                future_covariates,
                _,
                _,
                future_target,
                ts
            ) = self[i]
            
            time_index = ts.time_index
            future_covariates = np.vstack(
                [historic_future_covariates, future_covariates]
            )
            series = np.vstack(
                [past_target, future_target]
            )
            future_covariates = TimeSeries.from_times_and_values(
                time_index,
                future_covariates
            )
            past_covariates = TimeSeries.from_times_and_values(
                time_index[:past_covariates.shape[0]], past_covariates
            )
            ts = TimeSeries.with_values(ts, series)
            
            if self._index_mapping:
                i = self._index_mapping[i]
                
            labels = self.ds['ts_names'][i // self.outputs_per_array]

            dataset_dict[f'{prefix}series'].append(ts)
            dataset_dict[f'{prefix}future_covariates'].append(future_covariates)
            dataset_dict[f'{prefix}past_covariates'].append(past_covariates)
            dataset_dict[f'{prefix}labels'].append(labels)
        
        self.return_ts = False
        return dataset_dict

    @property
    def monthly_sales(self):
        return self._monthly_sales[self.start:self.end]

    @property
    def outputs_per_array(self):
        return self.L - self.input_chunk_length - self.output_chunk_length + 1
    
    @property
    def L(self):
        return self.end - self.start
    
    @abstractmethod
    def _get_slice(self, original_array_index, start_index, end_index):
        raise NotImplementedError

    @abstractmethod
    def _validate_sample(self, index):
        return True

    def __len__(self):
        if self._index_mapping:
            return len(self._index_mapping)
        else:
            return len(self.ds['target_series']) * self.outputs_per_array

    def __getitem__(self, item):
        if self._index_mapping:
            item = self._index_mapping[item]
        length = self.input_chunk_length + self.output_chunk_length
        original_array_index = item // self.outputs_per_array
        start_index = self.start + item % self.outputs_per_array
        end_index = (start_index + length) if (start_index + length) < self._n_time_steps else None
        return self._get_slice(original_array_index, start_index, end_index)


class PlastinkaTrainingTSDataset(PlastinkaBaseTSDataset, MixedCovariatesTrainingDataset):
    def __init__(self,
                 stocks: pd.DataFrame,
                 monthly_sales: pd.DataFrame,
                 input_chunk_length: int = 12,
                 output_chunk_length: int = 1,
                 minimum_sales_months: int = 4,
                 static_transformer: BaseEstimator = None,
                 static_features: Optional[Sequence[str]] = None,
                 scaler: BaseEstimator = None,
                 weight_coef: float = 0.,
                 start: int = 0,
                 end: Optional[int] = None,
                 past_covariates_fnames: Sequence[str] = ('Конверт', 'Стиль', 'Ценовая категория'),
                 past_covariates_span: int = 3,
                 save_ts: bool = True,
                 save_dir: Optional[str] = None,
                 dataset_name: Optional[str] = None,
                 dtype: Union[str, np.dtype] = np.float32):
        MixedCovariatesTrainingDataset.__init__(self)

        self.minimum_sales_months = minimum_sales_months
        self.save_ts = save_ts

        PlastinkaBaseTSDataset.__init__(
            self, 
            stocks=stocks, 
            monthly_sales=monthly_sales,
            static_transformer=static_transformer, 
            static_features=static_features, 
            scaler=scaler,
            weight_coef=weight_coef, 
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length, 
            start=start, 
            end=end,
            past_covariates_fnames=past_covariates_fnames,
            past_covariates_span=past_covariates_span,
            save_dir=save_dir,
            dataset_name=dataset_name, 
            dtype=dtype
        )

    def _process_sample(self, ds, item_multiidx):
        item_sales = self._monthly_sales[item_multiidx]
        if item_sales.values.nonzero()[0].shape[0] >= self.minimum_sales_months:
            value_series = np.expand_dims(
                item_sales, 1
            )
            ds['ts_names'].append(item_multiidx)
            ds['target_series'].append(value_series)
            ds['time_index'].append(item_sales.index)

    def _get_slice(self, original_array_index, start_index, end_index):
        item = original_array_index
        item_multiidx = self.ds['ts_names'][item]
        series_item = self.ds['target_series'][item][
            start_index:end_index
        ].astype(self.dtype)
        future_covariates_item = self.get_future_covariates(item_multiidx)[
            start_index:end_index
        ].astype(self.dtype)
        past_covariates_item = self.get_past_covariates(item_multiidx)[
            start_index:end_index
        ].astype(self.dtype)
        static_covariates_item = self.static_covariates_mapping.loc[
            item_multiidx
        ].astype(self.dtype)

        if self.scaler is not None:
            series_item = self.scaler.transform(series_item)

        soft_availability = past_covariates_item[:, -1]
        output = [
            series_item[:-self.output_chunk_length],
            past_covariates_item[:-self.output_chunk_length],
            future_covariates_item[:-self.output_chunk_length],
            future_covariates_item[-self.output_chunk_length:],
            np.expand_dims(static_covariates_item.values, 1).T,
            self.reweight_fn(
                series_item[:-self.output_chunk_length] *
                soft_availability
            ),
            series_item[-self.output_chunk_length:]
        ]

        if self.return_ts:
            time_series = TimeSeries.from_times_and_values(
                self.ds['time_index'][item][start_index:end_index],
                series_item,
                static_covariates=static_covariates_item
            )
            output.append(time_series)

        return output

    def _validate_sample(self, index):
        item_multiidx = self.ds['ts_names'][
            index // self.outputs_per_array
        ]
        in_stock = self.stocks.loc[:, pd.IndexSlice[
            (slice(None),) + tuple(item_multiidx)
        ]].values
        in_stock = in_stock[-self.output_chunk_length:]
        return np.any(in_stock)


class MultiColumnLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, separator='/'):
        self.separator = separator
        self.encoders = {}

    def fit(self, X, y=None):
        for column in X.columns:
            mlb = MultiLabelBinarizer(sparse_output=True)
            mlb.fit(X[column].str.split(self.separator).map(lambda x: list(map(lambda y: y.strip(), x))))
            self.encoders[column] = mlb

        return self

    def transform(self, X, index=None):
        idx = list(X.columns)
        transformed_columns = []
        for column in idx:
            binarized = self.encoders[column].transform(
                X[column].str.split(self.separator).map(lambda x: list(map(lambda y: y.strip(), x)))
            )
            binarized_df = pd.DataFrame.sparse.from_spmatrix(
                binarized,
                columns=[f"{column}_{cls}" for cls in self.encoders[column].classes_]
            )
            transformed_columns.append(binarized_df)

        dummy_df = pd.concat([X, *transformed_columns], axis=1)

        if index is None:
            index = idx
        dummy_df = dummy_df.set_index(index)

        return dummy_df

    def is_fit(self):
        return len(self.encoders) > 0


class OrdinalEncoder(OrdinalEncoder):
    def __init__(self):
        super().__init__()

    def is_fit(self):
        return hasattr(self, 'n_features_in_')

    def get_feature_names(self):
        if self.is_fit():
            return self.feature_names_in_


class GlobalLogMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, feature_range=(0, 1)):
        # Задаем диапазон масштабирования
        self.feature_range = feature_range
        self.global_min_ = None
        self.global_max_ = None

    def fit(self, X, y=None):
        # Проверяем, что данные в формате DataFrame или массива numpy
        X = self._validate_data(X)

        # Вычисляем глобальный минимум и максимум по всем данным
        logx = np.log1p(X)
        self.global_min_ = np.min(logx)
        self.global_max_ = np.max(logx)
        return self

    def transform(self, X):
        # Проверяем, что трансформер был обучен
        if self.global_min_ is None or self.global_max_ is None:
            raise ValueError("This GlobalMinMaxScaler instance is not fitted yet. Call 'fit' before using this method.")

        X = self._validate_data(X)

        # Применяем глобальное масштабирование
        scale = self.feature_range[1] - self.feature_range[0]
        X = np.log1p(X)
        X_scaled = (X - self.global_min_) / (self.global_max_ - self.global_min_)
        X_scaled = X_scaled * scale + self.feature_range[0]

        return X_scaled + 1e-6

    def inverse_transform(self, X_scaled):
        # Обратное преобразование из масштабированного диапазона в исходный
        if self.global_min_ is None or self.global_max_ is None:
            raise ValueError("This GlobalMinMaxScaler instance is not fitted yet. Call 'fit' before using this method.")

        scale = self.feature_range[1] - self.feature_range[0]
        X_scaled = X_scaled - 1e-6
        X = (X_scaled - self.feature_range[0]) / scale
        X = X * (self.global_max_ - self.global_min_) + self.global_min_
        X = np.expm1(X)

        return X

    def _validate_data(self, X):
        # Приводим данные к DataFrame для совместимости с pandas или numpy
        if not isinstance(X, (pd.DataFrame, pd.Series, np.ndarray)):
            raise ValueError("Input should be a pandas DataFrame, Series or numpy array.")
        if not isinstance(X, np.ndarray):
            X = X.values
        return X

    def is_fit(self):
        return self.global_min_ is not None and self.global_max_ is not None


def validate_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    def fill_partial_years_np_where(series):
        try:
            is_not_20 = ~series.str.startswith('2')
            valid_str = series.str.extract('(^\d+)')[0]
            padding_9 = (4 - valid_str.str.len().astype(int)).map(lambda x: '9' * x)
            padding_1 = (4 - valid_str.str.len().astype(int)).map(lambda x: '1' * x)
            filled = np.where(is_not_20, valid_str + padding_9, valid_str + padding_1)
        except:
            pass
        return filled

    validated = df.copy()

    validated['Год записи'] = pd.to_numeric(validated['Год записи'], errors='coerce')
    validated = validated.dropna(subset=['Год записи'])

    uncert_rerelease_year_idxs = ~validated['Год выпуска'].fillna('1234').astype(str).str.match(r'^(\d{4}|[^0-9]*)$')
    unvalid_rereleases = validated.loc[uncert_rerelease_year_idxs, 'Год выпуска'].astype(str)
    validated.loc[uncert_rerelease_year_idxs, 'Год выпуска'] = fill_partial_years_np_where(unvalid_rereleases)
    validated['Год выпуска'] = pd.to_numeric(validated['Год выпуска'], errors='coerce')

    validated.loc[validated['Тип'] == 'Оригинал', 'Год выпуска'] = validated.loc[
        validated['Тип'] == 'Оригинал', 'Год записи']

    valid_rereleases = validated[(validated['Тип'] != 'Оригинал') & (validated['Год выпуска'].notna())]
    if len(valid_rereleases) > 10:
        mean_gap = (valid_rereleases['Год выпуска'] - valid_rereleases['Год записи']).mean().round().astype(
            int)
    else:
        mean_gap = 15

    rerelease_nans = (validated['Тип'] != 'Оригинал') & (validated['Год выпуска'].isna())
    validated.loc[rerelease_nans, 'Год выпуска'] = validated.loc[rerelease_nans, 'Год записи'] + mean_gap

    invalid_release_dates = validated['Год выпуска'] < validated['Год записи']
    validated.loc[invalid_release_dates, 'Год выпуска'] = validated.loc[invalid_release_dates, 'Год записи'] + mean_gap

    current_year = datetime.now().year
    validated['Год записи'] = validated['Год записи'].clip(1950, current_year)
    validated['Год выпуска'] = validated['Год выпуска'].clip(1950, current_year)

    return validated


def validate_categories(df: pd.DataFrame) -> pd.DataFrame:
    validated = df.copy()
    nan_idx = validated['Конверт'].isna()
    nan_df = validated[nan_idx]
    for gpouping_cols in [['Исполнитель', 'Альбом', 'Год выпуска'], ['Ценовая категория', 'Год выпуска']]:
        for gl, g_nan in nan_df.groupby(gpouping_cols, observed=True):
            filter_condition = (validated[gpouping_cols] == pd.Series(gl, index=gpouping_cols)).all(axis=1)
            g_known = validated.loc[filter_condition, 'Конверт'].dropna()
            idxs = g_nan.index
            if g_known.shape[0]:
                validated.loc[idxs] = validated.loc[idxs].fillna({'Конверт': g_known.mode().iloc[0]})
    validated['Конверт'] = validated['Конверт'].map(lambda x: 'Sealed' if x == 'SS' else 'Opened')
    return validated


def validate_styles(df: pd.DataFrame) -> pd.DataFrame:
    validated = df.copy()
    validated = validated.fillna({'Стиль': 'None'})
    for (artist, album), group in validated.groupby(['Исполнитель', 'Альбом',]):
        try:
            if len(group) > 1:
                validated.loc[(validated['Исполнитель'] == artist) & (validated['Альбом'] == album), 'Стиль'] = \
                    group['Стиль'].mode().values[0]
        except Exception:
            pass

    return validated


def process_raw(df: pd.DataFrame, bins=None) -> pd.DataFrame:
    rename = {}
    if 'Штрихкод' not in df.columns:
        rename['Barcode'] = 'Штрихкод'

    if 'Дата создания' not in df.columns:
        rename['Дата добавления'] = 'Дата создания'

    validated = df.rename(columns=rename)
    validated = validated.fillna({'Штрихкод': 'None'})
    validated.loc[:, 'Штрихкод'] = validated['Штрихкод'].map(lambda x: x.replace(" ", "").lstrip('0'))
    validated = validated.dropna(subset=[
        'Исполнитель',
        'Альбом',
        'Цена, руб.',
        'Дата создания',
    ])

    validated, bins = categorize_prices(validated, bins)
    validated = validate_date_columns(validated)
    validated = validate_categories(validated)
    validated = validated.assign(precise_record_year=validated['Год записи'])
    validated = categorize_dates(validated)
    validated = validate_styles(validated)

    coltypes = dict(filter(lambda x: x[0] in validated.columns, COLTYPES.items()))
    datecols = [col for col in validated.columns if col.startswith('Дата')]
    validated = validated[list(coltypes.keys())]
    validated = validated.dropna()

    for col in datecols:
        temp_date_col = pd.to_datetime(validated[col], dayfirst=True, errors='coerce')
        temp_date_col[temp_date_col.isna()] = pd.to_datetime(validated.loc[temp_date_col.isna(), col],
                                                             unit='s', errors='coerce')
        validated[col] = temp_date_col

    validated = validated.astype(coltypes)
    if len(datecols) == 2:
        idx = validated['Дата создания'] > validated['Дата продажи']
        validated.loc[idx, 'Дата создания'] = validated.loc[idx, 'Дата продажи']

    return validated


def count_by_category(df: pd.DataFrame, by: Union[str, Sequence[str]]) -> pd.DataFrame:
    grouped = df.groupby(list(by), observed=True)

    if 'Экземпляры' in df.columns:
        counted = grouped.agg(count=('Экземпляры', 'sum'))
        counted = counted.query('count > 0')

    else:
        counted = pd.DataFrame(grouped.size().rename('count'))

    return counted


def categorize_dates(df: pd.DataFrame) -> pd.DataFrame:
    validated = df.copy()
    for col in ['Год записи', 'Год выпуска']:
        validated[col] = pd.cut(
            validated[col],
            bins=[-np.inf] +
                 list(range(1950, (int(datetime.now().year) - int(datetime.now().year) % 10) + 1, 10)) +
                 [np.inf],
            labels=['<1950'] +
                   [f'{decade}s' for decade in range(1950, (int(datetime.now().year) - int(datetime.now().year) % 10), 10)] +
                   [f'>{(int(datetime.now().year) - int(datetime.now().year) % 10)}']
        )

    return validated


def categorize_prices(df: pd.DataFrame, bins=None, q=(0.05, 0.3, 0.5, 0.75, 0.9, 0.95, 0.99)) -> Sequence:
    df = df.copy()
    prices = df['Цена, руб.'].astype('int64')
    if bins is None:
        df['Ценовая категория'], bins = pd.qcut(prices, q=q, retbins=True)
        #df['Ценовая категория'] = df['Ценовая категория'].astype(str)

    else:
        df['Ценовая категория'] = pd.cut(prices, bins=bins, include_lowest=True)

    df.drop('Цена, руб.', axis=1, inplace=True)
    #df['Ценовая категория'] = df['Ценовая категория'].astype(str)

    return df, bins


def filter_by_date(df: pd.DataFrame, cutoff_date: Optional[str], late=False) -> pd.DataFrame:
    if cutoff_date is None:
        return df

    cutoff_date = pd.to_datetime(cutoff_date, dayfirst=True)
    df['Дата создания'] = pd.to_datetime(df['Дата создания'], dayfirst=True)
    idx = df['Дата создания'] > cutoff_date if late else df['Дата создания'] <= cutoff_date
    filtered_df = df[idx]

    return filtered_df.reset_index(drop=True)


def count_stocks(df: pd.DataFrame, cutoff_date: Optional[str], group_keys: Sequence[str], bins=None, late=False) -> Sequence:
    validated_df = process_raw(df, bins)
    filtered_df = filter_by_date(validated_df, cutoff_date, late=late)
    counted = count_by_category(filtered_df, group_keys)

    return counted, bins


def get_starting_stocks(df: pd.DataFrame, cutoff_date: str, data_path: str, bins: Optional[pd.Series]) -> pd.DataFrame:
    group_keys = GROUP_KEYS

    stocks, bins = count_stocks(
        df=df,
        cutoff_date=cutoff_date,
        group_keys=group_keys,
        bins=bins
    )

    for p in Path(data_path).glob('*.xls'):
        monthly_sales, _ = count_stocks(pd.read_excel(p, dtype='str'), cutoff_date, group_keys, bins)
        stocks = (stocks + monthly_sales).fillna(stocks).fillna(monthly_sales)

    stocks = pd.DataFrame(stocks)
    stocks.rename(columns={'count': pd.to_datetime(cutoff_date, dayfirst=True)}, inplace=True)
    return stocks


def get_stock_history(data_path: str, bins=None, cutoff_date=None) -> pd.DataFrame:
    dfs = []
    keys = ['Дата создания', 'Дата продажи', *GROUP_KEYS]

    for p in Path(data_path).glob('*.xls*'):
        movements, _ = count_stocks(
            df=pd.read_excel(p, dtype='str'),
            cutoff_date=cutoff_date,
            group_keys=keys,
            bins=bins,
            late=True
        )

        movements = movements.reset_index()
        sold = movements.groupby([k for k in keys if k != 'Дата создания']).agg(outflow=('count', 'sum')) * -1
        arrived = movements.groupby([k for k in keys if k != 'Дата продажи']).agg(inflow=('count', 'sum'))
        arrived.rename_axis(index={'Дата создания': '_date'}, inplace=True)
        sold.rename_axis(index={'Дата продажи': '_date'}, inplace=True)
        movements = pd.concat([arrived, sold], axis=1).fillna(0)
        dfs.append(movements)

    stock_history = dfs[0]
    for df in dfs[1:]:
        stock_history = (stock_history + df).fillna(stock_history).fillna(df)

    stock_history = stock_history.sort_index()
    stock_history = stock_history.assign(change=lambda x: x['inflow'] + x['outflow'])
    month_year = (
        stock_history.index.get_level_values('_date').
        to_series(name='_month_year', index=stock_history.index).dt.to_period(freq='M')
    )
    stock_history = stock_history.assign(_month_year=month_year)

    return stock_history


def get_stock_features(
        stocks: pd.DataFrame,
        daily_movements: pd.DataFrame,
) -> pd.DataFrame:
    stock_features = defaultdict(list)
    idx = [i for i in daily_movements.index.names if not i.startswith('_')]
    groups = daily_movements.groupby('_month_year')
    for month, daily_data in groups:
        daily_data = daily_data.reset_index()
        shp = daily_data.pivot_table(
            index=idx,
            columns='_date',
            values='change',
            aggfunc='sum',
            fill_value=0,
        )
        
        stocks = stocks.join(shp, how='outer').fillna(0).cumsum(axis=1)
        stocks = stocks.sort_index(axis=1)
        conf = stocks.clip(0, 5) / 5
        month_conf = conf.iloc[:, 1:].mean(1).rename(month)

        in_stock = stocks.clip(0, 1)
        in_stock_frac = in_stock.iloc[:, 1:].mean(1).rename(month)

        stock_features['availability'].append(in_stock_frac)
        stock_features['confidence'].append(month_conf)

        stocks = stocks.iloc[:, -1:]

    for k, v in stock_features.items():
        k_df = pd.concat(v, axis=1).fillna(0).T
        k_df = k_df.set_index(k_df.index.to_timestamp('ms'))
        stock_features[k] = k_df

    stock_features = pd.concat(
        [stock_features['availability'], stock_features['confidence']],
        axis=1,
        keys=['availability', 'confidence']
    )
    return stock_features


def transform_months(series):
    msin = np.sin(2 * np.pi * series / 12).values
    mcos = np.cos(2 * np.pi * series / 12).values

    return msin, mcos


def get_monthly_sales_pivot(monthly_sales_df, start=None, end=None):
    monthly_sales_pivot = monthly_sales_df.copy()
    idx = [i for i in monthly_sales_df.index.names if not i.startswith('_')]
    monthly_sales_pivot = monthly_sales_pivot.reset_index()
    monthly_sales_pivot['outflow'] = monthly_sales_pivot['outflow'].abs()
    monthly_sales_pivot = monthly_sales_pivot.pivot_table(
        index='_month_year',
        columns=idx,
        values='outflow',
        aggfunc={'outflow': 'sum'},
        fill_value=0
    )
    #sort by index
    monthly_sales_pivot = monthly_sales_pivot.sort_index(axis=1)
    monthly_sales_pivot = monthly_sales_pivot.set_index(
        monthly_sales_pivot.index.to_timestamp()
    )
    if start is not None and end is not None:
        dt_idx = pd.period_range(start=start, end=end, freq='M').to_timestamp()
        monthly_sales_pivot = monthly_sales_pivot.reindex(dt_idx, fill_value=0.)

    return monthly_sales_pivot


def get_past_covariates_df(monthly_sales_df, feature_list, span):
    monthly_sales_df = monthly_sales_df.T
    pc_df = monthly_sales_df.reset_index(
        [i for i in monthly_sales_df.index.names if not i in [*feature_list, 'Год записи']],
        drop=True
    )
    pc_df = pc_df.reset_index()

    grouped = pc_df.groupby([*feature_list, 'Год записи'])

    pc_df_ema = grouped.apply(
        lambda x: x.iloc[:, 2:].mean(axis=0).ewm(span=span, adjust=False).mean()
    )
    pc_df_ema['aggregation'] = 'exponentially_weighted_moving_average'
    pc_df_ema = pc_df_ema.set_index('aggregation', append=True)

    pc_df_emv = grouped.apply(
        lambda x: x.iloc[:, 2:].mean(axis=0).ewm(span=span, adjust=False).var()
    ).bfill(axis=1)
    pc_df_emv['aggregation'] = 'exponentially_weighted_moving_variance'
    pc_df_emv = pc_df_emv.set_index('aggregation', append=True)

    return pd.concat([pc_df_ema, pc_df_emv], axis=0)



