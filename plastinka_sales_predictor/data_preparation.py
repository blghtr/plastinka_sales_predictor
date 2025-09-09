import logging
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Sequence
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import dill
import numpy as np
import pandas as pd
from darts.timeseries import TimeSeries
from darts.utils.data.torch_datasets.inference_dataset import TorchInferenceDataset
from darts.utils.data.torch_datasets.training_dataset import TorchTrainingDataset
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder, minmax_scale
from typing_extensions import Self

logger = logging.getLogger(__name__)

COLTYPES = {
    "Штрихкод": str,
    "Barcode": str,
    "Экземпляры": "int64",
    "Ценовая категория": "str",
    "Цена, руб.": "float64",
    "Конверт": str,
    "Альбом": str,
    "Исполнитель": str,
    "Год записи": str,
    "Год выпуска": str,
    "Тип": str,
    "Стиль": str,
    "Дата создания": "datetime64[ns]",
    "Дата продажи": "datetime64[ns]",
    "Дата добавления": "datetime64[ns]",
    "precise_record_year": "int64",
}

COLUMN_MAPPING = {
    "Штрихкод": "barcode",
    "Исполнитель": "artist",
    "Альбом": "album",
    "Конверт": "cover_type",
    "Ценовая категория": "price_category",
    "Тип": "release_type",
    "Год записи": "recording_decade",
    "Год выпуска": "release_decade",
    "Стиль": "style",
    "precise_record_year": "record_year",
    "Дата создания": "created_date",
    "Дата продажи": "sold_date",
}

GROUP_KEYS = [
    "barcode",
    "artist",
    "album",
    "cover_type",
    "price_category",
    "release_type",
    "recording_decade",
    "release_decade",
    "style",
    "record_year",
]


# hardcoded for now
BASE_FEATURES_DEFAULT = [
    "sales",
    "movement"
]

REPORT_FEATURES_DEFAULT = [
    "availability",
    "confidence",
    "lost_sales",
    "masked_mean_sales_items",
    "masked_mean_sales_rub"
]


class PlastinkaBaseTSDataset:
    def __init__(
        self,
        stock_features: pd.DataFrame,
        monthly_sales: pd.DataFrame,
        static_transformer: BaseEstimator = None,
        static_features: Sequence[str] | None = None,
        scaler: BaseEstimator = None,
        weight_coef: float = 0.0,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
        start: int = 0,
        end: int | None = None,
        past_covariates_fnames: Sequence[str] = (
            "release_type",
            "cover_type",
            "style",
            "price_category",
        ),
        past_covariates_span: int = 3,
        save_dir: str | None = None,
        dataset_name: str | None = None,
        dtype: str | np.dtype = np.float32,
        minimum_sales_months: int = 1,
    ):
        super().__init__()
        logger.info("Initializing PlastinkaBaseTSDataset...")
        # --- Регуляризация временного индекса ---
        monthly_sales = ensure_monthly_regular_index(monthly_sales)
        stock_features = ensure_monthly_regular_index(stock_features)

        self.dtype = dtype
        multiidxs = monthly_sales.columns.drop_duplicates().tolist()
        self._idx2multiidx = OrderedDict(
            {i: multiidxs[i] for i in range(len(multiidxs))}
        )
        self._multiidx2idx = OrderedDict(
            {tuple(multiidx): idx for idx, multiidx in enumerate(multiidxs)}
        )
        self._index_names_mapping = OrderedDict(
            {n: i for i, n in enumerate(monthly_sales.columns.names)}
        )
        self._time_index = monthly_sales.index
        self._monthly_sales = monthly_sales.loc[
            self._time_index, self._multiidx2idx.keys()
        ].values.astype(self.dtype)
        self._stock_features = self._get_stock_features_values(stock_features)
        self._n_time_steps = self._monthly_sales.shape[0]
        self._end = end
        self.start = start
        self._past_covariates_fnames = past_covariates_fnames
        self.minimum_sales_months = minimum_sales_months
        self.setup_dataset(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            span=past_covariates_span,
            weights_alpha=weight_coef,
            scaler=scaler,
            transformer=static_transformer,
            static_features=static_features,
            copy=False,
            reindex=False,
        )
        self._allow_empty_stock = False
        self._idx_mapping = self._build_index_mapping()
        self.save_dir = save_dir
        self.dataset_name = dataset_name

        if save_dir is not None:
            logger.info(f"Saving dataset to {save_dir}/{dataset_name}.dill")
            self.save(save_dir, dataset_name)
        logger.info("PlastinkaBaseTSDataset initialized.")

    @classmethod
    def from_dill(cls, dill_path: str | Path):
        with open(dill_path, "rb") as f:
            return dill.load(f)

    def _get_stock_features_values(self, stock_features):
        full_indices = []
        for level1 in stock_features.columns.get_level_values(0).unique():
            for combo in self._multiidx2idx.keys():
                full_indices.append((level1, *combo))

        valid_indices = [idx for idx in full_indices if idx in stock_features.columns]

        if not valid_indices:
            raise ValueError("No valid indices found")

        arr = stock_features.loc[self._time_index, valid_indices].values.astype(
            self.dtype
        )

        arr = arr.reshape(arr.shape[0], arr.shape[1] // len(self._multiidx2idx), -1)
        return arr

    def save(self, save_dir: str | Path | None = None, dataset_name: str | None = None):
        if save_dir is None:
            save_dir = self.save_dir

        if dataset_name is None:
            if not self.dataset_name:
                dataset_name = self.__class__.__name__
            else:
                dataset_name = self.dataset_name

        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        with open(save_dir / f"{dataset_name}.dill", "wb") as f:
            dill.dump(self, f)

    def prepare_past_covariates(self, span=None):
        if span is not None:
            self._past_covariates_span = span

        self._past_covariates_cached = {
            feat: get_past_covariates_df(
                self.monthly_sales_df,
                self.stock_features,
                (feat,),
                self._past_covariates_span,
            )
            for feat in self._past_covariates_fnames
        }

    def get_static_covariates(self):
        if self.static_transformer is None:
            return None

        unique_items = pd.DataFrame(
            self._multiidx2idx.keys(), columns=self._index_names_mapping.keys()
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
                index=pd.MultiIndex.from_frame(full_index),
            )

        else:
            static_covariates = static_covariates.set_index(
                pd.MultiIndex.from_frame(full_index)
            )

        static_covariates = static_covariates.astype(self.dtype)

        return static_covariates

    def get_future_covariates(self, item_multiidx):
        time_index = self.time_index
        is_hot = time_index.map(lambda x: int(x.year)) == item_multiidx[-1]
        msin, mcos = transform_months(time_index.map(lambda x: x.month))
        year = minmax_scale(time_index.map(lambda x: x.year))
        future_covariates = np.vstack([is_hot, msin, mcos, year]).T.astype(self.dtype)

        return future_covariates

    def get_past_covariates(self, item_multiidx):
        past_covariates = []
        release_year = item_multiidx[self._index_names_mapping["recording_decade"]]
        for feat_name in self._past_covariates_cached:
            feat_vals = item_multiidx[self._index_names_mapping[feat_name]]
            feat_df = self._past_covariates_cached[feat_name].loc[
                :, pd.IndexSlice[feat_vals, release_year, :]
            ]
            past_covariates.append(feat_df)

        past_covariates.append(
            pd.DataFrame(self._get_item_sales(item_multiidx))
            .ewm(span=self._past_covariates_span, adjust=False)
            .mean()
        )

        past_cov_arr = np.hstack(past_covariates)

        if self.scaler:
            past_cov_arr = self.scaler.transform(past_cov_arr)

        past_cov_arr = np.hstack([past_cov_arr, self._get_item_stock(item_multiidx)])

        past_cov_arr = past_cov_arr.astype(self.dtype)
        return past_cov_arr

    def _set_length(self, input_chunk_length, output_chunk_length):
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length

    def _set_window(self, start, end):
        self.start = start
        self._end = end

    def set_scaler(self, scaler):
        if not scaler.is_fit():
            scaler = scaler.fit(self.monthly_sales)
        self.scaler = scaler

    def set_static_transformer(self, transformer):
        self.static_transformer = transformer

    def setup_dataset(
        self,
        window: tuple[int, int] | None = None,
        input_chunk_length: int | None = None,
        output_chunk_length: int = 1,
        span: int | None = None,
        weights_alpha: float | None = None,
        scaler: BaseEstimator | None = None,
        transformer: BaseEstimator | None = None,
        static_features: Sequence[str] | None = None,
        copy: bool = True,
        reindex: bool = True,
    ) -> Self | None:
        def validate_boundaries(window, input_chunk_length, output_chunk_length):
            window = (0, self._n_time_steps) if window is None else window
            start, end = window
            if start < 0 or end > self._n_time_steps:
                raise ValueError(f"Invalid boundaries: start={start}, end={end}. \n" 
                                 f"start must be >= 0 and end must be <= {self._n_time_steps}")
            if input_chunk_length and output_chunk_length:
                if input_chunk_length < self.minimum_sales_months or output_chunk_length <= 0:
                    raise ValueError(f"Invalid length: input_chunk_length={input_chunk_length}, output_chunk_length={output_chunk_length}. \n" 
                                     f"input_chunk_length must be >= {self.minimum_sales_months} and output_chunk_length must be > 0")
                if (end - start) < input_chunk_length + output_chunk_length:
                    raise ValueError(f"Invalid boundaries: start={start}, end={end}, input_chunk_length={input_chunk_length}, output_chunk_length={output_chunk_length}. \n" 
                                     f"window length {end - start} must be >= input_chunk_length + output_chunk_length")
            return True

        if copy:
            ds = deepcopy(self)
        else:
            ds = self

        try:
            validate_boundaries(
                window,
                input_chunk_length,
                output_chunk_length
            )
            if input_chunk_length and output_chunk_length:
                ds._set_length(input_chunk_length, output_chunk_length)
            if window:
                ds._set_window(*window)

            if scaler:
                ds.set_scaler(scaler)

            self.static_features = static_features
            if transformer:
                ds.set_static_transformer(transformer)
                self.static_covariates_mapping = self.get_static_covariates()

            if weights_alpha is not None:
                ds.set_reweight_fn(weights_alpha)

            if reindex:
                ds._idx_mapping = ds._build_index_mapping()

            ds.prepare_past_covariates(span)

            if copy:
                return ds

        except Exception as e:
            logger.error(f"Error setting up dataset: {e}", exc_info=True)
            logger.debug(
                '--------------------------------DEBUG--------------------------------'
            )
            logger.debug(
                f"Window: {window}, input_chunk_length: {input_chunk_length}, "
                f"output_chunk_length: {output_chunk_length}, span: {span}, "
                f"weights_alpha: {weights_alpha}, scaler: {scaler}, "
                f"transformer: {transformer}, static_features: {static_features}, "
                f"copy: {copy}, reindex: {reindex}"
            )
            raise e

    def _build_index_mapping(self):
        current_index = 0
        self._index_mapping, _index_mapping = {}, {}
        for i in range(len(self)):
            if self._sample_is_valid(i):
                _index_mapping[current_index] = i
                current_index += 1
        self._index_mapping = _index_mapping

    def _sample_is_valid(self, index):
        index, start_index, end_index = self._project_index(index)

        item_sales = self.monthly_sales[start_index:end_index, index]
        enough_sales = (
            (
                item_sales[
                    item_sales > 0.0  # scaled zero
                ].shape[0]
            )
            >=
            self.minimum_sales_months
        )

        target_stock = self.stock_features[-self.output_chunk_length :, :, index]

        in_stock = np.any(target_stock)

        return enough_sales and (self._allow_empty_stock or in_stock)

    def set_reweight_fn(self, alpha):
        def reweight_fn(array):
            min_weight = 0.1
            if alpha == 0.0:
                min_weight = 1.0
            weights = min_weight + alpha * np.log1p(array)

            return weights

        self.reweight_fn = reweight_fn

    def reset_window(self, copy: bool = True):
        if self._n_time_steps:
            ds = self.setup_dataset(window=(0, self._n_time_steps), copy=copy)
            return ds

    def _get_raw_sample_arrays(self, array_index, start_index, end_index):
        """
        Extracts raw NumPy arrays for a given index, common to both training and inference datasets.
        """
        end_index_safe = end_index if end_index is not None else self._n_time_steps

        series_item = np.expand_dims(
            self.monthly_sales[start_index:end_index_safe, array_index], axis=1
        )
        item_multiidx = self._idx2multiidx[array_index]
        # Renamed for clarity as per user's suggestion
        future_covariates_item = self.get_future_covariates(item_multiidx)
        historic_future_covariates_item = future_covariates_item[
            start_index: end_index_safe - self.output_chunk_length
        ]
        future_covariates_item_output_chunk = future_covariates_item[
            end_index_safe - self.output_chunk_length: end_index_safe
        ]
        past_covariates_item = self.get_past_covariates(item_multiidx)[
            start_index:end_index_safe
        ]

        static_covariates_item = None
        if self.static_covariates_mapping is not None:
            static_covariates_item = self.static_covariates_mapping.loc[item_multiidx]

        if self.scaler is not None:
            series_item = self.scaler.transform(series_item)

        soft_availability = np.expand_dims(
            past_covariates_item[: -self.output_chunk_length, -1], 1
        )
        reweight_fn_output = self.reweight_fn(
            series_item[: -self.output_chunk_length] * soft_availability
        )

        return (
            series_item,
            past_covariates_item,
            historic_future_covariates_item,
            future_covariates_item_output_chunk,
            static_covariates_item,
            reweight_fn_output,
        )

    def __len__(self):
        if self._index_mapping:
            return len(self._index_mapping)
        else:
            return self.monthly_sales.shape[1] * self.outputs_per_array

    def _project_index(self, index):
        if self._index_mapping:
            index = self._index_mapping[index]
        item = index // self.outputs_per_array

        length = self.input_chunk_length + self.output_chunk_length
        start_index = index % self.outputs_per_array
        end_index = (
            (
                start_index + length
            )
            if (start_index + length) < self._n_time_steps
            else None
        )
        return item, start_index, end_index

    def _get_item_sales(self, item_multiidx):
        return self.monthly_sales[:, self._multiidx2idx[item_multiidx]]

    def _get_item_stock(self, item_multiidx):
        return self.stock_features[:, :, self._multiidx2idx[item_multiidx]]

    @property
    def monthly_sales(self):
        if self._monthly_sales is not None:
            return self._monthly_sales[self.start : self.end]
        return None

    @property
    def monthly_sales_df(self):
        if self._monthly_sales is not None:
            return pd.DataFrame(
                self.monthly_sales,
                index=self.time_index,
                columns=pd.MultiIndex.from_tuples(
                    self._multiidx2idx.keys(), names=self._index_names_mapping.keys()
                ),
            )
        return None

    @property
    def stock_features(self):
        if self._stock_features is not None:
            return self._stock_features[self.start : self.end]
        return None

    @property
    def time_index(self):
        return self._time_index[self.start : self.end]

    @property
    def outputs_per_array(self):
        return self.L - self.input_chunk_length - self.output_chunk_length + 1

    @property
    def L(self):
        return self.end - self.start

    @property
    def end(self):
        end = self._end
        if end is None:
            return self._n_time_steps
        return end

    @property
    def labels(self):
        labels = []
        for idx in range(len(self)):
            array_index, _, _ = self._project_index(idx)
            item_multiidx = self._idx2multiidx[array_index]
            labels.append(item_multiidx)
        return labels


class PlastinkaTrainingTSDataset(PlastinkaBaseTSDataset, TorchTrainingDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        (array_index, start_index, end_index) = self._project_index(idx)
        (
            series_item,
            past_covariates_item,
            historic_future_covariates_item,
            future_covariates_item_output_chunk,
            static_covariates_item,
            reweight_fn_output,
        ) = self._get_raw_sample_arrays(array_index, start_index, end_index)

        past_target = series_item[: -self.output_chunk_length]
        future_target = series_item[-self.output_chunk_length :]

        static_covariates_item_array = None
        if static_covariates_item is not None:
            static_covariates_item_array = np.expand_dims(
                static_covariates_item.values, 1
            ).T
        output = (
            past_target,
            past_covariates_item[: -self.output_chunk_length], # Past part of past_covariates
            historic_future_covariates_item,
            future_covariates_item_output_chunk,
            static_covariates_item_array,
            reweight_fn_output,
            future_target,
        )
        return output


class PlastinkaInferenceTSDataset(PlastinkaBaseTSDataset, TorchInferenceDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._allow_empty_stock = True # This flag was part of the original training dataset but makes sense for inference too

        # --- Restore padding logic directly in __init__ for inference dataset ---
        pad = self.output_chunk_length

        # Create padding arrays
        zero_sales = np.zeros(
            (pad, self._monthly_sales.shape[1]), dtype=self._monthly_sales.dtype
        )
        zero_stock = np.zeros(
            (pad, *self._stock_features.shape[1:]), dtype=self._stock_features.dtype
        )

        # Extend arrays
        self._monthly_sales = np.vstack([self._monthly_sales, zero_sales])
        self._stock_features = np.vstack([self._stock_features, zero_stock])
        self._n_time_steps = self._monthly_sales.shape[0]

        # Extend time index
        last_date = self._time_index[-1]
        new_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1), periods=pad, freq="MS"
        )
        self._time_index = self._time_index.append(new_dates)

        self.reset_window()

        if self.save_dir is not None:
            logger.info(f"Saving inference dataset to {self.save_dir}/{self.dataset_name}.dill")
            self.save(self.save_dir, self.dataset_name)

    def __getitem__(self, idx):
        (array_index, start_index, end_index) = self._project_index(idx)

        (
            series_item,
            past_covariates_item,
            historic_future_covariates_item,
            future_covariates_item_output_chunk,
            static_covariates_item,
            _, # reweight_fn_output not needed for inference __getitem__
        ) = self._get_raw_sample_arrays(array_index, start_index, end_index)

        # Darts InferenceDataset expects a SeriesSchema for target_series (7th element)
        # and pred_time (8th element).
        time_index_for_ts = self.time_index[start_index: end_index] # Use unpacked start_index, end_index

        # This is a workaround because Darts doesn't provide a direct way to build SeriesSchema
        # without a TimeSeries object from raw data, but expects a SeriesSchema from __getitem__.

        static_covariates_item_array = None
        if static_covariates_item is not None:
            static_covariates_item_array = np.expand_dims(
                static_covariates_item.values, 1
            ).T

        dummy_ts = TimeSeries.from_times_and_values(
            times=time_index_for_ts,
            values=series_item,
            static_covariates=static_covariates_item,
        )

        target_series_schema = dummy_ts.schema  # Extract the schema

        pred_time = dummy_ts.time_index[-self.output_chunk_length] # The time of the first point in the forecast horizon

        # Align with TorchInferenceDatasetOutput signature:
        # (past_target, past_covariates, future_past_covariates, historic_future_covariates, future_covariates, static_covariates, target_series_schema, pred_time)
        output = (
            series_item[: -self.output_chunk_length], # past_target
            past_covariates_item[: -self.output_chunk_length], # past_covariates
            None, # future_past_covariates (not explicitly generated by your current logic)
            historic_future_covariates_item, # historic_future_covariates
            future_covariates_item_output_chunk, # future_covariates
            static_covariates_item_array, # static_covariates
            target_series_schema, # target_series (now SeriesSchema)
            pred_time, # pred_time
        )
        return output

    def reset_window(self):
        if self._n_time_steps:
            self.setup_dataset(
                window=(self._n_time_steps - (
                    self.input_chunk_length + self.output_chunk_length
                ), self._n_time_steps),
                copy=False,
                reindex=True,
            )

    def _set_length(self, input_chunk_length, output_chunk_length):
        super()._set_length(input_chunk_length, output_chunk_length)
        self.reset_window()


class MultiColumnLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, separator="/"):
        self.separator = separator
        self.encoders = {}

    def fit(self, X, y=None):
        X = self.validate_data(X)
        for column in X.columns:
            mlb = MultiLabelBinarizer(sparse_output=True)
            mlb.fit(
                X[column]
                .str.split(self.separator)
                .map(lambda x: [y.strip() for y in x])
            )
            self.encoders[column] = mlb

        return self

    def transform(self, X, index=None):
        X = self.validate_data(X)
        idx = list(X.columns)
        transformed_columns = []
        for column in idx:
            encoder = self.encoders.get(column, None)
            if encoder is None:
                continue
            binarized = encoder.transform(
                X[column]
                .str.split(self.separator)
                .map(lambda x: [y.strip() for y in x])
            )
            binarized_df = pd.DataFrame.sparse.from_spmatrix(
                binarized,
                columns=[f"{column}_{cls}" for cls in self.encoders[column].classes_],
            )
            transformed_columns.append(binarized_df)

        dummy_df = pd.concat([X, *transformed_columns], axis=1)

        if index is None:
            index = idx
        dummy_df = dummy_df.set_index(index)

        return dummy_df

    def is_fit(self):
        return len(self.encoders) > 0

    def validate_data(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input should be a pandas DataFrame.")
        return X.astype(str, copy=True)


class OrdinalEncoder(OrdinalEncoder):
    def __init__(self):
        super().__init__()

    def is_fit(self):
        return hasattr(self, "n_features_in_")

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
            raise ValueError(
                "This GlobalMinMaxScaler instance is not fitted yet. Call 'fit' before using this method."
            )

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
            raise ValueError(
                "This GlobalMinMaxScaler instance is not fitted yet. Call 'fit' before using this method."
            )

        X_scaled = self._validate_data(X_scaled)

        scale = self.feature_range[1] - self.feature_range[0]
        X_scaled = X_scaled - 1e-6
        X = (X_scaled - self.feature_range[0]) / scale
        X = X * (self.global_max_ - self.global_min_) + self.global_min_
        X = np.expm1(X)

        return X

    def _validate_data(self, X):
        # Приводим данные к DataFrame для совместимости с pandas или numpy
        if not isinstance(X, pd.DataFrame | pd.Series | np.ndarray):
            raise ValueError(
                "Input should be a pandas DataFrame, Series or numpy array."
            )
        if not isinstance(X, np.ndarray):
            X = X.values
        return X

    def is_fit(self):
        return self.global_min_ is not None and self.global_max_ is not None


def validate_feature_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    def fill_partial_years_np_where(series):
        try:
            is_not_20 = ~series.str.startswith("2")
            valid_str = series.str.extract(r"(^\\d+)")[0]
            padding_9 = (4 - valid_str.str.len().astype(int)).map(lambda x: "9" * x)
            padding_1 = (4 - valid_str.str.len().astype(int)).map(lambda x: "1" * x)
            return np.where(is_not_20, valid_str + padding_9, valid_str + padding_1)

        except Exception as e:
            
            logger.warning(
                f"Error in fill_partial_years_np_where: {e}", exc_info=True
            )
            return series  # Return original series on error instead of silently failing

    validated = df.copy()

    validated["Год записи"] = pd.to_numeric(validated["Год записи"], errors="coerce")
    validated = validated.dropna(subset=["Год записи"])

    uncert_rerelease_year_idxs = ~validated["Год выпуска"].fillna("1234").astype(
        str
    ).str.match(r"^(\\d{4}|[^0-9]*)$")
    unvalid_rereleases = validated.loc[
        uncert_rerelease_year_idxs, "Год выпуска"
    ].astype(str)
    validated.loc[uncert_rerelease_year_idxs, "Год выпуска"] = (
        fill_partial_years_np_where(unvalid_rereleases)
    )
    validated["Год выпуска"] = pd.to_numeric(validated["Год выпуска"], errors="coerce")

    validated.loc[validated["Тип"] == "Оригинал", "Год выпуска"] = validated.loc[
        validated["Тип"] == "Оригинал", "Год записи"
    ]

    valid_rereleases = validated[
        (validated["Тип"] != "Оригинал") & (validated["Год выпуска"].notna())
    ]
    if len(valid_rereleases) > 10:
        diff_years = valid_rereleases["Год выпуска"] - valid_rereleases["Год записи"]
        # Check for valid (non-NaN) differences before calculating mean
        if not diff_years.empty and not diff_years.isnull().all():
            mean_gap = diff_years.mean()
            if pd.notna(mean_gap):
                mean_gap = int(round(mean_gap))
            else:
                mean_gap = 15  # Default if mean is NaN
        else:
            mean_gap = 15  # Default if no valid differences
    else:
        mean_gap = 15

    rerelease_nans = (validated["Тип"] != "Оригинал") & (
        validated["Год выпуска"].isna()
    )
    validated.loc[rerelease_nans, "Год выпуска"] = (
        validated.loc[rerelease_nans, "Год записи"] + mean_gap
    )

    invalid_release_dates = validated["Год выпуска"] < validated["Год записи"]
    validated.loc[invalid_release_dates, "Год выпуска"] = (
        validated.loc[invalid_release_dates, "Год записи"] + mean_gap
    )

    current_year = datetime.now().year
    validated["Год записи"] = validated["Год записи"].clip(1950, current_year)
    validated["Год выпуска"] = validated["Год выпуска"].clip(1950, current_year)

    return validated


def validate_date_columns(
        df: pd.DataFrame, 
        datecols: Sequence[str], 
    ) -> pd.DataFrame:
     
    validated = df.copy()

    if len(datecols) == 2:
        idx = validated["Дата создания"] > validated["Дата продажи"]
        validated.loc[idx, "Дата создания"] = validated.loc[idx, "Дата продажи"]

        dominant_month_period = (
            validated["Дата продажи"]
            .dt
            .to_period('M')
            .mode()[0]
        )
        outlier_mask = (
            validated["Дата продажи"]
            .dt
            .to_period('M') != dominant_month_period
        )
        validated.loc[outlier_mask, "Дата продажи"] = None
        validated["Дата продажи"] = validated["Дата продажи"].dt.floor("D")

    validated = validated.dropna(subset=datecols)
    validated["Дата создания"] = validated["Дата создания"].dt.floor("D")
    
    return validated


def validate_categories(df: pd.DataFrame) -> pd.DataFrame:
    validated = df.copy()
    nan_idx = validated["Конверт"].isna()
    nan_df = validated[nan_idx]
    for gpouping_cols in [
        ["Исполнитель", "Альбом", "Год выпуска"],
        ["Ценовая категория", "Год выпуска"],
    ]:
        for gl, g_nan in nan_df.groupby(gpouping_cols, observed=True):
            filter_condition = (
                validated[gpouping_cols] == pd.Series(gl, index=gpouping_cols)
            ).all(axis=1)
            g_known = validated.loc[filter_condition, "Конверт"].dropna()
            idxs = g_nan.index
            if g_known.shape[0]:
                mode_val = g_known.mode()
                if not mode_val.empty:
                    validated.loc[idxs] = validated.loc[idxs].fillna(
                        {"Конверт": mode_val.iloc[0]}
                    )
                else:
                    # If no mode found (all values unique or other edge case), use the first value
                    
                    logger.warning(
                        "No mode found for 'Конверт' in group. Using first value if available."
                    )
                    if len(g_known) > 0:
                        validated.loc[idxs] = validated.loc[idxs].fillna(
                            {"Конверт": g_known.iloc[0]}
                        )
    validated["Конверт"] = validated["Конверт"].map(
        lambda x: "Sealed" if x == "SS" else "Opened"
    )
    return validated

def validate_styles(df: pd.DataFrame) -> pd.DataFrame:
    validated = df.copy()
    validated = validated.fillna({"Стиль": "None"})
    for (artist, album), group in validated.groupby(
        [
            "Исполнитель",
            "Альбом",
        ]
    ):
        try:
            if len(group) > 1:
                validated.loc[
                    (validated["Исполнитель"] == artist)
                    & (validated["Альбом"] == album),
                    "Стиль",
                ] = group["Стиль"].mode().values[0]
        except Exception:
            pass

    return validated

def process_raw(df: pd.DataFrame, bins=None) -> pd.DataFrame:
    
    rename = {}
    if "Штрихкод" not in df.columns:
        rename["Barcode"] = "Штрихкод"

    if "Дата создания" not in df.columns:
        rename["Дата добавления"] = "Дата создания"

    validated = df.rename(columns=rename)
    validated = validated.fillna({"Штрихкод": "None"})
    validated.loc[:, "Штрихкод"] = validated["Штрихкод"].map(
        lambda x: x.replace(" ", "").lstrip("0")
    )

    validated = validated.dropna(
        subset=[
            "Исполнитель",
            "Альбом",
            "Цена, руб.",
            "Дата создания",
        ]
    )

    validated.loc[:, "Цена, руб."] = pd.to_numeric(
        validated["Цена, руб."], errors='coerce'
    ).astype("int64")

    validated, bins = categorize_prices(validated, bins)
    validated = validate_feature_date_columns(validated)
    validated = validate_categories(validated)
    if "precise_record_year" not in validated.columns:
        validated = validated.assign(precise_record_year=validated["Год записи"])
    validated = categorize_dates(validated)
    validated = validate_styles(validated)

    coltypes = dict(filter(lambda x: x[0] in validated.columns, COLTYPES.items()))
    datecols = [col for col in validated.columns if col.startswith("Дата")]
    
    # Ensure 'Конверт' is always included if it exists in the original df
    if "Конверт" in df.columns and "Конверт" not in coltypes:
        coltypes["Конверт"] = str

    validated = validated[list(coltypes.keys())]
    validated = validated.dropna()
    
    # Validate that we have data after processing
    if validated.empty:
        raise ValueError(
            f"All data was filtered out during processing. "
            f"Original shape: {df.shape}, "
            f"After dropna: {validated.shape}. "
            f"Check input data quality and required columns."
        )

    for col in datecols:
        temp_date_col = pd.to_datetime(
            validated[col], 
            dayfirst=True, 
            errors="coerce"
        )
        temp_date_col[temp_date_col.isna()] = pd.to_datetime(
            validated.loc[temp_date_col.isna(), col], 
            unit="s", errors="coerce"
        )
        validated[col] = temp_date_col
    validated = validated.astype(coltypes)

    validated = validate_date_columns(validated, datecols)

    # Rename columns to match DB schema
    validated = validated.rename(columns=COLUMN_MAPPING)

    return validated

def categorize_dates(df: pd.DataFrame) -> pd.DataFrame:
    validated = df.copy()
    for col in ["Год записи", "Год выпуска"]:
        validated[col] = pd.cut(
            validated[col],
            bins=[-np.inf]
            +
            list(
                range(
                    1950,
                    (int(datetime.now().year) - int(datetime.now().year) % 10) + 1,
                    10,
                )
            )
            +
            [np.inf],
            labels=["<1950"]
            +
            [
                f"{decade}s"
                for decade in range(
                    1950, (int(datetime.now().year) - int(datetime.now().year) % 10), 10
                )
            ]
            +
            [f">{(int(datetime.now().year) - int(datetime.now().year) % 10)}"],
        )

    return validated

def categorize_prices(
    df: pd.DataFrame, bins=None, q=(0.05, 0.3, 0.5, 0.75, 0.9, 0.95, 0.99)
) -> Sequence:
    df = df.copy()
    prices = df["Цена, руб."]

    if bins is None:
        try:
            df["Ценовая категория"], bins = pd.qcut(prices, q=q, retbins=True, duplicates='drop')
        except ValueError as e:
            if "Bin edges must be unique" in str(e) or "Too many duplicate edges" in str(e):
                raise ValueError(
                    f"Недостаточно уникальных цен для создания категорий. "
                    f"Требуется минимум {len(q) + 1} уникальных цен, "
                    f"но найдено только {prices.nunique()}. "
                    f"Уникальные цены: {sorted(prices.unique())}"
                )
            else:
                raise e
    else:
        df["Ценовая категория"] = pd.cut(prices, bins=bins, include_lowest=True)

    return df, bins

def filter_by_date(
    df: pd.DataFrame | pd.Series,
    cutoff_date: pd.Timestamp | None,
) -> pd.DataFrame:
    if cutoff_date is None:
        return df
    cutoff_date = pd.to_datetime(cutoff_date, dayfirst=True)

    created_date = None
    if isinstance(df, pd.DataFrame) and "created_date" in df.columns:
        created_date = pd.to_datetime(df["created_date"])
    elif "created_date" in df.index.names:
        created_date = df.index.get_level_values("created_date")
    else:
        logger.error(
            "filter_by_date - created_date not in df.columns or df.index.names"
        )
        return df
    
    mask = pd.IndexSlice[:] if created_date is None else created_date > cutoff_date
    filtered_df = df.loc[mask]

    return filtered_df

def process_stock(df):
    """Sum 'Экземпляры' field in a group for stock calculation"""
    return pd.Series({
        "count": df["Экземпляры"].astype("int64").sum(),
        "price": df["Цена, руб."].astype("float64").mean()
    })

def process_movements(df):
    """
    Count the number of rows in a group and calculate the mean price
    for stock movement calculation
    """
    return pd.Series(
        {
            "count": len(df),
            "price": (df["Цена, руб."].astype("float64").mean()),
        }
    )

def _process_stock(
        preprocessed_df: pd.DataFrame, 
        keys_no_dates: Sequence[str], 
        cutoff_date: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process stock data to create stock and prices DataFrames."""
    
    # Use all data without filtering
    filtered_df = filter_by_date(preprocessed_df, cutoff_date)

    # Group by non-date keys for monthly sales counting
    monthly_df = filtered_df.groupby(keys_no_dates).agg(
        count=("count", "sum"), price=("price", "mean")
    )

    monthly_stock, monthly_prices = (
        monthly_df.loc[:, "count"],
        monthly_df.loc[:, "price"],
    )

    monthly_stock = monthly_stock.rename("stock").to_frame()


    return monthly_stock, monthly_prices

def _init_stock(
    stock_path: str,
    all_keys: Sequence[str],
    cutoff_date: pd.Timestamp | None = None,
    bins: pd.IntervalIndex | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stock_df = read_data_file(stock_path)

    preprocessed_stock_df = process_raw(stock_df, bins=bins)

    filtered_stock_df = filter_by_date(preprocessed_stock_df, cutoff_date)
    
    grouped_stock_df = _get_grouped_df(filtered_stock_df, all_keys)

    # Fork for stock (everything before cutoff)
    keys_no_dates = [
        k for k in all_keys if k not in {"created_date", "sold_date"}
    ]
    
    stock, prices_from_stock = _process_stock(
        grouped_stock_df,
        keys_no_dates,
        cutoff_date
    )
    
    
    return stock, prices_from_stock

def _get_sales_files_paths(sales_path: str) -> list[Path]:
    sales_files: list[Path] = []
    sales_path_str = str(sales_path)
    if sales_path_str.lower().endswith((".csv", ".xlsx", ".xls")):
        sales_files.append(Path(sales_path))
    elif Path(sales_path).is_file():
        raise ValueError("Sales path is an unsupported file, not a directory")
    elif Path(sales_path).is_dir():
        sales_files.extend(Path(sales_path).glob("*.xls*"))
        sales_files.extend(Path(sales_path).glob("*.xlsx"))
        sales_files.extend(Path(sales_path).glob("*.csv"))
    else:
        raise ValueError("Sales path does not lead to a valid file or directory.")
    return sales_files

def read_data_file(path: Path) -> pd.DataFrame:
    if not isinstance(path, Path):
        path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, dtype="str")
    elif path.suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path, dtype="str")
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

def _preprocess_sort_sales(sales_files_paths: list[Path], bins: pd.IntervalIndex) -> OrderedDict:
    monthly_dfs = {}
    for path in sales_files_paths:
        df = process_raw(read_data_file(path), bins=bins)
        if not df.empty and "sold_date" in df.columns:
            month = pd.to_datetime(
                df["sold_date"]
                .mode()[0]
                .to_period('M')
                .start_time
            )
            monthly_dfs[month] = df
    return OrderedDict(sorted(monthly_dfs.items(), reverse=True))  # reverse=True для обработки от позднего к раннему

def _get_grouped_df(
    processed_df: pd.DataFrame,
    group_keys: Sequence[str],
) -> pd.DataFrame:
    
    group_keys = [k for k in group_keys if k in processed_df.columns]
    preprocessed_df = (
        processed_df
        .groupby(group_keys)
        .apply(process_movements)
        .sort_index(axis=1, ascending=False)
    )

    return preprocessed_df

def _process_grouped_df(
        grouped_df: pd.DataFrame, 
        all_keys: Sequence[str], 
        keys_no_dates: Sequence[str],
        cutoff_date: pd.Timestamp | None = None
    ):
    """Process sales data to create sales, movement, and prices DataFrames."""

    def _prune_arrival_dates(
            arrived: pd.DataFrame, 
            sold: pd.DataFrame
    ) -> pd.DataFrame:
        """Prune arrival dates that are before the sold dates."""
        return arrived.loc[
            arrived
            .index
            .get_level_values('_date')
            .to_series()
            .ge(
                sold
                .index
                .get_level_values('_date')
                .min()
            )
            .values
        ]

    prices = grouped_df.groupby(
        keys_no_dates
    ).agg(
        price=("price", "mean")
    )

    # Prepare grouping keys for sold / arrived ensuring presence in dataframe
    sold_keys = [
        k for k in all_keys if k != "created_date" and k in grouped_df.index.names
    ]
    arrived_keys = [
        k for k in all_keys if k != "sold_date" and k in grouped_df.index.names
    ]

    # Process movements for stock history
    sold = grouped_df.groupby(
        sold_keys
    ).agg(
        outflow=(
            "count", 
            "sum"
        )
    ).squeeze()

    arrived = (
        grouped_df
        .pipe(
            (filter_by_date, 'df'), 
            cutoff_date=cutoff_date
        ).groupby(
            arrived_keys
        ).agg(
            inflow=(
                "count", 
                "sum"
            )
        ).squeeze())

    arrived.rename_axis(index={"created_date": "_date"}, inplace=True)
    sold.rename_axis(index={"sold_date": "_date"}, inplace=True)

    arrived = _prune_arrival_dates(arrived, sold)

    sales, arrived = sold.align(
        arrived,
        join='outer',
        axis=0
    )

    movements = (
        sales
        .fillna(0)
        .mul(-1)
        .add(
            arrived.fillna(0)
        )
        .rename('movement')
    )

    sales_pivot = sales.unstack(
        level='_date'
    ).fillna(0).astype(int)
    
    movement_pivot = movements.unstack(
        level='_date'
    ).fillna(0).astype(int)

    return sales_pivot, movement_pivot, prices

def _combine_prices(prices_from_stock: pd.DataFrame, prices_from_sales: pd.DataFrame) -> pd.DataFrame:
    """Combine prices from stock and sales sources using mean."""
    
    prices = pd.concat(
        [prices_from_stock, prices_from_sales],
        axis=1, sort=True
    ).mean(axis=1)
    
    return prices

def _calculate_cumulative_stock(stock: pd.DataFrame, movement: pd.DataFrame):
    """Joins previous stock with monthly movement and calculates the cumulative sum in reverse order."""

    # For reverse calculation: invert signs and reverse cumsum direction
    inverted_movement = -movement  # Invert signs for reverse calculation
    reversed_movement = inverted_movement.sort_index(axis=1, ascending=False)

    if not isinstance(stock.columns, pd.DatetimeIndex):
        latest_date = reversed_movement.columns[0]
        stock_date = latest_date + pd.Timedelta(days=1)
        stock = stock.rename(columns={stock.columns[0]: stock_date})

    cumulative_stock = stock.join(reversed_movement, how="outer").fillna(0)
    # Apply cumsum in reverse direction (from right to left)
    cumulative_stock = cumulative_stock.cumsum(axis=1)


    return cumulative_stock

def _split_stock(cumulative_stock):
    """Splits the stock into the current month's view and the end-of-month carry-over stock."""
    # немного неочевидно с датами: cumulative_stock содержит остатки на конец каждого дня,
    # крайне правое значение - первое число предыдущего месяца,
    # поэтому для получения остатков на конец месяца нужно взять крайне правое значение
    # и для получения остатков за фактический месяц нужно взять все значения, кроме крайне правого
    # хоть это по дате и часть месяца
    
    # сдвинем на один день вперед, чтобы получить остатки на начало каждого дня
    last_stock = cumulative_stock.iloc[:, -1:]
    monthly_stock = cumulative_stock.iloc[:, :-1]
    monthly_stock.columns = monthly_stock.columns - pd.Timedelta(days=1)

    # гарантируем, что last_stock указывает на начало месяца
    last_stock.columns = last_stock.columns.to_period('M').start_time

    
    return last_stock, monthly_stock

def _calculate_availability_and_confidence(monthly_stock: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Рассчитывает долю наличия и уверенность в данных, возвращая Series."""
    
    in_stock = monthly_stock.clip(0, 1)
    availability = in_stock.mean(1)
    confidence = (monthly_stock.clip(0, 5) / 5).mean(1)
    availability_mask = in_stock > 0
    
    
    return availability, confidence, availability_mask

def _calculate_masked_sales(
        sales: pd.DataFrame, 
        availability_mask: pd.DataFrame, 
        prices: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series]:
    """Рассчитывает средние продажи с учетом наличия и их стоимость, возвращая Series."""
    
    availability_mask = availability_mask.reindex(sales.index)
    masked_mean_sales_items = (
        sales
        .where(availability_mask)
        .mean(axis=1, skipna=True)
        .fillna(0)
    )
    try:
        prices_for_index = prices.loc[masked_mean_sales_items.index]
    except KeyError:
        prices_for_index = pd.Series(0, index=masked_mean_sales_items.index)

    masked_mean_sales_rub = (
        masked_mean_sales_items
        .mul(prices_for_index, axis=0)
        .fillna(0)
        .astype(int)
    )
    
    
    return masked_mean_sales_items, masked_mean_sales_rub

def _calculate_lost_sales(
        availability_mask: pd.DataFrame, 
        masked_mean_sales_rub: pd.Series
    ) -> pd.Series:
    """Рассчитывает упущенные продажи, возвращая Series."""
    
    availability_mask = availability_mask.reindex(
        masked_mean_sales_rub.index
    )
    lost_sales = (
        (~availability_mask)
        .astype(int)
        .sum(axis=1, skipna=True)
        .mul(
            masked_mean_sales_rub
            .squeeze()
        )
        .fillna(0)
    )
    
    return lost_sales

def _gather_base_features(
        base_features_lists: dict[str, list[pd.Series | pd.DataFrame]],
    ) -> dict[str, pd.Series | pd.DataFrame]:
    """Gather base features from artifact pool."""
    
    def _process_is_time_agnostic(
        _feature_df: pd.Series | pd.DataFrame
    ) -> pd.Series | pd.DataFrame:
        return _feature_df.mean(axis=1).fillna(0).astype(int)
    
    def _process_time_dependent(
        _feature_df: pd.Series | pd.DataFrame
    ) -> pd.Series | pd.DataFrame:
        return (
            _feature_df
            .groupby(
                _feature_df
                .columns
                .to_period('M')
                .start_time, 
                axis=1
            )
            .agg('sum')
            .fillna(0)
        )
    
    def _get_correct_processing_fn(
        feature_df: pd.Series | pd.DataFrame
    ) -> Callable:
        if isinstance(
            feature_df.columns, pd.DatetimeIndex
        ):
            return _process_time_dependent
        else:
            return _process_is_time_agnostic
    
    base_features_dict = {}
    for feature_name, feature_list in base_features_lists.items():
        if feature_list:
            feature_df = pd.concat(
                feature_list, 
                axis=1
            )
            processing_fn = _get_correct_processing_fn(feature_df)
            base_features_dict[feature_name] = processing_fn(feature_df)
    return base_features_dict

def run_data_processing_pipeline(config: dict, initial_inputs: dict) -> dict:
    """ 
    Runs a declarative, three-stage data processing pipeline with built-in logic 
    for base and report feature aggregation.
    """
    artifact_pool = initial_inputs.copy()
    base_features_names = artifact_pool.get('base_features_names', [])
    report_features_names = artifact_pool.get('report_features_names', [])
    base_features_lists = {f: [] for f in base_features_names}
    artifact_pool['base_features_lists'] = base_features_lists
    artifact_pool['pipeline_outputs'] = None
    monthly_reports_list = []

    def _apply_step(step_config: dict):
        func = step_config['func']
        inputs = {key: artifact_pool.get(key) for key in step_config.get('inputs', [])}
        output_names = step_config.get('outputs', [])
        results = func(**inputs)
        if not isinstance(results, tuple):
            results = (results,)
        for name, value in zip(output_names, results):
            artifact_pool[name] = value

    # --- 1. Init Stage ---
    logger.info("=== INIT STAGE ===")
    for step_name, step_config in config.get('init_steps', {}).items():
        logger.info(f"Running init step: {step_name}")
        _apply_step(step_config)

    # --- 2. Iterative Steps Stage ---
    logger.info("=== ITERATIVE STEPS STAGE ===")
    sorted_sales_dfs = artifact_pool.get('sorted_sales_dfs', {})
    
    for month_date, processed_df in sorted_sales_dfs.items():
        logger.info(f"Processing month: {month_date}")
        artifact_pool['processed_df'] = processed_df
        artifact_pool['month_date'] = month_date
        
        for step_name, step_config in config.get('steps', {}).items():
            logger.info(f"Running step: {step_name}")
            _apply_step(step_config)

        # --- Built-in Aggregation Logic ---
        for key in base_features_names:
            if key in artifact_pool:
                base_features_lists[key].append(
                    artifact_pool[key]
                )

        series_list = [
            artifact_pool[name].rename(name) 
            for name in report_features_names 
            if name in artifact_pool 
            and isinstance(artifact_pool[name], pd.Series)
        ]
        if series_list:
            monthly_report_df = pd.concat(series_list, axis=1)
            monthly_report_df = monthly_report_df.assign(_date=month_date)
            monthly_reports_list.append(monthly_report_df.reset_index())

    # --- 3. Closure Stage ---
    logger.info("=== CLOSURE STAGE ===")
    # --- Built-in Report Aggregation Logic ---
    report_features_df = None
    if monthly_reports_list:
        report_features_df = pd.concat(
            monthly_reports_list, 
            axis=0, 
            ignore_index=True
        )
        final_index_cols = ['_date'] + [
            key for key in GROUP_KEYS 
            if key in report_features_df.columns
        ]
        report_features_df = report_features_df.set_index(
            final_index_cols
        ).fillna(0).abs()
    artifact_pool['report_features'] = report_features_df

    # --- Built-in Base Features Aggregation Logic ---
    artifact_pool['base_features_dict'] = _gather_base_features(
        base_features_lists
    )

    # --- Dynamic closure Logic ---
    for step_name, step_config in config.get('closure', {}).items():
        logger.info(f"Running closure step: {step_name}")
        _apply_step(step_config)
        
    return artifact_pool.get('pipeline_outputs')

def process_data(
    stock_path: str, 
    sales_path: str, 
    bins: pd.IntervalIndex | None = None,
    base_features_names: list[str] = [],
    report_features_names: list[str] = []
) -> dict[str, pd.DataFrame]:
    
    if not base_features_names:
        base_features_names = BASE_FEATURES_DEFAULT
    if not report_features_names:
        report_features_names = REPORT_FEATURES_DEFAULT
    
    initial_inputs = {
        'stock_path': stock_path,
        'sales_path': sales_path,
        'bins': bins,
        'all_keys': ["created_date", "sold_date", *GROUP_KEYS],
        'group_keys': ["created_date", "sold_date", *GROUP_KEYS],
        'keys_no_dates': GROUP_KEYS,
        'base_features_names': base_features_names,
        'report_features_names': report_features_names,
    }
    
    final_artifacts = run_data_processing_pipeline(UNIFIED_PIPELINE_CONFIG, initial_inputs)
    
    base_features_dict = final_artifacts.get('base_features_dict', {})
    report_features = final_artifacts.get('report_features', None)

    if base_features_dict:
        for feature_name, df_pivot in base_features_dict.items():
            if isinstance(df_pivot, pd.DataFrame):
                df_pivot.columns.name = '_date'
                base_features_dict[feature_name] = df_pivot

    return {
        **base_features_dict,
        'report_features': report_features
    }


UNIFIED_PIPELINE_CONFIG = {
    'init_steps': OrderedDict([
        ('_get_sales_files_paths', {
            'func': _get_sales_files_paths,
            'inputs': ['sales_path'],
            'outputs': ['sales_files_paths']
        }),
        ('_preprocess_sort_sales', {
            'func': _preprocess_sort_sales,
            'inputs': ['sales_files_paths', 'bins'],
            'outputs': ['sorted_sales_dfs']
        }),
        ('_extract_cutoff_date', {
            'func': (
                lambda sorted_sales_dfs: 
                list(
                    sorted_sales_dfs.keys()
                )[-1]
                .to_period('M')
                .start_time
                .floor("D") - pd.Timedelta(days=1)
            ),
            'inputs': ['sorted_sales_dfs'],
            'outputs': ['cutoff_date']
        }),
        ('_init_stock', {
            'func': _init_stock,
            'inputs': ['stock_path', 'all_keys', 'cutoff_date', 'bins'],
            'outputs': ['stock', 'prices_from_stock']
        }),
    ]),
    'steps': OrderedDict([
        ('_get_grouped_df', {
            'func': _get_grouped_df,
            'inputs': ['processed_df', 'group_keys'],
            'outputs': ['grouped_df']
        }),
        ('_process_grouped_df', {
            'func': _process_grouped_df,
            'inputs': ['grouped_df', 'all_keys', 'keys_no_dates', 'cutoff_date'],
            'outputs': ['sales', 'movement', 'prices_from_sales']
        }),
        ('_combine_prices', {
            'func': _combine_prices,
            'inputs': ['prices_from_stock', 'prices_from_sales'],
            'outputs': ['prices']
        }),
        ('_calculate_cumulative_stock', {
            'func': _calculate_cumulative_stock,
            'inputs': ['stock', 'movement'],
            'outputs': ['cumulative_stock']
        }),
        ('_split_stock', {
            'func': _split_stock,
            'inputs': ['cumulative_stock'],
            'outputs': ['stock', 'monthly_stock']
        }),
        ('_calculate_availability_confidence', {
            'func': _calculate_availability_and_confidence,
            'inputs': ['monthly_stock'],
            'outputs': ['availability', 'confidence', 'availability_mask']
        }),
        ('_calculate_masked_sales', {
            'func': _calculate_masked_sales,
            'inputs': ['sales', 'availability_mask', 'prices'],
            'outputs': ['masked_mean_sales_items', 'masked_mean_sales_rub']
        }),
        ('_calculate_lost_sales', {
            'func': _calculate_lost_sales,
            'inputs': ['availability_mask', 'masked_mean_sales_rub'],
            'outputs': ['lost_sales']
        }),
    ]),
    'closure': OrderedDict([
        ('package_outputs', {
            'func': lambda base_features_dict, report_features: {
                'base_features_dict': base_features_dict,
                'report_features': report_features
            },
            'inputs': ['base_features_dict', 'report_features'],
            'outputs': ['pipeline_outputs']
        })
    ])
}


def ensure_monthly_regular_index(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = df.index.to_timestamp()
        except Exception:
            raise ValueError("Index must be DatetimeIndex")
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="MS")
    return df.reindex(full_range, fill_value=0)


def transform_months(series):
    msin = np.sin(2 * np.pi * series / 12).values
    mcos = np.cos(2 * np.pi * series / 12).values

    return msin, mcos

def get_monthly_sales_pivot(monthly_sales_df):
    monthly_sales_pivot = monthly_sales_df.groupby(
        monthly_sales_df.index.get_level_values("_date").to_period("M")
    ).agg('sum')
    monthly_sales_pivot = monthly_sales_pivot.abs()
    monthly_sales_pivot = monthly_sales_pivot.sort_index(axis=1)

    return monthly_sales_pivot

def get_past_covariates_df(monthly_sales_df, stock_features, feature_list, span):
    boolean_mask = stock_features.T
    boolean_mask = boolean_mask[:, 0].astype(bool)
    monthly_sales_df = monthly_sales_df.T
    masked_data = monthly_sales_df.where(boolean_mask)

    pc_df = masked_data.reset_index(
        [
            i
            for i in masked_data.index.names
            if i not in [*feature_list, "recording_decade"]
        ],
        drop=True,
    )
    pc_df = pc_df.reset_index()

    grouped = pc_df.groupby([*feature_list, "recording_decade"])

    pc_df_ema = (
        grouped.apply(
            lambda x: x.iloc[:, 2:].mean(axis=0).ewm(span=span, adjust=False).mean()
        )
        .bfill(axis=1, limit=3)
        .fillna(0)
    )
    pc_df_ema["aggregation"] = "exponentially_weighted_moving_average"
    pc_df_ema = pc_df_ema.set_index("aggregation", append=True)

    pc_df_emv = (
        grouped.apply(
            lambda x: x.iloc[:, 2:].mean(axis=0).ewm(span=span, adjust=False).var()
        )
        .bfill(axis=1, limit=3)
        .fillna(0)
    )
    pc_df_emv["aggregation"] = "exponentially_weighted_moving_variance"
    pc_df_emv = pc_df_emv.set_index("aggregation", append=True)

    return pd.concat([pc_df_ema, pc_df_emv], axis=0).T
