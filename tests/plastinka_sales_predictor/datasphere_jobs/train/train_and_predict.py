
    def test_load_datasets_success(
        self, monkeypatch, mock_inference_dataset_class, mock_training_dataset_class, mock_validate_file
    ):
        """Test successful dataset loading."""
        # Arrange
        monkeypatch.setattr(
            "plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.validate_dataset_file",
            mock_validate_file,
        )
        monkeypatch.setattr(
            "plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.PlastinkaTrainingTSDataset",
            mock_training_dataset_class,
        )
        monkeypatch.setattr(
            "plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.PlastinkaInferenceTSDataset",
            mock_inference_dataset_class,
        )
        input_dir = "/test/input"
        mock_train_dataset = MagicMock()
        mock_inference_dataset = MagicMock()
        
        # Configure the class methods to return different datasets
        mock_training_dataset_class.from_dill.return_value = mock_train_dataset
        mock_inference_dataset_class.from_dill.return_value = mock_inference_dataset

        # Act
        train_ds, inference_ds = train_and_predict.load_datasets(input_dir)

        # Assert
        assert train_ds == mock_train_dataset
        assert inference_ds == mock_inference_dataset

        # Verify validation calls (now 2 files)
        assert mock_validate_file.call_count == 2

        # Verify dataset loading calls
        expected_train_path = os.path.join(input_dir, "train.dill")
        expected_inference_path = os.path.join(input_dir, "inference.dill")
        
        mock_training_dataset_class.from_dill.assert_called_once_with(expected_train_path)
        mock_inference_dataset_class.from_dill.assert_called_once_with(expected_inference_path)
