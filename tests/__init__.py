"""
plastinka_sales_predictior: Общая стратегия тестирования

- Все тесты изолированы с помощью pytest fixtures (scope='function' по умолчанию).
- Для работы с БД используется PostgreSQL с тестовым пулом соединений, схема применяется автоматически через фикстуры.
- Для файловой системы используется pyfakefs (scope='function'), с патчингом aiofiles для асинхронных операций.
- Все внешние сервисы (DataSphere, DAL, FastAPI endpoints) мокируются через monkeypatch и MagicMock.
- Для интеграционных тестов используются реальные временные директории и файлы, cleanup гарантируется.
- Все тесты следуют Arrange-Act-Assert, используют keyword arguments в mock-assertions.
- Документация по best practices и антипаттернам — см. testing_guidelines.mdc и deploy_testing.
"""
