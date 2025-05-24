import pytest
import os
import sqlite3
from pathlib import Path
import tempfile
import sys
import importlib

def test_sqlite3_schema_directly():
    """
    Тест для проверки проблемы со схемой SQLite напрямую, без pyfakefs.
    """
    # Создаем временный файл для базы данных
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp_file:
        db_path = tmp_file.name
    
    try:
        # Подключаемся к БД
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Сначала попробуем выполнить схему по частям, чтобы найти проблемное место
        
        # Шаг 1: Создаем таблицу parameter_sets
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS parameter_sets (
            parameter_set_id TEXT PRIMARY KEY,
            parameters TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            is_active BOOLEAN DEFAULT 0
        )
        """)
        conn.commit()
        
        # Проверяем, что таблица создана
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='parameter_sets'")
        assert cursor.fetchone() is not None, "parameter_sets не была создана"
        
        # Шаг 2: Создаем таблицу training_results с внешним ключом
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_results (
            result_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            model_id TEXT,
            parameter_set_id TEXT,
            metrics TEXT,
            parameters TEXT,
            duration INTEGER,
            FOREIGN KEY (parameter_set_id) REFERENCES parameter_sets(parameter_set_id)
        )
        """)
        conn.commit()
        
        # Проверяем, что таблица создана
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='training_results'")
        assert cursor.fetchone() is not None, "training_results не была создана"
        
        # Шаг 3: Создаем индекс на parameter_set_id
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_training_results_parameter_set ON training_results(parameter_set_id)
        """)
        conn.commit()
        
        # Проверяем, что индекс создан
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_training_results_parameter_set'")
        assert cursor.fetchone() is not None, "Индекс не был создан"
        
        # Теперь импортируем и запускаем SCHEMA_SQL напрямую
        # Это нужно для проверки, не является ли проблема в том, как мы используем executescript
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from deployment.app.db.schema import SCHEMA_SQL
        
        # Создаем новое соединение с чистой базой
        conn.close()
        os.unlink(db_path)
        
        # Открываем новую базу
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Выполняем полную схему целиком
        try:
            cursor.executescript(SCHEMA_SQL)
            conn.commit()
            print("SCHEMA_SQL выполнена успешно")
        except sqlite3.OperationalError as e:
            # Если ошибка, возьмем часть SCHEMA_SQL до ошибки и попробуем выполнить ее
            error_msg = str(e)
            print(f"Ошибка при выполнении полной SCHEMA_SQL: {error_msg}")
            
            # Разбиваем SCHEMA_SQL на отдельные операторы
            statements = SCHEMA_SQL.split(';')
            for i, stmt in enumerate(statements):
                if not stmt.strip():
                    continue
                try:
                    print(f"Выполняю оператор {i+1}")
                    cursor.execute(stmt)
                    conn.commit()
                except sqlite3.OperationalError as e2:
                    print(f"Ошибка в операторе {i+1}: {str(e2)}")
                    print(f"Проблемный SQL: {stmt}")
                    break
            
            # Тест должен упасть с ошибкой
            assert False, f"SCHEMA_SQL не может быть выполнена из-за: {error_msg}"
    
    finally:
        # Закрываем соединение и удаляем файл
        if 'conn' in locals() and conn:
            conn.close()
        if os.path.exists(db_path):
            os.unlink(db_path) 