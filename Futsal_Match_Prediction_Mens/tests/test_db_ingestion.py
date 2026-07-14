import pytest
from unittest.mock import patch, MagicMock
import os
import pandas as pd
from db.db_connection import DB
from components.data_ingestion import DataIngestion

def test_tunnel_start_success():
    db = DB()
    with patch("db.db_connection.os.path.exists", return_value=True):
        with patch("db.db_connection.SSHTunnelForwarder") as mock_ssh:
            mock_tunnel = MagicMock()
            mock_ssh.return_value = mock_tunnel
            
            result = db.db_test_connection()
            
            assert result is True
            mock_tunnel.start.assert_called_once()
            assert db.tunnel == mock_tunnel

def test_tunnel_start_failure():
    db = DB()
    with patch("db.db_connection.os.path.exists", return_value=True):
        with patch("db.db_connection.SSHTunnelForwarder", side_effect=Exception("Timeout")):
            result = db.db_test_connection()
            assert result is False

def test_ssh_key_not_found():
    db = DB()
    with patch("db.db_connection.os.path.exists", return_value=False):
        result = db.db_test_connection()
        assert result is False
        assert db.tunnel is None

def test_engine_creation():
    db = DB()
    mock_tunnel = MagicMock()
    mock_tunnel.is_active = True
    mock_tunnel.local_bind_port = 3306
    db.tunnel = mock_tunnel
    
    with patch("db.db_connection.create_engine") as mock_create_engine:
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        result = db.db_create_engine()
        assert result == mock_engine

def test_engine_no_tunnel():
    db = DB()
    db.tunnel = None
    result = db.db_create_engine()
    assert result is None

def test_close_connection():
    db = DB()
    mock_tunnel = MagicMock()
    db.tunnel = mock_tunnel
    
    db.db_close_connection()
    mock_tunnel.stop.assert_called_once()

# --- Data Ingestion Tests ---
@pytest.fixture
def mock_db_connection():
    with patch("components.data_ingestion.DB") as mock_db_class:
        mock_db = MagicMock()
        mock_db.db_test_connection.return_value = True
        
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()
        
        # Setup fetchall to return some dummy data
        mock_result.fetchall.return_value = [(1, "A"), (2, "B")]
        mock_result.keys.return_value = ["id", "name"]
        
        mock_conn.execute.return_value = mock_result
        # Context manager for connection
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = False
        
        mock_engine.connect.return_value = mock_conn
        mock_db.db_create_engine.return_value = mock_engine
        
        mock_db_class.return_value = mock_db
        yield mock_db_class

def test_ingest_match_data(mock_db_connection):
    di = DataIngestion()
    
    df = di.ingest_match_data()
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "id" in df.columns
    assert "name" in df.columns

def test_ingest_player_data(mock_db_connection):
    di = DataIngestion()
    
    df = di.ingest_player_data()
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "id" in df.columns
    assert "name" in df.columns

def test_connection_failure_handling():
    with patch("components.data_ingestion.DB") as mock_db_class:
        mock_db = MagicMock()
        mock_db.db_test_connection.return_value = False
        mock_db_class.return_value = mock_db
        
        di = DataIngestion()
        assert di.engine is None
        
        with pytest.raises(ConnectionError):
            di.ingest_match_data()

def test_query_structure():
    with patch("components.data_ingestion.DB") as mock_db_class:
        mock_db = MagicMock()
        mock_db.db_test_connection.return_value = False
        mock_db_class.return_value = mock_db
        
        di = DataIngestion()
        
        assert "futsaloz.cmp_matches" in di.match_query
        assert "futsaloz.live_match_score_cards" in di.player_query

def test_comp_id_filter():
    with patch("components.data_ingestion.DB") as mock_db_class:
        mock_db = MagicMock()
        mock_db.db_test_connection.return_value = False
        mock_db_class.return_value = mock_db
        
        di = DataIngestion()
        
        assert "1028" in di.match_query
        assert "948" in di.match_query
        assert "1028" in di.player_query
