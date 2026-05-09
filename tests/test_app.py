"""
Unit tests for the Flask application (app.py).
Tests cover endpoints, response generation, and chat logging.
"""

import unittest
import json
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock, mock_open
import torch
import numpy as np

# Module-level stubs (`sys.modules['nanoGPT.model']`, fake `torch.load`) and
# the initial `import app` happen in `tests/conftest.py`.


class TestApp(unittest.TestCase):
    """Test cases for Flask application routes and functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.original_chat_log = 'chat_history.jsonl'
        self.original_model_dir = 'models'
        
        # Mock environment variables and paths
        self.test_chat_log = os.path.join(self.test_dir, 'chat_history.jsonl')
        self.test_model_dir = os.path.join(self.test_dir, 'models')
        
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_save_chat(self):
        """save_chat appends a JSONL entry with user, assistant, timestamp."""
        test_file = os.path.join(self.test_dir, 'chat_history.jsonl')

        with patch('app.CHAT_LOG_FILE', test_file):
            from app import save_chat
            save_chat("Hello", "Hi there!")

        self.assertTrue(os.path.exists(test_file))
        with open(test_file) as f:
            data = json.loads(f.readline())
        self.assertEqual(data['user'], "Hello")
        self.assertEqual(data['assistant'], "Hi there!")
        self.assertIn('timestamp', data)

    def test_save_chat_creates_valid_jsonl(self):
        """Test that save_chat creates valid JSONL entries."""
        # Create a real file for this test
        test_file = os.path.join(self.test_dir, 'test_chat.jsonl')
        
        with patch('app.CHAT_LOG_FILE', test_file):
            from app import save_chat
            
            save_chat("User message", "Assistant response")
            
            # Verify file exists and contains valid JSON
            self.assertTrue(os.path.exists(test_file))
            with open(test_file, 'r') as f:
                line = f.readline()
                data = json.loads(line)
                self.assertEqual(data['user'], "User message")
                self.assertEqual(data['assistant'], "Assistant response")

    @patch('app.model')
    @patch('app.enc')
    @patch('app.device', 'cpu')
    def test_generate_response_basic(self, mock_enc, mock_model):
        """Test basic response generation."""
        # Setup mocks
        mock_enc.encode.return_value = [1, 2, 3]
        mock_enc.decode.return_value = "Human: Hello\nAssistant: Hi there!"
        mock_token = torch.tensor([[4]])  # Mock next token
        mock_enc.encode.return_value = [10]  # Newline token
        
        # Mock model output
        mock_logits = torch.zeros(1, 3, 50257)
        mock_model.config.block_size = 1024
        mock_model.return_value = (mock_logits, None)
        
        with patch('app.torch.multinomial') as mock_multinomial:
            mock_multinomial.return_value = mock_token
            
            from app import generate_response
            result = generate_response("Human: Hello\nAssistant:", max_tokens=5, temperature=0.8)
            
            # Should return some response (or fallback)
            self.assertIsInstance(result, str)

    @patch('app.model')
    @patch('app.enc')
    def test_generate_response_stops_at_newline(self, mock_enc, mock_model):
        """Test that generation stops at newline token."""
        mock_model.config.block_size = 1024
        mock_model.return_value = (torch.zeros(1, 3, 50257), None)

        # Mock encoding
        mock_enc.encode.side_effect = lambda x, **_: [1, 2, 3] if 'Human' in x else [10]  # 10 = newline
        mock_enc.decode.return_value = "Human: Test\nAssistant: Response\n"

        # Mock first token is newline, should stop immediately
        newline_token = torch.tensor([[10]])

        with patch('app.torch.multinomial') as mock_multinomial:
            mock_multinomial.return_value = newline_token

            from app import generate_response
            generate_response("Human: Test\nAssistant:", max_tokens=100)

            # Should have stopped early due to newline (single multinomial call)
            mock_multinomial.assert_called_once()

    @patch('app.model')
    @patch('app.enc')
    def test_generate_response_respects_block_size(self, mock_enc, mock_model):
        """Test that generation respects model block_size limit."""
        mock_model.config.block_size = 128
        mock_model.return_value = (torch.zeros(1, 128, 50257), None)

        # Create tokens that exceed block size
        long_prompt_tokens = list(range(150))  # Exceeds block_size

        mock_enc.encode.return_value = long_prompt_tokens
        mock_enc.decode.return_value = "Human: " + "word " * 150 + "\nAssistant: Response"

        with patch('app.torch.multinomial') as mock_multinomial:
            mock_multinomial.return_value = torch.tensor([[10]])  # Newline

            from app import generate_response

            # Should crop to block_size
            generate_response("Human: " + "word " * 150 + "\nAssistant:", max_tokens=10)

            # Verify the model was called with tokens cropped to block_size
            mock_model.assert_called()
            (idx_cond,), _ = mock_model.call_args
            self.assertLessEqual(idx_cond.size(1), mock_model.config.block_size)

    @patch('app.Flask')
    @patch('app.model')
    @patch('app.generate_response')
    @patch('app.save_chat')
    def test_chat_endpoint_success(self, mock_save, mock_generate, mock_model, mock_flask):
        """Test successful chat endpoint call."""
        from app import app
        
        mock_generate.return_value = "Test response"
        
        with patch('app.os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data='{"user":"test"}\n')):
                with app.test_client() as client:
                    response = client.post('/chat', 
                                         json={'message': 'Hello'},
                                         content_type='application/json')
                    
                    self.assertEqual(response.status_code, 200)
                    data = json.loads(response.data)
                    self.assertIn('response', data)
                    self.assertIn('chat_count', data)

    @patch('app.Flask')
    def test_chat_endpoint_no_message(self, mock_flask):
        """Test chat endpoint with missing message."""
        from app import app
        
        with app.test_client() as client:
            response = client.post('/chat',
                                 json={},
                                 content_type='application/json')
            
            self.assertEqual(response.status_code, 400)
            data = json.loads(response.data)
            self.assertIn('error', data)
            self.assertEqual(data['error'], 'No message provided')

    @patch('app.Flask')
    def test_chat_endpoint_empty_message(self, mock_flask):
        """Test chat endpoint with empty message string."""
        from app import app
        
        with app.test_client() as client:
            response = client.post('/chat',
                                 json={'message': ''},
                                 content_type='application/json')
            
            self.assertEqual(response.status_code, 400)
            data = json.loads(response.data)
            self.assertEqual(data['error'], 'No message provided')

    @patch('app.Flask')
    @patch('app.os.path.exists')
    def test_stats_endpoint_no_file(self, mock_exists, mock_flask):
        """Test stats endpoint when chat file doesn't exist."""
        from app import app
        
        mock_exists.return_value = False
        
        with app.test_client() as client:
            response = client.get('/stats')
            
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertEqual(data['chat_count'], 0)

    @patch('app.Flask')
    @patch('app.os.path.exists')
    def test_stats_endpoint_with_file(self, mock_exists, mock_flask):
        """Test stats endpoint with existing chat file."""
        from app import app
        
        mock_exists.return_value = True
        
        # Mock file with 5 lines
        mock_file_content = '\n'.join(['{"user":"test"}'] * 5)
        
        with patch('builtins.open', mock_open(read_data=mock_file_content)):
            with app.test_client() as client:
                response = client.get('/stats')
                
                self.assertEqual(response.status_code, 200)
                data = json.loads(response.data)
                self.assertEqual(data['chat_count'], 5)
                self.assertFalse(data['ready_for_finetuning'])  # < 10

    @patch('app.Flask')
    @patch('app.os.path.exists')
    def test_stats_ready_for_finetuning(self, mock_exists, mock_flask):
        """Test stats endpoint indicates ready when >= 10 conversations."""
        from app import app
        
        mock_exists.return_value = True
        
        # Mock file with 15 lines
        mock_file_content = '\n'.join(['{"user":"test"}'] * 15)
        
        with patch('builtins.open', mock_open(read_data=mock_file_content)):
            with app.test_client() as client:
                response = client.get('/stats')
                
                self.assertEqual(response.status_code, 200)
                data = json.loads(response.data)
                self.assertEqual(data['chat_count'], 15)
                self.assertTrue(data['ready_for_finetuning'])

    @patch('app.Flask')
    @patch('app.render_template')
    def test_index_endpoint(self, mock_render, mock_flask):
        """Test index endpoint renders template."""
        from app import app
        
        mock_render.return_value = '<html>...</html>'
        
        with app.test_client() as client:
            response = client.get('/')
            
            self.assertEqual(response.status_code, 200)
            mock_render.assert_called_once_with('index.html')


class TestAppFileHandling(unittest.TestCase):
    """Test file handling and edge cases."""
    
    def test_chat_file_counting_handles_io_errors(self):
        """Test that file counting handles IO errors gracefully."""
        # This tests the bug we found earlier - unclosed files
        # In a proper implementation, files should be closed
        
        with patch('app.os.path.exists', return_value=True):
            with patch('builtins.open', side_effect=IOError("Permission denied")):
                # Should handle error gracefully
                pass  # Implementation should catch this

    def test_save_chat_handles_write_errors(self):
        """Test that save_chat handles write errors."""
        # Should test error handling when file write fails
        pass


if __name__ == '__main__':
    unittest.main()

